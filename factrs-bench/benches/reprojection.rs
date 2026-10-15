//! Camera calibration: views of a planar target, one reprojection factor per
//! target corner.
//!
//! A factor projects its corner through the view's pose (SE(3)) and a pinhole
//! camera with Brown-Conrady distortion (fx, fy, cx, cy, k1, k2, p1, p2, k3),
//! so it is differentiated over 15 tangent dimensions. That makes it far more
//! expensive to linearize than a pose-graph factor, and the factor evaluation,
//! not the sparse solve, dominates the optimization.

use std::hint::black_box;

use factrs::{
    assign_symbols,
    core::{Graph, LevenMarquardt, Values},
    dtype, fac,
    linalg::{Const, ForwardProp, Numeric, Vector, Vector3, VectorX},
    residuals::Residual2,
    traits::*,
    variables::{MatrixLieGroup, VectorVar, SE3, SO3},
};

type Camera = VectorVar<9>;
assign_symbols!(K: Camera; X: SE3);

/// Views of the target: (views, corners per row, corners per column).
const SIZES: [(usize, usize, usize); 3] = [(10, 9, 6), (20, 11, 8), (40, 13, 9)];

#[derive(Clone, Debug)]
pub struct Reprojection {
    corner: Vector3,
    pixel: [dtype; 2],
}

/// Pixel of a target point `p` (in the camera frame) through the camera `k`.
fn project<T: Numeric>(k: &[T], p: &Vector3<T>) -> [T; 2] {
    let (x, y) = (p[0] / p[2], p[1] / p[2]);
    let (fx, fy, cx, cy) = (k[0], k[1], k[2], k[3]);
    let (k1, k2, p1, p2, k3) = (k[4], k[5], k[6], k[7], k[8]);
    let r2 = x * x + y * y;
    let radial = r2 * (k1 + r2 * (k2 + r2 * k3)) + 1.0;
    let xd = x * radial + x * y * p1 * 2.0 + p2 * (r2 + x * x * 2.0);
    let yd = y * radial + p1 * (r2 + y * y * 2.0) + x * y * p2 * 2.0;
    [fx * xd + cx, fy * yd + cy]
}

#[factrs::mark]
impl Residual2 for Reprojection {
    type Differ = ForwardProp<Const<15>>;
    type V1 = Camera;
    type V2 = SE3;
    type DimIn = Const<15>;
    type DimOut = Const<2>;

    fn residual2<T: Numeric>(&self, camera: VectorVar<9, T>, pose: SE3<T>) -> VectorX<T> {
        let p = pose.apply(self.corner.cast::<T>().as_view());
        let [u, v] = project(camera.0.as_slice(), &p);
        VectorX::from_vec(vec![u - self.pixel[0], v - self.pixel[1]])
    }
}

/// Target pose for view `i` of `n`: tilted up to ~30° about an axis that
/// turns with `i`, 0.5–0.7 m in front of the camera.
fn view_pose(i: usize, n: usize) -> SE3 {
    let a = i as dtype / n as dtype * std::f64::consts::TAU;
    let tilt = 0.25 + 0.25 * (i % 3) as dtype / 2.0;
    let axis = Vector3::new(a.cos() * tilt, a.sin() * tilt, 0.05 * a.cos());
    let xyz = Vector3::new(
        0.04 * a.sin(),
        -0.03 * a.cos(),
        0.5 + 0.2 * (i % 4) as dtype / 3.0,
    );
    SE3::from_rot_trans(SO3::exp(axis.as_view()), xyz)
}

/// The problem and a perturbed initial guess.
fn calibration(views: usize, cols: usize, rows: usize) -> (Graph, Values) {
    let truth = [
        1200.0, 1190.0, 640.0, 480.0, -0.12, 0.05, 0.001, -0.0005, 0.0,
    ];
    let mut graph = Graph::new();
    let mut values = Values::new();
    let guess = [1150.0, 1240.0, 620.0, 500.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    values.insert(K(0), Camera::from(Vector::<9>::from(guess)));
    let nudge = SE3::exp(VectorX::from_vec(vec![0.02, -0.01, 0.015, 0.01, -0.01, 0.02]).as_view());
    for i in 0..views {
        let pose = view_pose(i, views);
        for r in 0..rows {
            for c in 0..cols {
                let corner = Vector3::new(
                    0.03 * (c as dtype - (cols - 1) as dtype / 2.0),
                    0.03 * (r as dtype - (rows - 1) as dtype / 2.0),
                    0.0,
                );
                let pixel = project(&truth, &pose.apply(corner.as_view()));
                let residual = Reprojection { corner, pixel };
                graph.add_factor(fac![residual, (K(0), X(i as u32)), 0.5 as std]);
            }
        }
        values.insert(X(i as u32), pose.compose(&nudge));
    }
    (graph, values)
}

/// Levenberg-Marquardt from the perturbed guess to the true calibration.
#[divan::bench(args = SIZES)]
fn optimize(bencher: divan::Bencher, size: (usize, usize, usize)) {
    let (graph, init) = calibration(size.0, size.1, size.2);
    bencher.bench_local(|| {
        let mut opt: LevenMarquardt = LevenMarquardt::new_default(graph.clone());
        black_box(opt.optimize(init.clone()).unwrap());
    });
}

/// One linearization, once per iteration.
#[divan::bench(args = SIZES)]
fn linearize(bencher: divan::Bencher, size: (usize, usize, usize)) {
    let (graph, init) = calibration(size.0, size.1, size.2);
    bencher.bench_local(|| black_box(graph.linearize(&init)));
}

/// One error evaluation, once per step the optimizer tries.
#[divan::bench(args = SIZES)]
fn error(bencher: divan::Bencher, size: (usize, usize, usize)) {
    let (graph, init) = calibration(size.0, size.1, size.2);
    bencher.bench_local(|| black_box(graph.error(&init)));
}

fn main() {
    // Keep faer sequential, as in the g2o benchmark: with the `rayon`
    // feature, only the factor evaluation runs in parallel.
    faer::set_global_parallelism(faer::Par::Seq);
    divan::main();
}
