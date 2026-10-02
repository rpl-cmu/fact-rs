//! CodSpeed / divan benchmarks for the core factrs operations.
//!
//! Run locally with `cargo bench --bench factrs`, or with CodSpeed via
//! `cargo codspeed build && cargo codspeed run`.
use std::hint::black_box;

use factrs::{
    containers::ValuesOrder,
    core::{GaussNewton, Graph, LevenMarquardt, SE3, SO3, Values},
    linalg::{DiffResult, VectorX},
    linear::{CholeskySolver, LinearSolver},
    optimizers::{BaseOptParams, LevenParams},
    traits::{Optimizer, Variable},
    utils::load_g20,
};

const DATA_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/examples/data/");
const DATASETS: &[&str] = &["M3500.g2o", "sphere2500.g2o", "parking-garage.g2o"];

fn main() {
    // Keep everything single-threaded for reproducible measurements
    faer::set_global_parallelism(faer::Par::Seq);
    divan::main();
}

fn load(file: &str) -> (Graph, Values) {
    load_g20(&format!("{DATA_DIR}{file}"))
}

// ------------------------- Data loading ------------------------- //
#[divan::bench(args = DATASETS)]
fn load_g2o(file: &str) -> (Graph, Values) {
    load(black_box(file))
}

// ------------------------- Graph operations ------------------------- //
mod graph {
    use super::*;

    #[divan::bench(args = DATASETS)]
    fn error(bencher: divan::Bencher, file: &str) {
        let (graph, values) = load(file);
        bencher.bench_local(|| black_box(&graph).error(black_box(&values)));
    }

    #[divan::bench(args = DATASETS)]
    fn linearize(bencher: divan::Bencher, file: &str) {
        let (graph, values) = load(file);
        bencher.bench_local(|| black_box(&graph).linearize(black_box(&values)));
    }

    #[divan::bench(args = DATASETS)]
    fn sparsity_pattern(bencher: divan::Bencher, file: &str) {
        let (graph, values) = load(file);
        bencher.bench_local(|| {
            black_box(&graph).sparsity_pattern(ValuesOrder::from_values(black_box(&values)))
        });
    }

    /// Single linearization + sparse Cholesky solve, i.e. the core of one
    /// Gauss-Newton step.
    #[divan::bench(args = DATASETS)]
    fn linear_solve(bencher: divan::Bencher, file: &str) {
        let (graph, values) = load(file);
        let order = ValuesOrder::from_values(&values);
        let graph_order = graph.sparsity_pattern(order.clone());
        let linear_graph = graph.linearize(&values);
        bencher.bench_local(|| {
            let DiffResult { value: r, diff: j } = linear_graph.residual_jacobian(&graph_order);
            let mut solver = CholeskySolver::default();
            solver.solve_lst_sq(j.as_ref(), r.as_ref())
        });
    }
}

// ------------------------- Full optimization ------------------------- //
mod optimize {
    use super::*;

    // Fixed number of iterations so the work done is deterministic
    const MAX_ITERS: usize = 5;

    #[divan::bench(args = DATASETS)]
    fn gauss_newton(bencher: divan::Bencher, file: &str) {
        let (graph, values) = load(file);
        bencher
            .with_inputs(|| (graph.clone(), values.clone()))
            .bench_local_values(|(graph, values)| {
                let params = BaseOptParams {
                    max_iterations: MAX_ITERS,
                    error_tol_relative: 0.0,
                    error_tol_absolute: 0.0,
                    error_tol: 0.0,
                };
                let mut opt = GaussNewton::new(params, graph);
                opt.optimize(values)
            });
    }

    #[divan::bench(args = DATASETS)]
    fn levenberg_marquardt(bencher: divan::Bencher, file: &str) {
        let (graph, values) = load(file);
        bencher
            .with_inputs(|| (graph.clone(), values.clone()))
            .bench_local_values(|(graph, values)| {
                let mut params = LevenParams::default();
                params.base.max_iterations = MAX_ITERS;
                params.base.error_tol_relative = 0.0;
                params.base.error_tol_absolute = 0.0;
                params.base.error_tol = 0.0;
                let mut opt = LevenMarquardt::new(params, graph);
                opt.optimize(values)
            });
    }
}

// ------------------------- Lie group operations ------------------------- //
mod lie {
    use super::*;

    const N: usize = 1000;

    fn tangents(dim: usize) -> Vec<VectorX> {
        (0..N)
            .map(|i| {
                let t = i as factrs::dtype / N as factrs::dtype;
                VectorX::from_fn(dim, |j, _| (t + 0.1 * j as factrs::dtype).sin())
            })
            .collect()
    }

    #[divan::bench]
    fn so3_exp(bencher: divan::Bencher) {
        let xi = tangents(3);
        bencher.bench_local(|| {
            xi.iter()
                .map(|x| SO3::exp(black_box(x.as_view())))
                .collect::<Vec<_>>()
        });
    }

    #[divan::bench]
    fn se3_exp(bencher: divan::Bencher) {
        let xi = tangents(6);
        bencher.bench_local(|| {
            xi.iter()
                .map(|x| SE3::exp(black_box(x.as_view())))
                .collect::<Vec<_>>()
        });
    }

    #[divan::bench]
    fn se3_log(bencher: divan::Bencher) {
        let poses: Vec<SE3> = tangents(6).iter().map(|x| SE3::exp(x.as_view())).collect();
        bencher.bench_local(|| poses.iter().map(|p| black_box(p).log()).collect::<Vec<_>>());
    }

    #[divan::bench]
    fn se3_compose(bencher: divan::Bencher) {
        let poses: Vec<SE3> = tangents(6).iter().map(|x| SE3::exp(x.as_view())).collect();
        bencher.bench_local(|| {
            poses
                .windows(2)
                .map(|w| black_box(&w[0]).compose(black_box(&w[1])))
                .collect::<Vec<_>>()
        });
    }

    #[divan::bench]
    fn se3_ominus(bencher: divan::Bencher) {
        let poses: Vec<SE3> = tangents(6).iter().map(|x| SE3::exp(x.as_view())).collect();
        bencher.bench_local(|| {
            poses
                .windows(2)
                .map(|w| black_box(&w[0]).ominus(black_box(&w[1])))
                .collect::<Vec<_>>()
        });
    }
}
