use core::fmt;

use super::{AccelUnbiased, Gravity, GyroUnbiased, ImuState};
use crate::{
    dtype,
    linalg::{Matrix, Matrix3, Numeric, SupersetOf, Vector3},
    variables::{ImuBias, MatrixLieGroup, SO3, Variable},
};

/// Struct to hold the preintegrated Imu delta
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub(crate) struct ImuDelta<T: Numeric = dtype> {
    pub(crate) dt: T,
    pub(crate) xi_theta: Vector3<T>,
    pub(crate) xi_vel: Vector3<T>,
    pub(crate) xi_pos: Vector3<T>,
    bias_init: ImuBias<T>,
    h_bias_accel: Matrix<9, 3, T>,
    h_bias_gyro: Matrix<9, 3, T>,
    gravity: Gravity<T>,
}

impl<T: Numeric> ImuDelta<T> {
    pub(crate) fn new(gravity: Gravity<T>, bias_init: ImuBias<T>) -> Self {
        Self {
            dt: T::from(0.0),
            xi_theta: Vector3::zeros(),
            xi_vel: Vector3::zeros(),
            xi_pos: Vector3::zeros(),
            bias_init,
            h_bias_accel: Matrix::zeros(),
            h_bias_gyro: Matrix::zeros(),
            gravity,
        }
    }

    pub fn bias_init(&self) -> &ImuBias<T> {
        &self.bias_init
    }

    pub(crate) fn first_order_update(&self, bias_diff: &ImuBias<T>, idx: usize) -> Vector3<T> {
        self.h_bias_accel.fixed_rows(idx) * bias_diff.accel()
            + self.h_bias_gyro.fixed_rows(idx) * bias_diff.gyro()
    }

    pub(crate) fn predict(&self, x1: &ImuState<T>) -> ImuState<T> {
        let ImuState { r, v, p, bias } = x1;

        // Compute first-order updates to preint with bias
        let bias_diff = bias - &self.bias_init;
        let xi_theta = self.xi_theta + self.first_order_update(&bias_diff, 0);
        let xi_v = self.xi_vel + self.first_order_update(&bias_diff, 3);
        let xi_p = self.xi_pos + self.first_order_update(&bias_diff, 6);

        // Estimate x2
        // R2_meas = R1 * exp(xi_theta)
        let r2_meas = r.compose(&SO3::exp(xi_theta.as_view()));

        // v2_meas = v + g * dt + R1 * xi_v
        let v2_meas = v + self.gravity.0 * self.dt + r.apply(xi_v.as_view());

        // p2_meas = p1 + v * dt + 0.5 * g * dt^2 + R1 * xi_p
        let p2_meas = p
            + v * self.dt
            + self.gravity.0 * self.dt * self.dt * T::from(0.5)
            + r.apply(xi_p.as_view());

        let b2_meas = bias.clone();

        ImuState {
            r: r2_meas,
            v: v2_meas,
            p: p2_meas,
            bias: b2_meas,
        }
    }

    #[allow(non_snake_case)]
    pub(crate) fn integrate(
        &mut self,
        gyro: &GyroUnbiased<T>,
        accel: &AccelUnbiased<T>,
        dt: T,
    ) -> Matrix<15, 15, T> {
        // Construct jacobian
        let A = self.A(gyro, accel, dt);

        // Setup
        self.dt += dt;
        let accel_world = SO3::exp(self.xi_theta.as_view()).apply(accel.0.as_view());
        let H = SO3::dexp_right(self.xi_theta.as_view());
        let Hinv = H.try_inverse().expect("Failed to invert H(theta)");

        // Integrate (make sure integration occurs in the correct order)
        self.xi_pos += self.xi_vel * dt + accel_world * (dt * dt * 0.5);
        self.xi_vel += accel_world * dt;
        self.xi_theta += Hinv * gyro.0 * dt;

        // Propagate H
        self.propagate_H(&A);

        A
    }

    #[allow(non_snake_case)]
    fn propagate_H(&mut self, A: &Matrix<15, 15, T>) {
        let Amini = A.fixed_view::<9, 9>(0, 0);
        // This should come from B, turns out it's identical as A
        let Bgyro = A.fixed_view::<9, 3>(0, 9);
        let Baccel = A.fixed_view::<9, 3>(0, 12);
        self.h_bias_gyro = Amini * self.h_bias_gyro + Bgyro;
        self.h_bias_accel = Amini * self.h_bias_accel + Baccel;
    }

    // TODO(Easton): Should this just be auto-diffed? Need to benchmark to see if
    // that's faster
    #[allow(non_snake_case)]
    fn A(&self, gyro: &GyroUnbiased<T>, accel: &AccelUnbiased<T>, dt: T) -> Matrix<15, 15, T> {
        let H = SO3::dexp_right(self.xi_theta.as_view());
        let Hinv = H.try_inverse().expect("Failed to invert H(theta)");
        let R: nalgebra::Matrix<
            T,
            nalgebra::Const<3>,
            nalgebra::Const<3>,
            nalgebra::ArrayStorage<T, 3, 3>,
        > = SO3::exp(self.xi_theta.as_view()).to_matrix();

        let mut A = Matrix::<15, 15, T>::identity();

        // First column (wrt theta)
        let M = Matrix3::<T>::identity() - SO3::hat(gyro.0.as_view()) * dt / T::from(2.0);
        A.fixed_view_mut::<3, 3>(0, 0).copy_from(&M);
        let mut M = -R * SO3::hat(accel.0.as_view()) * H * dt;
        A.fixed_view_mut::<3, 3>(3, 0).copy_from(&M);
        M *= dt / 2.0;
        A.fixed_view_mut::<3, 3>(6, 0).copy_from(&M);

        // Second column (wrt vel)
        let M = Matrix3::<T>::identity() * dt;
        A.fixed_view_mut::<3, 3>(6, 3).copy_from(&M);

        // Third column (wrt pos)

        // Fourth column (wrt gyro bias)
        let M = -Hinv * dt;
        A.fixed_view_mut::<3, 3>(0, 9).copy_from(&M);

        // Fifth column (wrt accel bias)
        let mut M = -R * dt;
        A.fixed_view_mut::<3, 3>(3, 12).copy_from(&M);
        M *= dt / 2.0;
        A.fixed_view_mut::<3, 3>(6, 12).copy_from(&M);

        A
    }

    pub fn cast<TT: Numeric + SupersetOf<T>>(&self) -> ImuDelta<TT> {
        ImuDelta {
            dt: TT::from_subset(&self.dt),
            xi_theta: self.xi_theta.cast(),
            xi_vel: self.xi_vel.cast(),
            xi_pos: self.xi_pos.cast(),
            bias_init: self.bias_init.cast(),
            h_bias_accel: self.h_bias_accel.cast(),
            h_bias_gyro: self.h_bias_gyro.cast(),
            gravity: Gravity(self.gravity.0.cast()),
        }
    }
}

impl<T: Numeric> fmt::Display for ImuDelta<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "ImuDelta(dt: {}, theta: {}, v: {}, p: {})",
            self.dt, self.xi_theta, self.xi_vel, self.xi_pos
        )
    }
}

#[cfg(test)]
mod test {
    use matrixcompare::{assert_matrix_eq, assert_scalar_eq};
    use nalgebra::Const;

    use super::*;
    use crate::{
        linalg::{Diff, DualVector, ForwardProp, Vector, VectorX},
        residuals::{Accel, Gyro},
        variables::VectorVar,
    };

    #[cfg(not(feature = "f32"))]
    const TOL: f64 = 1e-5;
    #[cfg(feature = "f32")]
    const TOL: f32 = 1e-3;

    // Helper function to integrate a constant gyro / accel for a given amount of
    // time
    fn integrate<T: Numeric>(
        gyro: &Gyro<T>,
        accel: &Accel<T>,
        bias: &ImuBias<T>,
        n: i32,
        t: dtype,
    ) -> ImuDelta<T> {
        let mut delta = ImuDelta::new(Gravity::up(), bias.clone());
        // Remove the bias
        let accel = accel.remove_bias(bias);
        let gyro = gyro.remove_bias(bias);
        let dt = T::from(t / n as dtype);

        for _ in 0..n {
            delta.integrate(&gyro, &accel, dt);
        }

        delta
    }

    // Test contast acceleration
    #[test]
    fn integrate_accel() {
        let a = 5.0;
        let t = 3.0;
        let n = 100;

        let accel = Accel(Vector3::new(a, 0.0, 0.0));
        let gyro = Gyro(Vector3::zeros());

        let delta = integrate(&gyro, &accel, &ImuBias::identity(), n, t);

        println!("Delta: {delta}");
        assert_scalar_eq!(delta.dt, t, comp = abs, tol = TOL);
        assert_matrix_eq!(delta.xi_vel, accel.0 * t, comp = abs, tol = TOL);
        assert_matrix_eq!(delta.xi_pos, accel.0 * t * t / 2.0, comp = abs, tol = TOL);
    }

    // Test constant angular velocity
    #[test]
    fn integrate_gyro() {
        let a = 5.0;
        let t = 3.0;
        let n = 100;

        let accel = Accel(Vector3::zeros());
        let gyro = Gyro(Vector3::new(0.0, 0.0, a));

        let delta = integrate(&gyro, &accel, &ImuBias::identity(), n, t);

        println!("Delta: {delta}");
        assert_scalar_eq!(delta.dt, t, comp = abs, tol = TOL);
        assert_matrix_eq!(delta.xi_theta, gyro.0 * t, comp = abs, tol = TOL);
    }

    // Test construction of state transtition matrix A
    #[test]
    fn make_a() {
        let dt = 0.1;
        let v = Vector::<15>::from_fn(|i, _| i as dtype / 10.0);
        let gyro = Gyro::new(3.0, 2.0, 1.0);
        let accel: Accel = Accel::new(1.0, 2.0, 3.0);

        fn delta_from_vec<T: Numeric>(v: Vector<15, T>) -> ImuDelta<T> {
            let xi_theta = v.fixed_rows::<3>(0).into_owned();
            let xi_vel = v.fixed_rows::<3>(3).into_owned();
            let xi_pos = v.fixed_rows::<3>(6).into_owned();
            let bias_init = ImuBias::new(
                Gyro(v.fixed_rows::<3>(9).into_owned()),
                Accel(v.fixed_rows::<3>(12).into_owned()),
            );
            ImuDelta {
                dt: T::from(0.0),
                xi_theta,
                xi_vel,
                xi_pos,
                bias_init,
                h_bias_accel: Matrix::zeros(),
                h_bias_gyro: Matrix::zeros(),
                gravity: Gravity::up(),
            }
        }

        let f = |v: VectorVar<15, DualVector<Const<15>>>| {
            // construct measurements
            let gyro = Gyro(gyro.0.map(|g| g.into()));
            let accel = Accel(accel.0.map(|a| a.into()));

            // make delta from vector
            let mut delta = delta_from_vec(v.0);

            // Integrate
            let gyro = gyro.remove_bias(&delta.bias_init);
            let accel = accel.remove_bias(&delta.bias_init);
            delta.integrate(&gyro, &accel, DualVector::<Const<15>>::from(dt));

            // Return the delta as a vector
            let mut out = VectorX::zeros(15);
            out.fixed_rows_mut::<3>(0).copy_from(&delta.xi_theta);
            out.fixed_rows_mut::<3>(3).copy_from(&delta.xi_vel);
            out.fixed_rows_mut::<3>(6).copy_from(&delta.xi_pos);
            out.fixed_rows_mut::<3>(9).copy_from(delta.bias_init.gyro());
            out.fixed_rows_mut::<3>(12)
                .copy_from(delta.bias_init.accel());

            out
        };

        // Make expected A
        let vv: VectorVar<15> = VectorVar::from(v);
        let a_exp = ForwardProp::<Const<15>>::jacobian_1(f, &vv).diff;

        // Make got A
        let delta = delta_from_vec(v);
        let gyro = gyro.remove_bias(&delta.bias_init);
        let accel = accel.remove_bias(&delta.bias_init);
        let a_got = delta.A(&gyro, &accel, dt);

        println!("A_exp: {a_exp:.4}");
        println!("A_got: {a_got:.4}");
        // First column is an approximation, loosen the tolerance on those
        assert_matrix_eq!(
            a_exp.fixed_view::<12, 3>(3, 0),
            a_got.fixed_view::<12, 3>(3, 0),
            comp = abs,
            tol = 1e-3
        );
        assert_matrix_eq!(
            a_exp.fixed_view::<15, 12>(0, 3),
            a_got.fixed_view::<15, 12>(0, 3),
            comp = abs,
            tol = TOL
        );
    }

    // Test we get correct jacobian H for bias propagation
    #[test]
    #[allow(non_snake_case)]
    fn propagate_h() {
        let t = 1.0;
        let n = 2;
        let accel = Accel::new(1.0, 2.0, 3.0);
        let gyro = Gyro::new(0.1, 0.2, 0.3);
        let bias = ImuBias::new(Gyro::new(1.0, 2.0, 3.0), Accel::new(0.1, 0.2, 0.3));

        // Compute the H matrix
        let delta = integrate(&gyro, &accel, &bias, n, t);
        let H_accel_got = delta.h_bias_accel;
        let H_gyro_got = delta.h_bias_gyro;

        // Compute the H matrix via forward prop
        let integrate_diff = |bias: ImuBias<DualVector<Const<6>>>| {
            let accel = Accel(accel.0.map(|a| a.into()));
            let gyro = Gyro(gyro.0.map(|g| g.into()));
            let delta = integrate(&gyro, &accel, &bias, n, t);
            let mut preint = VectorX::zeros(9);
            preint.fixed_rows_mut::<3>(0).copy_from(&delta.xi_theta);
            preint.fixed_rows_mut::<3>(3).copy_from(&delta.xi_vel);
            preint.fixed_rows_mut::<3>(6).copy_from(&delta.xi_pos);
            preint
        };
        let H_exp = ForwardProp::<Const<6>>::jacobian_1(integrate_diff, &bias).diff;
        let H_gyro_exp = H_exp.fixed_view::<9, 3>(0, 0);
        let H_accel_exp = H_exp.fixed_view::<9, 3>(0, 3);

        println!("H_accel_got: {H_accel_got:.4}");
        println!("H_accel_exp: {H_accel_exp:.4}");
        assert_matrix_eq!(H_accel_got, H_accel_exp, comp = abs, tol = TOL);

        println!("H_gyro_got: {H_gyro_got:.4}");
        println!("H_gyro_exp: {H_gyro_exp:.4}");
        // Skip top 3 rows, it's an approximation
        assert_matrix_eq!(
            H_gyro_got.fixed_view::<6, 3>(3, 0),
            H_gyro_exp.fixed_view::<6, 3>(3, 0),
            comp = abs,
            tol = TOL
        );
    }
}
