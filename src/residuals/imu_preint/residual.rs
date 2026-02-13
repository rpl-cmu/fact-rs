use super::{delta::ImuDelta, Accel, Gravity, Gyro, ImuState};
use crate::{
    containers::{Factor, FactorBuilder, Symbol, TypedSymbol},
    dtype,
    linalg::{Const, ForwardProp, Matrix, Matrix3, VectorX},
    noise::GaussianNoise,
    residuals::Residual6,
    variables::{ImuBias, MatrixLieGroup, Variable, VectorVar3, SE3, SO3},
};
// ------------------------- Covariances ------------------------- //

/// Covariance parameters for the IMU preintegration
///
/// Tries to come with semi-reasonable defaults for the covariance parameters
/// ```
/// use factrs::residuals::ImuCovariance;
/// let cov = ImuCovariance::default();
/// ```
#[derive(Clone, Debug)]
pub struct ImuCovariance {
    pub cov_accel: Matrix3,
    pub cov_gyro: Matrix3,
    pub cov_accel_bias: Matrix3,
    pub cov_gyro_bias: Matrix3,
    pub cov_integration: Matrix3,
    pub cov_winit: Matrix3,
    pub cov_ainit: Matrix3,
}

/// Implements reasonable parameters for ImuCovariance
impl Default for ImuCovariance {
    fn default() -> Self {
        Self {
            cov_accel: Matrix3::identity() * 1e-5,
            cov_gyro: Matrix3::identity() * 1e-5,
            cov_accel_bias: Matrix3::identity() * 1e-6,
            cov_gyro_bias: Matrix3::identity() * 1e-6,
            cov_integration: Matrix3::identity() * 1e-7,
            cov_winit: Matrix3::identity() * 1e-7,
            cov_ainit: Matrix3::identity() * 1e-7,
        }
    }
}

impl ImuCovariance {
    /// Set $\epsilon_a$ covariance to a diagonal
    pub fn set_scalar_accel(&mut self, val: dtype) {
        self.cov_accel = Matrix3::identity() * val;
    }

    /// Set $\epsilon_\omega$ covariance to a diagonal
    pub fn set_scalar_gyro(&mut self, val: dtype) {
        self.cov_gyro = Matrix3::identity() * val;
    }

    /// Set $\epsilon_{a^b}$ covariance to a diagonal
    pub fn set_scalar_accel_bias(&mut self, val: dtype) {
        self.cov_accel_bias = Matrix3::identity() * val;
    }

    /// Set $\epsilon_{\omega^b}$ covariance to a diagonal
    pub fn set_scalar_gyro_bias(&mut self, val: dtype) {
        self.cov_gyro_bias = Matrix3::identity() * val;
    }

    /// Set $\epsilon_{int}$ covariance to a diagonal
    pub fn set_scalar_integration(&mut self, val: dtype) {
        self.cov_integration = Matrix3::identity() * val;
    }

    // In practice, I think everyone makes cov_winit = cov_ainit
    // For now, the public interface assumes they are the same, but behind the
    // scenes we're using both
    /// Set $\epsilon_{init}$ covariance to a diagonal
    pub fn set_scalar_init(&mut self, val: dtype) {
        self.cov_winit = Matrix3::identity() * val;
        self.cov_ainit = Matrix3::identity() * val;
    }
}

// ------------------------- The Preintegrator ------------------------- //
/// Performs Imu preintegration
///
/// This is the main entrypoint for the IMU preintegration functionality. See
/// [IMU Preintegration module](crate::residuals::imu_preint) for more details
/// on the theory.
/// ```
/// use factrs::residuals::{ImuPreintegrator, Accel, Gravity, Gyro, ImuCovariance};
/// use factrs::variables::{SE3, VectorVar3, ImuBias};
/// use factrs::assign_symbols;
///
/// assign_symbols!(X: SE3; V: VectorVar3; B: ImuBias);
///
/// let mut preint =
///     ImuPreintegrator::new(ImuCovariance::default(), ImuBias::zeros(), Gravity::up());
///
/// let accel = Accel::new(0.1, 0.2, 9.81);
/// let gyro = Gyro::new(0.1, 0.2, 0.3);
/// let dt = 0.01;
/// // Integrate measurements for 100 steps
/// for _ in 0..100 {
///    preint.integrate(&gyro, &accel, dt);
/// }
///
/// // Build the factor
/// let factor = preint.build(X(0), V(0), B(0), X(1), V(1), B(1));
/// ```
#[derive(Clone, Debug)]
pub struct ImuPreintegrator {
    // Mutable state that will change as we integrate
    delta: ImuDelta,
    cov: Matrix<15, 15>,
    // Constants
    params: ImuCovariance,
}

impl ImuPreintegrator {
    /// Construct a new ImuPreintegrator
    ///
    /// Requires the covariance parameters, initial bias, and gravity vector
    pub fn new(params: ImuCovariance, bias_init: ImuBias, gravity: Gravity) -> Self {
        let delta = ImuDelta::new(gravity, bias_init);
        Self {
            delta,
            // init with small value to avoid singular matrix
            cov: Matrix::identity() * 1e-14,
            params,
        }
    }

    // TODO: Test (make sure dts are all correct)
    #[allow(non_snake_case)]
    fn B_Q_BT(&self, dt: dtype) -> Matrix<15, 15> {
        let p = &self.params;
        let H = SO3::dexp(self.delta.xi_theta.as_view());
        let Hinv = H.try_inverse().expect("Failed to invert H(theta)");
        let R = SO3::exp(self.delta.xi_theta.as_view()).to_matrix();

        // Construct all partials
        let H_theta_w = Hinv * dt;
        let H_theta_winit = -Hinv * dt;

        let H_v_a = R * dt;
        let H_v_ainit = -R * dt;

        let H_p_a = H_v_a * dt / 2.0;
        let H_p_int: Matrix3<dtype> = Matrix3::identity();
        let H_p_ainit = H_v_ainit * dt / 2.0;

        // Copy them into place
        let mut B = Matrix::<15, 15>::zeros();
        // First column (wrt theta)
        let M = H_theta_w * p.cov_gyro_bias * H_theta_w.transpose()
            + H_theta_winit * p.cov_winit * H_theta_winit.transpose();
        B.fixed_view_mut::<3, 3>(0, 0).copy_from(&M);

        // Second column (wrt vel)
        let M = H_v_a * p.cov_accel * H_v_a.transpose()
            + H_v_ainit * p.cov_ainit * H_v_ainit.transpose();
        B.fixed_view_mut::<3, 3>(3, 3).copy_from(&M);
        let M = H_p_a * p.cov_accel * H_v_a.transpose()
            + H_p_ainit * p.cov_ainit * H_v_ainit.transpose();
        B.fixed_view_mut::<3, 3>(6, 3).copy_from(&M);

        // Third column (wrt pos)
        let M = H_v_a * p.cov_accel * H_p_a.transpose()
            + H_v_ainit * p.cov_ainit * H_p_ainit.transpose();
        B.fixed_view_mut::<3, 3>(3, 6).copy_from(&M);
        let M = H_p_a * p.cov_accel * H_p_a.transpose()
            + H_p_int * p.cov_integration * H_p_int.transpose()
            + H_p_ainit * p.cov_ainit * H_p_ainit.transpose();
        B.fixed_view_mut::<3, 3>(6, 6).copy_from(&M);

        // Fourth column (wrt gyro bias)
        B.fixed_view_mut::<3, 3>(9, 9).copy_from(&p.cov_gyro_bias);

        // Fifth column (wrt accel bias)
        B.fixed_view_mut::<3, 3>(12, 12)
            .copy_from(&p.cov_accel_bias);

        B
    }

    /// Integrate a single IMU measurement
    /// ```
    /// # use factrs::residuals::imu_preint::*;
    /// # use factrs::variables::ImuBias;
    /// # let mut preint = ImuPreintegrator::new(ImuCovariance::default(), ImuBias::zeros(), Gravity::up());
    /// let gyro = Gyro::new(0.1, 0.2, 0.3);
    /// let accel = Accel::new(0.1, 0.2, 0.3);
    /// preint.integrate(&gyro, &accel, 0.01);
    /// ```
    #[allow(non_snake_case)]
    pub fn integrate(&mut self, gyro: &Gyro, accel: &Accel, dt: dtype) {
        // Remove bias estimate
        let gyro = gyro.remove_bias(self.delta.bias_init());
        let accel = accel.remove_bias(self.delta.bias_init());

        // Construct all matrices before integrating
        let B_Q_BT = self.B_Q_BT(dt);

        // Update preintegration
        let A = self.delta.integrate(&gyro, &accel, dt);

        // Update covariance
        self.cov = A * self.cov * A.transpose() + B_Q_BT;
    }

    /// Build a corresponding factor
    ///
    /// This consumes the preintegrator and returns a
    /// [factor](crate::containers::Factor) with the proper noise model.
    /// Requires properly typed symbols, likely created via
    /// [assign_symbols](crate::assign_symbols).
    /// ```
    /// # use factrs::residuals::imu_preint::*;
    /// # use factrs::variables::*;
    /// # use factrs::assign_symbols;
    /// # assign_symbols!(X: SE3; V: VectorVar3; B: ImuBias);
    /// # let preint = ImuPreintegrator::new(ImuCovariance::default(), ImuBias::zeros(), Gravity::up());
    /// let factor = preint.build(X(0), V(0), B(0), X(1), V(1), B(1));
    /// ```
    pub fn build<X1, V1, B1, X2, V2, B2>(
        self,
        x1: X1,
        v1: V1,
        b1: B1,
        x2: X2,
        v2: V2,
        b2: B2,
    ) -> Factor
    where
        X1: TypedSymbol<SE3>,
        V1: TypedSymbol<VectorVar3>,
        B1: TypedSymbol<ImuBias>,
        X2: TypedSymbol<SE3>,
        V2: TypedSymbol<VectorVar3>,
        B2: TypedSymbol<ImuBias>,
    {
        // Create noise from our covariance matrix
        let noise = GaussianNoise::from_matrix_cov(self.cov.as_view())
            .expect("IMU preintegration covariance must be positive definite");
        // Create the residual
        let res = ImuPreintegrationResidual { delta: self.delta };
        // Build the factor
        FactorBuilder::new6(res, x1, v1, b1, x2, v2, b2)
            .noise(noise)
            .build()
    }

    /// Build a corresponding factor, with unchecked symbols
    ///
    /// Same as [build](ImuPreintegrator::build), but without the symbol type
    /// checking
    pub fn build_unchecked<X1, V1, B1, X2, V2, B2>(
        self,
        x1: X1,
        v1: V1,
        b1: B1,
        x2: X2,
        v2: V2,
        b2: B2,
    ) -> Factor
    where
        X1: Symbol,
        V1: Symbol,
        B1: Symbol,
        X2: Symbol,
        V2: Symbol,
        B2: Symbol,
    {
        // Create noise from our covariance matrix
        let noise = GaussianNoise::from_matrix_cov(self.cov.as_view())
            .expect("IMU preintegration covariance must be positive definite");
        // Create the residual
        let res = ImuPreintegrationResidual { delta: self.delta };
        // Build the factor
        FactorBuilder::new6_unchecked(res, x1, v1, b1, x2, v2, b2)
            .noise(noise)
            .build()
    }
}

// ------------------------- The Residual ------------------------- //

#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub struct ImuPreintegrationResidual {
    delta: ImuDelta,
}

#[factrs::mark]
impl Residual6 for ImuPreintegrationResidual {
    type Differ = ForwardProp<Const<30>>;
    type DimIn = Const<30>;
    type DimOut = Const<15>;
    type V1 = SE3;
    type V2 = VectorVar3;
    type V3 = ImuBias;
    type V4 = SE3;
    type V5 = VectorVar3;
    type V6 = ImuBias;

    fn residual6<T: crate::linalg::Numeric>(
        &self,
        x1: SE3<T>,
        v1: VectorVar3<T>,
        b1: ImuBias<T>,
        x2: SE3<T>,
        v2: VectorVar3<T>,
        b2: ImuBias<T>,
    ) -> VectorX<T> {
        // Add dual types to all of our fields
        let delta = &self.delta.cast();

        // Pull out the measurements
        let start = ImuState {
            r: x1.rot().clone(),
            v: v1.0,
            p: x1.xyz().into_owned(),
            bias: b1,
        };
        let ImuState {
            r: r2_meas,
            v: v2_meas,
            p: p2_meas,
            bias: b2_meas,
        } = delta.predict(&start);
        let p_2: VectorVar3<T> = x2.xyz().into_owned().into();
        let p2_meas: VectorVar3<T> = p2_meas.into();
        let v2_meas: VectorVar3<T> = v2_meas.into();

        // Compute residuals
        // Because of how the noise is integrated,
        // we have to use the right version of ominus here
        // This won't matter so much for the vector elements (right = left for vectors)
        let r_r = r2_meas.ominus_right(x2.rot());
        let r_vel = v2_meas.ominus_right(&v2);
        let r_p = p2_meas.ominus_right(&p_2);
        let r_bias = b2_meas.ominus_right(&b2);

        let mut residual = VectorX::zeros(15);
        residual.fixed_rows_mut::<3>(0).copy_from(&r_r);
        residual.fixed_rows_mut::<3>(3).copy_from(&r_vel);
        residual.fixed_rows_mut::<3>(6).copy_from(&r_p);
        residual.fixed_rows_mut::<6>(9).copy_from(&r_bias);

        residual
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{
        assert_variable_eq, assign_symbols,
        containers::{Graph, Values},
        fac,
        linalg::Vector3,
        optimizers::{GaussNewton, Optimizer},
        residuals::{Accel, Gyro, PriorResidual},
        variables::{ImuBias, VectorVar3, SE3},
    };

    assign_symbols!(X: SE3; V: VectorVar3; B: ImuBias);

    // Test using residual in an optimization problem
    #[test]
    fn optimize() {
        // Initial conditions
        let accel = Accel::new(0.1, 0.0, -9.81);
        let gyro = Gyro::zeros();
        let x0 = SE3::identity();
        let v0 = VectorVar3::identity();
        let b0 = ImuBias::identity();
        let dt = 0.01;
        let n = 100;

        // Integrate measurements
        let mut preint = ImuPreintegrator::new(ImuCovariance::default(), b0.clone(), Gravity::up());
        for _ in 0..n {
            preint.integrate(&gyro, &accel, dt);
        }
        println!("xi_pos : {}", preint.delta);

        // Build factor and graph
        let mut graph = Graph::new();
        let factor = preint.build(X(0), V(0), B(0), X(1), V(1), B(1));
        let prior_x0 = fac!(PriorResidual::new(x0.clone()), X(0), 1e-3 as cov);
        let prior_v0 = fac!(PriorResidual::new(v0.clone()), V(0), 1e-3 as cov);
        let prior_b0 = fac!(PriorResidual::new(b0.clone()), B(0), 1e-3 as cov);
        graph.add_factor(factor);
        graph.add_factor(prior_x0);
        graph.add_factor(prior_v0);
        graph.add_factor(prior_b0);

        // println!("{:?}", graph);

        let mut values = Values::new();
        values.insert(X(0), x0);
        values.insert(V(0), v0);
        values.insert(B(0), b0);
        values.insert(X(1), SE3::identity());
        values.insert(V(1), VectorVar3::identity());
        values.insert(B(1), ImuBias::identity());

        // Optimize
        let mut opt: GaussNewton = GaussNewton::new_default(graph);
        let results = opt.optimize(values).expect("Optimization failed");

        // Check results
        let t = n as dtype * dt;
        let xyz = Vector3::new(accel.0.x * t * t / 2.0, 0.0, 0.0);

        let x1_exp = SE3::from_rot_trans(SO3::identity(), xyz);
        let x1_got = results.get(X(1)).expect("Somehow missing X(1)").clone();
        println!("x1_exp: {}", x1_exp);
        println!("x1_got: {}", x1_got);
        assert_variable_eq!(x1_got, x1_exp, comp = abs, tol = 1e-5);

        let v1_exp = VectorVar3::new(accel.0.x * t, 0.0, 0.0);
        let v1_got = results.get(V(1)).expect("Somehow missing V(1)").clone();
        println!("v1_exp: {}", v1_exp);
        println!("v1_got: {}", v1_got);
        assert_variable_eq!(v1_got, v1_exp, comp = abs, tol = 1e-5);
    }
}
