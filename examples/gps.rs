/*
This is a port of the "LocalizationExample.cpp" found in the gtsam repository
https://github.com/borglab/gtsam/blob/11142b08fc842f1fb79ccf3017946d70f5173335/examples/LocalizationExample.cpp

A simple 2D pose slam example with "GPS" measurements
 - The robot moves forward 2 meter each iteration
 - The robot initially faces along the X axis (horizontal, to the right in 2D)
 - We have full odometry between pose
 - We have "GPS-like" measurements implemented with a custom factor
*/

#![allow(unused_imports)]
// Our state will be represented by SE2 -> theta, x, y
// VectorVar2 is a newtype around Vector2 for optimization purposes
use factrs::{
    assign_symbols,
    core::{BetweenResidual, GaussNewton, Graph, Values},
    dtype, fac,
    linalg::{Const, ForwardProp, Numeric, NumericalDiff, VectorX},
    residuals::Residual1,
    traits::*,
    variables::{SE2, VectorVar2},
};

#[derive(Clone, Debug)]
// Enable serialization if it's desired
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GpsResidual {
    meas: VectorVar2,
}

impl GpsResidual {
    pub fn new(x: dtype, y: dtype) -> Self {
        Self {
            meas: VectorVar2::new(x, y),
        }
    }
}

// The `mark` macro handles serialization stuff and some custom impl as well
#[factrs::mark]
impl Residual1 for GpsResidual {
    // Use forward propagation for differentiation
    type Differ = ForwardProp<<Self as Residual1>::DimIn>;
    // Alternatively, could use numerical differentiation (6 => 10^-6 as
    // denominator)
    // type Differ = NumericalDiff<6>;

    // The input variable type, input dimension of variable(s), and output dimension
    // of residual
    type V1 = SE2;
    type DimIn = Const<3>;
    type DimOut = Const<2>;

    // D is a custom numeric type that can be leveraged for autodiff
    fn residual1<T: Numeric>(&self, v: SE2<T>) -> VectorX<T> {
        // Convert measurement from dtype to T
        let p_meas = self.meas.cast();
        // Convert p to VectorVar2 as well
        let p = VectorVar2::from(v.xy().into_owned());

        p.ominus(&p_meas)
    }

    // You can also hand-code the jacobian by hand if extra efficiency is desired.
    // fn residual1_jacobian(&self, values: &Values, keys: &[Key]) ->
    // DiffResult<VectorX, MatrixX> {     let p: &SE2 = values
    //         .get_unchecked(keys[0])
    //         .expect("got wrong variable type");
    //     let s = p.theta().sin();
    //     let c = p.theta().cos();
    //     let diff = MatrixX::from_row_slice(2, 3, &[0.0, c, -s, 0.0, s, c]);
    //     DiffResult {
    //         value: self.residual1(p.clone()),
    //         diff,
    //     }
    // }
    // As a note - the above jacobian is only valid if running with the "left"
    // feature disabled
    // Enabling the left feature will change the jacobian
}

// Here we assign X to always represent SE2 variables
// We'll get a compile-time error if we try anything else
assign_symbols!(X: SE2);

fn main() {
    let mut graph = Graph::new();

    // Add odometry factors
    let res = BetweenResidual::new(SE2::new(0.0, 2.0, 0.0));
    let odometry_01 = fac![res.clone(), (X(0), X(1)), (0.1, 0.2) as cov];
    let odometry_12 = fac![res, (X(1), X(2)), (0.1, 0.2) as cov];
    graph.add_factor(odometry_01);
    graph.add_factor(odometry_12);

    // Add gps factors
    let g0 = fac![GpsResidual::new(0.0, 0.0), X(0), 1.0 as std];
    let g1 = fac![GpsResidual::new(2.0, 0.0), X(1), 1.0 as std];
    let g2 = fac![GpsResidual::new(4.0, 0.0), X(2), 1.0 as std];
    graph.add_factor(g0);
    graph.add_factor(g1);
    graph.add_factor(g2);

    // Make values
    let mut values = Values::new();
    values.insert(X(0), SE2::new(1.0, 2.0, 3.0));
    values.insert(X(1), SE2::identity());
    values.insert(X(2), SE2::identity());

    // // These will all compile-time error
    // // mismatched symbol-variable types
    // values.insert(X(5), VectorVar2::identity());
    // // wrong number of keys
    // let f = fac![GpsResidual::new(0.0, 0.0), (X(0), X(1))];
    // // wrong noise-model dimension
    // let n = factrs::noise::GaussianNoise::<5>::from_scalar_sigma(0.1);
    // let f = fac![GpsResidual::new(0.0, 0.0), X(0), n];
    // // mismatched symbol-variable types
    // assign_symbols!(Y : VectorVar2);
    // let f = fac![GpsResidual::new(0.0, 0.0), Y(0), 0.1 as std];

    // optimize
    let mut opt: GaussNewton = GaussNewton::new_default(graph);
    let result = opt.optimize(values).expect("Optimization failed");

    println!("Final Result: {:#?}", result);
}
