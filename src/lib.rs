//! fact.rs (pronounced factors) is a nonlinear least squares optimization
//! library over factor graphs written in Rust.
//!
//! It is specifically geared toward sensor fusion in robotics. It aims to be
//! fast, easy to use, and safe. The fact.rs API takes heavy inspiration from
//! the [gtsam library](https://gtsam.org/).
//!
//! Currently, it supports the following features
//! - Gauss-Newton & Levenberg-Marquadt Optimizers
//! - Common Lie Groups supported (SO2, SO3, SE2, SE3) with optimization in Lie
//!   Algebras
//! - Automatic differentiation via dual numbers
//! - Serialization of graphs & variables via optional serde support
//! - Easy conversion to rerun types for straightforward visualization
//!
//! # Background
//!
//! Specifically, we solve the following problem,
//!
//! $$
//! \blue{\Theta^*} = \red{\argmin_{\Theta}}
//! \sum_{i} \green{\rho_i(||r_i(\Theta)||_{\Sigma_i} )}
//! $$
//!
//! The fact.rs API takes heavy inspiration from the [gtsam library](https://gtsam.org/),
//! and is loosely structured as follows,
//! - <blue>Variables</blue>: These are the unknowns in the optimization
//!   problem. They are all lie-group based (even if trivially so). See the
//!   [variable](crate::variables) module for details on custom implementations.
//!   A collection of variables is stored in a
//!   [Values](crate::containers::Values) container.
//! - <red>Optimizers</red>: The optimizer is responsible for finding the
//!   optimal variables that minimize the factors. More info on factor graph
//!   optimizers in [Optimizers](crate::optimizers).
//! - <green>Factors</green>: Each factor represents a probabilistic constraint
//!   in the optimization. More info in [Factor](crate::containers::Factor). A
//!   collection of factors is stored in a [Graph](crate::containers::Graph)
//!   container.
//!
//! # Example
//! ```
//! use factrs::{
//!     assign_symbols,
//!     core::{BetweenResidual, GaussNewton, Graph, Huber, PriorResidual, Values, SO2},
//!     fac,
//!     traits::*,
//! };
//!
//! // Assign symbols to variable types
//! assign_symbols!(X: SO2);
//!
//! // Make all the values
//! let mut values = Values::new();
//!
//! let x = SO2::from_theta(1.0);
//! let y = SO2::from_theta(2.0);
//! values.insert(X(0), SO2::identity());
//! values.insert(X(1), SO2::identity());
//!
//! // Make the factors & insert into graph
//! let mut graph = Graph::new();
//! let res = PriorResidual::new(x.clone());
//! let factor = fac![res, X(0)];
//! graph.add_factor(factor);
//!
//! let res = BetweenResidual::new(y.minus(&x));
//! let factor = fac![res, (X(0), X(1)), 0.1 as std, Huber::default()];
//! graph.add_factor(factor);
//!
//! // Optimize!
//! let mut opt: GaussNewton = GaussNewton::new_default(graph);
//! let result = opt.optimize(values).unwrap();
//! println!("Results {:#}", result);
//! ```

#![warn(clippy::unwrap_used)]
#![cfg_attr(docsrs, feature(doc_cfg))]

/// The default floating point type used in the library
#[cfg(not(feature = "f32"))]
#[allow(non_camel_case_types)]
pub type dtype = f64;

#[cfg(feature = "f32")]
#[allow(non_camel_case_types)]
pub type dtype = f32;

// Hack to be able to use our proc macro inside and out of our crate
// https://users.rust-lang.org/t/how-to-express-crate-path-in-procedural-macros/91274/10
#[doc(hidden)]
extern crate self as factrs;
/// Easiest way to create a factor
///
/// Similar to `vec!` in the std library, this is a macro to create a factor
/// from a residual, keys, and alternatively noise and robust kernel. A simple
/// usage would be
/// ```
/// # use factrs::{assign_symbols, fac, core::{SO2, PriorResidual, BetweenResidual}, traits::*};
/// # let prior = PriorResidual::new(SO2::identity());
/// # let between = BetweenResidual::new(SO2::identity());
/// # assign_symbols!(X: SO2);
/// let prior_factor = fac![prior, X(0)];
/// let between_factor = fac![between, (X(0), X(1))];
/// ```
/// Additionally, there is a number of helper options for specifying a noise
/// model,
/// ```
/// # use factrs::{assign_symbols, fac, core::{SO2, PriorResidual, GaussianNoise}, traits::*};
/// # let prior = PriorResidual::new(SO2::identity());
/// # assign_symbols!(X: SO2);
/// let noise = GaussianNoise::from_scalar_sigma(0.1);
/// let f1a = fac![prior, X(0), noise];
/// # let prior = PriorResidual::new(SO2::identity());
/// let f1b = fac![prior, X(0), 0.1 as std];
/// # let prior = PriorResidual::new(SO2::identity());
/// let f2 = fac![prior, X(0), 0.1 as cov];
/// # let prior = PriorResidual::new(SO2::identity());
/// let f3 = fac![prior, X(0), (0.1, 0.3) as std];
/// ```
/// where `f1a` and `f1b` are identical, and where `f3` uses
/// [from_split_sigma](factrs::noise::GaussianNoise::from_split_sigma)
/// to specify the rotation and translation noise separately. (where rotation is
/// ALWAYS first in factrs)
///
/// Finally, a robust kernel can be specified as well,
/// ```
/// # use factrs::{assign_symbols, fac, core::{SO2, PriorResidual, Huber}, traits::*};
/// # let prior = PriorResidual::new(SO2::identity());
/// # assign_symbols!(X: SO2);
/// let f1 = fac![prior, X(0), 0.1 as std, Huber::default()];
/// # let prior = PriorResidual::new(SO2::identity());
/// let f2 = fac![prior, X(0), _, Huber::default()];
/// ```
/// where `f2` uses [UnitNoise](factrs::noise::UnitNoise) as the noise model.
pub use factrs_proc::fac;
/// Mark an implementation of [Variable](factrs::traits::Variable),
/// [Residual](factrs::traits::Residual), [Noise](factrs::traits::NoiseModel),
/// or [Robust](factrs::traits::RobustCost).
///
/// This is mostly to aid in serialization when the `serde` feature is enabled.
/// Since these items are boxed inside [Factor](factrs::core::Factor), we use
/// `typetag` for serialization which requires a little bit of manual work.
///
/// For examples of usage, check out the [tests](https://github.com/rpl-cmu/factrs/tree/dev/tests) folder.
///
/// Specifically, it does the following for each trait:
///
/// ### [Variable](factrs::traits::Variable)
/// If serde is disabled, does nothing. Otherwise, it does the following:
/// - Checks there is a single generic for the datatype
/// - Add tag for serialization
/// - Add tag for serializing
///   [PriorResidual\<Type\>](factrs::core::PriorResidual) and
///   [BetweenResidual\<Type\>](factrs::core::BetweenResidual) as well.
///
/// ### [Residual](factrs::traits::Residual)
/// This should be applied on a numbered residual such as
/// [Residual2](factrs::residuals::Residual2) and will automatically derive
/// [Residual](factrs::traits::Residual). Additionally, if serde is
/// enabled, it will add a tag for serialization.
///
/// ### [Noise](factrs::traits::NoiseModel)
/// If serde is disabled, does nothing. Otherwise, it will tag the noise model
/// for serialization, up to size 32.
///
/// ### [Robust](factrs::traits::RobustCost)
/// If serde is disabled, does nothing. Otherwise, it will tag the robust
/// kernel.
pub use factrs_proc::mark;

pub mod containers;
pub mod linalg;
pub mod linear;
pub mod noise;
pub mod optimizers;
pub mod residuals;
pub mod robust;
pub mod utils;
pub mod variables;

/// Untagged symbols if `unchecked` API is desired.
///
/// We strongly recommend using [assign_symbols] to
/// create and tag symbols with the appropriate types. However, we provide a
/// number of pre-defined symbols if desired. Note these objects can't be tagged
/// due to the orphan rules.
pub mod symbols {
    crate::assign_symbols!(
        A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z
    );
}

/// Helper module to import common traits
///
/// This module is meant to be glob imported to make it easier to use the traits
/// and functionality in the library.
/// ```
/// use factrs::traits::*;
/// ```
pub mod traits {
    pub use crate::{
        linalg::Diff, noise::NoiseModel, optimizers::Optimizer, residuals::Residual,
        robust::RobustCost, variables::Variable,
    };
}

/// Helper module to group together most commonly used types
///
/// Specifically, this contains everything that would be needed to implement a
/// simple pose graph. While we recommend against it, for quick usage it can be
/// glob imported as
/// ```
/// use factrs::core::*;
/// ```
pub mod core {
    pub use crate::{
        assign_symbols,
        containers::{Factor, Graph, Values},
        fac,
        linalg::{Vector1, Vector2, Vector3},
        noise::{GaussianNoise, UnitNoise},
        optimizers::{GaussNewton, LevenMarquardt},
        residuals::{BetweenResidual, PriorResidual},
        robust::{GemanMcClure, Huber, L2},
        variables::{VectorVar, VectorVar1, VectorVar2, VectorVar3, SE2, SE3, SO2, SO3},
    };
}

#[cfg(feature = "rerun")]
#[cfg_attr(docsrs, doc(cfg(feature = "rerun")))]
/// Conversion from fact.rs types to rerun types
///
/// Most the fact.rs types can be converted into rerun types for visualization
/// purposes. The following conversions are supported,
/// - VectorVar2 -> Vec2D, Points2D
/// - VectorVar3 -> Vec3D, Points3D
/// - SO2 -> Arrows2D
/// - SE2 -> Arrows2D, Points2D
/// - SO3 -> Rotation3D, Arrows3D
/// - SE3 -> Transform3D, Arrows3D, Points3D
///
/// Furthermore, we can also convert iterators of these types into the
/// corresponding rerun types. This is useful for visualizing multiple objects
/// at once.
/// - Iterator of VectorVar2 -> Points2D
/// - Iterator of VectorVar3 -> Points3D
/// - Iterator of SE2 -> Arrows2D, Points2D
/// - Iterator of SE3 -> Arrows3D, Points3D
pub mod rerun;

#[cfg(feature = "serde")]
#[cfg_attr(docsrs, doc(cfg(feature = "serde")))]
/// Macros to help with serde serialization
///
/// In case you are using a [marked](crate::mark) custom implementation along
/// with serialization, you'll have to manually "tag" each type for
/// serialization. This module provides a number of helper functions to do so.
pub mod serde {
    #[doc(inline)]
    pub use crate::{
        noise::tag_noise, residuals::tag_residual, robust::tag_robust, variables::tag_variable,
    };
}
