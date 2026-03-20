//! Noise model representations
//!
//! Represent Gaussian noise models in a factor graph, specifically used when
//! constructing a [factor](crate::containers::Factor).

use std::fmt::Debug;

use dyn_clone::DynClone;

use crate::linalg::{DimName, MatrixX, VectorX};

/// The trait for a noise model.
#[cfg_attr(feature = "serde", typetag::serde(tag = "tag"))]
pub trait NoiseModel: Debug + DynClone {
    /// The dimension of the noise model
    type Dim: DimName
    where
        Self: Sized;

    fn dim(&self) -> usize
    where
        Self: Sized,
    {
        Self::Dim::DIM
    }

    /// Whiten a vector
    fn whiten_vec(&self, v: VectorX) -> VectorX;

    /// Whiten a matrix
    fn whiten_mat(&self, m: MatrixX) -> MatrixX;
}

dyn_clone::clone_trait_object!(NoiseModel);

#[cfg(feature = "serde")]
pub use register_noisemodel as tag_noise;

mod gaussian;
pub use gaussian::GaussianNoise;

mod unit;
pub use unit::UnitNoise;
