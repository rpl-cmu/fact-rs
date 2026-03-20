use core::fmt;

use super::NoiseModel;
use crate::linalg::{Const, MatrixX, VectorX};

/// A unit noise model.
///
/// Represents a noise model that does not modify the input, or equal weighting
/// in a [factor](crate::containers::Factor).
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct UnitNoise<const N: usize>;

#[factrs::mark]
impl<const N: usize> NoiseModel for UnitNoise<N> {
    type Dim = Const<N>;

    fn whiten_vec(&self, v: VectorX) -> VectorX {
        v
    }

    fn whiten_mat(&self, m: MatrixX) -> MatrixX {
        m
    }
}

impl<const N: usize> fmt::Display for UnitNoise<N> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{self:?}")
    }
}
