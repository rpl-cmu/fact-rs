//! Various helpers for linear algebra structures.
//!
//! Specifically this module contains the following,
//! - re-alias all nalgebra types to use our dtype by default.
//! - re-alias num-dual types for identical reasons
//! - a [MatrixBlock] struct to help with block matrix operations after
//!   linearization
//! - a [Diff] trait to help with numerical and forward-mode differentiation
//! - Forward mode differentiator [ForwardProp]
//! - Numerical differentiator [NumericalDiff]
use crate::dtype;

mod dual;
pub use dual::{DualAllocator, DualScalar, DualVector, Numeric};
// Dual numbers
pub use num_dual::Derivative;

mod nalgebra_wrap;
pub use nalgebra_wrap::*;

// ------------------------- MatrixBlocks ------------------------- //
/// A struct to help with block matrix operations after linearization
///
/// This struct is used to store a matrix and a set of indices that
/// represent the start of each block in the matrix. This is useful
/// when linearizing a factor graph, where the Jacobian is a block
/// matrix with each block corresponding to a different variable.
#[derive(Debug, Clone)]
pub struct MatrixBlock {
    mat: MatrixX,
    idx: Vec<usize>,
}

impl MatrixBlock {
    pub fn new(mat: MatrixX, idx: Vec<usize>) -> Self {
        Self { mat, idx }
    }

    pub fn get_idx(&self, idx: usize) -> usize {
        self.idx[idx]
    }

    pub fn get_block(&self, idx: usize) -> MatrixViewX<'_> {
        let idx_start = self.idx[idx];
        let idx_end = if idx + 1 < self.idx.len() {
            self.idx[idx + 1]
        } else {
            self.mat.ncols()
        };
        self.mat.columns(idx_start, idx_end - idx_start)
    }

    pub fn mul(&self, idx: usize, x: VectorViewX<'_>) -> VectorX {
        self.get_block(idx) * x
    }

    pub fn mat(&self) -> MatrixViewX<'_> {
        self.mat.as_view()
    }

    pub fn idx(&self) -> &[usize] {
        &self.idx
    }
}

// ------------------------- Derivatives ------------------------- //
use paste::paste;

use crate::variables::VariableDtype;

/// A struct to hold the result of a differentiation operation
#[derive(Debug, Clone)]
pub struct DiffResult<V, G> {
    pub value: V,
    pub diff: G,
}

macro_rules! fn_maker {
    (grad, $num:expr, $( ($name:ident: $var:ident) ),*) => {
        paste! {
            fn [<gradient_ $num>]<$( $var: VariableDtype, )* F: Fn($($var::Alias<Self::T>,)*) -> Self::T>
                    (f: F, $($name: &$var,)*) -> DiffResult<dtype, VectorX>{
                    let f_wrapped = |$($name: $var::Alias<Self::T>,)*| vectorx![f($($name.clone(),)*)];
                    let DiffResult { value, diff } = Self::[<jacobian_ $num>](f_wrapped, $($name,)*);
                    let diff = VectorX::from_iterator(diff.len(), diff.iter().cloned());
                    DiffResult { value: value[0], diff }
                }
        }
    };

    (jac, $num:expr, $( ($name:ident: $var:ident) ),*) => {
        paste! {
            fn [<jacobian_ $num>]<$( $var: VariableDtype, )* F: Fn($($var::Alias<Self::T>,)*) -> VectorX<Self::T>>
                    (f: F, $($name: &$var,)*) -> DiffResult<VectorX, MatrixX>;
        }
    };
}

/// A trait to abstract over different differentiation methods
///
/// Specifically, this trait works for multi-input functions (where each input
/// is a variable) with scalar output (gradient) or vector output (Jacobian).
///
/// This trait is implemented for both numerical and forward-mode in
/// [NumericalDiff] and [ForwardProp], respectively. Where possible, we
/// recommend [ForwardProp] which functions using dual numbers.
pub trait Diff {
    /// The dtype of the variables
    type T: Numeric;

    fn_maker!(grad, 1, (v1: V1));
    fn_maker!(grad, 2, (v1: V1), (v2: V2));
    fn_maker!(grad, 3, (v1: V1), (v2: V2), (v3: V3));
    fn_maker!(grad, 4, (v1: V1), (v2: V2), (v3: V3), (v4: V4));
    fn_maker!(grad, 5, (v1: V1), (v2: V2), (v3: V3), (v4: V4), (v5: V5));
    fn_maker!(grad, 6, (v1: V1), (v2: V2), (v3: V3), (v4: V4), (v5: V5), (v6: V6));

    fn_maker!(jac, 1, (v1: V1));
    fn_maker!(jac, 2, (v1: V1), (v2: V2));
    fn_maker!(jac, 3, (v1: V1), (v2: V2), (v3: V3));
    fn_maker!(jac, 4, (v1: V1), (v2: V2), (v3: V3), (v4: V4));
    fn_maker!(jac, 5, (v1: V1), (v2: V2), (v3: V3), (v4: V4), (v5: V5));
    fn_maker!(jac, 6, (v1: V1), (v2: V2), (v3: V3), (v4: V4), (v5: V5), (v6: V6));
}

/// Compute the derivative of a scalar function using numerical derivatives.
pub fn numerical_derivative<F: Fn(dtype) -> dtype>(
    f: F,
    x: dtype,
    eps: dtype,
) -> DiffResult<dtype, dtype> {
    let r = f(x);
    let d = (f(x + eps) - f(x - eps)) / (2.0 * eps);

    DiffResult { value: r, diff: d }
}

/// Compute the derivative of a scalar function using forward derivatives.
pub fn forward_prop_derivative<F: Fn(DualScalar) -> DualScalar>(
    f: F,
    x: dtype,
) -> DiffResult<dtype, dtype> {
    let xd = x.into();
    let r = f(xd);
    DiffResult {
        value: r.re,
        diff: r.eps,
    }
}

mod numerical_diff;
pub use numerical_diff::NumericalDiff;

mod forward_prop;
pub use forward_prop::ForwardProp;
