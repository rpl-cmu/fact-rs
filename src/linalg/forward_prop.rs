use paste::paste;

use super::{
    AllocatorBuffer, Diff, MatrixDim,
    dual::{DualAllocator, DualVector},
};
use crate::{
    linalg::{Const, DefaultAllocator, DiffResult, DimName, Dyn, MatrixX, VectorDim, VectorX},
    variables::{Variable, VariableDtype},
};

/// Forward mode differentiator
///
/// It requires a function that takes in variables with a dtype of [DualVector]
/// and outputs a vector of the same dtype. The generic parameter `N` is used to
/// specify the dimension of the DualVector.
///
/// This struct is used to compute the Jacobian of a function using forward mode
/// differentiation via dual-numbers. It can operate on functions with up to 6
/// inputs and with vector-valued outputs.
///
/// ```
/// use factrs::{
///     linalg::{Const, DiffResult, ForwardProp, Numeric, VectorX, vectorx},
///     traits::*,
///     variables::SO2,
/// };
///
/// fn f<T: Numeric>(x: SO2<T>, y: SO2<T>) -> VectorX<T> {
///     x.ominus(&y)
/// }
///
/// let x = SO2::from_theta(2.0);
/// let y = SO2::from_theta(1.0);
///
/// // 2 as the generic since we have 2 dimensions going in
/// let DiffResult { value, diff } = ForwardProp::<Const<2>>::jacobian_2(f, &x, &y);
/// assert_eq!(value, vectorx![1.0]);
/// ```
pub struct ForwardProp<N: DimName> {
    _phantom: std::marker::PhantomData<N>,
}

macro_rules! forward_maker {
    ($num:expr, $( ($name:ident: $var:ident) ),*) => {
        paste! {
            #[allow(unused_assignments)]
            fn [<jacobian_ $num>]<$( $var: VariableDtype, )* F: Fn($($var::Alias<Self::T>,)*) -> VectorX<Self::T>>
                    (f: F, $($name: &$var,)*) -> DiffResult<VectorX, MatrixX>{
                // Prepare variables
                let mut curr_dim = 0;
                $(
                    let $name: $var::Alias<Self::T> = $name.dual(curr_dim);
                    curr_dim += $name.dim();
                )*

                // Compute residual
                let res = f($($name,)*);

                // Compute Jacobian
                let n = VectorDim::<N>::zeros().shape_generic().0;
                let eps1 = MatrixDim::<Dyn, N>::from_rows(
                    res.map(|r| r.eps.unwrap_generic(n, Const::<1>).transpose())
                        .as_slice(),
                );

                let mut eps = MatrixX::zeros(res.len(), N::DIM);
                eps.copy_from(&eps1);

                DiffResult {
                    value: res.map(|r| r.re),
                    diff: eps,
                }
            }
        }
    };
}

impl<N: DimName> Diff for ForwardProp<N>
where
    AllocatorBuffer<N>: Sync + Send,
    DefaultAllocator: DualAllocator<N>,
    DualVector<N>: Copy,
{
    type T = DualVector<N>;

    forward_maker!(1, (v1: V1));
    forward_maker!(2, (v1: V1), (v2: V2));
    forward_maker!(3, (v1: V1), (v2: V2), (v3: V3));
    forward_maker!(4, (v1: V1), (v2: V2), (v3: V3), (v4: V4));
    forward_maker!(5, (v1: V1), (v2: V2), (v3: V3), (v4: V4), (v5: V5));
    forward_maker!(6, (v1: V1), (v2: V2), (v3: V3), (v4: V4), (v5: V5), (v6: V6));
}
