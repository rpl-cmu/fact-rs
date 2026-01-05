use paste::paste;

use crate::{
    dtype,
    linalg::{Diff, DiffResult, MatrixX, VectorX},
    variables::{Variable, VariableDtype},
};

/// Forward mode differentiator
///
/// It operates on functions with regular dtype inputs and outputs, no dual
/// numbers required. The generic parameter `PWR` is used to specify the power
/// of the step size, it PWR=6 uses 1e-6 as a step size.
///
/// This struct is used to compute the Jacobian of a function using forward mode
/// differentiation via dual-numbers. It can operate on functions with up to 6
/// inputs and with vector-valued outputs.
///
/// ```
/// use factrs::{
///     linalg::{DiffResult, NumericalDiff, VectorX, vectorx},
///     traits::*,
///     variables::SO2,
/// };
///
/// // We can also be generic over Numeric as in [ForwardProp] as well if desired
/// fn f(x: SO2, y: SO2) -> VectorX {
///     x.ominus(&y)
/// }
///
/// let x = SO2::from_theta(2.0);
/// let y = SO2::from_theta(1.0);
///
/// // 2 as the generic since we have 2 dimensions going in
/// let DiffResult { value, diff } = NumericalDiff::<6>::jacobian_2(f, &x, &y);
/// assert_eq!(value, vectorx![1.0]);
/// ```
pub struct NumericalDiff<const PWR: i32 = 6>;

macro_rules! numerical_maker {
    ($num:expr, $( ($idx:expr, $name:ident, $var:ident) ),*) => {
        paste! {
            #[allow(unused_assignments)]
            fn [<jacobian_$num>]<$( $var: VariableDtype, )* F: Fn($($var,)*) -> VectorX>
                    (f: F, $($name: &$var,)*) -> DiffResult<VectorX, MatrixX> {
                let eps = dtype::powi(10.0, -PWR);

                // Get Dimension
                let mut dim = 0;
                $(dim += Variable::dim($name);)*

                let res = f($( $name.clone(), )*);

                // Compute gradient
                let mut jac: MatrixX = MatrixX::zeros(res.len(), dim);
                let mut tvs = [$( VectorX::zeros(Variable::dim($name)), )*];

                for i in 0..$num {
                    let mut curr_dim = 0;
                    for j in 0..tvs[i].len() {
                        tvs[i][j] = eps;
                        // TODO: It'd be more efficient to not have to add tangent vectors to each variable
                        // However, I couldn't find a way to do this for a single vector without having to
                        // do a nested iteration of $name which isn't allowed
                        $(let [<$name _og>] = $name.oplus(tvs[$idx].as_view());)*
                        let plus = f($( [<$name _og>], )*);

                        tvs[i][j] = -eps;
                        $(let [<$name _og>] = $name.oplus(tvs[$idx].as_view());)*
                        let minus = f($( [<$name _og>], )*);

                        let delta = (plus - minus) / (2.0 * eps);
                        jac.columns_mut(curr_dim + j, 1).copy_from(&delta);

                        tvs[i][j] = 0.0;
                    }
                    curr_dim += tvs[i].len();
                }

                DiffResult { value: res, diff: jac }
            }
        }
    };
}

impl<const PWR: i32> Diff for NumericalDiff<PWR> {
    type T = dtype;

    numerical_maker!(1, (0, v1, V1));
    numerical_maker!(2, (0, v1, V1), (1, v2, V2));
    numerical_maker!(3, (0, v1, V1), (1, v2, V2), (2, v3, V3));
    numerical_maker!(4, (0, v1, V1), (1, v2, V2), (2, v3, V3), (3, v4, V4));
    numerical_maker!(
        5,
        (0, v1, V1),
        (1, v2, V2),
        (2, v3, V3),
        (3, v4, V4),
        (4, v5, V5)
    );
    numerical_maker!(
        6,
        (0, v1, V1),
        (1, v2, V2),
        (2, v3, V3),
        (3, v4, V4),
        (4, v5, V5),
        (5, v6, V6)
    );
}

macro_rules! numerical_variable_maker {
    ($num:expr, $( ($idx:expr, $name:ident, $var:ident) ),*) => {
        paste! {
            #[allow(unused_assignments)]
            pub fn [<jacobian_variable_$num>]<$( $var: VariableDtype, )* VOut: VariableDtype, F: Fn($($var,)*) -> VOut>
                    (f: F, $($name: &$var,)*) -> DiffResult<VOut, MatrixX> {
                let eps = dtype::powi(10.0, -PWR);

                // Get Dimension
                let mut dim = 0;
                $(dim += Variable::dim($name);)*

                let res = f($( $name.clone(), )*);

                // Compute gradient
                let mut jac: MatrixX = MatrixX::zeros(VOut::DIM, dim);
                let mut tvs = [$( VectorX::zeros(Variable::dim($name)), )*];

                for i in 0..$num {
                    let mut curr_dim = 0;
                    for j in 0..tvs[i].len() {
                        tvs[i][j] = eps;
                        // TODO: It'd be more efficient to not have to add tangent vectors to each variable
                        // However, I couldn't find a way to do this for a single vector without having to
                        // do a nested iteration of $name which isn't allowed
                        $(let [<$name _og>] = $name.oplus(tvs[$idx].as_view());)*
                        let plus = f($( [<$name _og>], )*);

                        tvs[i][j] = -eps;
                        $(let [<$name _og>] = $name.oplus(tvs[$idx].as_view());)*
                        let minus = f($( [<$name _og>], )*);

                        let delta = plus.ominus(&minus) / (2.0 * eps);
                        jac.columns_mut(curr_dim + j, 1).copy_from(&delta);

                        tvs[i][j] = 0.0;
                    }
                    curr_dim += tvs[i].len();
                }

                DiffResult { value: res, diff: jac }
            }
        }
    };
}

impl<const PWR: i32> NumericalDiff<PWR> {
    numerical_variable_maker!(1, (0, v1, V1));
    numerical_variable_maker!(2, (0, v1, V1), (1, v2, V2));
    numerical_variable_maker!(3, (0, v1, V1), (1, v2, V2), (2, v3, V3));
    numerical_variable_maker!(4, (0, v1, V1), (1, v2, V2), (2, v3, V3), (3, v4, V4));
    numerical_variable_maker!(
        5,
        (0, v1, V1),
        (1, v2, V2),
        (2, v3, V3),
        (3, v4, V4),
        (4, v5, V5)
    );
    numerical_variable_maker!(
        6,
        (0, v1, V1),
        (1, v2, V2),
        (2, v3, V3),
        (3, v4, V4),
        (4, v5, V5),
        (5, v6, V6)
    );
}
