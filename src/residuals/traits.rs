use std::fmt::Debug;

use dyn_clone::DynClone;

use crate::{
    containers::{Key, Values},
    linalg::{Diff, DiffResult, DimName, MatrixX, Numeric, VectorX},
    variables::{Variable, VariableDtype},
};

type Alias<V, T> = <V as Variable>::Alias<T>;

// ------------------ Base Residual Trait & Helpers ------------------ //
/// Base trait for residuals
///
/// This trait is used to implement custom residuals. It is recommended to use
/// implement one of the `ResidualN` traits, and then [mark](factrs::mark) it to
/// implement this.
#[cfg_attr(feature = "serde", typetag::serde(tag = "tag"))]
pub trait Residual: Debug + DynClone + Send {
    fn dim_in(&self) -> usize;

    fn dim_out(&self) -> usize;

    fn residual(&self, values: &Values, keys: &[Key]) -> VectorX;

    fn residual_jacobian(&self, values: &Values, keys: &[Key]) -> DiffResult<VectorX, MatrixX>;
}

dyn_clone::clone_trait_object!(Residual);

// -------------- Use Macro to create residuals with set sizes -------------- //
use paste::paste;
#[cfg(feature = "serde")]
pub use register_residual as tag_residual;
macro_rules! residual_maker {
    ($num:expr, $( ($idx:expr, $name:ident, $var:ident) ),*) => {
        paste! {
            #[doc=concat!("Residual trait for ", $num, " variables")]
            pub trait [<Residual $num>]: Residual
            {
                $(
                    #[doc=concat!("Type of variable ", $idx)]
                    type $var: VariableDtype;
                )*
                /// The total input dimension
                type DimIn: DimName;
                /// The output dimension of the residual
                type DimOut: DimName;
                /// Differentiator type (see [Diff](crate::linalg::Diff))
                type Differ: Diff;

                /// Main residual computation
                ///
                /// If implementing your own residual, this is the only method you need to implement.
                /// It is generic over the dtype to allow for differentiable types.
                fn [<residual $num>]<T: Numeric>(&self, $($name: Alias<Self::$var, T>,)*) -> VectorX<T>;

                #[doc="Wrapper that unpacks and calls [" [<residual $num>] "](Self::" [<residual $num>] ")."]
                fn [<residual $num _values>](&self, values: &Values, keys: &[Key]) -> VectorX
                where
                    $(
                        Self::$var: 'static,
                    )*
                 {
                    // Unwrap everything
                    $(
                        let $name: &Self::$var = values.get_unchecked(keys[$idx]).unwrap_or_else(|| {
                            panic!("Key not found in values: {:?} with type {}", keys[$idx], std::any::type_name::<Self::$var>())
                        });
                    )*
                    self.[<residual $num>]($($name.clone(),)*)
                }


                #[doc="Wrapper that unpacks variables and computes jacobians using [" [<residual $num>] "](Self::" [<residual $num>] ")."]
                fn [<residual $num _jacobian>](&self, values: &Values, keys: &[Key]) -> DiffResult<VectorX, MatrixX>
                where
                    $(
                        Self::$var: 'static,
                    )*
                {
                    // Unwrap everything
                    $(
                        let $name: &Self::$var = values.get_unchecked(keys[$idx]).unwrap_or_else(|| {
                            panic!("Key not found in values: {:?} with type {}", keys[$idx], std::any::type_name::<Self::$var>())
                        });
                    )*
                    Self::Differ::[<jacobian_ $num>](|$($name,)*| self.[<residual $num>]($($name,)*), $($name,)*)
                }
            }
        }
    };
}

residual_maker!(1, (0, v1, V1));
residual_maker!(2, (0, v1, V1), (1, v2, V2));
residual_maker!(3, (0, v1, V1), (1, v2, V2), (2, v3, V3));
residual_maker!(4, (0, v1, V1), (1, v2, V2), (2, v3, V3), (3, v4, V4));
residual_maker!(
    5,
    (0, v1, V1),
    (1, v2, V2),
    (2, v3, V3),
    (3, v4, V4),
    (4, v5, V5)
);
residual_maker!(
    6,
    (0, v1, V1),
    (1, v2, V2),
    (2, v3, V3),
    (3, v4, V4),
    (4, v5, V5),
    (5, v6, V6)
);
