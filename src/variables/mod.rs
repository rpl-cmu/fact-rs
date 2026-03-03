//! Variables to optimize
//!
//! This module contains the definition of the variables that can be optimized
//! using the optimization algorithms. We model each variable as a Lie group
//! $\mathcal{G}$ [^@solaMicroLieTheory2021], even if it is trivially so.
//! Because of this, each type $X \in \mathcal{G}$ and it's group action $\cdot$
//! must satisfy the following properties,
//!
//! | Property         | Description |
//! | :---:            | :---: |
//! | Identity         | $\exists I \ni X \cdot I = X = I \cdot X$ |
//! | Inverse          | $\exists X^{-1} \ni X \cdot X^{-1} = I = X^{-1} \cdot X$ |
//! | Closed           | $\forall X_1, X_2 \ni X_1 \cdot X_2 = X_3$ |
//! | Associativity    | $\forall X_1, X_2, X_3 \ni (X_1 \cdot X_2) \cdot X_3 = X_1 \cdot (X_2 \cdot X_3)$ |
//! | Exponential maps | $\xi \in \mathfrak{g} \implies \exp(\xi) \in \mathcal{G}$ |
//! | Logarithm maps   | $X \in \mathcal{G} \implies \log(X) \in \mathfrak{g}$ |
//!
//! Additionally, for optimization purposes, we adopt $\oplus$ and $\ominus$
//! operators [^@solaMicroLieTheory2021],
//!
//! | Property  | Right | Left |
//! | :---:     | :---: | :---: |
//! | $\oplus$  | $x \oplus \xi = x \cdot \exp(\xi)$   | $x \oplus \xi = \exp(\xi) \cdot x$ |
//! | $\ominus$ | $x \ominus y = \log(y^{-1} \cdot x)$ | $x \ominus y = \log(x \cdot y^{-1})$ |
//!
//! fact.rs defaults to using the right formulation, but the left formulation
//! can be enabled using the `left` feature. (Note this does have significant
//! consequences, including changing covariance interpretations)
//!
//! All these properties are encapsulated in the [Variable] trait. Additionally,
//! we parametrized each variable over its datatype to allow for dual numbers to
//! be propagated through residuals for automatic jacobian computation.
//!
//! If you want to implement a custom variable, you'll need to implement
//! [Variable] and [mark](factrs::mark) if using serde. We also recommend using
//! the [test_variable](crate::test_variable) macro to ensure the above
//! properties are satisfied.
//!
//! As an implementation detail, [Variable] is the main trait with all the
//! important corresponding methods. [VariableSafe] is a dyn-compatible version
//! that is implemented via a blanket. [VariableDtype] has both [Variable] and
//! [VariableSafe] as supertraits with the added constraint that the datatype
//! is [dtype](factrs::dtype).
//!
//! [^@solaMicroLieTheory2021]: Solà, Joan, et al. “A Micro Lie Theory for State Estimation in Robotics.” Arxiv:1812.01537, Dec. 2021
mod traits;
#[cfg(feature = "serde")]
pub use traits::tag_variable;
pub use traits::{MatrixLieGroup, Variable, VariableDtype, VariableSafe};

mod so2;
pub use so2::SO2;

mod se2;
pub use se2::SE2;

mod so3;
pub use so3::SO3;

mod se3;
pub use se3::SE3;

mod se23;
pub use se23::SE23;

mod vector;
pub use vector::{
    VectorVar, VectorVar1, VectorVar2, VectorVar3, VectorVar4, VectorVar5, VectorVar6,
};

mod imu_bias;
pub use imu_bias::ImuBias;

mod macros;
