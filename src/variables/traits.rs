use std::fmt::{Debug, Display};

use downcast_rs::{Downcast, impl_downcast};

use crate::{
    dtype,
    linalg::{
        AllocatorBuffer, Const, DefaultAllocator, DimName, DualAllocator, DualVector, MatrixDim,
        MatrixViewDim, Numeric, SupersetOf, VectorDim, VectorViewX, VectorX,
    },
};

/// Variable trait for Lie groups
///
/// All variables must implement this trait to be used in the optimization
/// algorithms. See [module level documentation](crate::variables) for more
/// details.
pub trait Variable: Clone + Sized + Display + Debug {
    /// Datatype of the variable
    type T: Numeric;
    /// Dimension of the Lie group / Tangent space
    type Dim: DimName;
    const DIM: usize = Self::Dim::DIM;
    /// Alias for the type for dual conversion
    type Alias<TT: Numeric>: Variable<T = TT>;

    // Group operations
    /// Identity element of the group
    fn identity() -> Self;
    /// Inverse of the group element
    fn inverse(&self) -> Self;
    /// Composition of two group elements
    fn compose(&self, other: &Self) -> Self;
    /// Exponential map (trivial if a vector space)
    fn exp(delta: VectorViewX<Self::T>) -> Self;
    /// Logarithm map (trivial if a vector space)
    fn log(&self) -> VectorX<Self::T>;

    /// Conversion to dual space
    ///
    /// Simply convert all interior values of dtype to DD.
    fn cast<TT: Numeric + SupersetOf<Self::T>>(&self) -> Self::Alias<TT>;

    /// Dimension helper
    fn dim(&self) -> usize {
        Self::DIM
    }

    /// Adds value from the tangent space to the group element
    ///
    /// By default this uses the "right" version [^@solaMicroLieTheory2021]
    /// $$
    /// x \oplus \xi = x \cdot \exp(\xi)
    /// $$
    /// If the "left" feature is enabled, instead this turns to
    /// $$
    /// x \oplus \xi = \exp(\xi) \cdot x
    /// $$
    ///
    /// [^@solaMicroLieTheory2021]: Solà, Joan, et al. “A Micro Lie Theory for State Estimation in Robotics.” Arxiv:1812.01537, Dec. 2021
    #[inline]
    fn oplus(&self, xi: VectorViewX<Self::T>) -> Self {
        if cfg!(feature = "left") {
            self.oplus_left(xi)
        } else {
            self.oplus_right(xi)
        }
    }

    #[inline]
    fn oplus_right(&self, xi: VectorViewX<Self::T>) -> Self {
        self.compose(&Self::exp(xi))
    }

    #[inline]
    fn oplus_left(&self, xi: VectorViewX<Self::T>) -> Self {
        Self::exp(xi).compose(self)
    }

    /// Compares two group elements in the tangent space
    ///
    /// By default this uses the "right" version [^@solaMicroLieTheory2021]
    /// $$
    /// x \ominus y = \log(y^{-1} \cdot x)
    /// $$
    /// If the "left" feature is enabled, instead this turns to
    /// $$
    /// x \ominus y = \log(x \cdot y^{-1})
    /// $$
    ///
    /// [^@solaMicroLieTheory2021]: Solà, Joan, et al. “A Micro Lie Theory for State Estimation in Robotics.” Arxiv:1812.01537, Dec. 2021
    #[inline]
    fn ominus(&self, y: &Self) -> VectorX<Self::T> {
        if cfg!(feature = "left") {
            self.ominus_left(y)
        } else {
            self.ominus_right(y)
        }
    }

    #[inline]
    fn ominus_right(&self, y: &Self) -> VectorX<Self::T> {
        y.inverse().compose(self).log()
    }

    #[inline]
    fn ominus_left(&self, y: &Self) -> VectorX<Self::T> {
        self.compose(&y.inverse()).log()
    }

    /// Subtract out portion from other variable.
    ///
    /// This can be seen as a "tip-to-tail" computation. IE it computes the
    /// transformation between two poses. I like to think of it as "taking away"
    /// the portion subtracted out, for example given a chain of poses $a, b,
    /// c$, the following "removes" the portion from $a$ to $b$.
    ///
    /// $$
    /// {}_a T_c \boxminus {}_a T_b = ({}_a T_b)^{-1} {}_a T_c = {}_b T_c
    /// $$
    ///
    /// This operation is NOT effected by the left/right feature.
    #[inline]
    fn minus(&self, other: &Self) -> Self {
        other.inverse().compose(self)
    }

    // TODO: This function is kind of ugly still
    // It'd be nice if it used the dtype of the type itself instead of making a
    // dtype with a generic

    /// Setup group element correctly using the tangent space
    ///
    /// By default this uses the exponential map to propagate dual numbers to
    /// the variable to setup the jacobian properly. Can be hardcoded to avoid
    /// the repeated computation.
    #[inline]
    fn dual_exp<N: DimName>(idx: usize) -> Self::Alias<DualVector<N>>
    where
        AllocatorBuffer<N>: Sync + Send,
        DefaultAllocator: DualAllocator<N>,
        DualVector<N>: Copy,
    {
        let mut tv: VectorX<DualVector<N>> = VectorX::zeros(Self::DIM);
        let n = VectorDim::<N>::zeros().shape_generic().0;
        for (i, tvi) in tv.iter_mut().enumerate() {
            tvi.eps = num_dual::Derivative::derivative_generic(n, Const::<1>, idx + i)
        }
        Self::Alias::<DualVector<N>>::exp(tv.as_view())
    }

    /// Applies the tangent vector in dual space
    ///
    /// Takes the results from [dual_exp](Self::dual_exp) and applies the
    /// tangent vector using the right/left oplus operator.
    #[inline]
    fn dual<N: DimName>(&self, idx: usize) -> Self::Alias<DualVector<N>>
    where
        AllocatorBuffer<N>: Sync + Send,
        DefaultAllocator: DualAllocator<N>,
        DualVector<N>: Copy + SupersetOf<Self::T>,
    {
        // Setups tangent vector -> exp, then we compose here
        let casted: Self::Alias<DualVector<N>> = self.cast::<DualVector<N>>();
        let setup: Self::Alias<DualVector<N>> = Self::dual_exp(idx);
        if cfg!(feature = "left") {
            setup.compose(&casted)
        } else {
            casted.compose(&setup)
        }
    }
}

/// The object safe version of [Variable].
///
/// This trait is used to allow for dynamic dispatch of noise models.
/// Implemented for all types that implement [Variable].
// TODO: Rename to VariableGeneric? Something like that
#[cfg_attr(feature = "serde", typetag::serde(tag = "tag"))]
pub trait VariableSafe: Debug + Display + Downcast {
    fn clone_box(&self) -> Box<dyn VariableSafe>;

    fn dim(&self) -> usize;

    fn oplus_mut(&mut self, delta: VectorViewX);
}

#[cfg_attr(feature = "serde", typetag::serde)]
impl<V: Variable<T = dtype> + 'static> VariableSafe for V {
    fn clone_box(&self) -> Box<dyn VariableSafe> {
        Box::new((*self).clone())
    }

    fn dim(&self) -> usize {
        self.dim()
    }

    fn oplus_mut(&mut self, delta: VectorViewX) {
        *self = self.oplus(delta);
    }
}

impl_downcast!(VariableSafe);

impl Clone for Box<dyn VariableSafe> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}
#[cfg(feature = "serde")]
pub use register_variablesafe as tag_variable;

/// Alias for variable with T = dtype
///
/// This trait is 100% for convenience. It wraps all types that implements
/// [VariableSafe] and [Variable] (with proper aliases) into a single trait.
pub trait VariableDtype: VariableSafe + Variable<T = dtype, Alias<dtype> = Self> {}
impl<V: VariableSafe + Variable<T = dtype, Alias<dtype> = V>> VariableDtype for V {}

use nalgebra as na;

/// Properties specific to matrix Lie groups
///
/// Many variables used in robotics state estimation are specific Lie Groups
/// that consist of matrix elements. We encapsulate a handful of their
/// properties here.
pub trait MatrixLieGroup: Variable
where
    na::DefaultAllocator: na::allocator::Allocator<Self::TangentDim, Self::TangentDim>,
    na::DefaultAllocator: na::allocator::Allocator<Self::MatrixDim, Self::MatrixDim>,
    na::DefaultAllocator: na::allocator::Allocator<Self::VectorDim, Self::TangentDim>,
    na::DefaultAllocator: na::allocator::Allocator<Self::TangentDim, Const<1>>,
    na::DefaultAllocator: na::allocator::Allocator<Self::VectorDim, Const<1>>,
{
    /// Dimension of the tangent space
    type TangentDim: DimName;
    /// Dimension of the corresponding matrix representation
    type MatrixDim: DimName;
    /// Dimension of vectors that can be transformed
    type VectorDim: DimName;

    /// Adjoint operator
    fn adjoint(&self) -> MatrixDim<Self::TangentDim, Self::TangentDim, Self::T>;

    /// Hat operator
    ///
    /// Converts a vector from $\xi \in \mathbb{R}^n$ to the Lie algebra
    /// $\xi^\wedge \in \mathfrak{g}$
    fn hat(
        xi: MatrixViewDim<'_, Self::TangentDim, Const<1>, Self::T>,
    ) -> MatrixDim<Self::MatrixDim, Self::MatrixDim, Self::T>;

    /// Vee operator
    ///
    /// Inverse of the hat operator. Converts a matrix from the Lie algebra
    /// $\xi^\wedge \in \mathfrak{g}$ to a vector $\xi \in \mathbb{R}^n$
    fn vee(
        xi: MatrixViewDim<'_, Self::MatrixDim, Self::MatrixDim, Self::T>,
    ) -> MatrixDim<Self::TangentDim, Const<1>, Self::T>;

    /// Hat operator for swapping
    ///
    /// This is our own version of the hat operator used for swapping with
    /// vectors to be rotated. For many common Lie groups, this encodes the
    /// following "swap"
    ///
    /// $$
    /// \xi^\wedge p = \text{hat\\_swap}(p) \xi
    /// $$
    ///
    ///
    /// For example, in SO(3) $\text{hat\\_swap}(p) = -p^\wedge$.
    fn hat_swap(
        xi: MatrixViewDim<'_, Self::VectorDim, Const<1>, Self::T>,
    ) -> MatrixDim<Self::VectorDim, Self::TangentDim, Self::T>;

    /// Transform a vector
    ///
    /// Transform/rotate a vector using the group element. In SO(3), this is
    /// rotation, in SE(3) this is a rigid body transformation.
    fn apply(
        &self,
        v: MatrixViewDim<'_, Self::VectorDim, Const<1>, Self::T>,
    ) -> MatrixDim<Self::VectorDim, Const<1>, Self::T>;

    /// Transform group element to a matrix
    fn to_matrix(&self) -> MatrixDim<Self::MatrixDim, Self::MatrixDim, Self::T>;

    /// Create a group element from a matrix
    fn from_matrix(mat: MatrixViewDim<'_, Self::MatrixDim, Self::MatrixDim, Self::T>) -> Self;
}
