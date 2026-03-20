use std::{fmt, ops};

use crate::{
    dtype,
    linalg::{
        AllocatorBuffer, Const, DefaultAllocator, Derivative, DimName, DualAllocator, DualVector,
        Matrix1, Matrix2, MatrixView, Numeric, SupersetOf, Vector1, Vector2, VectorDim,
        VectorView1, VectorView2, VectorViewX, VectorX, vectorx,
    },
    variables::{MatrixLieGroup, Variable},
};

/// Special Orthogonal Group in 2D
///
/// Implementation of SO(2) for 2D rotations. Specifically, we use complex
/// numbers to represent rotations.
#[derive(Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SO2<T: Numeric = dtype> {
    a: T,
    b: T,
}

impl<T: Numeric> SO2<T> {
    /// Create a new SO2 from an angle in radians
    #[allow(clippy::needless_borrow)]
    pub fn from_theta(theta: T) -> Self {
        SO2 {
            a: (&theta).cos(),
            b: (&theta).sin(),
        }
    }

    /// Convert SO2 to an angle in radians
    pub fn to_theta(&self) -> T {
        self.b.atan2(self.a)
    }
}

#[factrs::mark]
impl<T: Numeric> Variable for SO2<T> {
    type T = T;
    type Dim = Const<1>;
    type Alias<TT: Numeric> = SO2<TT>;

    fn identity() -> Self {
        SO2 {
            a: T::from(1.0),
            b: T::from(0.0),
        }
    }

    fn inverse(&self) -> Self {
        SO2 {
            a: self.a,
            b: -self.b,
        }
    }

    fn compose(&self, other: &Self) -> Self {
        SO2 {
            a: self.a * other.a - self.b * other.b,
            b: self.a * other.b + self.b * other.a,
        }
    }

    fn exp(xi: VectorViewX<T>) -> Self {
        let theta = xi[0];
        SO2::from_theta(theta)
    }

    fn log(&self) -> VectorX<T> {
        vectorx![self.b.atan2(self.a)]
    }

    fn cast<TT: Numeric + SupersetOf<Self::T>>(&self) -> Self::Alias<TT> {
        SO2 {
            a: TT::from_subset(&self.a),
            b: TT::from_subset(&self.b),
        }
    }

    fn dual_exp<N: DimName>(idx: usize) -> Self::Alias<DualVector<N>>
    where
        AllocatorBuffer<N>: Sync + Send,
        DefaultAllocator: DualAllocator<N>,
        DualVector<N>: Copy,
    {
        let mut a = DualVector::<N>::from_re(1.0);
        a.eps = Derivative::new(Some(VectorDim::<N>::zeros()));

        let mut b = DualVector::<N>::from_re(0.0);
        let mut eps = VectorDim::<N>::zeros();
        eps[idx] = 1.0;
        b.eps = Derivative::new(Some(eps));

        SO2 { a, b }
    }
}

impl<T: Numeric> MatrixLieGroup for SO2<T> {
    type TangentDim = Const<1>;
    type MatrixDim = Const<2>;
    type VectorDim = Const<2>;

    fn adjoint(&self) -> Matrix1<T> {
        Matrix1::identity()
    }

    fn hat(xi: VectorView1<T>) -> Matrix2<T> {
        Matrix2::new(T::from(0.0), -xi[0], xi[0], T::from(0.0))
    }

    fn vee(xi: MatrixView<2, 2, T>) -> Vector1<T> {
        Vector1::new(xi[(1, 0)])
    }

    fn hat_swap(xi: VectorView2<T>) -> Vector2<T> {
        Vector2::new(-xi[1], xi[0])
    }

    fn apply(&self, v: VectorView2<T>) -> Vector2<T> {
        Vector2::new(v[0] * self.a - v[1] * self.b, v[0] * self.b + v[1] * self.a)
    }

    fn from_matrix(mat: MatrixView<2, 2, T>) -> Self {
        SO2 {
            a: mat[(0, 0)],
            b: mat[(1, 0)],
        }
    }

    fn to_matrix(&self) -> Matrix2<T> {
        Matrix2::new(self.a, -self.b, self.b, self.a)
    }
}

impl<T: Numeric> ops::Mul for SO2<T> {
    type Output = SO2<T>;

    #[inline]
    fn mul(self, other: Self) -> Self::Output {
        self.compose(&other)
    }
}

impl<T: Numeric> ops::Mul for &SO2<T> {
    type Output = SO2<T>;

    #[inline]
    fn mul(self, other: Self) -> Self::Output {
        self.compose(other)
    }
}

impl<T: Numeric> fmt::Display for SO2<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let precision: usize = f.precision().unwrap_or(3);
        write!(f, "SO2(theta: {:.p$})", self.log()[0], p = precision)
    }
}

impl<T: Numeric> fmt::Debug for SO2<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let precision: usize = f.precision().unwrap_or(3);
        write!(
            f,
            "SO2 {{ a: {:.p$}, b: {:.p$} }}",
            self.a,
            self.b,
            p = precision
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{test_lie, test_variable};

    test_variable!(SO2);

    test_lie!(SO2);
}
