use std::{fmt, ops};

use super::VectorVar2;
use crate::{
    dtype,
    linalg::{
        vectorx, AllocatorBuffer, Const, DefaultAllocator, DimName, DualAllocator, DualVector,
        Matrix2, Matrix2x3, Matrix3, MatrixView, Numeric, SupersetOf, Vector2, Vector3,
        VectorView2, VectorView3, VectorViewX, VectorX,
    },
    variables::{MatrixLieGroup, Variable, SO2},
};

/// Special Euclidean Group in 2D
///
/// Implementation of SE(2) for 2D transformations.
#[derive(Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SE2<T: Numeric = dtype> {
    rot: SO2<T>,
    xy: Vector2<T>,
}

impl<T: Numeric> SE2<T> {
    /// Create a new SE2 from an angle, x, and y coordinates
    pub fn new(theta: T, x: T, y: T) -> Self {
        SE2 {
            rot: SO2::from_theta(theta),
            xy: Vector2::new(x, y),
        }
    }

    pub fn xy(&self) -> VectorView2<'_, T> {
        self.xy.as_view()
    }

    pub fn x(&self) -> T {
        self.xy[0]
    }

    pub fn y(&self) -> T {
        self.xy[1]
    }

    pub fn rot(&self) -> &SO2<T> {
        &self.rot
    }

    pub fn theta(&self) -> T {
        self.rot.log()[0]
    }
}

#[factrs::mark]
impl<T: Numeric> Variable for SE2<T> {
    type T = T;
    type Dim = Const<3>;
    type Alias<TT: Numeric> = SE2<TT>;

    fn identity() -> Self {
        SE2 {
            rot: Variable::identity(),
            xy: Vector2::zeros(),
        }
    }

    fn compose(&self, other: &Self) -> Self {
        SE2 {
            rot: &self.rot * &other.rot,
            xy: self.rot.apply(other.xy.as_view()) + self.xy,
        }
    }

    fn inverse(&self) -> Self {
        let inv = self.rot.inverse();
        SE2 {
            xy: -&inv.apply(self.xy.as_view()),
            rot: inv,
        }
    }

    #[allow(non_snake_case)]
    #[allow(clippy::needless_borrow)]
    fn exp(xi: VectorViewX<T>) -> Self {
        let theta = xi[0];
        let xy = Vector2::new(xi[1], xi[2]);

        let rot = SO2::<T>::exp(xi.rows(0, 1));

        let xy = if cfg!(feature = "fake_exp") {
            xy
        } else {
            let A;
            let B;
            if theta < T::from(1e-5) {
                A = T::from(1.0);
                B = T::from(0.0);
            } else {
                A = theta.sin() / theta;
                B = (T::from(1.0) - theta.cos()) / theta;
            };
            let V = Matrix2::new(A, -B, B, A);
            V * xy
        };

        SE2 { rot, xy }
    }

    #[allow(non_snake_case)]
    #[allow(clippy::needless_borrow)]
    fn log(&self) -> VectorX<T> {
        let theta = self.rot.log()[0];

        let xy = if cfg!(feature = "fake_exp") {
            &self.xy
        } else {
            let A;
            let B;
            if theta < T::from(1e-5) {
                A = T::from(1.0);
                B = T::from(0.0);
            } else {
                A = theta.sin() / theta;
                B = (T::from(1.0) - theta.cos()) / theta;
            };
            let V = Matrix2::new(A, -B, B, A);

            let Vinv = V.try_inverse().expect("V is not invertible");
            &(Vinv * self.xy)
        };

        vectorx![theta, xy[0], xy[1]]
    }

    fn cast<TT: Numeric + SupersetOf<Self::T>>(&self) -> Self::Alias<TT> {
        SE2 {
            rot: self.rot.cast(),
            xy: self.xy.cast(),
        }
    }

    fn dual_exp<N: DimName>(idx: usize) -> Self::Alias<DualVector<N>>
    where
        AllocatorBuffer<N>: Sync + Send,
        DefaultAllocator: DualAllocator<N>,
        DualVector<N>: Copy,
    {
        SE2 {
            rot: SO2::<dtype>::dual_exp(idx),
            xy: VectorVar2::<dtype>::dual_exp(idx + 1).into(),
        }
    }
}

impl<T: Numeric> MatrixLieGroup for SE2<T> {
    type TangentDim = Const<3>;
    type MatrixDim = Const<3>;
    type VectorDim = Const<2>;

    fn adjoint(&self) -> Matrix3<T> {
        let mut mat = Matrix3::<T>::zeros();

        let r_mat = self.rot.to_matrix();

        mat.fixed_view_mut::<2, 2>(0, 0).copy_from(&r_mat);
        mat[(0, 2)] = self.xy[2];
        mat[(1, 2)] = -self.xy[1];

        mat
    }

    fn hat(xi: VectorView3<T>) -> Matrix3<T> {
        let mut mat = Matrix3::<T>::zeros();
        mat[(0, 1)] = -xi[0];
        mat[(1, 0)] = xi[0];

        mat[(0, 2)] = xi[1];
        mat[(1, 2)] = xi[2];

        mat
    }

    fn vee(xi: MatrixView<3, 3, T>) -> Vector3<T> {
        Vector3::new(xi[(1, 0)], xi[(0, 1)], xi[(0, 2)])
    }

    fn apply(&self, v: VectorView2<T>) -> Vector2<T> {
        self.rot.apply(v) + self.xy
    }

    fn hat_swap(xi: VectorView2<T>) -> Matrix2x3<T> {
        let mut mat = Matrix2x3::<T>::zeros();
        mat.fixed_view_mut::<2, 1>(0, 0)
            .copy_from(&SO2::hat_swap(xi.as_view()));

        mat.fixed_view_mut::<2, 2>(0, 1)
            .copy_from(&Matrix2::identity());
        mat
    }

    fn to_matrix(&self) -> Matrix3<T> {
        let mut mat = Matrix3::<T>::identity();
        mat.fixed_view_mut::<2, 2>(0, 0)
            .copy_from(&self.rot.to_matrix());
        mat.fixed_view_mut::<2, 1>(0, 2).copy_from(&self.xy);
        mat
    }

    fn from_matrix(mat: MatrixView<3, 3, T>) -> Self {
        let rot = mat.fixed_view::<2, 2>(0, 0).clone_owned();
        let rot = SO2::from_matrix(rot.as_view());

        let xy = mat.fixed_view::<2, 1>(0, 2).into();

        SE2 { rot, xy }
    }
}

impl<T: Numeric> ops::Mul for SE2<T> {
    type Output = SE2<T>;

    #[inline]
    fn mul(self, other: Self) -> Self::Output {
        self.compose(&other)
    }
}

impl<T: Numeric> ops::Mul for &SE2<T> {
    type Output = SE2<T>;

    #[inline]
    fn mul(self, other: Self) -> Self::Output {
        self.compose(other)
    }
}

impl<T: Numeric> fmt::Display for SE2<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let precision: usize = f.precision().unwrap_or(3);
        write!(
            f,
            "SE2(theta: {:.p$}, x: {:.p$}, y: {:.p$})",
            self.rot.log()[0],
            self.xy[0],
            self.xy[1],
            p = precision
        )
    }
}

impl<T: Numeric> fmt::Debug for SE2<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let precision: usize = f.precision().unwrap_or(3);
        write!(
            f,
            "SE2 {{ rot: {:?}, x: {:.p$}, y: {:.p$} }}",
            self.rot,
            self.xy[0],
            self.xy[1],
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
