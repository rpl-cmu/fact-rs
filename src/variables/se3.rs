use std::{fmt, ops};

use super::VectorVar3;
use crate::{
    dtype,
    linalg::{
        AllocatorBuffer, Const, DefaultAllocator, DimName, DualAllocator, DualVector, Matrix3,
        Matrix3x6, Matrix4, Matrix6, MatrixView, Numeric, SupersetOf, Vector3, Vector6,
        VectorView3, VectorView6, VectorViewX, VectorX,
    },
    variables::{MatrixLieGroup, Variable, SO3},
};

/// Special Euclidean Group in 3D
///
/// Implementation of SE(3) for 3D transformations.
#[derive(Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SE3<T: Numeric = dtype> {
    rot: SO3<T>,
    xyz: Vector3<T>,
}

impl<T: Numeric> SE3<T> {
    /// Create a new SE3 from an SO3 and a Vector3
    pub fn from_rot_trans(rot: SO3<T>, xyz: Vector3<T>) -> Self {
        SE3 { rot, xyz }
    }

    pub fn rot(&self) -> &SO3<T> {
        &self.rot
    }

    pub fn xyz(&self) -> VectorView3<'_, T> {
        self.xyz.as_view()
    }
}

#[factrs::mark]
impl<T: Numeric> Variable for SE3<T> {
    type T = T;
    type Dim = Const<6>;
    type Alias<TT: Numeric> = SE3<TT>;

    fn identity() -> Self {
        SE3 {
            rot: Variable::identity(),
            xyz: Vector3::zeros(),
        }
    }

    fn compose(&self, other: &Self) -> Self {
        SE3 {
            rot: &self.rot * &other.rot,
            xyz: self.rot.apply(other.xyz.as_view()) + self.xyz,
        }
    }

    fn inverse(&self) -> Self {
        let inv = self.rot.inverse();
        SE3 {
            xyz: -&inv.apply(self.xyz.as_view()),
            rot: inv,
        }
    }

    #[allow(non_snake_case)]
    fn exp(xi: VectorViewX<T>) -> Self {
        let xi_rot = xi.fixed_view::<3, 1>(0, 0).clone_owned();
        let rot = SO3::<T>::exp(xi.rows(0, 3));

        let xyz = Vector3::new(xi[3], xi[4], xi[5]);

        let xyz = if cfg!(feature = "fake_exp") {
            xyz
        } else {
            let w2 = xi_rot.norm_squared();
            let B;
            let C;
            if w2 < T::from(1e-5) {
                B = T::from(0.5);
                C = T::from(1.0 / 6.0);
            } else {
                let w = w2.sqrt();
                let A = w.sin() / w;
                B = (T::from(1.0) - w.cos()) / w2;
                C = (T::from(1.0) - A) / w2;
            };
            let I = Matrix3::identity();
            let wx = SO3::hat(xi_rot.as_view());
            let V = I + wx * B + wx * wx * C;
            V * xyz
        };

        SE3 { rot, xyz }
    }

    #[allow(non_snake_case)]
    fn log(&self) -> VectorX<T> {
        let mut xi = VectorX::zeros(6);
        let xi_theta = self.rot.log();

        let xyz = if cfg!(feature = "fake_exp") {
            self.xyz
        } else {
            let w2 = xi_theta.norm_squared();
            let B;
            let C;
            if w2 < T::from(1e-5) {
                B = T::from(0.5);
                C = T::from(1.0 / 6.0);
            } else {
                let w = w2.sqrt();
                let A = w.sin() / w;
                B = (T::from(1.0) - w.cos()) / w2;
                C = (T::from(1.0) - A) / w2;
            };

            let I = Matrix3::identity();
            let wx = SO3::hat(xi_theta.as_view());
            let V = I + wx * B + wx * wx * C;

            let Vinv = V.try_inverse().expect("V is not invertible");
            Vinv * self.xyz
        };

        xi.as_mut_slice()[0..3].clone_from_slice(xi_theta.as_slice());
        xi.as_mut_slice()[3..6].clone_from_slice(xyz.as_slice());

        xi
    }

    fn cast<TT: Numeric + SupersetOf<Self::T>>(&self) -> Self::Alias<TT> {
        SE3 {
            rot: self.rot.cast(),
            xyz: self.xyz.cast(),
        }
    }

    fn dual_exp<N: DimName>(idx: usize) -> Self::Alias<DualVector<N>>
    where
        AllocatorBuffer<N>: Sync + Send,
        DefaultAllocator: DualAllocator<N>,
        DualVector<N>: Copy,
    {
        SE3 {
            rot: SO3::<dtype>::dual_exp(idx),
            xyz: VectorVar3::<dtype>::dual_exp(idx + 3).into(),
        }
    }
}

impl<T: Numeric> MatrixLieGroup for SE3<T> {
    type TangentDim = Const<6>;
    type MatrixDim = Const<4>;
    type VectorDim = Const<3>;

    fn adjoint(&self) -> Matrix6<T> {
        let mut mat = Matrix6::zeros();

        let r_mat = self.rot.to_matrix();
        let t_r_mat = SO3::hat(self.xyz.as_view()) * r_mat;

        mat.fixed_view_mut::<3, 3>(0, 0).copy_from(&r_mat);
        mat.fixed_view_mut::<3, 3>(3, 3).copy_from(&r_mat);
        mat.fixed_view_mut::<3, 3>(3, 0).copy_from(&t_r_mat);

        mat
    }

    fn hat(xi: VectorView6<T>) -> Matrix4<T> {
        let mut mat = Matrix4::zeros();
        mat[(0, 1)] = -xi[2];
        mat[(0, 2)] = xi[1];
        mat[(1, 0)] = xi[2];
        mat[(1, 2)] = -xi[0];
        mat[(2, 0)] = -xi[1];
        mat[(2, 1)] = xi[0];

        mat[(0, 3)] = xi[3];
        mat[(1, 3)] = xi[4];
        mat[(2, 3)] = xi[5];

        mat
    }

    fn vee(xi: MatrixView<4, 4, T>) -> Vector6<T> {
        Vector6::new(
            xi[(2, 1)],
            xi[(0, 2)],
            xi[(1, 0)],
            xi[(0, 3)],
            xi[(1, 3)],
            xi[(2, 3)],
        )
    }

    fn hat_swap(xi: VectorView3<T>) -> Matrix3x6<T> {
        let mut mat = Matrix3x6::zeros();
        mat.fixed_view_mut::<3, 3>(0, 0)
            .copy_from(&SO3::hat_swap(xi.as_view()));
        mat.fixed_view_mut::<3, 3>(0, 3)
            .copy_from(&Matrix3::identity());
        mat
    }

    fn apply(&self, v: VectorView3<T>) -> Vector3<T> {
        self.rot.apply(v) + self.xyz
    }

    fn to_matrix(&self) -> Matrix4<T> {
        let mut mat = Matrix4::<T>::identity();
        mat.fixed_view_mut::<3, 3>(0, 0)
            .copy_from(&self.rot.to_matrix());
        mat.fixed_view_mut::<3, 1>(0, 3).copy_from(&self.xyz);
        mat
    }

    fn from_matrix(mat: MatrixView<4, 4, T>) -> Self {
        let rot = mat.fixed_view::<3, 3>(0, 0).clone_owned();
        let rot = SO3::from_matrix(rot.as_view());

        let xyz = mat.fixed_view::<3, 1>(0, 3).into();

        SE3 { rot, xyz }
    }
}

impl<T: Numeric> ops::Mul for SE3<T> {
    type Output = SE3<T>;

    #[inline]
    fn mul(self, other: Self) -> Self::Output {
        self.compose(&other)
    }
}

impl<T: Numeric> ops::Mul for &SE3<T> {
    type Output = SE3<T>;

    #[inline]
    fn mul(self, other: Self) -> Self::Output {
        self.compose(other)
    }
}

impl<T: Numeric> fmt::Display for SE3<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let precision = f.precision().unwrap_or(3);
        let rlog = self.rot.log();
        write!(
            f,
            "SE3(r: [{:.p$}, {:.p$}, {:.p$}], t: [{:.p$}, {:.p$}, {:.p$}])",
            rlog[0],
            rlog[1],
            rlog[2],
            self.xyz[0],
            self.xyz[1],
            self.xyz[2],
            p = precision
        )
    }
}

impl<T: Numeric> fmt::Debug for SE3<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let precision = f.precision().unwrap_or(3);
        write!(
            f,
            "SE3 {{ r: {:.p$?}, t: [{:.p$}, {:.p$}, {:.p$}] }}",
            self.rot,
            self.xyz[0],
            self.xyz[1],
            self.xyz[2],
            p = precision
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{test_lie, test_variable};

    test_variable!(SE3);

    test_lie!(SE3);
}
