use std::{fmt, ops};

use super::VectorVar3;
use crate::{
    dtype,
    linalg::{
        AllocatorBuffer, Const, DefaultAllocator, DimName, DualAllocator, DualVector, Matrix3,
        Matrix3x9, Matrix5, Matrix9, MatrixView, Numeric, SupersetOf, Vector3, Vector9,
        VectorView3, VectorView9, VectorViewX, VectorX,
    },
    variables::{MatrixLieGroup, SO3, Variable},
};

/// Extended pose Group SE_2(3)
///
/// Implementation of SE_2(3)
#[derive(Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SE23<T: Numeric = dtype> {
    rot: SO3<T>,
    uvw: Vector3<T>,
    xyz: Vector3<T>,
}

impl<T: Numeric> SE23<T> {
    /// Create a new SE23 from an SO3 and two Vector3s
    pub fn from_rot_vel_trans(rot: SO3<T>, uvw: Vector3<T>, xyz: Vector3<T>) -> Self {
        SE23 { rot, uvw, xyz }
    }

    pub fn rot(&self) -> &SO3<T> {
        &self.rot
    }

    pub fn uvw(&self) -> VectorView3<'_, T> {
        self.uvw.as_view()
    }

    pub fn xyz(&self) -> VectorView3<'_, T> {
        self.xyz.as_view()
    }
}

#[factrs::mark]
impl<T: Numeric> Variable for SE23<T> {
    type T = T;
    type Dim = Const<9>;
    type Alias<TT: Numeric> = SE23<TT>;

    fn identity() -> Self {
        SE23 {
            rot: Variable::identity(),
            uvw: Vector3::zeros(),
            xyz: Vector3::zeros(),
        }
    }

    fn compose(&self, other: &Self) -> Self {
        SE23 {
            rot: &self.rot * &other.rot,
            uvw: self.rot.apply(other.uvw.as_view()) + self.uvw,
            xyz: self.rot.apply(other.xyz.as_view()) + self.xyz,
        }
    }

    fn inverse(&self) -> Self {
        let inv = self.rot.inverse();
        SE23 {
            uvw: -&inv.apply(self.uvw.as_view()),
            xyz: -&inv.apply(self.xyz.as_view()),
            rot: inv,
        }
    }

    #[allow(non_snake_case)]
    fn exp(xi: VectorViewX<T>) -> Self {
        let xi_rot = xi.fixed_view::<3, 1>(0, 0).clone_owned();
        let rot = SO3::<T>::exp(xi.rows(0, 3));
        let uvw = Vector3::new(xi[3], xi[4], xi[5]);
        let xyz = Vector3::new(xi[6], xi[7], xi[8]);

        let (uvw, xyz) = if cfg!(feature = "fake_exp") {
            (uvw, xyz)
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
            }
            let I = Matrix3::identity();
            let wx = SO3::hat(xi_rot.as_view());
            let V = I + wx * B + wx * wx * C;
            (V * uvw, V * xyz)
        };

        SE23 { rot, uvw, xyz }
    }

    #[allow(non_snake_case)]
    fn log(&self) -> VectorX<T> {
        let mut xi = VectorX::zeros(9);
        let xi_theta = self.rot.log();

        let (uvw, xyz) = if cfg!(feature = "fake_exp") {
            (self.uvw, self.xyz)
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
            }
            let I = Matrix3::identity();
            let wx = SO3::hat(xi_theta.as_view());
            let V_inv = I - wx * B + wx * wx * C;
            (V_inv * self.uvw, V_inv * self.xyz)
        };

        xi.as_mut_slice()[0..3].copy_from_slice(xi_theta.as_slice());
        xi.as_mut_slice()[3..6].copy_from_slice(uvw.as_slice());
        xi.as_mut_slice()[6..9].copy_from_slice(xyz.as_slice());

        xi
    }

    fn cast<TT: Numeric + SupersetOf<Self::T>>(&self) -> Self::Alias<TT> {
        SE23 {
            rot: self.rot.cast(),
            uvw: self.uvw.cast(),
            xyz: self.xyz.cast(),
        }
    }

    fn dual_exp<N: DimName>(idx: usize) -> Self::Alias<DualVector<N>>
    where
        AllocatorBuffer<N>: Sync + Send,
        DefaultAllocator: DualAllocator<N>,
        DualVector<N>: Copy,
    {
        SE23 {
            rot: SO3::<dtype>::dual_exp(idx),
            uvw: VectorVar3::<dtype>::dual_exp(idx + 3).into(),
            xyz: VectorVar3::<dtype>::dual_exp(idx + 6).into(),
        }
    }
}

impl<T: Numeric> MatrixLieGroup for SE23<T> {
    type TangentDim = Const<9>;
    type MatrixDim = Const<5>;
    type VectorDim = Const<3>;

    fn adjoint(&self) -> Matrix9<T> {
        let mut mat = Matrix9::zeros();

        let r_mat = self.rot.to_matrix();
        let v_r_mat = SO3::hat(self.uvw.as_view()) * r_mat;
        let t_r_mat = SO3::hat(self.xyz.as_view()) * r_mat;

        mat.fixed_view_mut::<3, 3>(0, 0).copy_from(&r_mat);
        mat.fixed_view_mut::<3, 3>(3, 3).copy_from(&r_mat);
        mat.fixed_view_mut::<3, 3>(6, 6).copy_from(&r_mat);
        mat.fixed_view_mut::<3, 3>(3, 0).copy_from(&v_r_mat);
        mat.fixed_view_mut::<3, 3>(6, 0).copy_from(&t_r_mat);

        mat
    }

    fn hat(xi: VectorView9<T>) -> Matrix5<T> {
        let mut mat = Matrix5::zeros();
        mat[(0, 1)] = -xi[2];
        mat[(0, 2)] = xi[1];
        mat[(1, 0)] = xi[2];
        mat[(1, 2)] = -xi[0];
        mat[(2, 0)] = -xi[1];
        mat[(2, 1)] = xi[0];

        mat[(0, 3)] = xi[3];
        mat[(1, 3)] = xi[4];
        mat[(2, 3)] = xi[5];

        mat[(0, 4)] = xi[6];
        mat[(1, 4)] = xi[7];
        mat[(2, 4)] = xi[8];

        mat
    }

    fn vee(xi: MatrixView<5, 5, T>) -> Vector9<T> {
        let xi = Vector9::from_column_slice(&[
            xi[(2, 1)],
            xi[(0, 2)],
            xi[(1, 0)],
            xi[(0, 3)],
            xi[(1, 3)],
            xi[(2, 3)],
            xi[(0, 4)],
            xi[(1, 4)],
            xi[(2, 4)],
        ]);

        xi
    }

    fn hat_swap(xi: VectorView3<T>) -> Matrix3x9<T> {
        // The velocity term does not affect the rigid body transformation of a 3D vector
        let mut mat = Matrix3x9::zeros();
        mat.fixed_view_mut::<3, 3>(0, 0)
            .copy_from(&SO3::hat_swap(xi.as_view()));
        mat.fixed_view_mut::<3, 3>(0, 6)
            .copy_from(&Matrix3::identity());
        mat
    }

    fn apply(&self, v: VectorView3<T>) -> Vector3<T> {
        self.rot.apply(v) + self.xyz
    }

    fn to_matrix(&self) -> Matrix5<T> {
        let mut mat = Matrix5::identity();
        mat.fixed_view_mut::<3, 3>(0, 0)
            .copy_from(&self.rot.to_matrix());
        mat.fixed_view_mut::<3, 1>(0, 3).copy_from(&self.uvw);
        mat.fixed_view_mut::<3, 1>(0, 4).copy_from(&self.xyz);
        mat
    }

    fn from_matrix(mat: MatrixView<5, 5, T>) -> Self {
        let rot = mat.fixed_view::<3, 3>(0, 0).clone_owned();
        let rot = SO3::from_matrix(rot.as_view());

        let uvw = mat.fixed_view::<3, 1>(0, 3).clone_owned();
        let xyz = mat.fixed_view::<3, 1>(0, 4).clone_owned();

        SE23 { rot, uvw, xyz }
    }
}

impl<T: Numeric> ops::Mul for SE23<T> {
    type Output = Self;

    fn mul(self, other: Self) -> Self::Output {
        self.compose(&other)
    }
}

impl<T: Numeric> fmt::Display for SE23<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let precision = f.precision().unwrap_or(3);
        let rlog = self.rot.log();
        write!(
            f,
            "SE23 {{ r: [{:.p$?}, {:.p$?}, {:.p$?}], v: [{:.p$}, {:.p$}, {:.p$}], t: [{:.p$}, {:.p$}, {:.p$}] }}",
            rlog[0],
            rlog[1],
            rlog[2],
            self.uvw[0],
            self.uvw[1],
            self.uvw[2],
            self.xyz[0],
            self.xyz[1],
            self.xyz[2],
            p = precision
        )
    }
}

impl<T: Numeric> fmt::Debug for SE23<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let precision = f.precision().unwrap_or(3);
        write!(
            f,
            "SE23 {{ r: {:.p$?}, v: {:.p$}, {:.p$}, {:.p$}, t: [{:.p$}, {:.p$?}, {:.p$?}] }}",
            self.rot,
            self.uvw[0],
            self.uvw[1],
            self.uvw[2],
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

    test_variable!(SE23);

    test_lie!(SE23);
}
