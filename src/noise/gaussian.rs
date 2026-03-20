use std::fmt::{self, Debug};

use super::{NoiseModel, UnitNoise};
use crate::{
    dtype,
    linalg::{Const, Matrix, MatrixView, MatrixViewX, MatrixX, Vector, VectorView, VectorX},
};

/// A Gaussian noise model.
///
/// This noise model is used to represent Gaussian noise in a factor graph. This
/// will likely be the most used noise model.
#[derive(Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GaussianNoise<const N: usize> {
    sqrt_inf: Matrix<N, N>,
}

#[factrs::mark]
impl<const N: usize> NoiseModel for GaussianNoise<N> {
    type Dim = Const<N>;

    fn whiten_vec(&self, v: VectorX) -> VectorX {
        let mut out = VectorX::zeros(v.len());
        self.sqrt_inf.mul_to(&v, &mut out);
        out
    }

    fn whiten_mat(&self, m: MatrixX) -> MatrixX {
        let mut out = MatrixX::zeros(m.nrows(), m.ncols());
        self.sqrt_inf.mul_to(&m, &mut out);
        out
    }
}

impl<const N: usize> GaussianNoise<N> {
    /// Create a unit Gaussian noise.
    pub fn identity() -> UnitNoise<N> {
        UnitNoise
    }

    /// Create a Gaussian noise from a scalar sigma.
    pub fn from_scalar_sigma(sigma: dtype) -> Self {
        let sqrt_inf = Matrix::<N, N>::from_diagonal_element(1.0 / sigma);
        Self { sqrt_inf }
    }

    /// Create a Gaussian noise from a scalar covariance.
    pub fn from_scalar_cov(cov: dtype) -> Self {
        let sqrt_inf = Matrix::<N, N>::from_diagonal_element(1.0 / cov.sqrt());
        Self { sqrt_inf }
    }

    /// Create from split scalar sigmas.
    ///
    /// Will apply the first scalar to the first N/2 elements and the second
    /// scalar to the last N/2 elements. In the case of an odd N, the first N/2
    /// elements will have one less element than the last N/2 elements.
    pub fn from_split_sigma(sigma1: dtype, sigma2: dtype) -> Self {
        let mut sqrt_inf = Matrix::<N, N>::zeros();
        let inf1 = 1.0 / sigma1;
        let inf2 = 1.0 / sigma2;
        for i in 0..N / 2 {
            sqrt_inf[(i, i)] = inf1;
        }
        for i in N / 2..N {
            sqrt_inf[(i, i)] = inf2;
        }
        Self { sqrt_inf }
    }

    /// Create from split scalar covariances.
    ///
    /// Will apply the first scalar to the first N/2 elements and the second
    /// scalar to the last N/2 elements. In the case of an odd N, the first N/2
    /// elements will have one less element than the last N/2 elements.
    pub fn from_split_cov(cov1: dtype, cov2: dtype) -> Self {
        let mut sqrt_inf = Matrix::<N, N>::zeros();
        let inf1 = 1.0 / cov1.sqrt();
        let inf2 = 1.0 / cov2.sqrt();
        for i in 0..N / 2 {
            sqrt_inf[(i, i)] = inf1;
        }
        for i in N / 2..N {
            sqrt_inf[(i, i)] = inf2;
        }
        Self { sqrt_inf }
    }

    /// Create a diagonal Gaussian noise from a vector of sigmas.
    pub fn from_vec_sigma(sigma: VectorView<N>) -> Self {
        let sqrt_inf = Matrix::<N, N>::from_diagonal(&sigma.map(|x| 1.0 / x));
        Self { sqrt_inf }
    }

    /// Create a diagonal Gaussian noise from a vector of covariances.
    pub fn from_vec_cov(cov: VectorView<N>) -> Self {
        let sqrt_inf = Matrix::<N, N>::from_diagonal(&cov.map(|x| 1.0 / x.sqrt()));
        Self { sqrt_inf }
    }

    /// Create a diagonal Gaussian noise from a vector of information.
    pub fn from_vec_inf(inf: VectorView<N>) -> Self {
        let sqrt_inf = Matrix::<N, N>::from_diagonal(&inf.map(|x| x.sqrt()));
        Self { sqrt_inf }
    }

    /// Create a Gaussian noise from a covariance matrix.
    pub fn from_matrix_cov(cov: MatrixView<N, N>) -> Option<Self> {
        let sqrt_inf = cov.try_inverse()?.cholesky()?.l().transpose();
        Some(Self { sqrt_inf })
    }

    /// Create a Gaussian noise from an information matrix.
    pub fn from_matrix_inf(inf: MatrixView<N, N>) -> Option<Self> {
        let sqrt_inf = inf.cholesky()?.l().transpose();
        Some(Self { sqrt_inf })
    }

    /// Create a Gaussian noise from an upper-triangular sqrt information
    /// matrix.
    ///
    /// The sqrt information matrix R should satisfy `info = R^T * R`. This
    /// constructor uses R directly without performing Cholesky decomposition.
    pub fn from_matrix_sqrt_inf(sqrt_inf: Matrix<N, N>) -> Self {
        Self { sqrt_inf }
    }
}

fn is_diagonal(n: usize, m: MatrixViewX) -> bool {
    for i in 0..n {
        for j in 0..n {
            if i != j && m[(i, j)] != 0.0 {
                return false;
            }
        }
    }
    true
}

fn is_isotropic(n: usize, m: MatrixViewX) -> bool {
    let val = m[(0, 0)];
    for i in 1..n {
        if m[(i, i)] != val {
            return false;
        }
    }
    true
}

impl<const N: usize> Debug for GaussianNoise<N> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let precision = f.precision().unwrap_or(3);

        // If any type of diagonal, always print on a single line
        // Check if is a diagonal matrix
        if is_diagonal(N, self.sqrt_inf.as_view()) {
            // Check if all are the same
            if is_isotropic(N, self.sqrt_inf.as_view()) {
                return write!(
                    f,
                    "GaussianNoise{}(std: {:.p$})",
                    N,
                    1.0 / self.sqrt_inf[0],
                    p = precision
                );
            } else {
                write!(f, "GaussianNoise{N}(std: [")?;
                for i in 0..N {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{:.p$}", 1.0 / self.sqrt_inf[(i, i)], p = precision)?;
                }
                write!(f, "])")?;
            }
        } else if f.alternate() {
            writeln!(f, "GaussianNoise{N}(sqrt_inf:")?;
            let width = precision + 4;
            for i in 0..N {
                write!(f, "    [")?;
                for j in 0..N {
                    if j > 0 {
                        write!(f, ", ")?;
                    }
                    write!(
                        f,
                        "{:>w$.p$}",
                        self.sqrt_inf[(i, j)],
                        p = precision,
                        w = width
                    )?;
                }
                writeln!(f, "]")?;
            }
            write!(f, ")")?;
        } else {
            writeln!(
                f,
                "GaussianNoise{}(sqrt_inf: {:.p$?}",
                N,
                self.sqrt_inf,
                p = precision
            )?;
        }
        Ok(())
    }
}

macro_rules! make_gaussian_vector {
    ($($num:expr, [$($args:ident),*]);* $(;)?) => {$(
        impl GaussianNoise<$num> {
            /// Create a diagonal Gaussian noise from scalar sigmas.
            pub fn from_diag_sigmas($($args: dtype),*) -> Self {
                let sigmas = Vector::<$num>::new($($args,)*);
                Self::from_vec_sigma(sigmas.as_view())
            }

            /// Create a diagonal Gaussian noise from scalar covariances.
            pub fn from_diag_covs($($args: dtype,)*) -> Self {
                let sigmas = Vector::<$num>::new($($args,)*);
                Self::from_vec_cov(sigmas.as_view())
            }
        }
    )*};
}

make_gaussian_vector! {
    1, [s0];
    2, [s0, s1];
    3, [s0, s1, s2];
    4, [s0, s1, s2, s3];
    5, [s0, s1, s2, s3, s4];
    6, [s0, s1, s2, s3, s4, s5];
}

impl<const N: usize> fmt::Display for GaussianNoise<N> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "GaussianNoise{}: {:}", self.dim(), self.sqrt_inf)
    }
}
