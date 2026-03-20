// Re-export all nalgebra types to put default dtype on everything
// Misc imports
use nalgebra::{self as na, OVector};
pub use nalgebra::{
    ComplexField, Const, DefaultAllocator, Dim, DimName, Dyn, RealField, allocator::Allocator,
    dmatrix as matrixx, dvector as vectorx,
};
pub use simba::scalar::SupersetOf;

use crate::dtype;

// Make it easier to bind the buffer type
pub type AllocatorBuffer<N> = <DefaultAllocator as Allocator<N>>::Buffer<dtype>;

// ------------------------- Vector/Matrix Aliases ------------------------- //
// Vectors
pub type Vector<const N: usize, T = dtype> = na::SVector<T, N>;
pub type VectorX<T = dtype> = na::DVector<T>;
pub type Vector1<T = dtype> = na::SVector<T, 1>;
pub type Vector2<T = dtype> = na::SVector<T, 2>;
pub type Vector3<T = dtype> = na::SVector<T, 3>;
pub type Vector4<T = dtype> = na::SVector<T, 4>;
pub type Vector5<T = dtype> = na::SVector<T, 5>;
pub type Vector6<T = dtype> = na::SVector<T, 6>;
pub type Vector7<T = dtype> = na::SVector<T, 7>;
pub type Vector8<T = dtype> = na::SVector<T, 8>;
pub type Vector9<T = dtype> = na::SVector<T, 9>;

// Matrices
// square
pub type MatrixX<T = dtype> = na::DMatrix<T>;
pub type Matrix1<T = dtype> = na::Matrix1<T>;
pub type Matrix2<T = dtype> = na::Matrix2<T>;
pub type Matrix3<T = dtype> = na::Matrix3<T>;
pub type Matrix4<T = dtype> = na::Matrix4<T>;
pub type Matrix5<T = dtype> = na::Matrix5<T>;
pub type Matrix6<T = dtype> = na::Matrix6<T>;
pub type Matrix7<T = dtype> = na::Matrix<
    T,
    Const<7>,
    Const<7>,
    <na::DefaultAllocator as Allocator<Const<7>, Const<7>>>::Buffer<T>,
>;
pub type Matrix8<T = dtype> = na::Matrix<
    T,
    Const<8>,
    Const<8>,
    <na::DefaultAllocator as Allocator<Const<8>, Const<8>>>::Buffer<T>,
>;
pub type Matrix9<T = dtype> = na::Matrix<
    T,
    Const<9>,
    Const<9>,
    <na::DefaultAllocator as Allocator<Const<9>, Const<9>>>::Buffer<T>,
>;

// row
pub type Matrix1xX<T = dtype> = na::Matrix1xX<T>;
pub type Matrix1x2<T = dtype> = na::Matrix1x2<T>;
pub type Matrix1x3<T = dtype> = na::Matrix1x3<T>;
pub type Matrix1x4<T = dtype> = na::Matrix1x4<T>;
pub type Matrix1x5<T = dtype> = na::Matrix1x5<T>;
pub type Matrix1x6<T = dtype> = na::Matrix1x6<T>;
pub type Matrix1x7<T = dtype> = na::Matrix<
    T,
    Const<1>,
    Const<7>,
    <na::DefaultAllocator as Allocator<Const<1>, Const<7>>>::Buffer<T>,
>;
pub type Matrix1x8<T = dtype> = na::Matrix<
    T,
    Const<1>,
    Const<8>,
    <na::DefaultAllocator as Allocator<Const<1>, Const<8>>>::Buffer<T>,
>;
pub type Matrix1x9<T = dtype> = na::Matrix<
    T,
    Const<1>,
    Const<9>,
    <na::DefaultAllocator as Allocator<Const<1>, Const<9>>>::Buffer<T>,
>;

// two rows
pub type Matrix2xX<T = dtype> = na::Matrix2xX<T>;
pub type Matrix2x3<T = dtype> = na::Matrix2x3<T>;
pub type Matrix2x4<T = dtype> = na::Matrix2x4<T>;
pub type Matrix2x5<T = dtype> = na::Matrix2x5<T>;
pub type Matrix2x6<T = dtype> = na::Matrix2x6<T>;
pub type Matrix2x7<T = dtype> = na::Matrix<
    T,
    Const<2>,
    Const<7>,
    <na::DefaultAllocator as Allocator<Const<2>, Const<7>>>::Buffer<T>,
>;
pub type Matrix2x8<T = dtype> = na::Matrix<
    T,
    Const<2>,
    Const<8>,
    <na::DefaultAllocator as Allocator<Const<2>, Const<8>>>::Buffer<T>,
>;
pub type Matrix2x9<T = dtype> = na::Matrix<
    T,
    Const<2>,
    Const<9>,
    <na::DefaultAllocator as Allocator<Const<2>, Const<9>>>::Buffer<T>,
>;

// three rows
pub type Matrix3xX<T = dtype> = na::Matrix3xX<T>;
pub type Matrix3x2<T = dtype> = na::Matrix3x2<T>;
pub type Matrix3x4<T = dtype> = na::Matrix3x4<T>;
pub type Matrix3x5<T = dtype> = na::Matrix3x5<T>;
pub type Matrix3x6<T = dtype> = na::Matrix3x6<T>;
pub type Matrix3x7<T = dtype> = na::Matrix<
    T,
    Const<3>,
    Const<7>,
    <na::DefaultAllocator as Allocator<Const<3>, Const<7>>>::Buffer<T>,
>;
pub type Matrix3x8<T = dtype> = na::Matrix<
    T,
    Const<3>,
    Const<8>,
    <na::DefaultAllocator as Allocator<Const<3>, Const<8>>>::Buffer<T>,
>;
pub type Matrix3x9<T = dtype> = na::Matrix<
    T,
    Const<3>,
    Const<9>,
    <na::DefaultAllocator as Allocator<Const<3>, Const<9>>>::Buffer<T>,
>;

// four rows
pub type Matrix4xX<T = dtype> = na::Matrix4xX<T>;
pub type Matrix4x2<T = dtype> = na::Matrix4x2<T>;
pub type Matrix4x3<T = dtype> = na::Matrix4x3<T>;
pub type Matrix4x5<T = dtype> = na::Matrix4x5<T>;
pub type Matrix4x6<T = dtype> = na::Matrix4x6<T>;

// five rows
pub type Matrix5xX<T = dtype> = na::Matrix5xX<T>;
pub type Matrix5x2<T = dtype> = na::Matrix5x2<T>;
pub type Matrix5x3<T = dtype> = na::Matrix5x3<T>;
pub type Matrix5x4<T = dtype> = na::Matrix5x4<T>;
pub type Matrix5x6<T = dtype> = na::Matrix5x6<T>;

// six rows
pub type Matrix6xX<T = dtype> = na::Matrix6xX<T>;
pub type Matrix6x2<T = dtype> = na::Matrix6x2<T>;
pub type Matrix6x3<T = dtype> = na::Matrix6x3<T>;
pub type Matrix6x4<T = dtype> = na::Matrix6x4<T>;
pub type Matrix6x5<T = dtype> = na::Matrix6x5<T>;

// dynamic rows
pub type MatrixXx2<T = dtype> = na::MatrixXx2<T>;
pub type MatrixXx3<T = dtype> = na::MatrixXx3<T>;
pub type MatrixXx4<T = dtype> = na::MatrixXx4<T>;
pub type MatrixXx5<T = dtype> = na::MatrixXx5<T>;
pub type MatrixXx6<T = dtype> = na::MatrixXx6<T>;
pub type MatrixXxN<const N: usize, T = dtype> =
    na::Matrix<T, Dyn, Const<N>, na::VecStorage<T, Dyn, Const<N>>>;

// Views - aka references of matrices
pub type MatrixViewX<'a, T = dtype> = na::MatrixView<'a, T, Dyn, Dyn>;

pub type Matrix<const R: usize, const C: usize = 1, T = dtype> = na::Matrix<
    T,
    Const<R>,
    Const<C>,
    <na::DefaultAllocator as Allocator<Const<R>, Const<C>>>::Buffer<T>,
>;
pub type MatrixView<'a, const R: usize, const C: usize = 1, T = dtype> =
    na::MatrixView<'a, T, Const<R>, Const<C>>;

pub type VectorView<'a, const N: usize, T = dtype> = na::VectorView<'a, T, Const<N>>;
pub type VectorViewX<'a, T = dtype> = na::VectorView<'a, T, Dyn>;
pub type VectorView1<'a, T = dtype> = na::VectorView<'a, T, Const<1>>;
pub type VectorView2<'a, T = dtype> = na::VectorView<'a, T, Const<2>>;
pub type VectorView3<'a, T = dtype> = na::VectorView<'a, T, Const<3>>;
pub type VectorView4<'a, T = dtype> = na::VectorView<'a, T, Const<4>>;
pub type VectorView5<'a, T = dtype> = na::VectorView<'a, T, Const<5>>;
pub type VectorView6<'a, T = dtype> = na::VectorView<'a, T, Const<6>>;
pub type VectorView7<'a, T = dtype> = na::VectorView<'a, T, Const<7>>;
pub type VectorView8<'a, T = dtype> = na::VectorView<'a, T, Const<8>>;
pub type VectorView9<'a, T = dtype> = na::VectorView<'a, T, Const<9>>;

// Generic, taking in sizes with Const
pub type VectorDim<N, T = dtype> = OVector<T, N>;
pub type MatrixDim<R, C = Const<1>, T = dtype> =
    na::Matrix<T, R, C, <na::DefaultAllocator as Allocator<R, C>>::Buffer<T>>;
pub type MatrixViewDim<'a, R, C = Const<1>, T = dtype> = na::MatrixView<'a, T, R, C>;
