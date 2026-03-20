use core::fmt;

use factrs::{
    linalg::{MatrixX, VectorX},
    noise::NoiseModel,
};
use nalgebra::Const;

#[derive(Clone, Debug, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DoubleCovariance<const N: usize>;

#[factrs::mark]
impl<const N: usize> NoiseModel for DoubleCovariance<N> {
    type Dim = Const<N>;

    fn whiten_vec(&self, v: VectorX) -> VectorX {
        2.0 * v
    }

    fn whiten_mat(&self, m: MatrixX) -> MatrixX {
        2.0 * m
    }
}

impl<const N: usize> fmt::Display for DoubleCovariance<N> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "DoubleCovariance{}", self.dim())
    }
}

#[cfg(feature = "serde")]
mod ser_de {
    use factrs::linalg::vectorx;

    use super::*;

    // Make sure it serializes properly
    #[test]
    fn test_json_serialize() {
        let trait_object = &DoubleCovariance::<2> as &dyn NoiseModel;
        let json = serde_json::to_string(trait_object).unwrap();
        let expected = r#"{"tag":"DoubleCovariance<2>"}"#;
        println!("json: {json}");
        assert_eq!(json, expected);
    }

    #[test]
    fn test_json_deserialize() {
        let json = r#"{"tag":"DoubleCovariance<2>"}"#;
        let trait_object: Box<dyn NoiseModel> = serde_json::from_str(json).unwrap();
        let vec = vectorx![1.0, 2.0];
        assert_eq!(trait_object.whiten_vec(vec.clone()), 2.0 * vec);
    }
}
