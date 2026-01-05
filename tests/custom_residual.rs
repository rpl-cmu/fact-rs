use core::fmt;

use factrs::{
    dtype,
    linalg::{ForwardProp, Numeric, VectorX, vectorx},
    residuals::Residual1,
    traits::Variable,
    variables::SE2,
};
use nalgebra::Const;

#[derive(Clone, Debug, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct XPrior {
    x: dtype,
}

impl XPrior {
    pub fn new(x: dtype) -> XPrior {
        XPrior { x }
    }
}

#[factrs::mark]
impl Residual1 for XPrior {
    type Differ = ForwardProp<<Self as Residual1>::DimIn>;
    type V1 = SE2;
    type DimIn = <SE2 as Variable>::Dim;
    type DimOut = Const<1>;

    fn residual1<T: Numeric>(&self, v: SE2<T>) -> VectorX<T> {
        let z_meas = T::from(self.x);
        vectorx![z_meas - v.xy().x]
    }
}

impl fmt::Display for XPrior {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "XPrior({})", self.x)
    }
}

// TODO: Some tests to make sure it optimizes

#[cfg(feature = "serde")]
mod ser_de {
    use factrs::{containers::Values, symbols::X, traits::Residual};

    use super::*;

    // Make sure it serializes properly
    #[test]
    fn test_json_serialize() {
        let trait_object = &XPrior::new(1.2) as &dyn Residual;
        let json = serde_json::to_string(trait_object).unwrap();
        let expected = r#"{"tag":"XPrior","x":1.2}"#;
        println!("json: {}", json);
        assert_eq!(json, expected);
    }

    #[test]
    fn test_json_deserialize() {
        let json = r#"{"tag":"XPrior","x":1.2}"#;
        let trait_object: Box<dyn Residual> = serde_json::from_str(json).unwrap();

        let mut values = Values::new();
        values.insert_unchecked(X(0), SE2::new(0.0, 1.2, 0.0));
        let error = trait_object.residual(&values, &[X(0).into()])[0];

        assert_eq!(trait_object.dim_in(), 3);
        assert_eq!(trait_object.dim_out(), 1);
        assert_eq!(error, 0.0);
    }
}
