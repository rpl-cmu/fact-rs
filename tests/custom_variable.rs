use std::fmt;

use factrs::{
    dtype,
    linalg::{Numeric, SupersetOf, Vector1},
    traits::Variable,
};

#[derive(Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MyVar<T: Numeric = dtype> {
    val: Vector1<T>,
}

impl<T: Numeric> MyVar<T> {
    pub fn new(v: T) -> MyVar<T> {
        MyVar {
            val: Vector1::new(v),
        }
    }
}

#[factrs::mark]
impl<T: Numeric> Variable for MyVar<T> {
    type T = T;
    type Dim = factrs::linalg::Const<1>;
    type Alias<TT: Numeric> = MyVar<TT>;

    fn identity() -> Self {
        MyVar {
            val: Vector1::zeros(),
        }
    }

    fn inverse(&self) -> Self {
        MyVar { val: -self.val }
    }

    fn compose(&self, other: &Self) -> Self {
        MyVar {
            val: self.val + other.val,
        }
    }

    fn exp(delta: factrs::linalg::VectorViewX<T>) -> Self {
        let val = Vector1::new(delta[0]);
        MyVar { val }
    }

    fn log(&self) -> factrs::linalg::VectorX<T> {
        factrs::linalg::vectorx![self.val.x]
    }

    fn cast<TT: Numeric + SupersetOf<Self::T>>(&self) -> Self::Alias<TT> {
        MyVar {
            val: self.val.cast(),
        }
    }
}

impl<T: Numeric> fmt::Display for MyVar<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "MyVar(g: {:.3})", self.val.x)
    }
}

impl<T: Numeric> fmt::Debug for MyVar<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

factrs::test_variable!(MyVar);

#[cfg(feature = "serde")]
mod ser_de {
    use factrs::{
        assert_variable_eq, containers::Values, residuals::PriorResidual, symbols::X,
        traits::Residual, variables::VariableSafe,
    };

    use super::*;

    // Make sure it serializes properly
    #[test]
    fn test_json_serialize() {
        let trait_object = &MyVar::new(5.5) as &dyn VariableSafe;
        let json = serde_json::to_string(trait_object).unwrap();
        let expected = r#"{"tag":"MyVar","val":[5.5]}"#;
        println!("json: {json}");
        assert_eq!(json, expected);
    }

    #[test]
    fn test_json_deserialize() {
        let json = r#"{"tag":"MyVar","val":[4.5]}"#;
        let trait_object: Box<dyn VariableSafe> = serde_json::from_str(json).unwrap();
        let var: &MyVar = trait_object.downcast_ref::<MyVar>().unwrap();
        assert_variable_eq!(var, MyVar::new(4.5));
    }

    // Make sure the prior can as well
    #[test]
    fn test_prior_serialize() {
        let trait_object = &PriorResidual::new(MyVar::new(2.3)) as &dyn Residual;
        let json = serde_json::to_string(trait_object).unwrap();
        let expected = r#"{"tag":"PriorResidual<MyVar>","prior":{"val":[2.3]}}"#;
        println!("json: {json}");
        assert_eq!(json, expected);
    }

    #[test]
    fn test_prior_deserialize() {
        let json = r#"{"tag":"PriorResidual<MyVar>","prior":{"val":[1.2]}}"#;
        let trait_object: Box<dyn Residual> = serde_json::from_str(json).unwrap();

        let mut values = Values::new();
        values.insert_unchecked(X(0), MyVar::new(1.2));
        let error = trait_object.residual(&values, &[X(0).into()])[0];

        assert_eq!(trait_object.dim_in(), 1);
        assert_eq!(trait_object.dim_out(), 1);
        assert_eq!(error, 0.0);
    }
}
