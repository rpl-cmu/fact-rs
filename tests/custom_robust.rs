use factrs::{dtype, robust::RobustCost};

#[derive(Clone, Debug, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DoubleL2;

#[factrs::mark]
impl RobustCost for DoubleL2 {
    fn loss(&self, d2: dtype) -> dtype {
        d2
    }

    fn weight(&self, _d: dtype) -> dtype {
        2.0
    }
}

factrs::test_robust!(DoubleL2);

#[cfg(feature = "serde")]
mod ser_de {
    use super::*;

    // Make sure it serializes properly
    #[test]
    fn test_json_serialize() {
        let trait_object = &DoubleL2 as &dyn RobustCost;
        let json = serde_json::to_string(trait_object).unwrap();
        let expected = r#"{"tag":"DoubleL2"}"#;
        println!("json: {json}");
        assert_eq!(json, expected);
    }

    #[test]
    fn test_json_deserialize() {
        let json = r#"{"tag":"DoubleL2"}"#;
        let object = DoubleL2;
        let trait_object: Box<dyn RobustCost> = serde_json::from_str(json).unwrap();
        assert_eq!(trait_object.loss(1.0), object.loss(1.0));
        assert_eq!(trait_object.weight(1.0), object.weight(1.0));
    }
}
