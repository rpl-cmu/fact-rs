#[cfg(feature = "serde")]
mod ser_de {
    use factrs::{
        containers::Values, residuals::PriorResidual, symbols::X, traits::Residual,
        variables::VectorVar1,
    };

    #[test]
    fn test_vector_serialize() {
        let trait_object = &PriorResidual::new(VectorVar1::new(2.3)) as &dyn Residual;
        let json = serde_json::to_string(trait_object).unwrap();
        let expected = r#"{"tag":"PriorResidual<VectorVar<1>>","prior":[2.3]}"#;
        println!("json: {json}");
        assert_eq!(json, expected);
    }

    #[test]
    fn test_vector() {
        let json = r#"{"tag":"PriorResidual<VectorVar<1>>","prior":[1.2]}"#;
        let trait_object: Box<dyn Residual> = serde_json::from_str(json).unwrap();

        let mut values = Values::new();
        values.insert_unchecked(X(0), VectorVar1::new(1.2));
        let error = trait_object.residual(&values, &[X(0).into()])[0];

        assert_eq!(trait_object.dim_in(), 1);
        assert_eq!(trait_object.dim_out(), 1);
        assert_eq!(error, 0.0);
    }
}
