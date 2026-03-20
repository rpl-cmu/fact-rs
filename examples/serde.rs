use factrs::core::{
    BetweenResidual, GemanMcClure, Graph, PriorResidual, SE2, SO2, Values, assign_symbols, fac,
};

assign_symbols!(X: SO2; Y: SE2);

fn main() {
    // ------------------------- Serialize values ------------------------- //
    let x = SO2::from_theta(0.6);
    let y = SE2::new(1.0, 2.0, 0.3);
    let mut values = Values::new();
    values.insert(X(0), x.clone());
    values.insert(Y(1), y.clone());

    println!("------ Serializing Values ------");

    let serialized = serde_json::to_string_pretty(&values).unwrap();
    println!("serialized = {serialized}");

    // Convert the JSON string back to a Point.
    let deserialized: Values = serde_json::from_str(&serialized).unwrap();
    println!("deserialized = {deserialized:#}");

    // ------------------------- Serialize graph ------------------------- //
    let prior = PriorResidual::new(x);
    let bet = BetweenResidual::new(y);

    let prior = fac![prior, X(0), 0.1 as cov, GemanMcClure::default()];
    let bet = fac![bet, (Y(0), Y(1)), 10.0 as cov];

    let mut graph = Graph::new();
    graph.add_factor(prior);
    graph.add_factor(bet);

    println!("\n\n------ Serializing Graph ------");

    let serialized = serde_json::to_string_pretty(&graph).unwrap();
    println!("serialized = {serialized}");

    let deserialized: Graph = serde_json::from_str(&serialized).unwrap();
    println!("deserialized = {deserialized:#?}");
}
