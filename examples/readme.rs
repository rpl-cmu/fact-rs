use factrs::{
    assign_symbols,
    core::{BetweenResidual, GaussNewton, Graph, Huber, PriorResidual, Values, SO2},
    fac,
    traits::*,
};

// Assign symbols to variable types
assign_symbols!(X: SO2);

fn main() {
    // Make all the values
    let mut values = Values::new();

    let x = SO2::from_theta(1.0);
    let y = SO2::from_theta(2.0);
    values.insert(X(0), SO2::identity());
    values.insert(X(1), SO2::identity());

    // Make the factors & insert into graph
    let mut graph = Graph::new();
    let res = PriorResidual::new(x.clone());
    let factor = fac![res, X(0)];
    graph.add_factor(factor);

    let res = BetweenResidual::new(y.minus(&x));
    let factor = fac![res, (X(0), X(1)), 0.1 as std, Huber::default()];
    graph.add_factor(factor);

    // Optimize!
    let mut opt: GaussNewton = GaussNewton::new_default(graph);
    let result = opt.optimize(values).unwrap();
    println!("Results {:#}", result);
}
