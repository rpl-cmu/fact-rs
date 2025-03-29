use factrs::{core::*, linear::compute_conditional};

assign_symbols!(X: VectorVar3);

fn main() {
    let x0 = VectorVar3::new(1.0, 2.0, 3.0);
    let x1 = VectorVar3::new(4.0, 5.0, 6.0);

    let mut graph = Graph::new();
    graph.add_factor(fac![PriorResidual::new(x0.clone()), X(0)]);
    graph.add_factor(fac![BetweenResidual::new(x1.minus(&x0)), (X(0), X(1))]);

    let mut values = Values::new();
    values.insert(X(0), x0);
    values.insert(X(1), x1);

    let lin = graph.linearize(&values);

    let (cond, factor) = compute_conditional(&lin.factors, vec![X(1).into()]);

    println!("Conditional: {:?}", cond);
    println!("Factor: {:?}", factor);
}
