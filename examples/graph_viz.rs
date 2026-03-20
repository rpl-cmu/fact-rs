use factrs::{
    core::{BetweenResidual, Graph, PriorResidual, SE2, assign_symbols, fac},
    traits::Variable,
};

assign_symbols!(X: SE2);

#[cfg(not(feature = "rerun"))]
fn rerun_viz(_graph: &Graph) {}

#[cfg(feature = "rerun")]
fn rerun_viz(graph: &Graph) {
    // Setup the rerun
    let rec = rerun::RecordingStreamBuilder::new("factrs-graph-viz")
        .connect_grpc()
        .unwrap();

    // Send the graph
    let (nodes, edges) = graph.into();
    rec.log_static("graph", &[&nodes as &dyn rerun::AsComponents, &edges])
        .expect("log failed");
}

fn main() {
    let mut graph = Graph::new();
    let id = SE2::identity();

    graph.add_factor(fac![PriorResidual::new(id.clone()), X(1)]);
    graph.add_factor(fac![BetweenResidual::new(id.clone()), (X(1), X(2))]);
    graph.add_factor(fac![BetweenResidual::new(id.clone()), (X(2), X(3))]);

    graph.add_factor(fac![BetweenResidual::new(id.clone()), (X(1), X(4))]);
    graph.add_factor(fac![BetweenResidual::new(id.clone()), (X(2), X(4))]);
    graph.add_factor(fac![BetweenResidual::new(id.clone()), (X(3), X(5))]);

    rerun_viz(&graph);
}
