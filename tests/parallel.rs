//! `Graph::error` and `Graph::linearize` return the same bits as evaluating
//! the factors one by one, in order, whether or not the `rayon` feature is
//! enabled.

use factrs::{containers::ValuesOrder, dtype, linear::LinearGraph, utils::load_g20};

#[test]
fn graph_evaluation_matches_serial() {
    let (graph, values) = load_g20("examples/data/sphere2500.g2o");

    let serial_error: dtype = graph.iter().map(|f| f.error(&values)).sum();
    assert_eq!(graph.error(&values).to_bits(), serial_error.to_bits());

    let graph_order = graph.sparsity_pattern(ValuesOrder::from_values(&values));
    let parallel = graph.linearize(&values).residual_jacobian(&graph_order);
    let serial = LinearGraph::from_vec(graph.iter().map(|f| f.linearize(&values)).collect())
        .residual_jacobian(&graph_order);

    assert_eq!(parallel.value.nrows(), serial.value.nrows());
    for i in 0..serial.value.nrows() {
        assert_eq!(
            parallel.value[(i, 0)].to_bits(),
            serial.value[(i, 0)].to_bits()
        );
    }
    let (p, s) = (parallel.diff.val(), serial.diff.val());
    assert_eq!(p.len(), s.len());
    assert!(p.iter().zip(s).all(|(a, b)| a.to_bits() == b.to_bits()));
}
