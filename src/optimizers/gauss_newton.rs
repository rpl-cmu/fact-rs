use faer_ext::IntoNalgebra;

use super::{BaseOptParams, OptObserverVec, OptResult, Optimizer};
use crate::{
    containers::{Graph, GraphOrder, Values, ValuesOrder},
    dtype,
    linalg::DiffResult,
    linear::{LinearSolver, LinearValues},
};

// TODO: Build function to create?
/// The Gauss-Newton optimizer
///
/// Solves $A \Delta \Theta = b$ directly for each optimizer steps. Parameters
/// can be modified using the `params` field, and observers add using
/// `observers`. Additionally, is generic over the linear solver, but defaults
/// to [CholeskySolver]. See the [linear](crate::linear) module for more linear
/// solver options.
pub struct GaussNewton {
    graph: Graph,
    solver: Box<dyn LinearSolver>,
    /// Basic parameters for the optimizer
    pub params: BaseOptParams,
    /// Observers for the optimizer
    pub observers: OptObserverVec,
    // For caching computation between steps
    graph_order: Option<GraphOrder>,
}

impl GaussNewton {
    pub fn new(graph: Graph) -> Self {
        Self {
            graph,
            solver: Default::default(),
            observers: Default::default(),
            params: Default::default(),
            graph_order: None,
        }
    }

    pub fn graph(&self) -> &Graph {
        &self.graph
    }
}

impl Optimizer for GaussNewton {
    type Params = BaseOptParams;

    fn error(&self, values: &Values) -> dtype {
        self.graph.error(values)
    }

    fn params(&self) -> &BaseOptParams {
        &self.params
    }

    fn init(&mut self, _values: &Values) {
        // TODO: Some way to manual specify how to compute ValuesOrder
        // Precompute the sparsity pattern
        self.graph_order = Some(
            self.graph
                .sparsity_pattern(ValuesOrder::from_values(_values)),
        );
    }

    fn step(&mut self, mut values: Values, idx: usize) -> OptResult<Values> {
        // Solve the linear system
        let linear_graph = self.graph.linearize(&values);
        let DiffResult { value: r, diff: j } =
            linear_graph.residual_jacobian(self.graph_order.as_ref().expect("Missing graph order"));

        // Solve Ax = b
        let delta = self
            .solver
            .solve_lst_sq(j.as_ref(), r.as_ref())
            .as_ref()
            .into_nalgebra()
            .column(0)
            .clone_owned();

        // Update the values
        let dx = LinearValues::from_order_and_vector(
            self.graph_order
                .as_ref()
                .expect("Missing graph order")
                .order
                .clone(),
            delta,
        );
        values.oplus_mut(&dx);

        self.observers.notify(&values, idx);

        Ok(values)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::test_optimizer;

    test_optimizer!(GaussNewton);
}
