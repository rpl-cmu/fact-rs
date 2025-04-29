use faer_ext::IntoNalgebra;

use super::{BaseOptParams, OptObserverVec, OptResult, Optimizer};
use crate::{
    containers::{Graph, GraphOrder, Values, ValuesOrder},
    dtype,
    linalg::DiffResult,
    linear::{LinearSolver, LinearValues},
};

/// The Gauss-Newton optimizer
///
/// Solves $A \Delta \Theta = b$ directly for each optimizer steps. It defaults
/// to using [CholeskySolver](crate::linear::CholeskySolver) under the hood, but
/// this can be changed using [set_solver](GaussNewton::set_solver). See
/// the [linear](crate::linear) module for more linear solver options.
pub struct GaussNewton {
    graph: Graph,
    // TODO: Need to handle this in a better way?
    solver: Box<dyn LinearSolver>,
    /// Basic parameters for the optimizer
    params: BaseOptParams,
    /// Observers for the optimizer
    observers: OptObserverVec,
    // For caching computation between steps
    graph_order: Option<GraphOrder>,
}

impl GaussNewton {
    /// Sets the linear solver to use for the optimizer.
    pub fn set_solver(&mut self, solver: impl LinearSolver + 'static) {
        self.solver = Box::new(solver);
    }
}

impl Optimizer for GaussNewton {
    type Params = BaseOptParams;

    fn new(params: Self::Params, graph: Graph) -> Self {
        Self {
            graph,
            solver: Default::default(),
            observers: OptObserverVec::default(),
            params,
            graph_order: None,
        }
    }

    fn observers(&self) -> &OptObserverVec {
        &self.observers
    }

    fn observers_mut(&mut self) -> &mut OptObserverVec {
        &mut self.observers
    }

    fn graph(&self) -> &Graph {
        &self.graph
    }

    fn graph_mut(&mut self) -> &mut Graph {
        &mut self.graph
    }

    fn error(&self, values: &Values) -> dtype {
        self.graph.error(values)
    }

    fn params(&self) -> &BaseOptParams {
        &self.params
    }

    fn init(&mut self, _values: &Values) -> Vec<&'static str> {
        // TODO: Some way to manual specify how to compute ValuesOrder
        // Precompute the sparsity pattern
        self.graph_order = Some(
            self.graph
                .sparsity_pattern(ValuesOrder::from_values(_values)),
        );

        Vec::new()
    }

    fn step(&mut self, mut values: Values, _idx: usize) -> OptResult<(Values, String)> {
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

        Ok((values, String::new()))
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::test_optimizer;

    test_optimizer!(GaussNewton);
}
