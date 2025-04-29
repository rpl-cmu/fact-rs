use std::ops::Mul;

use faer::sparse::{SparseColMat, Triplet};
use faer_ext::IntoNalgebra;

use super::{BaseOptParams, OptError, OptObserverVec, OptParams, OptResult, Optimizer};
use crate::{
    containers::{Graph, GraphOrder, Values, ValuesOrder},
    dtype,
    linalg::DiffResult,
    linear::{LinearSolver, LinearValues},
};

/// Levenberg-Marquardt specific parameters
#[derive(Clone, Debug)]
pub struct LevenParams {
    pub lambda_min: dtype,
    pub lambda_max: dtype,
    pub lambda_factor: dtype,
    pub diagonal_damping: bool,
    pub base: BaseOptParams,
}

impl Default for LevenParams {
    fn default() -> Self {
        Self {
            lambda_min: 1e-10,
            lambda_max: 1e5,
            lambda_factor: 10.0,
            diagonal_damping: true,
            base: Default::default(),
        }
    }
}

impl OptParams for LevenParams {
    fn base_params(&self) -> &BaseOptParams {
        &self.base
    }
}

/// The Levenberg-Marquadt optimizer
///
/// Solves a damped version of the normal equations,  
/// $$A^\top A \Delta \Theta + \lambda diag(A) = A^\top b$$
/// each optimizer steps. It defaults
/// to using [CholeskySolver](crate::linear::CholeskySolver) under the hood, but
/// this can be changed using [set_solver](LevenMarquardt::set_solver). See the
/// [linear](crate::linear) module for more linear solver options.
pub struct LevenMarquardt {
    graph: Graph,
    solver: Box<dyn LinearSolver>,
    /// Levenberg-Marquardt specific parameters
    params: LevenParams,
    /// Observers for the optimizer
    observers: OptObserverVec,
    lambda: dtype,
    // For caching computation between steps
    graph_order: Option<GraphOrder>,
}

impl LevenMarquardt {
    /// Set the linear solver to use for solving the linear system
    pub fn set_solver(&mut self, solver: impl LinearSolver + 'static) {
        self.solver = Box::new(solver);
    }
}

impl Optimizer for LevenMarquardt {
    type Params = LevenParams;

    fn new(params: Self::Params, graph: Graph) -> Self {
        Self {
            graph,
            solver: Default::default(),
            observers: OptObserverVec::default(),
            params,
            lambda: 1e-5,
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

    fn params(&self) -> &BaseOptParams {
        &self.params.base
    }

    fn error(&self, values: &Values) -> crate::dtype {
        self.graph.error(values)
    }

    fn init(&mut self, values: &Values) -> Vec<&'static str> {
        // TODO: Some way to manual specify how to computer ValuesOrder
        // Precompute the sparsity pattern
        self.graph_order = Some(
            self.graph
                .sparsity_pattern(ValuesOrder::from_values(values)),
        );

        vec!["   Lambda   "]
    }

    // TODO: More sophisticated stopping criteria based on magnitude of the gradient
    fn step(&mut self, mut values: Values, _idx: usize) -> OptResult<(Values, String)> {
        // Make an ordering
        let order = ValuesOrder::from_values(&values);

        // Form the linear system
        let linear_graph = self.graph.linearize(&values);
        let DiffResult { value: r, diff: j } =
            linear_graph.residual_jacobian(self.graph_order.as_ref().expect("Missing graph order"));

        // Form A
        let jtj = j
            .as_ref()
            .transpose()
            .to_col_major()
            .expect("J failed to transpose")
            .mul(j.as_ref());

        // Form I
        let triplets_i = if self.params.diagonal_damping {
            (0..jtj.ncols())
                .map(|i| Triplet::new(i as isize, i as isize, jtj[(i, i)]))
                .collect::<Vec<_>>()
        } else {
            (0..jtj.ncols())
                .map(|i| Triplet::new(i as isize, i as isize, 1.0))
                .collect::<Vec<_>>()
        };
        let i = SparseColMat::<usize, dtype>::try_new_from_nonnegative_triplets(
            jtj.ncols(),
            jtj.ncols(),
            &triplets_i,
        )
        .expect("Failed to make damping terms");

        // Form b
        let b = j.as_ref().transpose().mul(&r);

        let mut dx = LinearValues::zero_from_order(order.clone());
        let old_error = linear_graph.error(&dx);

        loop {
            // Make Ax = b
            // Have to cast due to missing impl in faer for f32
            #[allow(clippy::unnecessary_cast)]
            let a = &jtj + (&i * self.lambda as f64);

            // Solve Ax = b
            let delta = self
                .solver
                .solve_symmetric(a.as_ref(), b.as_ref())
                .as_ref()
                .into_nalgebra()
                .column(0)
                .clone_owned();
            dx = LinearValues::from_order_and_vector(
                self.graph_order
                    .as_ref()
                    .expect("Missing graph order")
                    .order
                    .clone(),
                delta,
            );

            // Update our cost
            let curr_error = linear_graph.error(&dx);

            if curr_error < old_error {
                break;
            }

            self.lambda *= self.params.lambda_factor;
            if self.lambda > self.params.lambda_max {
                return Err(OptError::FailedToStep);
            }
        }

        // Update the values
        values.oplus_mut(&dx);
        self.lambda /= self.params.lambda_factor;
        if self.lambda < self.params.lambda_min {
            self.lambda = self.params.lambda_min;
        }

        Ok((values, format!("{:^12.4e} |", self.lambda)))
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::test_optimizer;

    test_optimizer!(LevenMarquardt);
}
