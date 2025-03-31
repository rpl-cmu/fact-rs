//! Optimizers for solving non-linear least squares problems.
//!
//! Specifically, given a nonlinear least squares problem of the form,
//!
//! $$
//! \Theta^* = \argmin_{\Theta} \sum_{i} \rho_i(||r_i(\Theta)||_{\Sigma_i} )
//! $$
//!
//! These optimizers function by first, linearizing to a linear least squares
//! problem,
//! $$
//! \Delta \Theta = \argmin_{\Delta \Theta} \sum_{i} ||A_i (\Delta \Theta)_i -
//! b_i ||^2
//! $$
//!
//! This can be rearranged to a large, sparse linear system given by,
//!
//! $$
//! A \Delta \Theta = b
//! $$
//!
//! which can then be solved. For example, Gauss-Newton solves this directly,
//! while Levenberg-Marquardt adds a damping term to the diagonal of $A^\top A$
//! to ensure positive definiteness.
//!
//! This module provides a set of optimizers that can be used to solve
//! non-linear least squares problems. Each optimizer implements the [Optimizer]
//! trait to give similar structure and usage.
//!
//! Additionally observers can be added to the optimizer to monitor the progress
//! of the optimization. A prebuilt [Rerun](https://rerun.io/) can be enabled via
//! the `rerun` feature.
//!
//! If you desire to implement your own optimizer, we additionally recommend
//! using the [test_optimizer](crate::test_optimizer) macro to run a handful of
//! simple tests over a few different variable types to ensure correctness.
mod traits;
pub use traits::{
    BaseOptParams, OptError, OptObserver, OptObserverVec, OptParams, OptResult, Optimizer,
};

mod macros;

mod gauss_newton;
pub use gauss_newton::GaussNewton;

mod levenberg_marquardt;
pub use levenberg_marquardt::{LevenMarquardt, LevenParams};

mod gnc;
pub use gnc::{GncGemanMcClure, GraduatedNonConvexity};

// These aren't tests themselves, but are helpers to test optimizers
#[cfg(test)]
pub mod test {
    use faer::assert_matrix_eq;
    use nalgebra::{DefaultAllocator, DimNameAdd, DimNameSum, ToTypenum};

    use super::*;
    use crate::{
        containers::{FactorBuilder, Graph, Values},
        dtype,
        linalg::{AllocatorBuffer, Const, DualAllocator, DualVector, VectorX},
        residuals::{BetweenResidual, PriorResidual, Residual},
        symbols::X,
        variables::VariableDtype,
    };

    pub fn optimize_prior<
        O,
        const DIM: usize,
        #[cfg(feature = "serde")] T: VariableDtype<Dim = nalgebra::Const<DIM>> + 'static + typetag::Tagged,
        #[cfg(not(feature = "serde"))] T: VariableDtype<Dim = nalgebra::Const<DIM>> + 'static,
    >(
        new: &dyn Fn(Graph) -> O,
    ) where
        PriorResidual<T>: Residual,
        O: Optimizer,
    {
        let t = VectorX::from_fn(T::DIM, |_, i| ((i + 1) as dtype) / 10.0);
        let p = T::exp(t.as_view());

        let mut values = Values::new();
        values.insert_unchecked(X(0), T::identity());

        let mut graph = Graph::new();
        let res = PriorResidual::new(p.clone());
        let factor = FactorBuilder::new1_unchecked(res, X(0)).build();
        graph.add_factor(factor);

        let mut opt = new(graph);
        values = opt.optimize(values).expect("Optimization failed");

        let out: &T = values.get_unchecked(X(0)).expect("Missing X(0)");
        assert_matrix_eq!(
            out.ominus(&p),
            VectorX::zeros(T::DIM),
            comp = abs,
            tol = 1e-6
        );
    }

    pub fn optimize_between<
        O,
        const DIM: usize,
        const DIM_DOUBLE: usize,
        #[cfg(feature = "serde")] T: VariableDtype<Dim = nalgebra::Const<DIM>> + 'static + typetag::Tagged,
        #[cfg(not(feature = "serde"))] T: VariableDtype<Dim = nalgebra::Const<DIM>> + 'static,
    >(
        new: &dyn Fn(Graph) -> O,
    ) where
        PriorResidual<T>: Residual,
        BetweenResidual<T>: Residual,
        O: Optimizer,
        Const<DIM>: ToTypenum,
        AllocatorBuffer<DimNameSum<Const<DIM>, Const<DIM>>>: Sync + Send,
        DefaultAllocator: DualAllocator<DimNameSum<Const<DIM>, Const<DIM>>>,
        DualVector<DimNameSum<Const<DIM>, Const<DIM>>>: Copy,
        Const<DIM>: DimNameAdd<Const<DIM>>,
    {
        let t = VectorX::from_fn(T::DIM, |_, i| ((i as dtype) - (T::DIM as dtype)) / 10.0);
        let p1 = T::exp(t.as_view());

        let t = VectorX::from_fn(T::DIM, |_, i| ((i + 1) as dtype) / 10.0);
        let p2 = T::exp(t.as_view());

        let mut values = Values::new();
        values.insert_unchecked(X(0), T::identity());
        values.insert_unchecked(X(1), T::identity());

        let mut graph = Graph::new();
        let res = PriorResidual::new(p1.clone());
        let factor = FactorBuilder::new1_unchecked(res, X(0)).build();
        graph.add_factor(factor);

        let diff = p2.minus(&p1);
        let res = BetweenResidual::new(diff);
        let factor = FactorBuilder::new2_unchecked(res, X(0), X(1)).build();
        graph.add_factor(factor);

        let mut opt = new(graph);
        values = opt.optimize(values).expect("Optimization failed");

        let out1: &T = values.get_unchecked(X(0)).expect("Missing X(0)");
        assert_matrix_eq!(
            out1.ominus(&p1),
            VectorX::zeros(T::DIM),
            comp = abs,
            tol = 1e-6
        );

        let out2: &T = values.get_unchecked(X(1)).expect("Missing X(1)");
        assert_matrix_eq!(
            out2.ominus(&p2),
            VectorX::zeros(T::DIM),
            comp = abs,
            tol = 1e-6
        );
    }
}
