#![allow(unused)]
use std::fmt::Debug;

use super::{BaseOptParams, OptObserverVec, OptParams, OptResult, Optimizer};
use crate::{
    containers::{GraphOrder, ValuesOrder},
    core::{Graph, Values},
    dtype,
    linalg::VectorViewX,
    linear::{CholeskySolver, LinearSolver},
    robust::RobustCost,
};

// ------------------------- Convexable Kernels ------------------------- //
fn chi2inv(p: dtype, dim: usize) -> dtype {
    todo!()
}

pub trait ConvexableKernel: RobustCost {
    // TODO: Do we really need a different mu for each factor?
    fn init_mu(d2: &[dtype], thresh: &[dtype]) -> dtype;

    fn new(mu: dtype, thresh: dtype) -> Self;

    // TODO: We're computing this for each factor, when it likely should be shared
    fn step_mu(&mut self);

    fn check_convergence(&self, w: dtype) -> bool;

    fn upcast(&self) -> Box<dyn RobustCost>
    where
        Self: Sized + 'static,
    {
        dyn_clone::clone_box(self)
    }
}

#[derive(Clone)]
struct GncGemanMcClure {
    mu: dtype,
    c2: dtype,
}

#[factrs::mark]
impl RobustCost for GncGemanMcClure {
    fn loss(&self, d2: dtype) -> dtype {
        let p = self.mu * self.c2;
        0.5 * p * d2 / (p + d2)
    }

    fn weight(&self, d2: dtype) -> dtype {
        let p = self.mu * self.c2;
        let denom = p + d2;
        let frac = p / denom;
        frac * frac
    }
}

impl ConvexableKernel for GncGemanMcClure {
    fn init_mu(d2: &[dtype], thresh: &[dtype]) -> dtype {
        0.5 * d2
            .iter()
            .zip(thresh)
            .fold(0.0, |mu, (d, t)| dtype::max(mu, d / t))
    }

    fn new(mu: dtype, thresh: dtype) -> Self {
        Self { mu, c2: thresh }
    }

    fn step_mu(&mut self) {
        self.mu = dtype::max(1.0, self.mu / 1.4);
    }

    fn check_convergence(&self, w: dtype) -> bool {
        dtype::abs(self.mu - 1.0) < 1e-8
    }
}

impl Debug for GncGemanMcClure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "GncGemanMcClure {{ mu: {}, c: {} }}",
            self.mu,
            self.c2.sqrt()
        )
    }
}

// ------------------------- GNC ------------------------- //
#[derive(Debug, Clone)]
pub struct GraduatedNonConvexityParams {
    base: BaseOptParams,
    // kernel: Box<dyn ConvexableKernel>,
}

impl Default for GraduatedNonConvexityParams {
    fn default() -> Self {
        Self {
            base: BaseOptParams::default(),
            // kernel: Box::new(GncGemanMcClure { mu: 1.0, c2: 1.0 }),
        }
    }
}

impl OptParams for GraduatedNonConvexityParams {
    fn base_params(&self) -> &BaseOptParams {
        &self.base
    }
}

// #[derive(Clone)]
pub struct GraduatedNonConvexity<K> {
    // Original graph
    graph: Graph,
    kernels: Vec<K>,
    /// Basic parameters for the optimizer
    pub params: GraduatedNonConvexityParams,
    /// Levenberg-Marquardt specific parameters
    // pub params_leven: LevenParams,
    /// Observers for the optimizer
    pub observers: OptObserverVec,
    // For caching computation between steps
    graph_order: Option<GraphOrder>,
}

impl<K> GraduatedNonConvexity<K> {
    pub fn new(graph: Graph) -> Self {
        Self {
            graph,
            kernels: Vec::new(),
            params: Default::default(),
            // params_leven: LevenParams::default(),
            observers: Default::default(),
            graph_order: None,
        }
    }
}

impl<K: ConvexableKernel> Optimizer for GraduatedNonConvexity<K> {
    type Params = GraduatedNonConvexityParams;

    fn params(&self) -> &BaseOptParams {
        &self.params.base
    }

    fn error(&self, values: &Values) -> dtype {
        self.graph.error(values)
    }

    fn init(&mut self, values: &Values) {
        // Initialize mu
        // TODO: Need chi2inv(0.95, dim) here to get inlier thresholds
        // TODO: Do I need a 1/2 like in gtsam?
        // - I think I do since our error is also prefixed by 1/2
        // c2 = 0.5 * chi2inv(0.95, dim);
        let e: Vec<_> = self.graph.iter().map(|f| f.error(values)).collect();
        let thresholds: Vec<_> = self
            .graph
            .iter()
            .map(|f| chi2inv(0.99, f.dim_out()))
            .collect();
        let mu = K::init_mu(&e, &thresholds);

        // Initialize the kernels
        self.kernels = self
            .graph
            .iter()
            .zip(thresholds)
            .map(|(f, t)| K::new(mu, t))
            .collect();

        // Precompute the sparsity pattern
        self.graph_order = Some(
            self.graph
                .sparsity_pattern(ValuesOrder::from_values(values)),
        );
    }

    fn step(&mut self, mut values: Values, idx: usize) -> OptResult<Values> {
        todo!()
    }
}
