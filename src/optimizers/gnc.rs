#![allow(unused)]
use std::fmt::Debug;

use statrs::distribution::{ChiSquared, ContinuousCDF};

use super::{BaseOptParams, LevenMarquardt, OptObserverVec, OptParams, OptResult, Optimizer};
use crate::{
    containers::{GraphOrder, ValuesOrder},
    core::{Graph, Values},
    dtype,
    linalg::VectorViewX,
    linear::{CholeskySolver, LinearSolver},
    robust::RobustCost,
};

// ------------------------- Convexable Kernels ------------------------- //
// Essentially this'll have a global mu that is shared, but each factor
// will have its own threshold.
pub trait ConvexableKernel: RobustCost + Clone {
    // TODO: Do we really need a different mu for each factor?
    fn init_mu(d2: &[dtype], thresh: &[dtype]) -> dtype;

    fn new(mu: dtype, thresh: dtype) -> Self;

    // TODO: We're computing this for each factor, when it likely should be
    fn step_mu(&mut self);

    fn upcast(&self) -> Box<dyn RobustCost>
    where
        Self: Sized + 'static,
    {
        dyn_clone::clone_box(self)
    }

    fn mu(&self) -> dtype;
}

#[derive(Clone)]
pub struct GncGemanMcClure {
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

    fn mu(&self) -> dtype {
        self.mu
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
#[derive(Debug)]
pub struct GraduatedNonConvexityParams<O: Optimizer = LevenMarquardt>
where
    O::Params: Clone,
{
    base: O::Params,
}

impl<O: Optimizer> Clone for GraduatedNonConvexityParams<O> {
    fn clone(&self) -> Self {
        Self {
            base: self.base.clone(),
        }
    }
}

impl<O: Optimizer> Default for GraduatedNonConvexityParams<O> {
    fn default() -> Self {
        Self {
            base: O::Params::default(),
        }
    }
}

impl<O: Optimizer> OptParams for GraduatedNonConvexityParams<O> {
    fn base_params(&self) -> &BaseOptParams {
        self.base.base_params()
    }
}

pub struct GraduatedNonConvexity<K = GncGemanMcClure, O: Optimizer = LevenMarquardt> {
    // Original graph
    kernels: Vec<K>,
    /// Basic parameters for the optimizer
    params: GraduatedNonConvexityParams<O>,
    /// Base optimizer
    optimizer: O,
}

impl<K: ConvexableKernel + 'static, O: Optimizer> Optimizer for GraduatedNonConvexity<K, O> {
    type Params = GraduatedNonConvexityParams<O>;

    fn new(params: Self::Params, graph: Graph) -> Self {
        Self {
            optimizer: O::new(params.base.clone(), graph.clone()),
            kernels: vec![],
            params,
        }
    }

    fn observers(&self) -> &OptObserverVec {
        self.optimizer.observers()
    }

    fn observers_mut(&mut self) -> &mut OptObserverVec {
        self.optimizer.observers_mut()
    }

    fn graph(&self) -> &Graph {
        self.optimizer.graph()
    }

    fn graph_mut(&mut self) -> &mut Graph {
        self.optimizer.graph_mut()
    }

    fn params(&self) -> &BaseOptParams {
        self.params.base.base_params()
    }

    fn error(&self, values: &Values) -> dtype {
        self.optimizer.error(values)
    }

    fn init(&mut self, values: &Values) {
        self.optimizer.init(values);

        // Initialize mu
        // TODO: Need chi2inv(0.95, dim) here to get inlier thresholds
        // TODO: Do I need a 1/2 like in gtsam?
        // - I think I do since our error is also prefixed by 1/2
        // c2 = 0.5 * chi2inv(0.95, dim);
        let e: Vec<_> = self.graph().iter().map(|f| f.error(values)).collect();
        let thresholds: Vec<_> = self
            .graph()
            .iter()
            .map(|f| {
                ChiSquared::new(f.dim_out() as f64)
                    .expect("")
                    .inverse_cdf(0.95)
            })
            .collect();

        let mu = K::init_mu(&e, &thresholds);

        // Initialize the kernels
        self.kernels = thresholds.iter().map(|t| K::new(mu, *t)).collect();
        println!("e = {:?}", e);
        println!("thresholds = {:?}", thresholds);
        println!("mu = {:?}", mu);
        println!("kernels = {:#?}", self.kernels);
    }

    fn step(&mut self, mut values: Values, idx: usize) -> OptResult<Values> {
        // Step the kernels
        self.kernels.iter_mut().for_each(|k| k.step_mu());
        // TODO: Do this to appease the borrow checker, but it's not great
        let kernels = self.kernels.clone();

        println!("mu = {:?}", kernels[0].mu());

        // Replace them in the graph
        self.graph_mut()
            .iter_mut()
            .zip(kernels)
            .for_each(|(f, k)| f.robust = k.upcast());

        // Optimize and return
        self.optimizer.step(values, idx)
    }
}
