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
/// A trait for kernels that can be iteratively "convexified"
///
/// This trait is used to define kernels that can be used in the Graduated
/// Non-Convexity (GNC) algorithm. Specifically throughout, `d2` is the error
/// squared and `thresh` is the inlier threshold, usually set to the 95th
/// percentile in [GncParams]
pub trait ConvexableKernel: RobustCost + Clone {
    /// How to initialize the mu parameter
    ///
    /// This will be done once at the start of the optimization, and will likely
    /// involve some form of maximum.
    fn init_mu(d2: &[dtype], thresh: &[dtype]) -> dtype;

    /// Create a new kernel with the given mu and threshold
    ///
    /// This threshold is often proportional to `c` parameter of the kernel.
    fn new(mu: dtype, thresh: dtype) -> Self;

    /// Step the mu parameter
    fn step_mu(&mut self, step_size: dtype);

    /// Convert the kernel to a boxed trait object
    fn upcast(&self) -> Box<dyn RobustCost>
    where
        Self: Sized + 'static,
    {
        dyn_clone::clone_box(self)
    }

    /// Get the current mu parameter
    fn mu(&self) -> dtype;
}

/// A Geman-McClure kernel
///
/// Given by,
/// $$
/// \frac{\mu c^2 x^2}{\mu c^2 + x^2}
/// $$
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
        2.0 * d2
            .iter()
            .zip(thresh)
            .fold(0.0, |mu, (d, t)| dtype::max(mu, d / t))
    }

    fn new(mu: dtype, thresh: dtype) -> Self {
        Self { mu, c2: thresh }
    }

    fn step_mu(&mut self, step_size: dtype) {
        self.mu = dtype::max(1.0, self.mu / step_size);
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
pub struct GncParams<O: Optimizer = LevenMarquardt>
where
    O::Params: Clone,
{
    base: O::Params,
    mu_step_size: dtype,
    percentile: dtype,
}

impl<O: Optimizer> Clone for GncParams<O> {
    fn clone(&self) -> Self {
        Self {
            base: self.base.clone(),
            mu_step_size: self.mu_step_size,
            percentile: self.percentile,
        }
    }
}

impl<O: Optimizer> Default for GncParams<O> {
    fn default() -> Self {
        Self {
            base: O::Params::default(),
            mu_step_size: 1.4,
            percentile: 0.95,
        }
    }
}

impl<O: Optimizer> OptParams for GncParams<O> {
    fn base_params(&self) -> &BaseOptParams {
        self.base.base_params()
    }
}

// TODO: This is supposed to iterate entirely over the inner function, but it
// seems to be failing for me
// TODO: Probably need to specify odometry as not an outlier

pub struct GraduatedNonConvexity<K = GncGemanMcClure, O: Optimizer = LevenMarquardt> {
    // Original graph
    kernels: Vec<K>,
    /// Basic parameters for the optimizer
    params: GncParams<O>,
    /// Base optimizer
    optimizer: O,
}

impl<K: ConvexableKernel + 'static, O: Optimizer> Optimizer for GraduatedNonConvexity<K, O> {
    type Params = GncParams<O>;

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

    fn init(&mut self, values: &Values) -> Vec<&'static str> {
        let mut base_append = self.optimizer.init(values);
        base_append.push("     Mu     ");

        // Gather error and thresholds
        let e: Vec<_> = self.graph().iter().map(|f| f.error(values)).collect();
        let thresholds: Vec<_> = self
            .graph()
            .iter()
            .map(|f| {
                ChiSquared::new(f.dim_out() as f64)
                    .expect("")
                    .inverse_cdf(self.params.percentile)
            })
            .collect();

        // Initialize the mu parameter
        let mu = K::init_mu(&e, &thresholds);

        // Initialize the kernels
        self.kernels = thresholds.iter().map(|t| K::new(mu, *t)).collect();

        base_append
    }

    fn step(&mut self, mut values: Values, idx: usize) -> OptResult<(Values, String)> {
        // Step the kernels
        self.kernels
            .iter_mut()
            .for_each(|k| k.step_mu(self.params.mu_step_size));
        // TODO: Do this to appease the borrow checker, but it's not great
        let kernels = self.kernels.clone();
        let mu = kernels[0].mu();

        // Replace them in the graph
        self.graph_mut()
            .iter_mut()
            .zip(kernels)
            .for_each(|(f, k)| f.robust = k.upcast());

        // Optimize and return
        let (values, mut info) = self.optimizer.step(values, idx)?;
        info.push_str(&format!(" {:^12.4e} |", mu));
        // let values: Values = self.optimizer.optimize(values).unwrap();
        // let info = format!(" {:^12.4e} |", mu);
        // println!("Finished step {}: {}", idx, info);

        Ok((values, info))
    }
}
