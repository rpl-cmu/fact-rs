#![allow(unused)]
use std::fmt::Debug;

use statrs::distribution::{ChiSquared, ContinuousCDF};

use super::{
    BaseOptParams, LevenMarquardt, OptError, OptObserverVec, OptParams, OptResult, Optimizer,
};
use crate::{
    containers::{GraphOrder, ValuesOrder},
    core::{Graph, L2, Values},
    dtype,
    linalg::VectorViewX,
    linear::{CholeskySolver, LinearSolver},
    robust::RobustCost,
};

// ------------------------- Convexable Kernels ------------------------- //

// TODO: Instead of upcasting, would it be better if this just outputted a dyn
// RobustCost?

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
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
        let frac = p / (p + d2);
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
/// Parameters for the Graduated Non-Convexity optimizer
///
/// This is a wrapper around the base optimizer parameters and the
/// optimizer parameters for the inner optimizer. Generic over the type of the
/// inner optimizer.
#[derive(Debug)]
pub struct GncParams<O: Optimizer = LevenMarquardt>
where
    O::Params: Clone,
{
    /// Basic parameters for GNC Optimizer
    pub base: BaseOptParams,
    /// Parameters for the inner optimizer
    ///
    /// Will likely want to lower the max number of iterations, and increase the
    /// tolerances.
    pub inner: O::Params,
    /// Step size for the mu parameter
    ///
    /// This is the step size for the mu parameter. Defaults to 1.4.
    pub mu_step_size: dtype,
    /// Percentile for the inlier threshold
    ///
    /// This is the percentile for the inlier threshold. Defaults to 0.95. Will
    /// be used to compute kernel parameters for robust kernels.
    pub percentile: dtype,
}

impl<O: Optimizer> Clone for GncParams<O> {
    fn clone(&self) -> Self {
        Self {
            base: self.base.clone(),
            inner: self.inner.clone(),
            mu_step_size: self.mu_step_size,
            percentile: self.percentile,
        }
    }
}

impl<O: Optimizer> Default for GncParams<O> {
    fn default() -> Self {
        Self {
            base: Default::default(),
            inner: Default::default(),
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

/// Graduated Non-Convexity [^@yangGraduatedNonConvexityRobust2020] optimizer
///
/// This optimizer uses a "convexification" approach to reduce initialization
/// sensitivity for robust nonlinear least-squares.
///
/// Specifically, it uses a set of robust kernels that can be convexified using
/// a parameter $\mu$. This looks like,
/// $$
/// \Theta^* = \argmin_{\Theta}
/// \sum_{i} \rho_i(||r_i(\Theta)||_{\Sigma_i}; \mu) )
/// $$
/// Note, our implementation (like the original) uses the same $\mu$ for each
/// factor. The optimizer begins with a $\mu$ for such that $\rho(\cdot; \mu)$
/// is convex, and progressively steps $\mu$, until $\rho(\cdot; \mu)$ is an
/// M-estimator, generally with constant asymptotic behavior. While a heuristic,
/// this has been shown to decrease sensitivity to initialization, a known
/// problem for M-estimation and outlier rejection.
///
/// [^@yangGraduatedNonConvexityRobust2020]: Yang, Heng, et al. “Graduated Non-Convexity for Robust Spatial Perception: From Non-Minimal Solvers to Global Outlier Rejection.” IEEE Robotics and Automation Letters, vol. 5, no. 2, Apr. 2020, pp. 1127–34
pub struct GraduatedNonConvexity<K = GncGemanMcClure, O: Optimizer = LevenMarquardt> {
    /// Holds the kernels
    ///
    ///  These will be iteratively updated as the optimization progresses. Any
    /// that are None are known inliers and their kernels won't be changed.
    kernels: Vec<Option<K>>,
    /// Basic parameters for the optimizer
    params: GncParams<O>,
    /// Graph to optimize
    graph: Graph,
    /// Base optimizer
    observers: OptObserverVec,
}

impl<K: ConvexableKernel + 'static, O: Optimizer> Optimizer for GraduatedNonConvexity<K, O> {
    type Params = GncParams<O>;

    fn new(params: Self::Params, graph: Graph) -> Self {
        Self {
            observers: OptObserverVec::default(),
            kernels: Vec::new(),
            graph,
            params,
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

    fn error(&self, values: &Values) -> dtype {
        self.graph.error(values)
    }

    fn init(&mut self, values: &Values) -> Vec<&'static str> {
        // Gather error and thresholds
        let e: Vec<_> = self.graph().iter().map(|f| f.error(values)).collect();
        #[allow(clippy::unnecessary_cast)]
        let thresholds: Vec<_> = self
            .graph()
            .iter()
            .map(|f| {
                ChiSquared::new(f.dim_out() as f64)
                    .expect("")
                    .inverse_cdf(self.params.percentile as f64) as dtype
            })
            .collect();

        // Initialize the mu parameter
        let mu = K::init_mu(&e, &thresholds);

        // Infer inliers from between factors with consecutive keys
        let is_odometry = self
            .graph()
            .iter()
            .enumerate()
            .map(|(i, f)| f.keys().len() == 2 && f.keys()[0].0 + 1 == f.keys()[1].0)
            .collect::<Vec<_>>();

        if is_odometry.iter().all(|&x| x) {
            log::warn!("All factors are odometry, no kernels will be created");
        }

        // Initialize the kernels
        self.kernels = thresholds
            .iter()
            .zip(is_odometry)
            .map(|(t, inlier)| if (inlier) { None } else { Some(K::new(mu, *t)) })
            .collect();

        vec!["     Mu     "]
    }

    fn step(&mut self, mut values: Values, idx: usize) -> OptResult<(Values, String)> {
        // Step the kernels
        self.kernels
            .iter_mut()
            .filter_map(|k| k.as_mut())
            .for_each(|k| k.step_mu(self.params.mu_step_size));

        // Get the most recent mu
        let mut mu = 0.0;
        for (i, k) in self.kernels.iter().enumerate() {
            if let Some(k) = k {
                mu = k.mu();
            }
        }

        // Replace the robust kernels in the graph
        #[allow(clippy::unwrap_used)]
        self.graph
            .iter_mut()
            .zip(self.kernels.clone())
            .filter(|(f, k)| k.is_some())
            .for_each(|(f, k)| f.robust = k.unwrap().upcast());

        // Optimize and return
        let error = self.error(&values);
        let mut info = String::new();
        // let inner_params = self.params.inner.base_params();

        // TODO: We leave a lot of performance on the table here, since a lot of
        // orderings and symbolic decomp will be recomputed each step.
        let mut opt = O::new(self.params.inner.clone(), self.graph().clone());
        let result = opt.optimize(values.clone());
        match result {
            Ok(v) => values = v,
            Err(OptError::MaxIterations(v)) => {
                values = v;
            }
            Err(e) => {
                log::warn!("Inner optimizer failed");
                return Err(e);
            }
        }
        info.push_str(&format!(" {:^12.4e} |", mu));

        Ok((values, info))
    }

    // Have to re-implement this because we need to do some extra stuff
    // Namely, to allow for increases to the error
    fn optimize(&mut self, mut values: Values) -> OptResult<Values> {
        // Setup up everything from our values
        let append = self.init(&values);

        // Check if we need to optimize at all
        let mut error_old = self.error(&values);
        if error_old <= self.params().error_tol {
            log::info!("Error is already below tolerance, skipping optimization");
            return Ok(values);
        }

        let extra = if append.is_empty() { "" } else { " |" };

        log::info!(
            "{:^5} | {:^12} | {:^12} | {:^12} | {}",
            "Iter",
            "Error",
            "ErrorAbs",
            "ErrorRel",
            append.join(" | ") + extra,
        );
        log::info!(
            "{:^5} | {:^12} | {:^12} | {:^12} | {}",
            "-----",
            "------------",
            "------------",
            "------------",
            append
                .iter()
                .map(|s| "-".repeat(s.len()))
                .collect::<Vec<_>>()
                .join(" | ")
                + extra
        );
        log::info!(
            "{:^5} | {:^12.4e} | {:^12} | {:^12} | {}",
            0,
            error_old,
            "-",
            "-",
            append
                .iter()
                .map(|s| format!("{:^width$}", "-", width = s.len()))
                .collect::<Vec<_>>()
                .join(" | ")
                + extra
        );

        // Begin iterations
        let mut error_new = error_old;
        for i in 1..self.params().max_iterations + 1 {
            error_old = error_new;
            let (temp, info) = self.step(values, i)?;
            values = temp;
            self.observers().notify(&values, i);

            // Evaluate error again to see how we did
            error_new = self.error(&values);

            // NOTE: This is the difference, we need to be ok with increases in error due to
            // changing the kernels
            let error_decrease_abs = dtype::abs(error_old - error_new);
            let error_decrease_rel = error_decrease_abs / error_old;

            log::info!(
                "{:^5} | {:^12.4e} | {:^12.4e} | {:^12.4e} | {}",
                i,
                error_new,
                error_decrease_abs,
                error_decrease_rel,
                info
            );

            // Check if we need to stop
            if error_new <= self.params().error_tol {
                log::info!("Error is below tolerance, stopping optimization");
                return Ok(values);
            }
            if error_decrease_abs <= self.params().error_tol_absolute {
                log::info!("Error decrease is below absolute tolerance, stopping optimization");
                return Ok(values);
            }
            if error_decrease_rel <= self.params().error_tol_relative {
                log::info!("Error decrease is below relative tolerance, stopping optimization");
                return Ok(values);
            }
        }

        Err(OptError::MaxIterations(values))
    }
}
