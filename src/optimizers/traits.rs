use crate::{
    core::{Graph, Values},
    dtype,
};

/// Error types for optimizers
#[derive(Debug)]
pub enum OptError {
    MaxIterations(Values),
    InvalidSystem,
    FailedToStep,
}

/// Result type for optimizers
pub type OptResult<T> = Result<T, OptError>;

// ------------------------- Optimizer Params ------------------------- //
pub trait OptParams: Default + Clone {
    fn base_params(&self) -> &BaseOptParams;
}

/// Parameters for the optimizer
#[derive(Debug, Clone)]
pub struct BaseOptParams {
    pub max_iterations: usize,
    pub error_tol_relative: dtype,
    pub error_tol_absolute: dtype,
    pub error_tol: dtype,
}

impl Default for BaseOptParams {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            error_tol_relative: 1e-6,
            error_tol_absolute: 1e-6,
            error_tol: 0.0,
        }
    }
}

impl OptParams for BaseOptParams {
    fn base_params(&self) -> &BaseOptParams {
        self
    }
}

// ------------------------- Optimizer Observers ------------------------- //
/// Observer trait for optimization
///
/// This trait is used to observe the optimization process. It is called at each
/// step of the optimization process.
pub trait OptObserver {
    fn on_step(&self, values: &Values, time: f64);
}

/// Observer collection for optimization
///
/// This struct holds a collection of observers for optimization. It is used to
/// notify all observers at each step of the optimization process.
#[derive(Default)]
pub struct OptObserverVec {
    observers: Vec<Box<dyn OptObserver>>,
}

impl OptObserverVec {
    pub fn add(&mut self, callback: impl OptObserver + 'static) {
        let boxed = Box::new(callback);
        self.observers.push(boxed);
    }

    pub fn notify(&self, values: &Values, idx: usize) {
        for callback in &self.observers {
            callback.on_step(values, idx as f64);
        }
    }
}

// ------------------------- Actual Trait Impl ------------------------- //
/// Trait for optimization algorithms
///
/// This trait is used to define the core optimization functions for an
/// optimizer, specifically a handful of stopping criteria and the main loop.
pub trait Optimizer {
    type Params: OptParams;

    // ------------------------- Required ------------------------- //
    /// Create a new optimizer
    fn new(params: Self::Params, graph: Graph) -> Self;

    /// Observers
    fn observers(&self) -> &OptObserverVec;

    /// Observers
    fn observers_mut(&mut self) -> &mut OptObserverVec;

    /// The graph we are optimizing
    fn graph(&self) -> &Graph;

    /// The graph we are optimizing
    ///
    /// This is mutable to allow for modifying the graph during optimization.
    /// BE CAREFUL! In most optimizer, the overall structure of the graph should
    /// remain the same between optimization steps.
    fn graph_mut(&mut self) -> &mut Graph;

    /// Parameters for the optimizer
    fn params(&self) -> &BaseOptParams;

    /// Perform a single step of optimization
    fn step(&mut self, values: Values, idx: usize) -> OptResult<Values>;

    /// Compute the error of the current values
    fn error(&self, values: &Values) -> dtype;

    /// Initialize the optimizer, optional
    fn init(&mut self, _values: &Values) {}

    // ------------------------- Derived from the above ------------------------- //
    // TODO: Custom logging based on optimizer
    /// Main optimization call function
    fn optimize(&mut self, mut values: Values) -> OptResult<Values> {
        // Setup up everything from our values
        self.init(&values);

        // Check if we need to optimize at all
        let mut error_old = self.error(&values);
        if error_old <= self.params().error_tol {
            log::info!("Error is already below tolerance, skipping optimization");
            return Ok(values);
        }

        log::info!(
            "{:^5} | {:^12} | {:^12} | {:^12}",
            "Iter",
            "Error",
            "ErrorAbs",
            "ErrorRel"
        );
        log::info!(
            "{:^5} | {:^12} | {:^12} | {:^12}",
            "-----",
            "------------",
            "------------",
            "------------"
        );
        log::info!(
            "{:^5} | {:^12.4e} | {:^12} | {:^12}",
            0,
            error_old,
            "-",
            "-"
        );

        // Begin iterations
        let mut error_new = error_old;
        for i in 1..self.params().max_iterations + 1 {
            error_old = error_new;
            values = self.step(values, i)?;
            self.observers().notify(&values, i);

            // Evaluate error again to see how we did
            error_new = self.error(&values);

            let error_decrease_abs = error_old - error_new;
            let error_decrease_rel = error_decrease_abs / error_old;

            log::info!(
                "{:^5} | {:^12.4e} | {:^12.4e} | {:^12.4e}",
                i,
                error_new,
                error_decrease_abs,
                error_decrease_rel
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

    fn add_observer(&mut self, observer: impl OptObserver + 'static) {
        self.observers_mut().add(observer);
    }

    /// Create a new optimizer with default params
    fn new_default(graph: Graph) -> Self
    where
        Self: Sized,
    {
        Self::new(Self::Params::default(), graph)
    }
}
