use std::{
    fmt::{Debug, Write},
    marker::PhantomData,
};

use faer::sparse::{Pair, SymbolicSparseColMat};
use pad_adapter::PadAdapter;

use super::{DefaultSymbolHandler, Idx, KeyFormatter, Values, ValuesOrder};
// Once "debug_closure_helpers" is stabilized, we won't need this anymore
// Need custom debug to handle pretty key printing at the moment
// Pad adapter helps with the pretty printing
use crate::containers::factor::FactorFormatter;
use crate::{containers::Factor, dtype, linear::LinearGraph};

/// Structure to represent a nonlinear factor graph
///
/// Main usage will be via `add_factor` to add new [factors](Factor) to the
/// graph. Also of note is the `linearize` function that returns a [linear (aka
/// Gaussian) factor graph](LinearGraph).
///
/// Since the graph represents a nonlinear least-squares problem, during
/// optimization it will be iteratively linearized about a set of variables and
/// solved iteratively.
///
/// ```
/// # use factrs::{
///    assign_symbols,
///    containers::{Graph, FactorBuilder},
///    residuals::PriorResidual,
///    robust::GemanMcClure,
///    traits::*,
///    variables::SO2,
/// };
/// # assign_symbols!(X: SO2);
/// # let factor = FactorBuilder::new1(PriorResidual::new(SO2::identity()), X(0)).build();
/// let mut graph = Graph::new();
/// graph.add_factor(factor);
/// ```
#[derive(Default, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Graph {
    factors: Vec<Factor>,
}

impl Graph {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            factors: Vec::with_capacity(capacity),
        }
    }

    pub fn add_factor(&mut self, factor: Factor) {
        self.factors.push(factor);
    }

    pub fn len(&self) -> usize {
        self.factors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.factors.is_empty()
    }

    pub fn error(&self, values: &Values) -> dtype {
        self.factors.iter().map(|f| f.error(values)).sum()
    }

    pub fn linearize(&self, values: &Values) -> LinearGraph {
        let factors = self.factors.iter().map(|f| f.linearize(values)).collect();
        LinearGraph::from_vec(factors)
    }

    pub fn sparsity_pattern(&self, order: ValuesOrder) -> GraphOrder {
        let total_rows = self.factors.iter().map(|f| f.dim_out()).sum();
        let total_columns = order.dim();

        let mut indices = Vec::<Pair<usize, usize>>::new();

        let _ = self.factors.iter().fold(0, |row, f| {
            f.keys().iter().for_each(|key| {
                (0..f.dim_out()).for_each(|i| {
                    let Idx {
                        idx: col,
                        dim: col_dim,
                    } = order.get(*key).expect("Key missing in values");
                    (0..*col_dim).for_each(|j| {
                        indices.push(Pair::new(row + i, col + j));
                    });
                });
            });
            row + f.dim_out()
        });

        let (sparsity_pattern, sparsity_order) =
            SymbolicSparseColMat::try_new_from_indices(total_rows, total_columns, &indices)
                .expect("Failed to make sparse matrix");
        GraphOrder {
            order,
            sparsity_pattern,
            sparsity_order,
        }
    }

    pub fn iter(&self) -> std::slice::Iter<'_, Factor> {
        self.factors.iter()
    }

    pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, Factor> {
        self.factors.iter_mut()
    }
}

impl Debug for Graph {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        GraphFormatter::<DefaultSymbolHandler>::new(self).fmt(f)
    }
}

impl IntoIterator for Graph {
    type Item = Factor;
    type IntoIter = std::vec::IntoIter<Factor>;

    fn into_iter(self) -> Self::IntoIter {
        self.factors.into_iter()
    }
}

/// Formatter for a graph
///
/// Specifically, this can be used if custom symbols are desired. See
/// [tests/custom_key](https://github.com/rpl-cmu/factrs/blob/dev/tests/custom_key.rs) for examples.
pub struct GraphFormatter<'g, KF> {
    pub graph: &'g Graph,
    kf: PhantomData<KF>,
}

impl<'g, KF> GraphFormatter<'g, KF> {
    pub fn new(graph: &'g Graph) -> Self {
        Self {
            graph,
            kf: Default::default(),
        }
    }
}

impl<KF: KeyFormatter> Debug for GraphFormatter<'_, KF> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if f.alternate() {
            f.write_str("Graph [\n")?;
            let mut pad = PadAdapter::new(f);
            for factor in self.graph.factors.iter() {
                writeln!(pad, "{:#?},", FactorFormatter::<KF>::new(factor))?;
            }
            f.write_str("]")
        } else {
            f.write_str("Graph [ ")?;
            for factor in self.graph.factors.iter() {
                write!(f, "{:?}, ", FactorFormatter::<KF>::new(factor))?;
            }
            f.write_str("]")
        }
    }
}

/// Simple structure to hold the order of the graph
///
/// Specifically this is used to cache linearization results such as the order
/// of the graph and the sparsity pattern of the Jacobian (allows use to avoid
/// resorting indices).
pub struct GraphOrder {
    // Contains the order of the variables
    pub order: ValuesOrder,
    // Contains the sparsity pattern of the jacobian
    pub sparsity_pattern: SymbolicSparseColMat<usize>,
    // Contains the order of values to put into the sparsity pattern
    pub sparsity_order: faer::sparse::Argsort<usize>,
}
