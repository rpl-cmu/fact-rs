#![allow(dead_code)]

use foldhash::HashMap;

use super::LinearGraph;
use crate::containers::{Key, ValuesOrder};

#[derive(Debug, Clone)]
struct Node {}

struct FactorTree {
    nodes: Vec<Node>,
}

#[derive(Debug, Clone, Default)]
struct ValuesIndex {
    pub index: HashMap<Key, Vec<usize>>,
}

// TODO: Make variable index
impl FactorTree {
    pub fn new(graph: &LinearGraph, order: &ValuesOrder) -> Self {
        let vi = ValuesIndex::default();

        let num_vars = order.len();
        let num_facs = graph.factors.len();

        let mut nodes = vec![Node {}; num_vars];

        for (k, idx) in order.iter() {
            let facs = vi.index.get(k).unwrap();
            for i in 0..facs.len() {
                let fac = &graph.factors[facs[i]];
                let dim = fac.dim_of_var(k);
                let idx = fac.a.get_idx(idx);
                let node = &mut nodes[idx];
            }
        }

        todo!()
    }
}
