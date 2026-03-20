use std::collections::hash_map::Iter as HashMapIter;

use foldhash::HashMap;

use super::{Key, Symbol, Values};

/// Location of a variable in a list
///
/// Since the map isn't ordered, we need to track both idx and len of each
/// variable
#[derive(Debug, Clone)]
pub struct Idx {
    pub idx: usize,
    pub dim: usize,
}

/// Tracks the location of each variable in the graph via an [Idx].
///
/// Likely won't need to ever interface with this unless a custom optimizer is
/// being implemented. Since the map isn't ordered, we need to track both idx
/// and len of each variable
#[derive(Debug, Clone)]
pub struct ValuesOrder {
    map: HashMap<Key, Idx>,
    dim: usize,
}

impl ValuesOrder {
    pub fn new(map: HashMap<Key, Idx>) -> Self {
        let dim = map.values().map(|idx| idx.dim).sum();
        Self { map, dim }
    }
    pub fn from_values(values: &Values) -> Self {
        let map = values
            .iter()
            .scan(0, |idx, (key, val)| {
                let order = *idx;
                *idx += val.dim();
                Some((
                    *key,
                    Idx {
                        idx: order,
                        dim: val.dim(),
                    },
                ))
            })
            .collect::<HashMap<Key, Idx>>();

        let dim = map.values().map(|idx| idx.dim).sum();

        Self { map, dim }
    }

    pub fn get(&self, symbol: impl Symbol) -> Option<&Idx> {
        self.map.get(&symbol.into())
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    pub fn iter(&self) -> HashMapIter<'_, Key, Idx> {
        self.map.iter()
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{
        containers::Values,
        symbols::X,
        variables::{Variable, VectorVar2, VectorVar3, VectorVar6},
    };

    #[test]
    fn from_values() {
        // Create some form of values
        let mut v = Values::new();
        v.insert_unchecked(X(0), VectorVar2::identity());
        v.insert_unchecked(X(1), VectorVar6::identity());
        v.insert_unchecked(X(2), VectorVar3::identity());

        // Create an order
        let order = ValuesOrder::from_values(&v);

        // Verify the order
        assert_eq!(order.len(), 3);
        assert_eq!(order.dim(), 11);
        assert_eq!(order.get(X(0)).expect("Missing key").dim, 2);
        assert_eq!(order.get(X(1)).expect("Missing key").dim, 6);
        assert_eq!(order.get(X(2)).expect("Missing key").dim, 3);
    }
}
