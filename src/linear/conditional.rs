use std::collections::HashMap;

use nalgebra::Cholesky;

use super::LinearFactor;
use crate::{
    containers::{Idx, Key},
    linalg::{MatrixBlock, MatrixX, VectorX},
};

#[derive(Debug, Clone)]
pub struct LinearConditional {
    frontals: Vec<Key>,
    factor: LinearFactor,
}

pub fn compute_conditional(
    factors: &[LinearFactor],
    frontal: Vec<Key>,
) -> (LinearConditional, LinearFactor) {
    // Get all the keys
    let mut keys = factors
        .iter()
        .flat_map(|f| f.keys.iter().map(|k| (k, f.dim_of_var(*k))))
        .collect::<HashMap<_, _>>();
    assert!(
        frontal.len() < keys.len(),
        "Frontal keys must be less than total keys"
    );

    println!("Keys {:?}", keys);

    // Put them in order
    let mut idx = 0;
    let mut order_frontal = HashMap::new();
    for key in &frontal {
        let dim = keys.remove(&key).unwrap();
        order_frontal.insert(key, Idx { idx, dim });
        idx += dim;
    }
    let mut order_other = HashMap::new();
    let mut idx = 0;
    for (key, dim) in keys.into_iter() {
        order_other.insert(*key, Idx { idx, dim });
        idx += dim;
    }

    println!("Order frontal {:?}", order_frontal);
    println!("Order other {:?}", order_other);

    // Make matrices
    let rows: usize = factors.iter().map(|f| f.dim_out()).sum();
    let frontal_dim = order_frontal.values().map(|idx| idx.dim).sum::<usize>();
    let other_dim = order_other.values().map(|idx| idx.dim).sum::<usize>();

    let mut a1 = MatrixX::zeros(rows, frontal_dim);
    let mut a2 = MatrixX::zeros(rows, other_dim);

    // copy everything in
    let mut row_idx = 0;
    for factor in factors.iter() {
        let row_dim = factor.dim_out();
        for (i, key) in factor.keys.iter().enumerate() {
            if frontal.contains(key) {
                let col_dim = order_frontal[key].dim;
                let col_idx = order_frontal[key].idx;
                a1.view_mut((row_idx, col_idx), (row_dim, col_dim))
                    .copy_from(&factor.a.get_block(i));
            } else {
                let col_dim = order_other[key].dim;
                let col_idx = order_other[key].idx;
                a2.view_mut((row_idx, col_idx), (row_dim, col_dim))
                    .copy_from(&factor.a.get_block(i));
            }
        }
        row_idx += factor.dim_out();
    }
    let b = VectorX::from_iterator(rows, factors.iter().flat_map(|f| f.b.iter()).cloned());

    println!("A1 {}", a1);
    println!("A2 {}", a2);
    println!("B {}", b);

    // Compute our blocks
    let adag = Cholesky::pack_dirty(a1.transpose() * &a1).solve(&a1.transpose());
    println!("Adag {}", adag);
    let r = &adag * &a2;
    let d = &adag * &b;

    // Make the new factors
    let factor = LinearFactor {
        a: MatrixBlock::new(
            a2 - &a1 * &r,
            order_other.values().map(|idx| idx.dim).collect(),
        ),
        b: b - &a1 * &d,
        keys: order_other.keys().cloned().collect(),
    };
    let conditional = LinearConditional {
        factor: LinearFactor {
            a: MatrixBlock::new(r, order_other.values().map(|idx| idx.dim).collect()),
            b: d,
            keys: order_other.keys().cloned().collect(),
        },
        frontals: frontal,
    };

    (conditional, factor)
}
