use diol::prelude::*;

const DATA_DIR: &str = "../examples/data/";

// ------------------------- factrs ------------------------- //
use factrs::{core::GaussNewton, traits::Optimizer, utils::load_g20};
fn factrs(bencher: Bencher, file: &str) {
    let (graph, init) = load_g20(&format!("{}{}", DATA_DIR, file));
    bencher.bench(|| {
        let mut opt: GaussNewton = GaussNewton::new_default(graph.clone());
        let mut results = opt.optimize(init.clone());
        black_box(&mut results);
    });
}

// ------------------------- tiny-solver ------------------------- //
use tiny_solver::{
    gauss_newton_optimizer, helper::read_g2o as load_tiny_g2o, optimizer::Optimizer as TSOptimizer,
};

fn tinysolver(bencher: Bencher, file: &str) {
    let (graph, init) = load_tiny_g2o(&format!("{}{}", DATA_DIR, file));
    bencher.bench(|| {
        let gn = gauss_newton_optimizer::GaussNewtonOptimizer::new();
        let mut results = gn.optimize(&graph, &init, None);
        black_box(&mut results);
    });
}

fn main() -> eyre::Result<()> {
    faer::set_global_parallelism(faer::Par::Seq);

    let to_run = list![factrs, tinysolver];

    let bench = Bench::from_args()?;
    bench.register_many("3d", to_run, ["sphere2500.g2o", "parking-garage.g2o"]);
    bench.register_many("2d", to_run, ["M3500.g2o"]);
    bench.run()?;

    Ok(())
}
