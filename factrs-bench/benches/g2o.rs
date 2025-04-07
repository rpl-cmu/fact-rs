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

// ------------------------- sophus ------------------------- //
fn sophus(bench: Bencher, file: &str) {
    let (graph, init) = if file.contains("M3500") {
        factrs_bench::sophus::load_g2o_2d(&format!("{}{}", DATA_DIR, file))
    } else {
        factrs_bench::sophus::load_g2o_3d(&format!("{}{}", DATA_DIR, file))
    };

    let params = sophus_opt::nlls::OptParams {
        num_iterations: 40,
        initial_lm_damping: 0.0, // force to be gauss-newton
        parallelize: false,
        solver: sophus_opt::nlls::LinearSolverType::SparseLdlt(Default::default()),
        error_tol_relative: 1e-6,
        error_tol_absolute: 1e-6,
        error_tol: 0.0,
    };
    bench.bench(|| {
        let mut results = sophus_opt::nlls::optimize_nlls(init.clone(), graph.clone(), params);
        black_box(&mut results);
    })
}

fn main() -> eyre::Result<()> {
    faer::set_global_parallelism(faer::Par::Seq);
    sophus_faer::set_global_parallelism(sophus_faer::Parallelism::None);

    let to_run = list![factrs, tinysolver, sophus];

    let bench = Bench::from_args()?;
    bench.register_many("3d", to_run, ["sphere2500.g2o", "parking-garage.g2o"]);
    bench.register_many("2d", to_run, ["M3500.g2o"]);
    bench.run()?;

    Ok(())
}
