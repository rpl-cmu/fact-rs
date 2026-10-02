use criterion::{criterion_group, criterion_main, Criterion};
use std::hint::black_box;

type BenchmarkGroup<'a> = criterion::BenchmarkGroup<'a, criterion::measurement::WallTime>;

const DATA_DIR: &str = "../examples/data/";
const DATASETS: [&str; 3] = ["M3500.g2o", "sphere2500.g2o", "parking-garage.g2o"];

// ------------------------- factrs ------------------------- //
#[cfg(feature = "factrs")]
use factrs::{core::GaussNewton, traits::Optimizer, utils::load_g20};
#[cfg(feature = "factrs")]
fn register_factrs(group: &mut BenchmarkGroup<'_>, file: &str) {
    let (graph, init) = load_g20(&format!("{}{}", DATA_DIR, file));
    group.bench_function("factrs", |bencher| {
        bencher.iter(|| {
            let mut opt: GaussNewton = GaussNewton::new_default(graph.clone());
            black_box(opt.optimize(init.clone()).unwrap());
        });
    });
}

// ------------------------- tiny-solver ------------------------- //
#[cfg(feature = "tiny-solver")]
use tiny_solver::{
    gauss_newton_optimizer, helper::read_g2o as load_tiny_g2o, optimizer::Optimizer as TSOptimizer,
};

#[cfg(feature = "tiny-solver")]
fn register_tinysolver(group: &mut BenchmarkGroup<'_>, file: &str) {
    let (graph, init) = load_tiny_g2o(&format!("{}{}", DATA_DIR, file));
    group.bench_function("tinysolver", |bencher| {
        bencher.iter(|| {
            let gn = gauss_newton_optimizer::GaussNewtonOptimizer::new();
            black_box(gn.optimize(&graph, &init, None));
        });
    });
}

// ------------------------- sophus ------------------------- //
#[cfg(feature = "sophus")]
fn register_sophus(group: &mut BenchmarkGroup<'_>, file: &str) {
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
    group.bench_function("sophus", |bencher| {
        bencher.iter(|| {
            black_box(
                sophus_opt::nlls::optimize_nlls(init.clone(), graph.clone(), params).unwrap(),
            );
        });
    });
}

// NOTE: tiny-solver is still using rayon under the hood for jacobian
// computation, Setting the number of rayon threads to 1 DRASTICALLY degrades
// it's performance.
fn g2o(criterion: &mut Criterion) {
    // set everything to single-threaded
    #[cfg(feature = "factrs")]
    faer::set_global_parallelism(faer::Par::Seq);
    #[cfg(feature = "sophus")]
    sophus_faer::set_global_parallelism(sophus_faer::Parallelism::None);
    #[cfg(feature = "tiny-solver")]
    rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build_global()
        .unwrap();

    for file in DATASETS {
        let mut group = criterion.benchmark_group(file);
        #[cfg(feature = "factrs")]
        register_factrs(&mut group, file);
        #[cfg(feature = "tiny-solver")]
        register_tinysolver(&mut group, file);
        #[cfg(feature = "sophus")]
        register_sophus(&mut group, file);
        group.finish();
    }
}

criterion_group!(benches, g2o);
criterion_main!(benches);
