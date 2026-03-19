use std::{env, time::Instant};

#[cfg(feature = "rerun")]
use factrs::rerun::RerunObserver;
use factrs::{
    core::{GaussNewton, LevenMarquardt, SE2, SE3},
    optimizers::{GncGemanMcClure, GncParams, GraduatedNonConvexity},
    traits::Optimizer,
    utils::load_g20,
};
#[cfg(feature = "rerun")]
use rerun::{Arrows2D, Arrows3D, Points2D, Points3D};

// Setups rerun and a callback for iteratively sending to rerun
// Must run with --features rerun for it to work
#[cfg(feature = "rerun")]
fn rerun_init(opt: &mut impl Optimizer, dim: &str, obj: &str) {
    // Setup the rerun & the callback
    let rec = rerun::RecordingStreamBuilder::new("factrs-g2o-example")
        .connect_grpc_opts("rerun+http://127.0.0.1:9876/proxy")
        .unwrap();

    // Log the graph
    let (nodes, edges) = opt.graph().into();
    rec.log_static("graph", &[&nodes as &dyn rerun::AsComponents, &edges])
        .expect("log failed");

    let topic = "base/solution";

    match (dim, obj) {
        ("se2", "points") => {
            let callback = RerunObserver::<SE2, Points2D>::new(rec, topic);
            opt.observers_mut().add(callback)
        }
        ("se2", "arrows") => {
            let callback = RerunObserver::<SE2, Arrows2D>::new(rec, topic);
            opt.observers_mut().add(callback)
        }
        ("se3", "points") => {
            let callback = RerunObserver::<SE3, Points3D>::new(rec, topic);
            opt.observers_mut().add(callback)
        }
        ("se3", "arrows") => {
            let callback = RerunObserver::<SE3, Arrows3D>::new(rec, topic);
            opt.observers_mut().add(callback)
        }
        _ => panic!("Invalid arguments"),
    };
}

#[cfg(not(feature = "rerun"))]
fn rerun_init(_opt: &mut impl Optimizer, _dim: &str, _obj: &str) {}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ---------------------- Parse Arguments & Load data ---------------------- //
    let mut args: Vec<String> = env::args().collect();
    match args.len() {
        2 => {
            args.push(String::from("gauss"));
            args.push(String::from("arrows"));
        }
        3 => {
            args.push(String::from("arrows"));
        }
        4 => {}
        _ => {
            println!(
                "Usage: {} <g2o file> <optimizer: [gauss|leven|gnc] = gauss> <vis: [points|arrows] = arrows>",
                args[0]
            );
            return Ok(());
        }
    }

    pretty_env_logger::init();

    // Load the graph from the g2o file
    let filename = &args[1];
    let (graph, init) = load_g20(filename);
    println!("File loaded, {} factors", graph.len());

    let obj = &args[3];
    let dim = if init.filter::<SE2>().count() != 0 {
        "se2"
    } else if init.filter::<SE3>().count() != 0 {
        "se3"
    } else {
        panic!("Graph doesn't have SE2 or SE3 variables");
    };

    // Make optimizer
    let mut optimizer: Box<dyn Optimizer> = match args[2].as_str() {
        "gauss" => {
            let mut opt = GaussNewton::new_default(graph);
            rerun_init(&mut opt, dim, obj);
            Box::new(opt)
        }
        "leven" => {
            let mut opt = LevenMarquardt::new_default(graph);
            rerun_init(&mut opt, dim, obj);
            Box::new(opt)
        }
        #[allow(clippy::field_reassign_with_default)]
        "gnc" => {
            let mut params: GncParams<LevenMarquardt> = GncParams::default();
            params.mu_step_size = 1.4;
            params.base.max_iterations = 200;
            params.inner.base.max_iterations = 5;
            params.inner.base.error_tol_absolute = 1e-4;
            params.inner.base.error_tol_relative = 1e-4;
            let mut opt: GraduatedNonConvexity<GncGemanMcClure, _> =
                GraduatedNonConvexity::new(params, graph);
            rerun_init(&mut opt, dim, obj);
            Box::new(opt)
        }
        _ => {
            println!("Optimizer not recognized");
            return Ok(());
        }
    };

    // ------------------------- Optimize ------------------------- //
    let start = Instant::now();
    let result = optimizer.optimize(init);
    let duration = start.elapsed();

    match result {
        Ok(_) => println!("Optimization converged!"),
        Err(_) => println!("Optimization failed!"),
    }
    println!("Optimization took: {:?}", duration);
    Ok(())
}
