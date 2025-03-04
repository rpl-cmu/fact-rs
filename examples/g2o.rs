#[cfg(feature = "rerun")]
use std::net::{SocketAddr, SocketAddrV4};
use std::{env, time::Instant};

#[cfg(feature = "rerun")]
use factrs::rerun::RerunObserver;
use factrs::{
    core::{GaussNewton, SE2, SE3},
    traits::Optimizer,
    utils::load_g20,
};
#[cfg(feature = "rerun")]
use rerun::{Arrows2D, Arrows3D, Points2D, Points3D};

// Setups rerun and a callback for iteratively sending to rerun
// Must run with --features rerun for it to work
#[cfg(feature = "rerun")]
fn rerun_init(opt: &mut GaussNewton, dim: &str, obj: &str) {
    // Setup the rerun & the callback
    let socket = SocketAddrV4::new("0.0.0.0".parse().unwrap(), 9876);
    let rec = rerun::RecordingStreamBuilder::new("factrs-g2o-example")
        .connect_tcp_opts(SocketAddr::V4(socket), rerun::default_flush_timeout())
        .unwrap();

    // Log the graph
    let (nodes, edges) = opt.graph().into();
    rec.log_static("graph", &[&nodes as &dyn rerun::AsComponents, &edges])
        .expect("log failed");

    let topic = "base/solution";

    match (dim, obj) {
        ("se2", "points") => {
            let callback = RerunObserver::<SE2, Points2D>::new(rec, topic);
            opt.observers.add(callback)
        }
        ("se2", "arrows") => {
            let callback = RerunObserver::<SE2, Arrows2D>::new(rec, topic);
            opt.observers.add(callback)
        }
        ("se3", "points") => {
            let callback = RerunObserver::<SE3, Points3D>::new(rec, topic);
            opt.observers.add(callback)
        }
        ("se3", "arrows") => {
            let callback = RerunObserver::<SE3, Arrows3D>::new(rec, topic);
            opt.observers.add(callback)
        }
        _ => panic!("Invalid arguments"),
    };
}

#[cfg(not(feature = "rerun"))]
fn rerun_init(_opt: &mut GaussNewton, _dim: &str, _obj: &str) {}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ---------------------- Parse Arguments & Load data ---------------------- //
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        println!("Usage: {} <g2o file> <points/arrows>", args[0]);
        return Ok(());
    }

    pretty_env_logger::init();

    // Load the graph from the g2o file
    let filename = &args[1];
    let (graph, init) = load_g20(filename);
    println!("File loaded");

    // Make optimizer
    let mut optimizer: GaussNewton = GaussNewton::new(graph);

    // Connect to rerun if it's been enabled
    if cfg!(feature = "rerun") {
        let obj = if args.len() >= 3 { &args[2] } else { "arrows" };
        let dim = if init.filter::<SE2>().count() != 0 {
            "se2"
        } else if init.filter::<SE3>().count() != 0 {
            "se3"
        } else {
            panic!("Graph doesn't have SE2 or SE3 variables");
        };
        rerun_init(&mut optimizer, dim, obj);
    }

    // ------------------------- Optimize ------------------------- //
    let start = Instant::now();
    let result = optimizer.optimize(init);
    let duration = start.elapsed();

    match result {
        Ok(_) => println!("Optimization converged!"),
        Err(e) => println!("Optimization failed: {:?}", e),
    }
    println!("Optimization took: {:?}", duration);
    Ok(())
}
