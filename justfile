# ---------------------- General helpers that should work for everyone ---------------------- #
# build and run rust benchmarks
bench-rust:
    cargo bench -p factrs-bench

# build and run cpp benchmarks
bench-cpp:
    cmake -B build factrs-bench/cpp
    cmake --build build
    ./build/bench

# profile the g2o example using flamegraph
profile:
    cargo flamegraph --profile profile --example g2o -- ./examples/data/parking-garage.g2o

# build docs with latex support
doc:
    RUSTDOCFLAGS="--cfg docsrs --html-in-header $PWD/assets/katex-header.html" cargo doc --features="serde rerun" -Zunstable-options -Zrustdoc-scrape-examples

bacon-doc:
    RUSTDOCFLAGS="--cfg docsrs --html-in-header $PWD/assets/katex-header.html" bacon doc --features="serde rerun" -- -Zunstable-options -Zrustdoc-scrape-examples

# ---------------------- Easton specific helpers that work on my system ---------------------- #
# tune the system for benchmarking using pyperf
# requires pyperf installed using uv
perf-tune:
    sudo ~/.local/share/uv/tools/pyperf/bin/pyperf system tune

# reset the system after benchmarking using pyperf
# requires pyperf installed using uv
perf-reset:
    sudo ~/.local/share/uv/tools/pyperf/bin/pyperf system reset

# make the benchmark plots
# requires uv and pyperf installed using uv
bench-plot: perf-tune
    # rust ones
    cargo bench -p factrs-bench --bench g2o -- --sample-count 100 --max-time 200 --output rust.json
    # cpp ones
    cmake -B build factrs-bench/cpp
    cmake --build build
    ./build/bench 100
    # make the plots
    uv run factrs-bench/plot.py