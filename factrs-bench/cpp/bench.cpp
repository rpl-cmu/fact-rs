#include "bench_ceres.h"
#include "ceres/types.h"
#include "gtsam/slam/dataset.h"

#include <cstdio>
#include <gtsam/nonlinear/GaussNewtonOptimizer.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>

#include <chrono>
#include <nanobench.h>

using namespace ankerl;

std::string directory = "examples/data/";
std::vector<std::string> files_3d{"sphere2500.g2o", "parking-garage.g2o"};
std::vector<std::string> files_2d{"M3500.g2o"};

// ------------------------- Ceres ------------------------- //
// TODO: Only values really need to be copied, is there a way around that?
template <typename DIM>
void run_ceres(nanobench::Bench *bench, std::string file) {
  std::map<int, typename DIM::Var> og_poses;
  std::vector<typename DIM::Constraint> og_constraints;
  std::tie(og_poses, og_constraints) = load_ceres<DIM>(directory + file);

  bench->context("benchmark", "ceres");
  bench->run(file, [&]() {
    // Copy the poses and constraints to avoid modifying the original data
    auto poses(og_poses);
    auto constraints(og_constraints);

    ceres::Problem problem;
    DIM::Build(constraints, &poses, &problem);

    ceres::Solver::Options options;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    // shrink as much as possible to basically do a Gauss-Newton step
    options.initial_trust_region_radius = 1e16;
    options.sparse_linear_algebra_library_type = ceres::SUITE_SPARSE;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.minimizer_progress_to_stdout = false;
    options.num_threads = 1;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    nanobench::doNotOptimizeAway(summary);
  });
}

// ------------------------- gtsam ------------------------- //
gtsam::GraphAndValues load_gtsam(std::string file, bool is3D) {
  auto read = gtsam::readG2o(file, is3D);

  if (is3D) {
    auto priorModel = gtsam::noiseModel::Diagonal::Variances(
        (gtsam::Vector(6) << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4).finished());
    read.first->addPrior(0, gtsam::Pose3::Identity(), priorModel);
  } else {
    auto priorModel = gtsam::noiseModel::Diagonal::Variances(
        gtsam::Vector3(1e-6, 1e-6, 1e-8));
    auto prior = gtsam::PriorFactor<gtsam::Pose2>(0, gtsam::Pose2());
    read.first->addPrior(0, gtsam::Pose2::Identity(), priorModel);
  }

  return read;
}

void run_gtsam(nanobench::Bench *bench, std::string file, bool is3D) {
  auto gv = load_gtsam(directory + file, is3D);

  bench->context("benchmark", "gtsam");
  bench->run(file, [&]() {
    gtsam::NonlinearFactorGraph graph(*gv.first);
    gtsam::Values values(*gv.second);

    gtsam::GaussNewtonOptimizer optimizer(graph, values);
    gtsam::Values result = optimizer.optimize();

    nanobench::doNotOptimizeAway(result);
  });
}

char const *markdown() {
  return R"DELIM(| benchmark | args | fastest | median | mean |
{{#result}}| {{context(benchmark)}} | {{name}} | {{minimum(elapsed)}} | {{median(elapsed)}} | {{average(elapsed)}} |
{{/result}})DELIM";
}

// ------------------------- Run benchmarks ------------------------- //
int main(int argc, char *argv[]) {

  nanobench::Bench b;
  b.timeUnit(std::chrono::milliseconds(1), "ms");

  // 3d benchmarks
  b.title("3d benchmarks");
  for (auto &file : files_3d) {
    run_gtsam(&b, file, true);
    run_ceres<ProblemSE3>(&b, file);
  }
  std::cout << "\nIn Markdown format:\n";
  b.render(markdown(), std::cout);

  // 2d benchmarks
  b.title("2d benchmarks");
  for (auto &file : files_2d) {
    run_gtsam(&b, file, false);
    run_ceres<ProblemSE2>(&b, file);
  }
  std::cout << "\nIn Markdown format:\n";
  b.render(markdown(), std::cout);
}