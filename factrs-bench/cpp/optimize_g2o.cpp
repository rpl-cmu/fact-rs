// Simple g2o pose-graph optimizer using GTSAM.
// Usage: optimize_g2o <path-to-g2o-file> [gn|lm]
//   gn = Gauss-Newton (default)
//   lm = Levenberg-Marquardt

#include "gtsam/linear/NoiseModel.h"
#include <boost/smart_ptr/shared_ptr.hpp>
#include <gtsam/nonlinear/GaussNewtonOptimizer.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/dataset.h>

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string>

// Detect whether a g2o file is 3D by scanning for VERTEX_SE3:QUAT lines.
bool is3D(const std::string &path) {
  std::ifstream in(path);
  if (!in) {
    std::cerr << "Error: cannot open file " << path << "\n";
    std::exit(1);
  }
  std::string token;
  while (in >> token) {
    if (token == "VERTEX_SE3:QUAT" || token == "EDGE_SE3:QUAT")
      return true;
    if (token == "VERTEX_SE2" || token == "EDGE_SE2")
      return false;
    // skip rest of line
    std::getline(in, token);
  }
  std::cerr << "Error: could not determine dimensionality of " << path << "\n";
  std::exit(1);
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <path-to-g2o-file> [gn|lm]\n";
    return 1;
  }

  const std::string path = argv[1];
  std::string method = "gn";
  if (argc >= 3) {
    method = argv[2];
  }
  if (method != "gn" && method != "lm") {
    std::cerr << "Unknown method '" << method << "'. Use 'gn' or 'lm'.\n";
    return 1;
  }

  // Detect 2D vs 3D
  const bool threeD = is3D(path);
  std::cout << "File:      " << path << "\n";
  std::cout << "Dimension: " << (threeD ? "3D" : "2D") << "\n";
  std::cout << "Method:    "
            << (method == "gn" ? "Gauss-Newton" : "Levenberg-Marquardt")
            << "\n";

  // Load graph and initial values from g2o
  gtsam::GraphAndValues gv = gtsam::readG2o(path, threeD);
  gtsam::NonlinearFactorGraph graph = *gv.first;
  gtsam::Values initial = *gv.second;

  // Add a prior on the first pose to anchor the graph (gauge freedom)
  if (threeD) {
    auto priorModel = gtsam::noiseModel::Diagonal::Variances(
        (gtsam::Vector(6) << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4).finished());
    graph.addPrior(0, gtsam::Pose3::Identity(), priorModel);
  } else {
    auto priorModel = gtsam::noiseModel::Diagonal::Variances(
        gtsam::Vector3(1e-6, 1e-6, 1e-8));
    graph.addPrior(0, gtsam::Pose2::Identity(), priorModel);
  }

  std::cout << "Poses:     " << initial.size() << "\n";
  std::cout << "Factors:   " << graph.size() << "\n";

  double errorBefore = graph.error(initial);
  std::cout << "\nInitial error: " << errorBefore << "\n";

  // figure out what the covariances look like
  auto fac = graph[2];
  auto fac_noise = boost::dynamic_pointer_cast<gtsam::NoiseModelFactor>(fac);
  auto gauss = boost::dynamic_pointer_cast<gtsam::noiseModel::Gaussian>(
      fac_noise->noiseModel());
  auto R = gauss->R();
  std::cout << R << "\n" << std::endl;
  std::cout << R.transpose() * R << "\n" << std::endl;
  // for (int i = 0; i < 5; i++) {
  //   auto fac = graph[i];
  //   fac->print();
  // }

  return 0;

  // Optimize
  auto t0 = std::chrono::high_resolution_clock::now();

  gtsam::Values result;
  if (method == "gn") {
    gtsam::GaussNewtonParams params;
    params.setVerbosity("ERROR");
    gtsam::GaussNewtonOptimizer optimizer(graph, initial, params);
    result = optimizer.optimize();
  } else {
    gtsam::LevenbergMarquardtParams params;
    params.setVerbosity("ERROR");
    gtsam::LevenbergMarquardtOptimizer optimizer(graph, initial, params);
    result = optimizer.optimize();
  }

  auto t1 = std::chrono::high_resolution_clock::now();
  double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

  double errorAfter = graph.error(result);
  std::cout << "Final error:   " << errorAfter << "\n";
  std::cout << "Time:          " << ms << " ms\n";

  return 0;
}
