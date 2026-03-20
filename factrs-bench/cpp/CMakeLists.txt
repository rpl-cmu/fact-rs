cmake_minimum_required(VERSION 3.16)

project(factrs-bench-cpp
    VERSION 0.1.0
    LANGUAGES CXX
)
include(FetchContent)

set(CMAKE_BUILD_TYPE RELEASE)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ------------------------- Include Benchmarking library ------------------------- #
FetchContent_Declare(
    nanobench
    GIT_REPOSITORY https://github.com/martinus/nanobench.git
    GIT_TAG v4.3.11
    GIT_SHALLOW TRUE
    GIT_PROGRESS   TRUE
    USES_TERMINAL_DOWNLOAD TRUE
)
FetchContent_MakeAvailable(nanobench)

# ------------------------- Include Ceres ------------------------- #
set(BUILD_TESTING OFF CACHE BOOL "")
set(BUILD_BENCHMARKS OFF CACHE BOOL "")
set(PROVIDE_UNINSTALL_TARGET OFF CACHE BOOL "")
set(BUILD_EXAMPLES OFF CACHE BOOL "") # I may need this...
set(USE_CUDA OFF CACHE BOOL "")

FetchContent_Declare(
    Ceres
    GIT_REPOSITORY https://github.com/ceres-solver/ceres-solver.git
    GIT_TAG        2.2.0
    GIT_SHALLOW TRUE
    GIT_PROGRESS   TRUE
    USES_TERMINAL_DOWNLOAD TRUE
)
FetchContent_MakeAvailable(Ceres)

# ------------------------- Include GTSAM ------------------------- #
# Enable features to make closer to factrs default
set(GTSAM_USE_QUATERNIONS ON CACHE BOOL "")
set(GTSAM_SLOW_BUT_CORRECT_BETWEENFACTOR ON CACHE BOOL "")
set(GTSAM_SLOW_BUT_CORRECT_EXPMAP ON CACHE BOOL "")
# Turn off extra building
set(GTSAM_BUILD_TESTS OFF CACHE BOOL "")
set(GTSAM_BUILD_DOCS OFF CACHE BOOL "")
set(GTSAM_BUILD_PYTHON OFF CACHE BOOL "")
set(GTSAM_INSTALL_MATLAB_TOOLBOX OFF CACHE BOOL "")
set(GTSAM_BUILD_UNSTABLE OFF CACHE BOOL "")
set(GTSAM_BUILD_EXAMPLES OFF CACHE BOOL "")
set(GTSAM_BUILD_EXAMPLES_ALWAYS OFF CACHE BOOL "")

# use single threaded comparisons
set(GTSAM_WITH_TBB OFF CACHE BOOL "")

# disable some gtsam warnings we don't care about
add_compile_options(-Wno-deprecated-declarations)
add_compile_options(-Wno-unused-but-set-variable)
add_compile_options(-Wno-cpp)

FetchContent_Declare(
    GTSAM
    GIT_REPOSITORY https://github.com/borglab/gtsam.git
    GIT_TAG        4.2.0
    GIT_SHALLOW TRUE
    GIT_PROGRESS   TRUE
    USES_TERMINAL_DOWNLOAD TRUE
)
FetchContent_MakeAvailable(GTSAM)

# ------------------------- Our executable ------------------------- #

add_executable(bench bench.cpp)
target_link_libraries(bench PRIVATE nanobench gtsam Ceres::ceres)
# Manually include some Ceres g2o helpers
target_include_directories(bench PRIVATE ${Ceres_SOURCE_DIR}/examples/slam)
set_property(TARGET bench PROPERTY CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)

add_executable(optimize_g2o optimize_g2o.cpp)
target_link_libraries(optimize_g2o PRIVATE gtsam)
set_property(TARGET optimize_g2o PROPERTY CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)