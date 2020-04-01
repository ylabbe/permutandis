cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(permutandis)

find_package(Torch REQUIRED)
find_package(Boost REQUIRED)

set(CONDA $ENV{CONDA_PREFIX} CACHE FILEPATH "path to anaconda")
message("Anaconda root: " ${CONDA})

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
include_directories(
  ${PROJECT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/third-party
  ${CONDA}/include/eigen3
  ${CONDA}/include
  ${Boost_INCLUDE_DIRS}
  )
link_directories(${CONDA}/lib)

add_executable(permutandis_test src/test.cpp src/env.cpp)
add_executable(solver src/solver.cpp src/env.cpp)

target_compile_options(permutandis_test PRIVATE -Wall -Wextra -pedantic)

target_link_libraries(permutandis_test Threads::Threads "${TORCH_LIBRARIES}")
target_link_libraries(solver Threads::Threads "${TORCH_LIBRARIES}")

set_property(TARGET permutandis_test solver PROPERTY CXX_STANDARD 17)
#target_compile_features(permutandis_test PRIVATE cxx_std_17)
