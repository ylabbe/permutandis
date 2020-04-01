#define CATCH_CONFIG_MAIN
#include "Catch2/catch.hpp"
#include "include/env.hpp"
#include "include/mcts.hpp"
#include "include/state.hpp"
#include <torch/torch.h>

using namespace torch;
using namespace permutandis;

std::tuple<Tensor, Tensor, Tensor> compute_moves_mcts(
    const Tensor current,
    const Tensor prev,
    const Tensor target,
    int max_iterations,
    float solve_radius,
    const std::string run_path);

TEST_CASE("create permutandis env", "[permutandis]") {
  PermutandisEnvironment env(4);
  env.reset();
  REQUIRE(env.current.possible());
  REQUIRE(env.target.possible());
  REQUIRE(env.current.distances(env.current).sum() == 0);
  for (int i=0; i < (int) env.action_mapping.size();  ++i) {
      env.compute_deterministic_action(i);
      auto action = env.action_mapping[i];
      env.step(action);
  }
}


bool run_test(int N) {
  MCTS::ComputeOptions options;
  options.max_iterations = 1e+6;
  options.verbose = true;
  options.nu = 1.0;

  PermutandisState<PermutandisEnvironment> state(N);
  REQUIRE(state.possible());

  auto outputs = MCTS::compute_move_with_outputs(state, options);
  auto moves = outputs.moves;

  for (auto v : moves) {
    state.compute_deterministic_action(v);
    state.do_move(v);
    if (state.solved())
      return true;
  }
  return state.solved();
}

TEST_CASE("MCTS test 3 objects", "[permutandis]") {
  REQUIRE(run_test(3));
}
TEST_CASE("MCTS test 4 objects", "[permutandis]") {
  REQUIRE(run_test(4));
}
TEST_CASE("MCTS test 5 objects", "[permutandis]") {
  REQUIRE(run_test(5));
}
TEST_CASE("MCTS test 6 objects", "[permutandis]") {
  REQUIRE(run_test(6));
}
TEST_CASE("MCTS test 7 objects", "[permutandis]") {
  REQUIRE(run_test(7));
}
TEST_CASE("MCTS test 8 objects", "[permutandis]") {
  REQUIRE(run_test(8));
}
TEST_CASE("MCTS test 10 objects", "[permutandis]") {
  REQUIRE(run_test(10));
}
TEST_CASE("MCTS test 15 objects", "[permutandis]") {
  REQUIRE(run_test(15));
}
TEST_CASE("MCTS test 20 objects", "[permutandis]") {
  REQUIRE(run_test(20));
}
TEST_CASE("MCTS test 25 objects", "[permutandis]") {
  REQUIRE(run_test(25));
}
TEST_CASE("MCTS test 30 objects", "[permutandis]") {
  REQUIRE(run_test(30));
}
