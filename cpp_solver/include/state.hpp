#ifndef PERMUTANDIS__STATE_HPP
#define PERMUTANDIS__STATE_HPP

#include <Eigen/Core>
#include <algorithm>
#include <bitset>
#include <cassert>
#include <random>
#include <torch/torch.h>
#include <tuple>

namespace permutandis {


template <typename TEnvironment>
class PermutandisState {
  // MCTS state compatible with the MCTS implementation
 public:
  using State = typename TEnvironment::State;
  TEnvironment env;
  typedef int Move;
  static const Move no_move = -1;

  PermutandisState(int n_objects) : env(n_objects), done(false) {
    reset();
  }

  PermutandisState(
      const torch::Tensor current,
      const torch::Tensor target)
    : env(current.size(0)), done(false) {
    set_locations(current, target);
    assert(possible());
    env.reset(false);
    update_possible_moves();
  }

  PermutandisState(
                   const torch::Tensor current,
                   const torch::Tensor target,
                   Eigen::Matrix<float, 2, 2> workspace,
                   const float radius)
    : env(current.size(0)), done(false) {

    env.current.workspace.min(0) = workspace(0, 0);
    env.current.workspace.min(1) = workspace(0, 1);
    env.current.workspace.max(0) = workspace(1, 0);
    env.current.workspace.max(1) = workspace(1, 1);
    env.current.radius = radius;

    env.target.workspace.min(0) = workspace(0, 0);
    env.target.workspace.min(1) = workspace(0, 1);
    env.target.workspace.max(0) = workspace(1, 0);
    env.target.workspace.max(1) = workspace(1, 1);
    env.target.radius = radius;

    set_locations(current, target);
    assert(possible());
    env.reset(false);
    update_possible_moves();
  }

  void reset() {
    env.reset();
    assert(possible());
    update_possible_moves();
    done = false;
  }

  void set_locations(
      const torch::Tensor current,
      const torch::Tensor target) {
    if (current.size(0) != target.size(0))
      std::runtime_error(
          "provided current and target configs have different size");
    if (current.size(0) != env.current.loc.size())
      std::runtime_error("env has different number of objects");

    for (int i = 0; i < current.size(0); ++i)
      for (int j = 0; j < 2; ++j) {
        env.current.loc(i, j) = current[i][j].item<float>();
        env.target.loc(i, j) = target[i][j].item<float>();
      }

    if (!possible())
      std::runtime_error("configuration is not possible");

    update_possible_moves();
    done = env.solved().all();
  }

  int compute_deterministic_action(int move) {
    return env.compute_deterministic_action(move);
  }

  void do_move(Move move) {
    done = env.step(env.action_mapping[move]);
    update_possible_moves();
  }

  bool solved() const {
    return env.solved().all();
  }

  const auto& get_moves() const {
    return possible_moves;
  }

  double get_result() const {
    if (env.solved().all())
      // High reward just to ensure that following the path that maximizes Q value yields the solution.
      // Only used for retrieving the action sequence *after* the tree has been built.
      // This is *not* reward shaping as we stop MCTS once a solution has been found.
      return 1e10;
    else
      // Actual reward used at every step during MCTS.
      return env.solved().count();
  }

  bool possible() const {
    return env.current.possible() && env.target.possible();
  }

 protected:
  std::vector<Move> possible_moves;
  bool done;
  void update_possible_moves() {
    possible_moves.clear();
    if (done)
      return;
    const auto solved = env.solved();
    int move = 0;
    for (const auto& a : env.action_mapping) {
      if (!solved(a.object))
        possible_moves.push_back(move);
      move++;
    }
  }

};

} // namespace permutandis

#endif // PERMUTANDIS__STATE_HPP
