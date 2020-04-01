#ifndef PERMUTANDIS__ENV_HPP
#define PERMUTANDIS__ENV_HPP

#include "state.hpp"
#include <Eigen/StdVector>
#include <torch/torch.h>

namespace permutandis {

using Vec2f = Eigen::Vector2f;

struct EnvironmentState {
  std::size_t N;
  using Location = Eigen::Matrix<float, Eigen::Dynamic, 2>;
  EnvironmentState(int n_objects);
  Location loc;
  float radius = 0.0375f;
  struct Workspace {
    Vec2f min = Vec2f(0.25f, -0.25f), max = Vec2f(0.75f, 0.25f);
  } workspace;

  void reset();
  Location from_workspace() const;
  Eigen::VectorXf distances(const EnvironmentState& other) const;
  bool possible() const;
  bool collides() const;
  std::size_t nearest_point_idx(const Vec2f& p) const;
  std::tuple<int, Vec2f, int> find_space(const Vec2f& p);
};

struct PermutandisEnvironment {
  typedef EnvironmentState State;

  PermutandisEnvironment(
      int n_objects,
      float solve_radius_ = 0.0001f);

  EnvironmentState current, target;
  int N, nactions, steps;
  float solve_radius;
  struct Action {
    int object, move_object;
    Eigen::Vector2f place_pos;
  };
  std::vector<Action> action_mapping;

  using Bitset = Eigen::Matrix<bool, Eigen::Dynamic, 1>;
  int compute_deterministic_action(int move);
  void reset(bool sample = true);
  auto state() const;
  const Bitset solved() const;
  const EnvironmentState make_attempt(const Action& a) const;
  bool step(Action a);
};

} // namespace permutandis

#endif // PERMUTANDIS__ENV_HPP
