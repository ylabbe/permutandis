#include "include/env.hpp"
#include "include/state.hpp"

namespace permutandis {

std::default_random_engine gen;

EnvironmentState::EnvironmentState(int n_objects) : N(n_objects) {
  loc = Location(N, 2);
}

void EnvironmentState::reset() {
  loc = from_workspace();
}

EnvironmentState::Location EnvironmentState::from_workspace() const {
  // Generate random configuration
  // std::default_random_engine gen;
  float eps = 1e-3f;
  std::uniform_real_distribution<float> dis_x(
      workspace.min.x() + eps, workspace.max.x() - eps);
  std::uniform_real_distribution<float> dis_y(
      workspace.min.y() + eps, workspace.max.y() - eps);

  std::vector<Vec2f, Eigen::aligned_allocator<Vec2f>> points;

  auto collides = [&points](const Vec2f& p, float collision_threshold) -> bool {
    for (auto& it : points)
      if ((p - it).norm() < collision_threshold)
        return true;
    return false;
  };

  int c = 0;
  while (points.size() < N) {
    if (c > 5000) {
      points.clear();
      c = 0;
    }
    auto p = Vec2f(dis_x(gen), dis_y(gen));
    c++;
    if (!collides(p, 2.f * radius))
      points.push_back(p);
  }
  Location base(N, 2);
  for (std::size_t i = 0; i < N; ++i)
    base.row(i) = points[i];
  return base;
}

bool EnvironmentState::collides() const {
  auto collision_threshold = 2.f * radius;
  for (std::size_t i = 0; i < N - 1; ++i) {
    float min_dist =
        (loc.bottomRightCorner(N - i - 1, 2).rowwise() - loc.row(i))
            .rowwise()
            .squaredNorm()
            .minCoeff();
    if (min_dist < collision_threshold * collision_threshold)
      return true;
  }
  return false;
}

Eigen::VectorXf EnvironmentState::distances(
    const EnvironmentState& other) const {
  const Eigen::VectorXf loc_dist = (this->loc - other.loc).rowwise().norm();
  return loc_dist;
}

bool EnvironmentState::possible() const {
  const Vec2f loc_min = loc.colwise().minCoeff();
  const Vec2f loc_max = loc.colwise().maxCoeff();
  bool in_workspace = (loc_min.array() > workspace.min.array()).all() &&
      (loc_max.array() < workspace.max.array()).all();
  return in_workspace & !collides();
}


std::size_t EnvironmentState::nearest_point_idx(const Vec2f& p) const {
  Eigen::Index index;
  (loc.transpose().colwise() - p).colwise().squaredNorm().minCoeff(&index);
  return index;
}

std::tuple<int, Vec2f, int> EnvironmentState::find_space(
    const Vec2f& p) {
  std::size_t idx = nearest_point_idx(p);
  EnvironmentState tmp = *this;
  float eps = 1e-2f;

  int n_collision_checks = 0;
  std::default_random_engine gen_fix(555);
  std::uniform_real_distribution<float> dis_x(
      workspace.min.x() + eps, workspace.max.x() - eps);
  std::uniform_real_distribution<float> dis_y(
      workspace.min.y() + eps, workspace.max.y() - eps);

  const float r = 2 * radius * 2 * radius;
  for (int i = 0; i < 4000; ++i) {
    n_collision_checks++;
    if (const Vec2f p_rnd(dis_x(gen_fix), dis_y(gen_fix));
        (p - p_rnd).squaredNorm() > r) {
      tmp.loc.row(idx) = p_rnd;
      if (!tmp.collides())
        return {int(idx), p_rnd, n_collision_checks};
    }
  }

  return {-1, Vec2f(), n_collision_checks};
}

PermutandisEnvironment::PermutandisEnvironment(
    int n_objects,
    float solve_radius_)
    : current(n_objects),
      target(n_objects),
      N(n_objects),
      steps(0),
      solve_radius(solve_radius_) {}

int PermutandisEnvironment::compute_deterministic_action(int k) {
  // Compute parameters of the pick & place movement corresponding to MCTS action "move"
  Action action = action_mapping[k];
  EnvironmentState attempt = make_attempt(action);
  bool is_possible = attempt.possible();
  int n_collision_checks = 1;
  // Object target is obstructed
  if (!is_possible) {
    // Find closest object j and position P
    auto [j, P, n_checks] =
        current.find_space(target.loc.row(k));
    n_collision_checks += n_checks;
    bool found = j >= 0;
    if (found) {
      // If no position is found, no object is moved
      action.place_pos = P;
      action.move_object = j;
    } else {
      // Otherwise, move object j to P
      action.place_pos = current.loc.row(k);
      action.move_object = k;
    }
  } else {
    // Object can be moved directly to its target
    action.place_pos = target.loc.row(k);
    action.move_object = k;
  }
  action_mapping[k] = action;
  return n_collision_checks;
}

void PermutandisEnvironment::reset(bool sample) {
  action_mapping.clear();

  // Initialize action mapping with dummy actions
  // None of these actions are actually used to solve the problem.
  // Actual actions are computed in "compute_deterministic_action"
  Eigen::Vector2f place_pos(-10., -10.);
  for (int i = 0; i < N; ++i) {
    action_mapping.push_back({i, i, place_pos});
  }

  // Option to sample random current and target arrangements.
  if (sample) {
    current.reset();
    target.reset();
    assert(current.possible());
    assert(target.possible());
  }
  steps = 0;
}

const PermutandisEnvironment::Bitset PermutandisEnvironment::solved() const {
  const PermutandisEnvironment::Bitset is_close =
      current.distances(target).array() < solve_radius;
  return is_close;
}

const EnvironmentState PermutandisEnvironment::make_attempt(
    const Action& a) const {
  EnvironmentState attempt(current);
  attempt.loc.row(a.object) = target.loc.row(a.object);
  return attempt;
}

bool PermutandisEnvironment::step(Action a) {
  current.loc.row(a.move_object) = a.place_pos;
  steps++;
  return solved().all();
}

} // namespace permutandis
