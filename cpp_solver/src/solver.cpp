
#include <iostream>
#include <nlohmann/json.hpp>
#include <cxxopts.hpp>
#include "include/env.hpp"
#include "include/mcts.hpp"
#include <Eigen/StdVector>

using namespace permutandis;

using nlohmann::json;
using std::cout;

template<class State>
bool check(State &state, const std::vector<int> &moves) {
  for(auto v: moves) {
    assert(!state.solved());
    state.env.compute_deterministic_action(v);
    state.do_move(v);
  }
  return state.solved();
}

MCTS::Outputs evaluate_mcts(torch::Tensor src,
                            torch::Tensor tgt,
                            Eigen::Matrix<float, 2, 2> workspace,
                            float radius,
                            MCTS::ComputeOptions options)
{
  PermutandisState<PermutandisEnvironment> state(src, tgt, workspace, radius);

  if (state.solved()){
    MCTS::Outputs outputs;
    outputs.n_iterations = 0;
    outputs.solved = true;
    outputs.time = 0.d;
    return outputs;
  }

	auto start = clock();
  auto outputs = MCTS::compute_move_with_outputs(state, options);
  auto end = clock();
  float clocks_per_sec = CLOCKS_PER_SEC;
  outputs.time = (end - start) / clocks_per_sec;
  bool solved = check(state, outputs.moves);
  outputs.solved = solved;
  return outputs;
}

void write_outputs(MCTS::Outputs outputs, std::string path)
{
  json j_out;

  // Can't find how to json serialize the vector of tuple
  std::vector<int> moves;
  std::vector<float> x, y;

  for (int i = 0; i < outputs.actions.size(); i++) {
    moves.push_back(std::get<0>(outputs.actions[i]));
    x.push_back((std::get<1>(outputs.actions[i]))(0));
    y.push_back((std::get<1>(outputs.actions[i]))(1));
  }
  j_out = {{"x", x}, {"y", y}, {"moves", moves},
           {"solved", outputs.solved}, {"time", outputs.time},
           {"solved_once", outputs.solved_once},
           {"n_iterations", outputs.n_iterations},
           {"n_collision_checks", outputs.n_collision_checks},
           {"n_moves", outputs.n_moves}};
  {
    std::ofstream ofs(path);
    ofs << j_out;
  }
}

template <class T>
auto default_value(T value) {
  return cxxopts::value<T>()->default_value(std::to_string(value));
}

int main(int argc, char* argv[]) {
  cxxopts::Options options("Rearrangement solver", "MCTS rearrangement planning");
  options.add_options()
    ("json_in", "json file containing dict(src,tgt,options)", cxxopts::value<std::string>())
    ("json_out", "json file containing outputs", cxxopts::value<std::string>())
    ;
  auto res = options.parse(argc, argv);
  json j;
  {
    std::ifstream ifs(res["json_in"].as<std::string>());
    ifs >> j;
  }

  auto src = j["src"];
  auto tgt = j["tgt"];
  int n_objects = src.size();
  torch::Tensor src_t = torch::zeros({n_objects, 2});
  torch::Tensor tgt_t = torch::zeros_like(src_t);
  for (int m = 0; m < n_objects; ++m) {
    for (int k = 0; k < 2; ++k) {
      src_t[m][k] = src[m][k].get<float>();
      tgt_t[m][k] = tgt[m][k].get<float>();
    }
  }

  auto workspace = j["workspace"];
  float radius = j["radius"];
  Eigen::Matrix<float, 2, 2> workspace_m;
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      workspace_m(i, j) = workspace[i][j].get<float>();
    }
  }

  int max_iterations = j["max_iterations"];
  double nu = j["nu"];
  MCTS::ComputeOptions mcts_options;
  mcts_options.nu = nu;
  mcts_options.max_iterations = max_iterations;

  std::string out_path = res["json_out"].as<std::string>();
  auto outputs = evaluate_mcts(src_t, tgt_t, workspace_m, radius, mcts_options);
  write_outputs(outputs, out_path);
  return 0;
}
