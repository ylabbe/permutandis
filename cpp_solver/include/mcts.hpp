#ifndef MCTS_HEADER_PETTER
#define MCTS_HEADER_PETTER

// Modified from
// Petter Strandmark 2013 https://github.com/PetterS/monte-carlo-tree-search
// Monte Carlo Tree Search for finite games.
// Originally based on Python code at
// http://mcts.ai/code/python.html


namespace MCTS {

  static void assertion_failed(const char* expr, const char* file, int line);

  static void
  assertion_failed(const char* expr, const char* file_cstr, int line) {
    using namespace std;

    // Extract the file name only.
    string file(file_cstr);
    auto pos = file.find_last_of("/\\");
    if (pos == string::npos) {
      pos = 0;
    }
    file = file.substr(pos + 1); // Returns empty string if pos + 1 == length.

    stringstream sout;
    sout << "Assertion failed: " << expr << " in " << file << ":" << line << ".";
    throw runtime_error(sout.str().c_str());
  }

#define attest(expr)                                      \
  if (!(expr)) {                                          \
    ::MCTS::assertion_failed(#expr, __FILE__, __LINE__);  \
  }
#ifndef NDEBUG
#define dattest(expr)                                     \
  if (!(expr)) {                                          \
    ::MCTS::assertion_failed(#expr, __FILE__, __LINE__);  \
  }
#else
#define dattest(expr) ((void)0)
#endif

struct ComputeOptions {
  int max_iterations;
  double nu;
  bool verbose;

  ComputeOptions()
      : max_iterations(10000),
        nu(1.0),
        verbose(false){}
};

struct Outputs {
  std::vector<int> moves;
  std::vector<std::tuple<int, Eigen::Vector2f>> actions;
  bool solved, solved_once;
  int n_iterations, n_iterations_first_solve, n_moves, n_collision_checks;
  double time;

  Outputs()
      : moves(),
        actions(),
        solved(false),
        solved_once(false),
        n_iterations(0),
        n_iterations_first_solve(-1),
        n_moves(0.),
        n_collision_checks(0),
        time(0.) {}
};

template <typename State>
std::vector<typename State::Move> compute_move(
    const State root_state,
    const ComputeOptions options = ComputeOptions());

} // namespace MCTS

#include <algorithm>
#include <cstdlib>
#include <future>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#ifdef USE_OPENMP
#include <omp.h>
#endif

namespace MCTS {
using std::cerr;
using std::endl;
using std::size_t;
using std::vector;


template <typename State>
class Node {
 public:
  typedef typename State::Move Move;
  State location_state;

  Node(const State& state);
  ~Node();

  bool has_untried_moves() const;
  template <typename RandomEngine>
  Move get_untried_move(RandomEngine* engine) const;
  bool has_children() const {
    return !children.empty();
  }
  Node* select_child_UCB(double nu = 1.0) const;
  Node* select_child_maxq() const;
  Node* add_child(const Move& move, const State& state);
  void update(double result);

  const Move move;
  Node* const parent;
  double wins;
  int visits;

  std::vector<Move> moves;
  std::vector<Node*> children;

 private:
  Node(const State& state, const Move& move, Node* parent);
  Node(const Node&);
  Node& operator=(const Node&);

  double UCB_score;
  torch::Tensor p;
};

template <typename State>
Node<State>::Node(const State& state)
    : location_state(state),
      move(State::no_move),
      parent(nullptr),
      wins(0),
      visits(0),
      moves(state.get_moves()),
      UCB_score(0) {}

template <typename State>
Node<State>::Node(const State& state, const Move& move_, Node* parent_)
    : location_state(state),
      move(move_),
      parent(parent_),
      wins(0),
      visits(0),
      moves(state.get_moves()),
      UCB_score(0) {}

template <typename State>
Node<State>::~Node() {
  for (auto child : children) {
    delete child;
  }
}

template <typename State>
bool Node<State>::has_untried_moves() const {
  return !moves.empty();
}

template <typename State>
template <typename RandomEngine>
typename State::Move Node<State>::get_untried_move(RandomEngine* engine) const {
  attest(!moves.empty());
  std::uniform_int_distribution<std::size_t> moves_distribution(0, moves.size() - 1);
  return moves[moves_distribution(*engine)];
}

template <typename State>
Node<State>* Node<State>::select_child_UCB(double nu)
    const {
  attest(!children.empty());
  for (auto child : children) {
    double base_score = double(child->wins) / double(child->visits);
    child->UCB_score = base_score + nu * std::sqrt(2.0 * std::log(double(this->visits)) / child->visits);
  }

  return *std::max_element(
      children.begin(), children.end(), [](Node* a, Node* b) {
        return a->UCB_score < b->UCB_score;
      });
}

template <typename State>
Node<State>* Node<State>::select_child_maxq() const {
  attest(!children.empty());

  return *std::max_element(
      children.begin(), children.end(), [](Node* a, Node* b) {
        return (
            (float(a->wins) / float(a->visits)) <
            (float(b->wins) / float(b->visits)));
      });
  ;
}

template <typename State>
Node<State>* Node<State>::add_child(const Move& move, const State& state) {
  auto node = new Node(state, move, this);
  children.push_back(node);
  attest(!children.empty());

  auto itr = moves.begin();
  for (; itr != moves.end() && *itr != move; ++itr)
    ;
  attest(itr != moves.end());
  moves.erase(itr);
  return node;
}

template <typename State>
void Node<State>::update(double result) {
  visits++;
  wins += result;
}

template <typename State>
std::unique_ptr<Node<State>> compute_tree(
    const State root_state,
    const ComputeOptions options,
    std::mt19937_64::result_type initial_seed) {
  Outputs outputs;
  return compute_tree(root_state, options, initial_seed, outputs);
}

template <typename State>
std::unique_ptr<Node<State>> compute_tree(
    const State root_state,
    const ComputeOptions options,
    std::mt19937_64::result_type initial_seed,
    Outputs& outputs) {
  std::mt19937_64 random_engine(initial_seed);
  attest(options.max_iterations >= 0);

  State eval_state = root_state;
  auto root = std::unique_ptr<Node<State>>(new Node<State>(eval_state));

  bool autostopping = false;
  int autostopping_iter = -1;
  outputs.n_iterations = 0;
  int n_moves = 0;
  // MCTS iterations loop
  for (int iter = 1;
       iter <= options.max_iterations || options.max_iterations < 0;
       ++iter) {
    n_moves = 0;
    outputs.n_iterations = iter;

    // Start from root
    auto node = root.get();
    State state = eval_state;

    // Selection stage: Select a path through the tree to a leaf node.
    while (!node->has_untried_moves() && node->has_children()) {
      // UCB selection: pick best child node according to UCB
      node = node->select_child_UCB(options.nu);
      // Recover parameters of pick & place motion cached inside child node
      state.env.action_mapping[node->move] =
          node->location_state.env.action_mapping[node->move];
      // Play action, update state inplace
      state.do_move(node->move);
      n_moves += 1;
    }

    // Expansion stage:
    // If current node is expandable, expand the
    // tree with a new node and move there.
    if (node->has_untried_moves()) {
      auto move = node->get_untried_move(&random_engine);
      // Compute motion parameters corresponding to MCTS action.
      // Motion parameters (object id and placing positions) are stored inside the state of the new node.
      int n_collision_checks = state.compute_deterministic_action(move);
      outputs.n_collision_checks += n_collision_checks;
      // Play action, update state inplace
      state.do_move(move);
      n_moves += 1;
      // Add new node
      node = node->add_child(move, state);
    }

    // If the the state is a final state (problem is solved)
    if (!outputs.solved_once && state.solved()) {
      // Store some informations in outputs struct
      outputs.n_moves = n_moves;
      outputs.solved_once = true;
      outputs.n_iterations_first_solve = iter;
      // Flags for stopping MCTS tree creation/update.
      autostopping = true;
      autostopping_iter = iter;
    }

    // Backpropagation stage: backpropagate the result (reward) up the tree to the root node.
    while (node != nullptr) {
      node->update(state.get_result());
      node = node->parent;
    }

    if (autostopping && (iter >= autostopping_iter)) {
      break;
    }

  }
  return root;
}


template <typename State>
Outputs compute_move_with_outputs(
    const State root_state,
    const ComputeOptions options) {
  Outputs outputs;
  compute_move(root_state, options, outputs);
  return outputs;
}

template <typename State>
std::vector<typename State::Move> compute_move(
    const State root_state,
    const ComputeOptions options) {
  Outputs outputs;
  return compute_move(root_state, options, outputs);
}

template <typename State>
std::vector<typename State::Move> compute_move(
    const State root_state,
    const ComputeOptions options,
    Outputs& outputs) {
  using namespace std;
  auto moves = root_state.get_moves();
  attest(moves.size() > 0);

  // Compute MCTS tree
  auto root_ptr = compute_tree(root_state, options, 12515, outputs);
  auto root = root_ptr.get();
  vector<typename State::Move> best_moves;
  // Recover actions and pick & place motions from tree by following the greedy path (max Q selection)
  vector<tuple<typename State::Move, Eigen::Vector2f>> actions;
  while (root->has_children()) {
    auto best_child = root->select_child_maxq();
    // Recover pick & place motion cached in the tree correspondig to MCTS action
    auto action = best_child->location_state.env.action_mapping[best_child->move];
    actions.push_back(make_tuple(action.move_object, action.place_pos));
    best_moves.push_back(best_child->move);
    root = best_child;
  }
  outputs.moves = best_moves;
  outputs.n_moves = best_moves.size();
  outputs.actions = actions;

  return best_moves;
}

} // namespace MCTS

#endif
