import sys
import numpy as np
from pathlib import Path

from permutandis.utils.state_sampler import StateSampler
from permutandis.utils.problem import RearrangementProblem

from permutandis.solvers.mcts import MCTSSolver
from permutandis.solvers.baseline import BaselineSolver

# Definition of the 2D workspace: ((x1, y1), (x2, y2))
WORKSPACE = np.array(((0.25, -0.25), (0.75, 0.25)))

# Object radius to consider for collision checking
RADIUS = 0.0375


def make_random_problem(n_objects):
    # Sample a random problem
    sampler = StateSampler(workspace=WORKSPACE, radius=RADIUS)
    src = sampler(n_objects=n_objects)
    tgt = sampler(n_objects=n_objects)
    problem = RearrangementProblem(src, tgt, workspace=WORKSPACE, radius=RADIUS)
    return problem


def mcts_solve_problem(problem):
    # Solve it with MCTS
    solver = MCTSSolver(max_iterations=10000, nu=1.0)
    outputs = solver(problem)
    actions = outputs['actions']
    dt = outputs['time']
    n_checks = outputs['n_collision_checks']
    n_objects = problem.src.shape[0]
    print_str = f"Problem with {n_objects} objects. "
    if outputs['solved']:
        problem.assert_solution_valid(actions)
        print_str += f"MCTS found solution with {len(actions)} actions in {dt * 1000:.2f} ms. ({n_checks})"
    else:
        print_str = "MCTS did not find a solution."
    print(print_str)


if __name__ == '__main__':
    np.random.seed(0)
    for n in range(1, 30):
        problem = make_random_problem(n)
        mcts_solve_problem(problem)
