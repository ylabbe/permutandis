MCTS permutandis
==============

MCTS rearrangement planning code for Y. Labbé, S. Zagoruyko, I. Kalevatykh, I. Laptev, J. Carpentier, M. Aubry and J.Sivic, "Monte-Carlo Tree Search for Efficient Visually Guided Rearrangement Planning", IEEE Robotics and Automation Letters.

[[arXiv](https://arxiv.org/abs/1904.10348)] [[Project Page](https://ylabbe.github.io/rearrangement-planning/index.html)] [[Video](https://youtu.be/vZ1B3JaL9Os)]

This repository contains the C++ implementation of our MCTS rearrangement solver presented in the paper. It also contains an interface for solving problems from python, the python code of the baseline presented in the paper as well as code for evaluating your own method.


# Building the MCTS C++ solver
The MCTS implementation is based on code from <https://github.com/PetterS/monte-carlo-tree-search>.

Requirements:
* c++17 compatible compiler (gcc-8 or llvm-7)
* Eigen 3.3.7 or higher
* json for c++
* PyTorch 1.0.1
* Python 3.7
* Boost 1.67

### Installing gcc-8 (tested on Ubuntu 18.04)

```bash
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt update
sudo apt install gcc-8 g++-8
```

### Install dependencies with anaconda

```bash
conda create -n permutandis python=3.7
conda activate permutandis
conda install -c conda-forge eigen=3.3.7 nlohmann_json=3.7.3 cmake=3.14.0
conda install -c anaconda boost=1.67.0
conda install -c pytorch pytorch-cpu=1.0.1
```

### Building the MCTS C++ solver

```bash
cd cpp_solver
mkdir build; cd build
CC=gcc-8 CXX=g++-8 CMAKE_PREFIX_PATH=$CONDA_PREFIX/lib/python3.7/site-packages/torch/share/cmake cmake -DCMAKE_BUILD_TYPE=Release ..
make -j4
```

### Running C++ tests

Check that the code has been properly compiled by running the C++ tests:

```bash
./permutandis_test
```

# Using the C++ solver from python
For conveniency and evaluation, we provide a python interface for solving tabletop rearrangement problem instances. 

### Installing the python package

```bash
conda activate permutandis
pip install -r requirements.txt
python setup.py install
```
We use the bokeh library for plotting. If you are interested in generating evaluation plots, please run:
```bash
conda install selenium geckodriver firefox -c conda-forge
```

### Solving a rearrangement problem
We now assume that you have compiled the C++ code and the binary executable `cpp_solver/build/solver` exists.

We provide an example that generates random rearrangement problems and solve them using our MCTS solver. You can run it using:
```bash
python -m permutandis.examples.solve_random_problems
```

Please see the code for this example directly if you are interested in using our solver for your own problems.

# Evaluating the solver

We provide evaluation code for evaluating our MCTS C++ solver which can be easily extended for also comparing it with your own solver if you are interested in building on top of our work.

### Evaluating our MCTS C++ solver
The evaluation is done using multiple processes, using dask for parallelization. It allows to easily scale the evaluation to a CPU cluster when evaluating on a high number complex problems with high object density.

You can run the evaluation for our MCTS C++ solver and the baseline described in the paper using the following command:
```bash
python -m permutandis.examples.evaluate_solvers
```
This will generate 350 problems with 1 to 35 objects in the workspace, evaluate MCTS and the baseline on each problem and save the results to `data/eval_results.json` as well as plots to `data/results_plots.png`. Note that in the paper, we evaluate on 3700 problems with 1 to 37 objects in the workspace. You change these parameters in the evaluation script for a fair comparison, but evaluation will take longer.

### Implementing your own solver
If you are interested in comparing your method with our MCTS solver, you only need to write a solver similar to `MCTSSolver` or `BaselineSolver`. Given a `RearrangementProblem` (as defined in `permutandis/utils/problem.py`) and outputs a dictionary with (at least) the following fields.
* `solved`: a boolean indicating wether your solver has found a solution for the problem.
* `actions`: if a solution is found, it is a list of tuples `[(object_id, place_pos), ...]` where each tuple correspond to a pick & place action. `object_id` is the index of the object to move and `place_pos` is the 2D position where the object should be placed.
* `n_collision_checks`: The number of collision checks used by your method if a solution is found. A collision check consists in checking wether an object can be placed at a specific location in the workspace. 

Please see `permutandis/solvers/baseline.py` for a simple solver written in python. Your solver can then be easily added to the evaluation script.

## Citation

If you find the code useful please cite this work:

```
@ARTICLE{labbe2020,
author={Y. {Labbe} and S. {Zagoruyko} and I. {Kalevatykh} and I. {Laptev} and J. {Carpentier} and M. {Aubry} and J. {Sivic}},
journal={IEEE Robotics and Automation Letters},
title={Monte-Carlo Tree Search for Efficient Visually Guided Rearrangement Planning},
year={2020}}
```
