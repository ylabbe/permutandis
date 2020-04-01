from pathlib import Path
from bokeh.io import export_png
from bokeh.layouts import gridplot
import permutandis
from permutandis.evaluation.rearrange_evaluation import  RearrangeEvaluation

from permutandis.solvers.mcts import MCTSSolver
from permutandis.solvers.baseline import BaselineSolver

from distributed import Client, LocalCluster

def make_client():
    # Change this if you have access to a cluster and you want to
    # run evaluation
    # See https://jobqueue.dask.org/en/latest/
    cluster = LocalCluster()
    client = Client(cluster)

def evaluate_solvers(results_path):
    rearrange_eval = RearrangeEvaluation()

    # Sample random problems
    # In the paper, we use n_max_objects=37 and n_configs=100
    rearrange_eval.sample_problems(
        n_min_objects=1, n_max_objects=35, n_configs=10,
        workspace=((0.25, -0.25), (0.75, 0.25)), radius=0.0375,
        timeout=2.0, n_max_iter=50)

    # Evaluate MCTS
    mcts_solver = MCTSSolver(nu=1.0, max_iterations=1e5)
    rearrange_eval.eval_solver(mcts_solver, method_name='mcts')

    # Evaluate baseline
    baseline_solver = BaselineSolver()
    rearrange_eval.eval_solver(baseline_solver, method_name='baseline')

    # Save results
    rearrange_eval.save(results_path)
    print("Wrote results to", results_path)

    return rearrange_eval

def make_plots(results_path):
    print("Reading results from", results_path)
    rearrange_eval = RearrangeEvaluation.from_json(results_path)
    figures = rearrange_eval.make_plots(methods=['mcts', 'baseline'])
    plot = gridplot([figures], toolbar_location=None)
    plot_path = results_path.parent / 'results_plots.png'
    export_png(plot, filename=plot_path)
    print("Saved plot to", plot_path)

if __name__ == '__main__':
    make_client()
    data_dir = Path(permutandis.__file__).parent.parent / 'data'
    results_path = data_dir / 'eval_results.json'
    evaluate_solvers(results_path)
    make_plots(results_path)
