import pandas as pd
from pathlib import Path
import json
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from distributed import get_client, as_completed

from permutandis.utils.problem import RearrangementProblem
from permutandis.utils.state_sampler import StateSampler
from permutandis.utils.errors import SamplerError

import bokeh
import seaborn as sns
from bokeh.plotting import figure
from bokeh.models import Range1d


class RearrangeEvaluation:
    def __init__(self, no_client=False):
        if no_client:
            self.client = None
        else:
            self.client = get_client()

    def sample_problems(self, n_min_objects=1, n_max_objects=12, n_configs=100,
                        workspace=((0.25, -0.25), (0.75, 0.25)), radius=0.0375,
                        timeout=2.0, n_max_iter=50):
        """
        n_min_objects, n_max_objects: interval of number of objects to consider for the problems
        n_configs: number of configurations for each number of objects
        workspace: workspace in which to sample objects
        radius: radius for collision checks
        timeout: time limit that defines when a configuration should be discarded
        because more objects cannot be added
        n_max_iter: Number of times we allow timeout to be reached
        """
        workspace = np.asarray(workspace).tolist()
        self.infos = dict(workspace=workspace, radius=radius)

        def sample_state(n_objects):
            sampler = StateSampler(workspace=workspace, radius=radius)
            n_iter = 0
            success = False
            while not success:
                if n_iter > n_max_iter:
                    raise SamplerError('Too many objects for this workspace.')
                try:
                    state = sampler(n_objects, max_time=timeout)
                    success = True
                except SamplerError:
                    success = False
                    n_iter += 1
            return state

        def sample_src_tgt(n_objects, seed):
            np.random.seed(seed)
            src = sample_state(n_objects)
            tgt = sample_state(n_objects)
            return src, tgt, n_objects

        eval_df = defaultdict(list)
        futures = []
        for n in np.arange(n_min_objects, n_max_objects + 1):
            for _ in range(n_configs):
                fut = self.client.submit(sample_src_tgt, n, np.random.randint(2**32-1))
                futures.append(fut)

        print("Sampling configurations ...")
        for fut in tqdm(as_completed(futures), total=len(futures)):
            src, tgt, n_objects = fut.result()
            eval_df['src'].append(src)
            eval_df['tgt'].append(tgt)
            eval_df['n_objects'].append(n_objects)

        self.eval_df = pd.DataFrame(eval_df)
        self.eval_df = self.eval_df.sort_values('n_objects').reset_index(drop=True)

    def eval_solver(self, solver, method_name='MyMethod', add_fields=[]):
        """
        solver: The solver to evaluate.
        method_name: Name of the method to evaluate, name that will appear in results dataframe.
        add_fields: List containing additionnal fields that should be saved. These fields
        must be present in the output dict of the solver.
        """
        all_outputs = defaultdict(list)
        save_fields = ['success', 'n_moves', 'n_collision_checks'] + list(add_fields)

        def run_solver(solver, problem, problem_idx):
            outputs = solver(problem)
            assert "solved" in outputs
            if outputs['solved']:
                assert 'actions' in outputs
                actions = outputs['actions']
                # Verify that the problem is indeed solved.
                # This will stop the evaluation if the action sequence found doesn't solve the problem
                # or doesn't respect the constraints (objects within workspace and no collisions)
                problem.assert_solution_valid(actions)
                outputs['n_moves'] = len(actions)
                outputs['success'] = True
            else:
                outputs['success'] = False
                outputs['n_moves'] = np.nan
                outputs['n_collision_checks'] = np.nan
            return outputs, problem_idx

        futures = []
        print(f"{method_name} evaluation ...")
        for row_idx, row in enumerate(self.eval_df.itertuples()):
            src = row.src
            tgt = row.tgt
            problem = RearrangementProblem(src=src, tgt=tgt,
                                           workspace=self.infos['workspace'],
                                           radius=self.infos['radius'])
            fut = self.client.submit(run_solver, solver, problem, row_idx)
            futures.append(fut)

        for fut in tqdm(as_completed(futures), total=len(futures)):
            outputs, problem_idx = fut.result()
            all_outputs['problem_idx'].append(problem_idx)
            for k in save_fields:
                all_outputs[f'{method_name}/{k}'].append(outputs[k])
        print("Done")
        sort_ids = np.argsort(all_outputs['problem_idx'])
        del all_outputs['problem_idx']
        all_outputs = pd.DataFrame(all_outputs)
        all_outputs = all_outputs.iloc[sort_ids].reset_index(drop=True)
        self.eval_df = pd.concat((self.eval_df, all_outputs), axis=1)
        return

    def save(self, p):
        save = json.dumps(dict(infos=self.infos, eval_df=self.eval_df.to_json()))
        Path(p).write_text(save)

    def load(self, p):
        save = json.loads(Path(p).read_text())
        self.infos = save['infos']
        self.eval_df = pd.read_json(save['eval_df'])

    @staticmethod
    def from_json(p):
        rearrange_eval = RearrangeEvaluation(no_client=True)
        rearrange_eval.load(p)
        return rearrange_eval

    def make_plots(self, methods=['mcts', 'baseline']):
        gb = self.eval_df.groupby('n_objects')
        gb_mean = gb.mean()
        colors = {method: color for method, color in zip(methods, sns.color_palette().as_hex())}
        line_width = 2

        def format_figure(f):
            f.legend.border_line_width = 0
            f.legend.border_line_alpha = 0
            f.legend.background_fill_alpha = 0
            f.xaxis.axis_label_text_font_style = 'normal'
            f.yaxis.axis_label_text_font_style = 'normal'
            f.toolbar.logo = None
            f.toolbar_location = None


        f = figure(title='Success rate')
        f.xaxis.axis_label = 'N objects'
        f.yaxis.axis_label = 'Success rate'
        f.y_range = Range1d(0, 105)
        for method in methods:
            f.line(gb_mean.index.values, gb_mean[f'{method}/success'].values * 100,
                   legend_label=method, color=colors[method], line_width=line_width)
        f.legend.location = 'bottom_left'
        format_figure(f)
        f_success = f

        f = figure(title='Number of object moves')
        f.xaxis.axis_label = 'N objects'
        f.yaxis.axis_label = 'Number of object moves'
        for method in methods:
            f.line(gb_mean.index.values, gb_mean[f'{method}/n_moves'].values,
                   legend_label=method, color=colors[method], line_width=line_width)
        f.legend.location = 'top_left'
        format_figure(f)
        f_moves = f

        f = figure(title='Number of collision checks', y_axis_type="log")
        f.xaxis.axis_label = 'N objects'
        f.yaxis.axis_label = 'Number of collision checks'
        for method in methods:
            f.line(gb_mean.index.values, gb_mean[f'{method}/n_collision_checks'].values,
                   legend_label=method, color=colors[method], line_width=line_width)
        f.legend.location = 'top_left'
        format_figure(f)
        f_collision_checks = f
        return [f_success, f_moves, f_collision_checks]
