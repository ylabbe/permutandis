import json
import tempfile
import subprocess
import numpy as np
import os
import random
from pathlib import Path


class MCTSSolver:
    def __init__(self,
                 max_iterations=10000,
                 nu=1.0,
                 solver_bin_path=None,
                 tmp_dir='/dev/shm/'):
        """
        max_iterations: Maximum number of MCTS iterations.
        nu: Term that balances exploration and exploitation in UCB (parameter $c$ in the paper).
        solver_bin_path: Path to "solver" binary
        tmp_dir: locations to save temporary json files used as input/outputs to the solver
        """
        if solver_bin_path is None:
            # Use default location
            solver_bin_path = Path(__file__).parent.parent.parent / 'cpp_solver/build/solver'
        self.solver_bin_path = solver_bin_path
        self.tmp_dir = tmp_dir
        self.max_iterations = max_iterations
        self.nu = nu

    def __call__(self, problem):
        src = problem.src.tolist()
        tgt = problem.tgt.tolist()
        workspace = problem.workspace.tolist()
        radius = problem.radius

        mcts_inputs = dict(src=src, tgt=tgt,
                           nu=self.nu, max_iterations=self.max_iterations,
                           workspace=workspace, radius=radius)

        input_path = Path(tempfile.NamedTemporaryFile(dir=self.tmp_dir, suffix='.json').name)
        input_path.write_text(json.dumps(mcts_inputs))
        output_path = Path(tempfile.NamedTemporaryFile(dir=self.tmp_dir, suffix='.json').name)
        subprocess.check_output([self.solver_bin_path,
                                 '--json_in', input_path.as_posix(),
                                 '--json_out', output_path.as_posix()])
        assert output_path.exists()
        outputs = json.loads(output_path.read_text())
        x = np.asarray(outputs['x'])[:, np.newaxis]
        y = np.asarray(outputs['y'])[:, np.newaxis]
        xy = np.concatenate((x, y), axis=1)
        del outputs['x']
        del outputs['y']
        outputs['place_pos'] = xy.tolist()
        outputs['actions'] = list(zip(outputs['moves'], outputs['place_pos']))
        input_path.unlink()
        output_path.unlink()
        return outputs
