import time
import numpy as np

from permutandis.utils.collisions import RadiusCollisionChecker
from permutandis.utils.errors import SamplerError

class StateSampler:
    def __init__(self, workspace, radius):
        self.collision_check = RadiusCollisionChecker(radius)
        self.workspace = workspace

    def sample_state(self, n_objects, max_time=20):
        state = []
        n_iter = 0
        start = time.time()
        while len(state) < n_objects:
            if time.time() - start > max_time:
                raise SamplerError('Cannot sample a collision-free configuration')
            pos = np.random.uniform(*self.workspace)
            if len(state) == 0 or not self.collision_check(np.asarray(state), pos).any():
                state.append(pos)
            n_iter += 1
        return np.asarray(state)

    def __call__(self, *args, **kwargs):
        return self.sample_state(*args, **kwargs)
