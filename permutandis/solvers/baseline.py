import time
import numpy as np
from permutandis.utils.collisions import RadiusCollisionChecker
from permutandis.utils.errors import SamplerError


class BaselineSolver:
    def __init__(self, max_time=2.0):
        self.max_time = max_time

    def __call__(self, problem):
        try:
            outputs = self.solve_or_error(problem)
        except SamplerError:
            outputs = dict(solved=False,
                           time=np.nan,
                           n_collision_checks=np.nan,
                           actions=None)
        return outputs


    def solve_or_error(self, problem):
        self.n_collision_checks = 0
        self.workspace = problem.workspace
        self.radius = problem.radius
        self.collision_check = RadiusCollisionChecker(self.radius)
        src, tgt = problem.src, problem.tgt
        self.start = time.time()

        actions = []
        state = src.copy()
        at_target = np.full(len(src), False)

        # Iterates over all objects
        for i in range(len(src)):
            # If object is already at target, don't do anything
            if at_target[i]:
                continue

            # Try to place object i at it's target position
            tgt_pos = tgt[i]
            tgt_free = False
            while not tgt_free:
                ids = np.where(self.check(state, tgt_pos))[0]
                # If target position (of object i) is free or there is only object i
                # which is in collision with the target, the object can be moved to its target
                if len(ids) == 0 or (len(ids) == 1 and ids[0] == i):
                    tgt_free = True
                # Otherwise, we need to move objects that are obstructing the target of i
                else:
                    # Pick one object that is obstructing the target of i
                    k = ids[0]
                    # Check if this object can be moved to its target directly
                    ids_ = np.where(self.check(state, tgt[k]))[0]
                    if len(ids_) == 0 or (len(ids_) == 1 and ids_[0] == k):
                        state[k] = tgt[k]
                        actions.append((k, tgt[k]))
                        at_target[k] = True
                    # Move object k to random position in the workspace that does
                    # not overlap with the target of i
                    else:
                        state_with_tgt = state.copy()
                        state_with_tgt[k] = tgt_pos
                        pos = self.find_freespace(state_with_tgt)
                        state[k] = pos
                        actions.append((k, pos))
            # Target of i is free, move object i to its target
            state[i] = tgt_pos
            actions.append((i, tgt_pos))
            at_target[i] = True
        dt = time.time() - self.start
        outputs = dict(time=dt,
                       solved=True,
                       n_collision_checks=self.n_collision_checks,
                       actions=actions)
        return outputs

    def find_freespace(self, state):
        valid = False
        while not valid:
            if time.time() - self.start > self.max_time:
                raise SamplerError('Failed to move object to freespace')
            pos = np.random.uniform(*self.workspace)
            valid = not np.any(self.check(state, pos))
        return pos

    def check(self, *args, **kwargs):
        self.n_collision_checks += 1
        return self.collision_check(*args, **kwargs)
