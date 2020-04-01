import numpy as np


class RadiusCollisionChecker:
    def __init__(self, radius):
        self.radius = radius

    def collisions(self, state, pos):
        return np.linalg.norm(state - pos, axis=1) < self.radius * 2

    def __call__(self, state, pos):
        return self.collisions(state, pos)


class WorkspaceChecker:
    def __init__(self, workspace):
        self.workspace = np.asarray(workspace)

    def __call__(self, state):
        return np.all(state >= self.workspace[0]) and np.all(state <= self.workspace[1])
