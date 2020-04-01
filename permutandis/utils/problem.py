import numpy as np

from .collisions import RadiusCollisionChecker, WorkspaceChecker

class RearrangementProblem:
    def __init__(self, src, tgt, workspace, radius):
        """
        Abstract class for 2D tabletop rearrangement planning problem with overhand grasps
        and simple collision models.

        src: Nx2 arrays of source positions, where N is the number of objects
        tgt: Nx2 arrays of target positions
        workspace: 2x2 array [[x1, y1], [x2, y2]] representing limits of the workspace
        radius: radius around each object to consider for collision checks
        """
        src = np.asarray(src)
        tgt = np.asarray(tgt)
        workspace = np.asarray(workspace)
        N = src.shape[0]
        assert src.shape == (N, 2)
        assert tgt.shape == (N, 2)
        assert workspace.shape == (2, 2)

        self.src = src
        self.tgt = tgt
        self.workspace = workspace
        self.radius = radius

    def assert_solution_valid(self, actions):
        cur = self.src.copy()
        tgt = self.tgt
        collisions = RadiusCollisionChecker(radius=self.radius)

        def collision_check(s):
            for n, x in enumerate(s):
                coll_ = collisions(s, x)
                coll_[n] = False
                if np.any(coll_):
                    return False
            return True

        workspace_check = WorkspaceChecker(workspace=self.workspace)
        assert collision_check(cur) and collision_check(tgt)
        assert workspace_check(cur) and workspace_check(tgt)
        for (object_id, place_pos) in actions:
            assert workspace_check(place_pos)
            cur[object_id] = place_pos
            assert collision_check(cur)
        d = np.linalg.norm(cur - tgt, axis=-1)
        cond = d <= 1e-3
        assert np.all(cond), f'Some objects are not at target.'
        return
