"""Falcon's robot baselines, ported to the 2D sim as holonomic velocity
policies (they emit (vx, vz) instead of Falcon's discrete stop/turn/forward,
since the target platform is a quadruped — see the fidelity discussion).

Both read the same privileged state Falcon's baselines do: A* waypoints toward
the goal (their oracle_shortest_path_sensor) and ground-truth human positions
and velocities (their human_velocity_sensor). Same act(obs, disks=None)
interface as the toy policies, so the eval loop and calibrators are shared.
"""

import numpy as np

from hm3d import astar, carrot
from vendor_falcon import compute_orca_velocity


class AStarPolicy:
    """Falcon's A* baseline: follow the shortest path to the goal, ignore
    humans. Plans once per episode (replans when the goal changes)."""

    def __init__(self, grid, speed=1.0):
        self.grid, self.speed, self.goal, self.path = grid, speed, None, None

    def _steer(self, obs):
        if self.goal is None or not np.allclose(obs["goal"], self.goal):
            self.goal = obs["goal"].copy()
            self.path = astar(self.grid, obs["robot"], obs["goal"])
        if not self.path:
            return np.zeros(2)
        d = carrot(self.path, obs["robot"]) - obs["robot"]
        return d / (np.linalg.norm(d) + 1e-9)

    def act(self, obs, disks=None):
        return self.speed * self._steer(obs)


class OrcaPolicy(AStarPolicy):
    """Falcon's ORCA baseline: A* path-following plus local avoidance of any
    human within SAFE metres, blending the ORCA velocity's heading 0.8 with the
    goal heading 0.2 (Falcon's weight); stop if the ORCA velocity collapses."""

    SAFE = 2.0

    def act(self, obs, disks=None):
        goal_dir = self._steer(obs)
        peds = obs["peds"]
        if len(peds) and np.linalg.norm(peds[:, :2] - obs["robot"], axis=1).min() < self.SAFE:
            v = self.speed * goal_dir
            orca_v = compute_orca_velocity(obs["robot"], v, peds[:, :2], peds[:, 2:], self.speed)
            if np.linalg.norm(orca_v) < 0.1:
                return np.zeros(2)  # blocked: stop, as in Falcon
            orca_dir = orca_v / np.linalg.norm(orca_v)
            blended = 0.8 * orca_dir + 0.2 * goal_dir
            return self.speed * blended / (np.linalg.norm(blended) + 1e-9)
        return self.speed * goal_dir
