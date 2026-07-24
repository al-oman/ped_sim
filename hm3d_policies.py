"""Falcon's robot baselines, ported to the 2D sim as holonomic velocity
policies (they emit (vx, vz) instead of Falcon's discrete stop/turn/forward,
since the target platform is a quadruped — see the fidelity discussion).

Both read the same privileged state Falcon's baselines do: A* waypoints toward
the goal (their oracle_shortest_path_sensor) and ground-truth human positions
and velocities (their human_velocity_sensor). Same act(obs, disks=None)
interface as the toy policies, so the eval loop and calibrators are shared.
"""

import numpy as np
import rvo2

from hm3d import astar, carrot
from policies import FlowPolicy as ToyFlowPolicy
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


class FlowPolicy(ToyFlowPolicy):
    """The flow-matching policy on HM3D: identical receding-horizon planning
    and constrained sampling to the toy FlowPolicy, but conditioned the way
    the HM3D model was trained (diffuser/datasets/hm3d.py) — carrot vector on
    an A* path plus the K nearest pedestrians, either their current state or
    their predicted futures depending on the checkpoint's cond_mode. Plans
    once per episode goal, like AStarPolicy.

    predictor: for futures-conditioned checkpoints, the predictor whose
    forecasts feed the condition. Defaults to a fresh instance matching the
    checkpoint's cond_mode; pass the eval loop's instance to share state
    (mandatory for stateful predictors like Social-LSTM, so the condition
    and the ACI disks see the same forecast)."""

    def __init__(self, grid, path="hm3d_flow.pt", predictor=None, **kw):
        super().__init__(path=path, **kw)
        self.grid, self.goal, self.astar_path = grid, None, None
        self.cond_mode = self.ckpt.get("cond_mode", "state")
        self.predictor = predictor
        if self.predictor is None and self.cond_mode != "state":
            self.predictor = make_predictor(self.cond_mode, self.model.horizon)

    def _condition(self, obs):
        from diffuser.datasets import hm3d_condition
        if self.goal is None or not np.allclose(obs["goal"], self.goal):
            self.goal = obs["goal"].copy()
            self.astar_path = astar(self.grid, obs["robot"], obs["goal"])
        pred = self.predictor.predict(obs["peds"]) if self.predictor is not None else None
        return hm3d_condition(obs, self.astar_path if self.astar_path else [obs["goal"]],
                              pred=pred)


def make_predictor(cond_mode, horizon, dt=0.1):
    """The predictor a futures-conditioned checkpoint was trained with."""
    from predictor import ConstantVelocity, SocialLSTM
    name = cond_mode.removeprefix("futures-")
    if name == "ConstantVelocity":
        return ConstantVelocity(dt, horizon)
    if name == "SocialLSTM":
        return SocialLSTM(dt, horizon)  # loads social_lstm.pt
    raise ValueError(f"unknown cond_mode {cond_mode}")


class RvoPolicy(AStarPolicy):
    """Competent ORCA baseline: same A*+carrot goal-following as AStarPolicy,
    but the commanded velocity comes from a one-step RVO2 solve (robot plus
    every human at its current position/velocity) instead of Falcon's weak
    hand-rolled compute_orca_velocity. Walls are not in the RVO2 sim; the A*
    carrot keeps the robot on the navmesh and the env slides along walls."""

    def __init__(self, grid, speed=1.0, dt=0.1, radius=0.3):
        super().__init__(grid, speed)
        # radius 0.3 > env's 0.25 so ORCA holds a margin over the 0.5 m
        # collision distance instead of grazing it
        self.dt, self.radius = dt, radius

    def act(self, obs, disks=None):
        pref = self.speed * self._steer(obs)
        peds = obs["peds"]
        if not len(peds):
            return pref
        sim = rvo2.PyRVOSimulator(self.dt, 5.0, 10, 2.0, 2.0, self.radius, self.speed)
        robot = sim.addAgent(tuple(obs["robot"]))
        sim.setAgentPrefVelocity(robot, tuple(pref))
        for p in peds:
            i = sim.addAgent(tuple(p[:2]))
            sim.setAgentVelocity(i, tuple(p[2:]))
            sim.setAgentPrefVelocity(i, tuple(p[2:]))  # assume they keep going
        sim.doStep()
        return np.array(sim.getAgentVelocity(robot))


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
