"""Robot control policies. A policy is anything with an
act(obs, disks=None) -> action method, where action is the robot's commanded
(vx, vy) in m/s. disks is an optional pair (centers (K, N, 2), radii (K, N)):
predicted pedestrian positions 1..K steps ahead with calibrated uncertainty
radii, which safety-aware policies may treat as keep-out regions. Policies
that don't use them just ignore the argument."""

import numpy as np
import rvo2


class WalkForward:
    """Walks straight to the right at constant speed, ignoring everything."""

    def __init__(self, speed=1.0):
        self.speed = speed

    def act(self, obs, disks=None):
        return np.array([self.speed, 0.0])


class FlowPolicy:
    """Steers the robot with the trained flow matching model: sample a
    TRAJ_LEN-step plan and follow it for replan_every steps before sampling a
    fresh one (resampling every step makes the robot jitter between the
    model's competing plans). The latest plan is kept in .plan as absolute
    positions, for rendering.

    Each replan draws n_samples cold candidates plus n_samples warm-started
    ones (seeded from the previous plan, noised to flow time tau) and keeps
    the cheapest by: plan jerkiness + disagreement with the previous plan
    + a large penalty per keep-out-disk violation.

    If disks are passed to act(), sampling is constrained: each disk is
    inflated by `margin` (the two body radii) and generated waypoints are
    kept out of it — see sample() in flow_model.py."""

    def __init__(self, path="flow.pt", dt=0.1, replan_every=4, margin=0.5,
                 n_samples=4, tau=0.6, stale=0.5, device=None):
        import torch
        from flow_model import UNet1D
        ckpt = torch.load(path, weights_only=False)
        # Default to CPU: this UNet is small enough that MPS dispatch overhead
        # makes it ~4x slower there (measured). Pass device="mps" if it grows.
        self.device = device or "cpu"
        self.model = UNet1D(ckpt["cond_dim"])
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()
        self.model.to(self.device)
        self.ckpt, self.dt, self.replan_every, self.margin = ckpt, dt, replan_every, margin
        self.n_samples, self.tau, self.stale = n_samples, tau, stale
        self.step = 0

    def act(self, obs, disks=None):
        import torch
        from flow_model import TRAJ_LEN, condition, sample
        # Resample on schedule, or if the plan no longer matches reality
        # (e.g., the env was reset under us, or the robot got blocked).
        if (self.step % self.replan_every == 0
                or np.linalg.norm(self.plan[self.step] - obs["robot"]) > self.stale):
            std = self.ckpt["traj_std"]
            c = (condition(obs) - self.ckpt["cond_mean"]) / self.ckpt["cond_std"]
            cond = torch.tensor(c, dtype=torch.float32, device=self.device).repeat(self.n_samples, 1)
            model_disks = None
            if disks is not None:
                centers, conformal = disks
                # Inflate by both body radii; disable uncalibrated disks (inf).
                R = np.where(np.isfinite(conformal), conformal + self.margin, -1.0)
                # Map into the model's normalized, robot-relative coordinates.
                model_disks = ((centers - obs["robot"]) / std, R / std)

            # Candidates: cold starts, plus warm starts seeded from what's
            # left of the previous plan (extended by its last displacement).
            cands = [sample(self.model, cond, disks=model_disks, hard=self.replan_every)]
            old = getattr(self, "plan", None)
            if old is not None:
                left = old[self.step:]
                pad = left[-1] + (left[-1] - left[-2]) * np.arange(1, TRAJ_LEN - len(left) + 1)[:, None]
                init = (np.vstack([left, pad]) - obs["robot"]) / std
                cands.append(sample(self.model, cond, disks=model_disks, hard=self.replan_every,
                                    init=torch.tensor(init.T, dtype=torch.float32,
                                                      device=self.device)[None],
                                    tau=self.tau))
            trajs = torch.cat(cands).cpu().numpy() * std    # (M, 2, TRAJ_LEN)
            plans = obs["robot"] + trajs.transpose(0, 2, 1)  # (M, TRAJ_LEN, 2)

            # Score and pick: smooth, consistent with the old plan, safe.
            cost = np.abs(np.diff(plans, 2, axis=1)).sum(axis=(1, 2))
            if old is not None:
                m = len(old) - self.step
                cost += np.linalg.norm(plans[:, :m] - old[self.step:], axis=2).mean(axis=1)
            if disks is not None:
                K = min(len(centers), self.replan_every)
                dist = np.linalg.norm(plans[:, :K, None] - centers[None, :K], axis=3)
                cost += 1e3 * (dist < R[None, :K] - 1e-3).sum(axis=(1, 2))
            self.plan = plans[cost.argmin()]
            self.step = 0
            # Waypoints whose disks overlap into an unavoidable union can't be
            # made safe by any projection — mark them infeasible.
            self.infeasible = np.zeros(len(self.plan), bool)
            if disks is not None:
                dist = np.linalg.norm(self.plan[:K, None] - centers[:K], axis=2)
                self.infeasible[:K] = (dist < R[:K] - 1e-3).any(axis=1)
        if self.infeasible[self.step]:
            self.step = 0     # brake, and replan with fresh disks next step
            return np.zeros(2)
        target = self.plan[self.step]  # next waypoint of the committed plan
        self.step += 1
        return (target - obs["robot"]) / self.dt


class OrcaExpert:
    """Steers the robot with ORCA: head for the goal, avoid the pedestrians.
    Builds a tiny one-step RVO2 sim from the observation each time — the robot
    plus every pedestrian at its current position and velocity — and returns
    the collision-free velocity ORCA picks for the robot."""

    def __init__(self, goal, dt, radius=0.4, speed=1.0, walls=()):
        self.goal, self.dt, self.radius, self.speed = np.array(goal), dt, radius, speed
        self.walls = walls

    def act(self, obs, disks=None):
        sim = rvo2.PyRVOSimulator(self.dt, 5.0, 10, 2.0, 2.0, self.radius, self.speed)
        for x0, x1, y0, y1 in self.walls:
            sim.addObstacle([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
        sim.processObstacles()
        robot = sim.addAgent(tuple(obs["robot"]))
        to_goal = self.goal - obs["robot"]
        pref = self.speed * to_goal / max(np.linalg.norm(to_goal), 1e-6)
        sim.setAgentPrefVelocity(robot, tuple(pref))
        for p in obs["peds"]:
            i = sim.addAgent(tuple(p[:2]))
            sim.setAgentVelocity(i, tuple(p[2:]))
            sim.setAgentPrefVelocity(i, tuple(p[2:]))  # assume they keep going
        sim.doStep()
        return np.array(sim.getAgentVelocity(robot))
