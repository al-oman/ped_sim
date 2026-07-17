"""Mirrors diffuser/datasets/sequence.py: a torch Dataset of expert
trajectory windows, normalized and ready for CFM.loss. Their loaders wrap
d4rl environments; ours collects episodes from the toy sim steered by the
ORCA expert. Each item is (trajectory, condition) — their
Batch(trajectories, conditions), except the condition is a flat vector for
the embedding side-door rather than an inpainting dict (see models/cfm.py).
"""

import numpy as np
import torch

from .normalization import GaussianNormalizer, IsotropicNormalizer


def condition(obs):
    """Flat conditioning vector: robot position + pedestrians relative to it."""
    peds = obs["peds"].copy()
    peds[:, :2] -= obs["robot"]
    return np.concatenate([obs["robot"], peds.ravel()])


class SequenceDataset(torch.utils.data.Dataset):
    """Collects episodes where the robot is steered by ORCA (the expert) and
    windows them for imitation: given the observation at step t, the target
    is the robot's next `horizon` positions relative to where it is now,
    channel-last (horizon, 2)."""

    def __init__(self, horizon=16, episodes=500, expert_radius=0.25, seed=999):
        self.horizon = horizon
        trajs, conds = self._collect(horizon, episodes, expert_radius, seed)
        self.traj_normalizer = IsotropicNormalizer(trajs.reshape(-1, 2))
        self.cond_normalizer = GaussianNormalizer(conds)
        self.trajectories = torch.tensor(self.traj_normalizer.normalize(trajs))
        self.conditions = torch.tensor(self.cond_normalizer.normalize(conds))
        self.cond_dim = conds.shape[1]

    @staticmethod
    def _collect(horizon, episodes, expert_radius, seed):
        # Imported here: policies/sim live at the repo root, above the package.
        from policies import OrcaExpert
        from sim import DT, HEIGHT, WALLS, WIDTH, Env, in_wall

        def random_start(rng):
            """Random wall-free start, so the data covers recovery situations."""
            while True:
                p = rng.uniform([0.5, 0.5], [WIDTH - 1.5, HEIGHT - 0.5])
                if not in_wall(p):
                    return p

        trajs, conds = [], []
        rng = np.random.default_rng(seed)
        for ep in range(episodes):
            env = Env(seed=ep)
            expert = OrcaExpert(goal=(WIDTH - 0.3, HEIGHT / 2), dt=DT, walls=WALLS,
                                radius=expert_radius)
            # Half the episodes start from the usual spot, half from anywhere.
            obs = env.reset() if ep % 2 == 0 else env.reset(robot_start=random_start(rng))
            pos, cs, done = [obs["robot"]], [condition(obs)], False
            while not done:
                obs, _, done = env.step(expert.act(obs))
                pos.append(obs["robot"])
                cs.append(condition(obs))
            pos = np.array(pos)
            for t in range(len(pos) - horizon):  # one training window per step
                trajs.append(pos[t + 1:t + 1 + horizon] - pos[t])  # (horizon, 2)
                conds.append(cs[t])
        return np.array(trajs, np.float32), np.array(conds, np.float32)

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, i):
        return self.trajectories[i], self.conditions[i]
