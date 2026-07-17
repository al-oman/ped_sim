"""HM3D counterpart of sequence.py (mirrors how diffuser/datasets keeps one
loader per environment family): expert trajectory windows collected from
Social-HM3D episodes in the 2D sim, with the expert being RvoPolicy — the
A*-waypoint-guided RVO2 ORCA baseline from hm3d_policies.py.

Conditioning differs from the toy in two ways (both robot-relative, so the
model transfers across scenes):
- goal-conditioning enters as the carrot vector — the pure-pursuit target on
  the A* path. That keeps the model local (it plans ~1.6 m) and pushes all
  navmesh/wall knowledge into the A* guidance outside the model.
- crowds vary in size, so the K nearest pedestrians are used, padded with a
  faraway dummy (PAD_DIST) when fewer are present.

hm3d_condition() is the shared definition: the dataset builds it at
collection time and the HM3D flow policy must build the identical vector at
planning time.
"""

import numpy as np
import torch

from .normalization import GaussianNormalizer, IsotropicNormalizer

K_PEDS = 4        # nearest pedestrians in the conditioning vector
PAD_DIST = 10.0   # relative position of a padded (absent) pedestrian slot, m
MAX_STEPS = 1000


def hm3d_condition(obs, path, K=K_PEDS):
    """Flat conditioning vector: carrot vector + K nearest pedestrians
    (relative position, velocity), padded to exactly K rows."""
    from hm3d import carrot
    c = carrot(path, obs["robot"]) - obs["robot"]
    rel = obs["peds"].copy() if len(obs["peds"]) else np.zeros((0, 4))
    if len(rel):
        rel[:, :2] -= obs["robot"]
        rel = rel[np.argsort(np.linalg.norm(rel[:, :2], axis=1))[:K]]
    pad = np.tile([PAD_DIST, PAD_DIST, 0.0, 0.0], (K - len(rel), 1))
    return np.concatenate([c, np.vstack([rel, pad]).ravel()]).astype(np.float32)


class HM3DSequenceDataset(torch.utils.data.Dataset):
    """Same windowing and normalization as the toy SequenceDataset: given the
    observation at step t, the target is the robot's next `horizon` positions
    relative to where it is now, channel-last (horizon, 2)."""

    def __init__(self, horizon=16, map_root="data",
                 ep_root="/Users/alexoman/workspaces/diffusion/socialnav_map_gen/"
                         "pointnav/social-hm3d/train",
                 n_scenes=30, eps_per_scene=10, dt=0.1, K=K_PEDS, seed=0):
        self.horizon = horizon
        trajs, conds = self._collect(horizon, map_root, ep_root, n_scenes,
                                     eps_per_scene, dt, K, seed)
        self.traj_normalizer = IsotropicNormalizer(trajs.reshape(-1, 2))
        self.cond_normalizer = GaussianNormalizer(conds)
        self.trajectories = torch.tensor(self.traj_normalizer.normalize(trajs))
        self.conditions = torch.tensor(self.cond_normalizer.normalize(conds))
        self.cond_dim = conds.shape[1]

    @staticmethod
    def _collect(horizon, map_root, ep_root, n_scenes, eps_per_scene, dt, K, seed):
        # Imported here: the HM3D modules live at the repo root, above the package.
        from hm3d import HM3DEnv, astar
        from hm3d_eval import scene_episodes, scene_index
        from hm3d_policies import RvoPolicy

        rng = np.random.default_rng(seed)
        idx = scene_index(map_root, ep_root)
        scenes = list(idx.items())
        rng.shuffle(scenes)
        trajs, conds, used = [], [], 0
        for scene, (maps, ep_files) in scenes[:n_scenes]:
            pairs = [(g, e) for g, e in scene_episodes(maps, ep_files)
                     if len(e["humans"]) > 0]
            if not pairs:
                continue
            take = rng.choice(len(pairs), min(eps_per_scene, len(pairs)), replace=False)
            experts = {}  # one expert per floor grid, as in hm3d_eval.evaluate
            for i in take:
                grid, ep = pairs[i]
                expert = experts.setdefault(id(grid), RvoPolicy(grid, dt=dt))
                env = HM3DEnv(grid, ep, dt=dt)
                obs = env.reset()
                if astar(grid, obs["robot"], obs["goal"]) is None:
                    continue  # unreachable goal (disconnected navmesh island)
                pos, cs, done, steps = [obs["robot"]], [], False, 0
                while not done and steps < MAX_STEPS:
                    a = expert.act(obs)              # also refreshes expert.path
                    cs.append(hm3d_condition(obs, expert.path, K))
                    obs, _, done = env.step(a)
                    pos.append(obs["robot"])
                    steps += 1
                    # Deadlock detector: Falcon's humans are robot-blind (they
                    # never yield), which breaks RVO2's reciprocity assumption —
                    # ~16% of episodes stall head-on forever, and their standing-
                    # still steps would swamp the dataset (median window
                    # displacement was 0 without this). 50 steps with < 0.1 m of
                    # progress ends the episode; ~20 stall steps are kept (so the
                    # model still learns to stop near people — trimming the whole
                    # stall taught it to plow through them), the endless tail is
                    # dropped.
                    if steps >= 50 and np.linalg.norm(pos[-1] - pos[-51]) < 0.1:
                        pos, cs = pos[:-30], cs[:-30]
                        break
                pos = np.array(pos)
                for t in range(len(pos) - horizon):  # one training window per step
                    trajs.append(pos[t + 1:t + 1 + horizon] - pos[t])
                    conds.append(cs[t])
                used += 1
        print(f"[ datasets/hm3d ] {used} episodes -> {len(trajs)} windows")
        return np.array(trajs, np.float32), np.array(conds, np.float32)

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, i):
        return self.trajectories[i], self.conditions[i]
