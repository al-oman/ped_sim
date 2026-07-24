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


def hm3d_condition(obs, path, K=K_PEDS, pred=None):
    """Flat conditioning vector: carrot vector + the K nearest pedestrians,
    padded to exactly K rows. Two modes:

    - pred=None ("state"): each pedestrian contributes (relative position,
      velocity) — 4 numbers.
    - pred=(horizon, N, 2) ("futures"): each pedestrian contributes its whole
      predicted trajectory, robot-relative and raw-flattened — horizon x 2
      numbers. Same padding scheme (a faraway dummy), same K-nearest-by-
      current-distance selection, so the only change vs "state" is what a
      pedestrian row contains."""
    from hm3d import carrot
    c = carrot(path, obs["robot"]) - obs["robot"]
    peds = obs["peds"] if len(obs["peds"]) else np.zeros((0, 4))
    order = (np.argsort(np.linalg.norm(peds[:, :2] - obs["robot"], axis=1))[:K]
             if len(peds) else [])
    if pred is None:
        rel = peds[order].copy()
        if len(rel):
            rel[:, :2] -= obs["robot"]
        pad = np.tile([PAD_DIST, PAD_DIST, 0.0, 0.0], (K - len(rel), 1))
        return np.concatenate([c, np.vstack([rel, pad]).ravel()]).astype(np.float32)
    rows = [(pred[:, n] - obs["robot"]).ravel() for n in order]
    rows += [np.full(pred.shape[0] * 2, PAD_DIST)] * (K - len(rows))
    return np.concatenate([c, *rows]).astype(np.float32)


class HM3DSequenceDataset(torch.utils.data.Dataset):
    """Same windowing and normalization as the toy SequenceDataset: given the
    observation at step t, the target is the robot's next `horizon` positions
    relative to where it is now, channel-last (horizon, 2)."""

    def __init__(self, horizon=16, map_root="data",
                 ep_root="/Users/alexoman/workspaces/diffusion/socialnav_map_gen/"
                         "pointnav/social-hm3d/train",
                 n_scenes=30, eps_per_scene=10, dt=0.1, K=K_PEDS, seed=0,
                 predictor=None):
        # predictor: None -> "state" conditioning (pos, vel per pedestrian);
        # "cv" (or a predictor instance) -> "futures" conditioning, the
        # predictor's horizon x 2 forecast per pedestrian (see hm3d_condition).
        # The deployed policy must be conditioned by the same predictor.
        self.horizon = horizon
        if predictor == "cv":
            from predictor import ConstantVelocity
            predictor = ConstantVelocity(dt, horizon)
        elif predictor == "slstm":
            from predictor import SocialLSTM
            predictor = SocialLSTM(dt, horizon)  # loads social_lstm.pt
        self.cond_mode = "state" if predictor is None else \
            f"futures-{type(predictor).__name__}"
        trajs, conds = self._collect(horizon, map_root, ep_root, n_scenes,
                                     eps_per_scene, dt, K, seed, predictor)
        self.traj_normalizer = IsotropicNormalizer(trajs.reshape(-1, 2))
        self.cond_normalizer = GaussianNormalizer(conds)
        self.trajectories = torch.tensor(self.traj_normalizer.normalize(trajs))
        self.conditions = torch.tensor(self.cond_normalizer.normalize(conds))
        self.cond_dim = conds.shape[1]

    @staticmethod
    def _collect(horizon, map_root, ep_root, n_scenes, eps_per_scene, dt, K, seed,
                 predictor=None):
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
                if predictor is not None and hasattr(predictor, "reset"):
                    predictor.reset()                # stateful predictors are per-episode
                while not done and steps < MAX_STEPS:
                    a = expert.act(obs)              # also refreshes expert.path
                    # N=0 gives a (horizon, 0, 2) forecast: still futures mode,
                    # so the condition keeps its futures-sized padding.
                    pred = predictor.predict(obs["peds"]) if predictor is not None else None
                    cs.append(hm3d_condition(obs, expert.path, K, pred))
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
