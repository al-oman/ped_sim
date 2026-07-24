"""Evaluate a policy over Social-HM3D/MP3D episodes in the 2D sim.

Iterates matched (map, episodes) scene pairs, runs N episodes each, and
reports the benchmark-style metrics: success rate, SPL, steps, collisions,
closest human approach. Baselines and the calibrated flow policy all go
through the same loop.

    python hm3d_eval.py <map_root> <episode_root> [n_per_scene] [max_scenes]
"""

import glob
import os
import time

import numpy as np

from hm3d import GridMap, HM3DEnv, astar, load_episodes

MAX_STEPS = 1000
DT = 0.1
HORIZON = 16  # prediction horizon = flow model TRAJ_LEN


def scene_index(map_root, ep_root):
    """scene -> (all floor-map paths, all episode-file paths), for scenes that
    have both. Minival stores one episode per file, so a scene can have many."""
    from collections import defaultdict
    maps, eps = defaultdict(list), defaultdict(list)
    for p in glob.glob(f"{map_root}/**/*_human_floor*.npz", recursive=True):
        maps[os.path.basename(p).split("_human_floor")[0]].append(p)
    for p in glob.glob(f"{ep_root}/**/*.json.gz", recursive=True):
        eps[os.path.basename(p).split(".")[0].split("_ep")[0]].append(p)
    return {s: (sorted(maps[s]), sorted(eps[s])) for s in sorted(maps) if s in eps}


def scene_episodes(map_paths, ep_paths):
    """Load every floor's grid and every episode, routing each episode to the
    nearest-height grid (scenes are multi-storey). Returns [(grid, episode)]."""
    grids = [GridMap(m) for m in map_paths]
    out = []
    for ep_file in ep_paths:
        for e in load_episodes(ep_file):
            g = min(grids, key=lambda g: abs(g.height - e["height"]))
            if abs(g.height - e["height"]) < 0.5:
                out.append((g, e))
    return out


def path_length(pts):
    return sum(np.linalg.norm(b - a) for a, b in zip(pts[:-1], pts[1:]))


def warm_calibrator(calib, predictor, grid, ep, steps=150, reactive=0.0):
    """Fill a fresh calibrator's score buffers before the scored episode, by
    rolling the episode's humans forward with the robot parked — they are
    robot-blind, so this is the same crowd the robot will face. Without this
    the disks are cold (tiny, noisy) exactly when the robot meets its first
    human, ~50 steps in. (With reactive>0 the parked robot still slightly
    perturbs nearby humans — acceptable: warmup is just a prior.)"""
    env = HM3DEnv(grid, ep, dt=DT, reactive=reactive)
    obs = env.reset()
    if hasattr(predictor, "reset"):
        predictor.reset()  # replay starts the episode's humans from frame 0
    for _ in range(steps):
        if not len(obs["peds"]):
            return
        pred = predictor.predict(obs["peds"])
        obs, _, _ = env.step(np.zeros(2))
        calib.update(pred, obs["peds"][:, :2])


def run_episode(env, policy, grid, predictor=None, calib=None):
    """One episode. If a calibrator is given, the policy also receives
    conformal keep-out disks (prediction, radii) each step, exactly as in the
    toy eval loop — the calibrator is fresh per episode (crowd size varies),
    so it warms up within the episode (radii start infinite = disabled)."""
    obs = env.reset()
    # Stateful predictors (SocialLSTM) carry per-episode history: clear it,
    # both on the disk predictor and on the policy's conditioning predictor
    # (often the same object — sharing is what keeps disks and conditioning
    # looking at the same forecast).
    for p in {id(p): p for p in (predictor, getattr(policy, "predictor", None))
              if hasattr(p, "reset")}.values():
        p.reset()
    optimal = astar(grid, obs["robot"], obs["goal"])
    if optimal is None:
        return None  # unreachable goal (disconnected navmesh island)
    opt_len = path_length(optimal)
    done, steps, collisions, closest, travelled, ever_collided = False, 0, 0, np.inf, 0.0, False
    miss, miss_n, tube_miss, tube_n, pending = 0.0, 0, 0.0, 0, []
    prev = obs["robot"].copy()
    while not done and steps < MAX_STEPS:
        disks = None
        if calib is not None and len(obs["peds"]):
            pred = predictor.predict(obs["peds"])
            disks = (pred, calib.radii())
        obs, reward, done = env.step(policy.act(obs, disks))
        if disks is not None:
            missed = calib.update(pred, obs["peds"][:, :2])
            # Realized coverage, as in the toy eval.py: certified disks
            # (k <= replan_every) per step, and whole-tube outcomes (a tube
            # fails if ANY of its HORIZON lookaheads missed).
            miss += missed[:getattr(policy, "replan_every", len(missed))].mean()
            miss_n += 1
            pending.insert(0, np.zeros((HORIZON, missed.shape[1]), bool))
            for j in range(len(missed)):
                pending[j][j] = missed[j]
            if len(pending) == HORIZON:
                tube_miss += pending.pop().any(axis=0).mean()
                tube_n += 1
        collisions += reward < 0
        ever_collided |= reward < 0
        if len(obs["peds"]):
            closest = min(closest, np.linalg.norm(obs["peds"][:, :2] - obs["robot"], axis=1).min())
        travelled += np.linalg.norm(obs["robot"] - prev)
        prev = obs["robot"].copy()
        steps += 1
    success = bool(env.reached)  # reached goal (not just "done", which may be a collision end)
    spl = success * opt_len / max(travelled, opt_len, 1e-6)
    return {"success": success, "human_collision": float(ever_collided),
            "steps": steps, "collisions": collisions,
            "closest": closest if np.isfinite(closest) else np.nan, "spl": spl,
            "coverage": 1 - miss / miss_n if miss_n else np.nan,
            "tube": 1 - tube_miss / tube_n if tube_n else np.nan}


def evaluate(policy_factory, scene_idx, n_per_scene, seed=0, collision_ends=False,
             min_humans=1,  # min_humans=0: keep human-free episodes (Falcon scores them)
             calibrator=None,  # aci.CALIBRATORS name -> conformal disks reach the policy
             reactive=0.0,  # crowd's robot-avoidance strength (0 = Falcon, Experiment 2)
             human_speed=1.0,   # deployment crowd speed; warmup always calibrates at
             gamma=0.01,        # 1.0, so human_speed != 1.0 is a calibration/deployment
             freeze_after_warm=False,  # shift. freeze -> split CP (radii never adapt)
             repeats=1, start_jitter=0.0, speed_jitter=0.0):  # per-rollout human
             # stochasticity: run each episode `repeats` times with jittered human
             # starts/speeds (rng seeded per repeat). Mirrors how habitat's stochastic
             # humans gave Falcon ~15 rollouts per episode. repeats=1 => deterministic.
    from collections import defaultdict
    rng = np.random.default_rng(seed)
    rows = []
    for scene, (maps, ep_files) in scene_idx.items():
        pairs = [(g, e) for g, e in scene_episodes(maps, ep_files) if len(e["humans"]) >= min_humans]
        if not pairs:
            continue
        idx = rng.choice(len(pairs), min(n_per_scene, len(pairs)), replace=False)  # random, not first-N
        by_grid = defaultdict(list)  # a scene spans floors; one policy per floor grid
        for i in idx:
            by_grid[id(pairs[i][0])].append(pairs[i])
        ep_i = 0
        for group in by_grid.values():
            policy = policy_factory(group[0][0])
            for grid, ep in group:
                # Seed the flow sampler per episode so the whole table is
                # reproducible and every policy meets identical noise on its
                # k-th episode (flow sampling is otherwise unseeded -> ~+-4pt
                # run-to-run SR variance at n~70). No-op for the classical
                # baselines, which draw no noise.
                for rep in range(repeats):
                    import torch
                    # seed the flow sampler per (episode, repeat) so the run is
                    # reproducible and every policy meets identical model noise
                    torch.manual_seed(1000 * seed + 97 * ep_i + rep)
                    # per-repeat human jitter (off when start/speed_jitter == 0)
                    ep_rng = np.random.default_rng(1000 * seed + 97 * ep_i + rep) \
                        if (start_jitter or speed_jitter) else None
                    predictor, calib = None, None
                    if calibrator is not None and len(ep["humans"]):
                        from aci import make_calibrator
                        from predictor import ConstantVelocity
                        # share the policy's conditioning predictor when it has one,
                        # so the ACI disks calibrate the same forecast the flow
                        # model is conditioned on
                        predictor = getattr(policy, "predictor", None) or \
                            ConstantVelocity(DT, HORIZON)
                        # split CP needs every calibration score kept (big window)
                        calib = make_calibrator(calibrator, alpha=0.1, horizon=HORIZON,
                                                n_peds=len(ep["humans"]), gamma=gamma,
                                                window=10 ** 6 if freeze_after_warm else 100)
                        warm_calibrator(calib, predictor, grid, ep, reactive=reactive)
                        calib.past = []  # pending predictions don't survive the reset
                        if freeze_after_warm:
                            calib.freeze()
                    r = run_episode(HM3DEnv(grid, ep, dt=DT, collision_ends=collision_ends,
                                            reactive=reactive, human_speed=human_speed,
                                            rng=ep_rng, start_jitter=start_jitter,
                                            speed_jitter=speed_jitter),
                                    policy, grid, predictor, calib)
                    if r:
                        rows.append(r)
                ep_i += 1
    return rows


def summary(name, rows):
    def m(k):
        return np.nanmean([r[k] for r in rows])
    ok = [r for r in rows if r["success"]] or rows
    coll_rate = np.mean([r["collisions"] > 0 for r in rows])   # episodes with any collision
    med_closest = np.nanmedian([r["closest"] for r in rows])   # median, not mean (bimodal)
    print(f"{name:10s}: SR {m('success'):.0%}, SPL {m('spl'):.2f}, "
          f"steps {np.mean([r['steps'] for r in ok]):.0f}, "
          f"coll-rate {coll_rate:.0%}, coll-steps {m('collisions'):.2f}, "
          f"closest med {med_closest:.2f} m  (n={len(rows)})")


if __name__ == "__main__":
    import sys

    from aci import CALIBRATORS
    from hm3d_policies import AStarPolicy, FlowPolicy, OrcaPolicy, RvoPolicy

    # --cal=aci|dtaci|max|max+ picks the calibrator for the "flow + <cal>" row
    cal = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--cal=")), "aci")
    assert cal in CALIBRATORS, f"--cal must be one of {list(CALIBRATORS)}"
    sys.argv = [a for a in sys.argv if not a.startswith("--cal=")]

    map_root = sys.argv[1] if len(sys.argv) > 1 else "data"
    ep_root = sys.argv[2] if len(sys.argv) > 2 else \
        "/Users/alexoman/workspaces/diffusion/socialnav_map_gen/pointnav"
    n_per_scene = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    max_scenes = int(sys.argv[4]) if len(sys.argv) > 4 else 5

    scene_idx = dict(list(scene_index(map_root, ep_root).items())[:max_scenes])
    print(f"{len(scene_idx)} scenes, up to {n_per_scene} episodes each")
    setups = [("astar", AStarPolicy, None), ("orca", OrcaPolicy, None),
              ("rvo", RvoPolicy, None)]
    if os.path.exists("hm3d_flow.pt"):  # trained HM3D flow model available
        setups += [("flow", FlowPolicy, None), (f"flow+{cal}", FlowPolicy, cal)]
    for name, factory, calibrator in setups:
        t0 = time.time()
        rows = evaluate(factory, scene_idx, n_per_scene, calibrator=calibrator)
        summary(name, rows)
        print(f"           [{time.time() - t0:.0f}s]")
