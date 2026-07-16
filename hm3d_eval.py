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


def run_episode(env, policy, grid):
    obs = env.reset()
    optimal = astar(grid, obs["robot"], obs["goal"])
    if optimal is None:
        return None  # unreachable goal (disconnected navmesh island)
    opt_len = path_length(optimal)
    done, steps, collisions, closest, travelled, ever_collided = False, 0, 0, np.inf, 0.0, False
    prev = obs["robot"].copy()
    while not done and steps < MAX_STEPS:
        obs, reward, done = env.step(policy.act(obs))
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
            "closest": closest if np.isfinite(closest) else np.nan, "spl": spl}


def evaluate(policy_factory, scene_idx, n_per_scene, seed=0, collision_ends=False,
             min_humans=1):  # min_humans=0: keep human-free episodes (Falcon scores them)
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
        for group in by_grid.values():
            policy = policy_factory(group[0][0])
            for grid, ep in group:
                r = run_episode(HM3DEnv(grid, ep, dt=DT, collision_ends=collision_ends), policy, grid)
                if r:
                    rows.append(r)
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

    from hm3d_policies import AStarPolicy, OrcaPolicy, RvoPolicy

    map_root = sys.argv[1] if len(sys.argv) > 1 else "data"
    ep_root = sys.argv[2] if len(sys.argv) > 2 else \
        "/Users/alexoman/workspaces/diffusion/socialnav_map_gen/pointnav"
    n_per_scene = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    max_scenes = int(sys.argv[4]) if len(sys.argv) > 4 else 5

    scene_idx = dict(list(scene_index(map_root, ep_root).items())[:max_scenes])
    print(f"{len(scene_idx)} scenes, up to {n_per_scene} episodes each")
    for name, factory in [("astar", AStarPolicy), ("orca", OrcaPolicy), ("rvo", RvoPolicy)]:
        t0 = time.time()
        rows = evaluate(factory, scene_idx, n_per_scene)
        summary(name, rows)
        print(f"           [{time.time() - t0:.0f}s]")
