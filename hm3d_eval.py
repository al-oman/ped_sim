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


def matched_scenes(map_root, ep_root):
    """(scene, map_path, ep_path) for scenes with both a human map and episodes."""
    maps = {os.path.basename(p).split("_human_floor0")[0]: p
            for p in glob.glob(f"{map_root}/**/*_human_floor0.npz", recursive=True)}
    out = []
    for ep in glob.glob(f"{ep_root}/**/*.json.gz", recursive=True):
        scene = os.path.basename(ep).split(".")[0].split("_ep")[0]
        if scene in maps:
            out.append((scene, maps[scene], ep))
    return sorted(set(out))


def path_length(pts):
    return sum(np.linalg.norm(b - a) for a, b in zip(pts[:-1], pts[1:]))


def run_episode(env, policy, grid):
    obs = env.reset()
    optimal = astar(grid, obs["robot"], obs["goal"])
    if optimal is None:
        return None  # unreachable goal (disconnected navmesh island)
    opt_len = path_length(optimal)
    done, steps, collisions, closest, travelled = False, 0, 0, np.inf, 0.0
    prev = obs["robot"].copy()
    while not done and steps < MAX_STEPS:
        obs, reward, done = env.step(policy.act(obs))
        collisions += reward < 0
        if len(obs["peds"]):
            closest = min(closest, np.linalg.norm(obs["peds"][:, :2] - obs["robot"], axis=1).min())
        travelled += np.linalg.norm(obs["robot"] - prev)
        prev = obs["robot"].copy()
        steps += 1
    spl = done * opt_len / max(travelled, opt_len, 1e-6)
    return {"success": done, "steps": steps, "collisions": collisions,
            "closest": closest if np.isfinite(closest) else np.nan, "spl": spl}


def evaluate(policy_factory, scenes, n_per_scene):
    rows = []
    for scene, map_path, ep_path in scenes:
        grid = GridMap(map_path)
        eps = [e for e in load_episodes(ep_path)
               if abs(e["height"] - grid.height) < 0.5 and len(e["humans"]) > 0]
        policy = policy_factory(grid)
        for ep in eps[:n_per_scene]:
            r = run_episode(HM3DEnv(grid, ep, dt=DT), policy, grid)
            if r:
                rows.append(r)
    return rows


def summary(name, rows):
    def m(k):
        return np.nanmean([r[k] for r in rows])
    ok = [r for r in rows if r["success"]] or rows
    print(f"{name:10s}: SR {m('success'):.0%}, SPL {m('spl'):.2f}, "
          f"steps {np.mean([r['steps'] for r in ok]):.0f}, "
          f"collisions {m('collisions'):.2f}, closest {m('closest'):.2f} m  (n={len(rows)})")


if __name__ == "__main__":
    import sys

    from hm3d_policies import AStarPolicy, OrcaPolicy

    map_root = sys.argv[1] if len(sys.argv) > 1 else "data"
    ep_root = sys.argv[2] if len(sys.argv) > 2 else \
        "/Users/alexoman/workspaces/diffusion/socialnav_map_gen/pointnav"
    n_per_scene = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    max_scenes = int(sys.argv[4]) if len(sys.argv) > 4 else 5

    scenes = matched_scenes(map_root, ep_root)[:max_scenes]
    print(f"{len(scenes)} scenes, {n_per_scene} episodes each")
    for name, factory in [("astar", AStarPolicy), ("orca", OrcaPolicy)]:
        t0 = time.time()
        rows = evaluate(factory, scenes, n_per_scene)
        summary(name, rows)
        print(f"           [{time.time() - t0:.0f}s]")
