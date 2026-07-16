"""Fidelity check: our 2D-holonomic baselines vs Falcon's native habitat-sim
baselines, on the same Social-HM3D minival split.

Falcon side (run on the cluster, where habitat-sim works) — point the existing
runners at minival and let them log to eval.log:

    cd habitat-baselines/habitat_baselines/rl/ddppo
    # edit *_hm3d.sh (or override on the CLI) to use the minival dataset:
    #   habitat/dataset/social_nav_v2=social-hm3d-minival
    bash single_node_astar_hm3d.sh   # -> evaluation/astar/hm3d/eval.log
    bash single_node_orca_hm3d.sh    # -> evaluation/orca/hm3d/eval.log

Then here (runs anywhere, no habitat): parse those logs and run our 2D
baselines on the same minival scenes, side by side.

    python fidelity_check.py <map_root> <minival_episode_root> \\
        [--falcon-astar path/to/astar_eval.log] [--falcon-orca path/to/orca_eval.log]

Our env is set collision_ends=True to match Falcon's human_collision semantics
(the episode ends, as a failure, on first contact). The comparison is
expected to be LOOSE — Falcon's robot is discrete (turn/forward), ours is
holonomic — so read it as "does the A*/ORCA ranking hold", not decimal match.
"""

import re
import sys

import numpy as np

from hm3d_eval import evaluate, scene_index
from hm3d_policies import AStarPolicy, OrcaPolicy

# Falcon's logged metric name -> our run_episode key (None: no clean analogue)
ALIGN = {"success": "success", "human_collision": "human_collision", "spl": "spl"}


def parse_falcon_log(path):
    """Pull 'Average episode <k>: <v>' lines from a Falcon eval.log."""
    out = {}
    with open(path) as f:
        for line in f:
            m = re.search(r"Average episode (\w+):\s*([-\d.]+)", line)
            if m:
                out[m.group(1)] = float(m.group(2))
    return out


def our_baselines(map_root, ep_root, n_per_scene, max_scenes):
    scene_idx = dict(list(scene_index(map_root, ep_root).items())[:max_scenes])
    print(f"our 2D eval: {len(scene_idx)} minival scenes, up to {n_per_scene} episodes each")
    results = {}
    for name, factory in [("astar", AStarPolicy), ("orca", OrcaPolicy)]:
        rows = evaluate(factory, scene_idx, n_per_scene, collision_ends=True,  # Falcon semantics
                        min_humans=0)  # Falcon scores human-free episodes too
        results[name] = {k: np.nanmean([r[k] for r in rows]) for k in rows[0]} | {"n": len(rows)}
    return results


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("map_root", nargs="?", default="data")
    ap.add_argument("ep_root", nargs="?",
                    default="/Users/alexoman/workspaces/diffusion/socialnav_map_gen/"
                            "pointnav/social-hm3d/minival_falcon")  # exact rsync of the
                            # cluster set Falcon evaluated (local copy had 1 file gunzipped)
    ap.add_argument("n_per_scene", nargs="?", type=int, default=25)
    ap.add_argument("max_scenes", nargs="?", type=int, default=20)
    ap.add_argument("--falcon-astar")
    ap.add_argument("--falcon-orca")
    a = ap.parse_args()

    ours = our_baselines(a.map_root, a.ep_root, a.n_per_scene, a.max_scenes)
    falcon = {"astar": parse_falcon_log(a.falcon_astar) if a.falcon_astar else {},
              "orca": parse_falcon_log(a.falcon_orca) if a.falcon_orca else {}}

    print(f"\n{'metric':16s} {'A* falcon':>10s} {'A* ours':>9s}   {'ORCA falcon':>11s} {'ORCA ours':>9s}")
    for fk, ok in ALIGN.items():
        fa = falcon["astar"].get(fk)
        fo = falcon["orca"].get(fk)
        print(f"{fk:16s} "
              f"{('%.3f' % fa) if fa is not None else '   n/a':>10s} "
              f"{ours['astar'][ok]:>9.3f}   "
              f"{('%.3f' % fo) if fo is not None else '     n/a':>11s} "
              f"{ours['orca'][ok]:>9.3f}")
    print("\nranking check: does ORCA beat A* on human_collision, in each column?")
