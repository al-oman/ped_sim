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


def our_baselines(map_root, ep_root, n_per_scene, max_scenes, repeats=1,
                  start_jitter=0.0, speed_jitter=0.0):
    scene_idx = dict(list(scene_index(map_root, ep_root).items())[:max_scenes])
    print(f"our 2D eval: {len(scene_idx)} minival scenes, up to {n_per_scene} episodes each, "
          f"x{repeats} stochastic repeats (start_jitter {start_jitter} m, speed_jitter {speed_jitter})")
    results = {}
    for name, factory in [("astar", AStarPolicy), ("orca", OrcaPolicy)]:
        rows = evaluate(factory, scene_idx, n_per_scene, collision_ends=True,  # Falcon semantics
                        min_humans=0,  # Falcon scores human-free episodes too
                        repeats=repeats, start_jitter=start_jitter, speed_jitter=speed_jitter)
        res = {}
        # mean +/- 95% CI (normal approx) per metric, for the write-up.
        # baselines have no coverage/tube (NaN), so filter before averaging.
        for k in rows[0]:
            vals = np.array([r[k] for r in rows], float)
            vals = vals[~np.isnan(vals)]
            res[k] = vals.mean() if len(vals) else np.nan
            res[k + "_ci"] = 1.96 * vals.std() / np.sqrt(len(vals)) if len(vals) else np.nan
        results[name] = res | {"n": len(rows)}
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
    # minival is only 10 distinct episodes and our sim is deterministic, so a
    # plain rerun repeats the same 10 outcomes. --repeats N re-rolls each episode
    # N times with jittered human starts/speeds (matching how habitat's stochastic
    # humans gave Falcon ~15 rollouts/episode) -> n = 10*N, with 95% CIs.
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--start-jitter", type=float, default=0.25, help="human start stddev, m")
    ap.add_argument("--speed-jitter", type=float, default=0.15, help="human speed +/- fraction")
    a = ap.parse_args()

    # jitter only bites when repeating; a single run stays deterministic
    sj, vj = (a.start_jitter, a.speed_jitter) if a.repeats > 1 else (0.0, 0.0)
    ours = our_baselines(a.map_root, a.ep_root, a.n_per_scene, a.max_scenes,
                         repeats=a.repeats, start_jitter=sj, speed_jitter=vj)
    falcon = {"astar": parse_falcon_log(a.falcon_astar) if a.falcon_astar else {},
              "orca": parse_falcon_log(a.falcon_orca) if a.falcon_orca else {}}

    print(f"\n(ours: mean +/- 95% CI over n={ours['astar']['n']} rollouts)")
    print(f"\n{'metric':16s} {'A* falcon':>10s} {'A* ours':>16s}   {'ORCA falcon':>11s} {'ORCA ours':>16s}")
    for fk, ok in ALIGN.items():
        fa = falcon["astar"].get(fk)
        fo = falcon["orca"].get(fk)
        print(f"{fk:16s} "
              f"{('%.3f' % fa) if fa is not None else '   n/a':>10s} "
              f"{ours['astar'][ok]:>8.3f}+/-{ours['astar'][ok + '_ci']:.3f}   "
              f"{('%.3f' % fo) if fo is not None else '     n/a':>11s} "
              f"{ours['orca'][ok]:>8.3f}+/-{ours['orca'][ok + '_ci']:.3f}")
    print("\nranking check: does ORCA beat A* on human_collision, in each column?")
