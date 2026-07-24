"""Experiment 1: the full calibrator comparison on Social-HM3D.

Falcon's robot-blind crowd (reactive=0). Every policy on the same held-out
scenes, same per-episode flow seeds (so flow rows are reproducible and each
calibrator meets identical model samples): the two classical baselines
(A*, RVO2), and — for BOTH flow models (state-conditioned hm3d_flow.pt and
Social-LSTM-futures hm3d_flow_slstm.pt) — raw flow plus flow with disks
calibrated by each of the five calibrators in aci.CALIBRATORS.

    union bound : aci, dtaci        (dtaci self-tunes gamma)
    max-over-h  : max, max+, maxdt+ (+ removes feedback lag; maxdt+ also self-tunes)

Held-out scenes (shuffled index >= 120; the models trained on 0..119), Falcon
collision_ends semantics, n = N_SCENES x N_PER_SCENE episodes/policy. Appends
one row per policy to the CSV, resumably (safe to interrupt/rerun).

Run: python hm3d_exp1.py [--models=classical,state,slstm] [--out=hm3d_exp1.csv]
The --models/--out flags let the slow slstm rows run as a separate parallel
process into their own CSV (concatenate afterwards; the header matches).
"""

import csv
import os
import sys
import time

import numpy as np

from hm3d_eval import evaluate, scene_index, summary
from hm3d_policies import AStarPolicy, FlowPolicy, RvoPolicy

N_SCENES = 40        # held-out scenes[120:120+N_SCENES]
N_PER_SCENE = 5      # episodes per scene (-> ~200 episodes/policy)

MODELS = {"state": "hm3d_flow.pt", "slstm": "hm3d_flow_slstm.pt"}
CALS = [None, "aci", "dtaci", "max", "max+", "maxdt+"]


def setups(groups):
    out = []
    if "classical" in groups:
        out += [("astar", "none", AStarPolicy, None), ("rvo", "none", RvoPolicy, None)]
    for tag, path in MODELS.items():
        if tag not in groups:
            continue
        for cal in CALS:
            name = f"flow[{tag}]" + (f"+{cal}" if cal else "")
            out.append((name, tag, lambda g, p=path: FlowPolicy(g, path=p), cal))
    return out


def held_out_scenes(ep_root=None):
    """HM3D: shuffled train scenes [120:160] (the models trained on 0..119).
    With --ep-root (e.g. MP3D val): a real held-out split, use every scene."""
    if ep_root:
        return scene_index("data", ep_root)
    idx = scene_index("data", "/Users/alexoman/workspaces/diffusion/socialnav_map_gen/"
                              "pointnav/social-hm3d/train")
    rng = np.random.default_rng(0)
    scenes = list(idx.items())
    rng.shuffle(scenes)
    return dict(scenes[120:120 + N_SCENES])


if __name__ == "__main__":
    groups = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--models=")),
                  "classical,state,slstm").split(",")
    out = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--out=")),
               "hm3d_exp1.csv")
    ep_root = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--ep-root=")), None)
    n_per = int(next((a.split("=", 1)[1] for a in sys.argv
                      if a.startswith("--n-per-scene=")), N_PER_SCENE))
    held = held_out_scenes(ep_root)
    print(f"{len(held)} held-out scenes, {n_per}/scene, reactive=0, "
          f"groups {groups} -> {out}", flush=True)
    fields = ["policy", "model", "aggregation", "success", "spl", "steps", "coll_rate",
              "coll_steps", "closest_med", "n"]
    # aggregation tag for the write-up: union-bound vs max-over-horizon vs none.
    AGG = {"aci": "union", "dtaci": "union", "max": "max", "max+": "max", "maxdt+": "max"}

    seen = set()
    if os.path.exists(out):
        with open(out) as f:
            seen = {row["policy"] for row in csv.DictReader(f)}
    else:
        with open(out, "w", newline="") as f:
            csv.DictWriter(f, fields).writeheader()

    for name, model, factory, cal in setups(groups):
        if name in seen:
            continue
        t0 = time.time()
        rows = evaluate(factory, held, n_per, collision_ends=True, calibrator=cal)
        ok = [r for r in rows if r["success"]] or rows
        line = dict(policy=name, model=model, aggregation=AGG.get(cal, "none"),
                    success=np.mean([r["success"] for r in rows]),
                    spl=np.mean([r["spl"] for r in rows]),
                    steps=np.mean([r["steps"] for r in ok]),
                    coll_rate=np.mean([r["collisions"] > 0 for r in rows]),
                    coll_steps=np.mean([r["collisions"] for r in rows]),
                    closest_med=np.nanmedian([r["closest"] for r in rows]),
                    n=len(rows))
        with open(out, "a", newline="") as f:
            csv.DictWriter(f, fields).writerow(line)
        summary(name, rows)
        print(f"           [{time.time() - t0:.0f}s]", flush=True)
    print(f"wrote {out}")
