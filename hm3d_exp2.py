"""Experiment 2: human->robot reactivity sweep on Social-HM3D.

Falcon's humans are robot-blind (reactive=0); `WaypointCrowd.reactive`
bends them away from the robot with an exp(-dist^2/8) kernel. The question:
does the safety layer's advantage survive as the crowd starts doing some of
the avoidance itself? Note the flow model and each episode's calibrator
warmup both assume the reactive=0 distribution, so reactive>0 is also a
deployment shift — the regime ACI's online adaptation is built for.

Held-out scenes (shuffled index >= 120; the model trained on 0..119), Falcon
collision_ends semantics. reactive=0 baselines come from today's held-out run
(same scenes/seeds; see HANDOFF). Appends one row per (reactive, policy) to
hm3d_exp2.csv, resumably. Run: python hm3d_exp2.py
"""

import csv
import os
import time

import numpy as np

from hm3d_eval import evaluate, scene_index, summary
from hm3d_policies import AStarPolicy, FlowPolicy, RvoPolicy

REACTIVE = [0.0, 0.1, 0.3, 1.0]   # 0 = Falcon (robot-blind) baseline
N_PER_SCENE = 5
OUT = "hm3d_exp2.csv"

SETUPS = [("astar", AStarPolicy, None), ("rvo", RvoPolicy, None),
          ("flow", FlowPolicy, None), ("flow+aci", FlowPolicy, "aci"),
          ("flow+maxdt+", FlowPolicy, "maxdt+")]


def held_out_scenes():
    idx = scene_index("data", "/Users/alexoman/workspaces/diffusion/socialnav_map_gen/"
                              "pointnav/social-hm3d/train")
    rng = np.random.default_rng(0)
    scenes = list(idx.items())
    rng.shuffle(scenes)
    return dict(scenes[120:140])


if __name__ == "__main__":
    held = held_out_scenes()
    print(f"{len(held)} held-out scenes, reactive sweep {REACTIVE}", flush=True)
    fields = ["reactive", "policy", "success", "spl", "steps", "coll_rate", "closest_med", "n"]
    seen = set()
    if os.path.exists(OUT):
        with open(OUT) as f:
            seen = {(row["reactive"], row["policy"]) for row in csv.DictReader(f)}
    else:
        with open(OUT, "w", newline="") as f:
            csv.DictWriter(f, fields).writeheader()

    for reactive in REACTIVE:
        for name, factory, cal in SETUPS:
            if (str(reactive), name) in seen:
                continue
            t0 = time.time()
            rows = evaluate(factory, held, N_PER_SCENE, collision_ends=True,
                            calibrator=cal, reactive=reactive)
            ok = [r for r in rows if r["success"]] or rows
            line = dict(reactive=reactive, policy=name,
                        success=np.mean([r["success"] for r in rows]),
                        spl=np.mean([r["spl"] for r in rows]),
                        steps=np.mean([r["steps"] for r in ok]),
                        coll_rate=np.mean([r["collisions"] > 0 for r in rows]),
                        closest_med=np.nanmedian([r["closest"] for r in rows]),
                        n=len(rows))
            with open(OUT, "a", newline="") as f:
                csv.DictWriter(f, fields).writerow(line)
            print(f"reactive {reactive}: ", end="")
            summary(name, rows)
            print(f"           [{time.time() - t0:.0f}s]", flush=True)
    print(f"wrote {OUT}")
