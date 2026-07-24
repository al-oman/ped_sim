"""Distribution-shift experiment on Social-HM3D: static (split) CP vs ACI.

The shift: every calibrator is warmed on the episode's crowd at the standard
human_speed 1.0 (warm_calibrator's replay), then the scored episode runs the
same humans at HUMAN_SPEED — faster people, worse constant-velocity errors,
exactly the calibration/deployment mismatch that breaks split CP's
exchangeability requirement. Split CP = same construction, frozen after
warmup (gamma=0, all scores kept); the ACI rows keep adapting online.

Reports realized coverage of the certified disks and of whole tubes (the
alpha-level guarantee event; target 1 - ALPHA) plus the usual task metrics.
Held-out scenes, Falcon collision_ends semantics. Appends one row per
(human_speed, calibrator) to hm3d_shift.csv, resumably.

Run: python hm3d_shift.py
"""

import csv
import os
import time

import numpy as np

from hm3d_eval import evaluate, scene_index
from hm3d_policies import FlowPolicy

ALPHA = 0.1
HUMAN_SPEEDS = [1.0, 1.5]   # 1.0 = no shift control, 1.5 = deployment shift
N_SCENES = 20
N_PER_SCENE = 5
OUT = "hm3d_shift.csv"

# (name, calibrator, gamma, freeze) — split CP is ACI's construction with the
# dial disabled and the radii frozen at warmup's end.
SETUPS = [("split-cp", "aci", 0.0, True),
          ("aci", "aci", 0.01, False),
          ("max+", "max+", 0.01, False),
          ("maxdt+", "maxdt+", 0.01, False)]


def held_out_scenes():
    idx = scene_index("data", "/Users/alexoman/workspaces/diffusion/socialnav_map_gen/"
                              "pointnav/social-hm3d/train")
    rng = np.random.default_rng(0)
    scenes = list(idx.items())
    rng.shuffle(scenes)
    return dict(scenes[120:120 + N_SCENES])


if __name__ == "__main__":
    held = held_out_scenes()
    print(f"{len(held)} held-out scenes, human_speed sweep {HUMAN_SPEEDS}", flush=True)
    fields = ["human_speed", "calibrator", "coverage", "tube", "success", "spl",
              "coll_rate", "closest_med", "n"]
    seen = set()
    if os.path.exists(OUT):
        with open(OUT) as f:
            seen = {(row["human_speed"], row["calibrator"]) for row in csv.DictReader(f)}
    else:
        with open(OUT, "w", newline="") as f:
            csv.DictWriter(f, fields).writeheader()

    for hs in HUMAN_SPEEDS:
        for name, cal, gamma, freeze in SETUPS:
            if (str(hs), name) in seen:
                continue
            t0 = time.time()
            rows = evaluate(FlowPolicy, held, N_PER_SCENE, collision_ends=True,
                            calibrator=cal, human_speed=hs, gamma=gamma,
                            freeze_after_warm=freeze)
            line = dict(human_speed=hs, calibrator=name,
                        coverage=np.nanmean([r["coverage"] for r in rows]),
                        tube=np.nanmean([r["tube"] for r in rows]),
                        success=np.mean([r["success"] for r in rows]),
                        spl=np.mean([r["spl"] for r in rows]),
                        coll_rate=np.mean([r["collisions"] > 0 for r in rows]),
                        closest_med=np.nanmedian([r["closest"] for r in rows]),
                        n=len(rows))
            with open(OUT, "a", newline="") as f:
                csv.DictWriter(f, fields).writerow(line)
            print(f"speed {hs} {name:9s}: cert-coverage {line['coverage']:.4f}, "
                  f"tube {line['tube']:.3f} (target {1 - ALPHA}), "
                  f"SR {line['success']:.0%}, coll {line['coll_rate']:.0%} "
                  f"[{time.time() - t0:.0f}s]", flush=True)
    print(f"wrote {OUT}")
