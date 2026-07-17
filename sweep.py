"""Parameter sweep for the ACI-constrained flow policy.

Sweeps each axis one-at-a-time around DEFAULTS, plus a small alpha x window
factorial. Each cell runs EPISODES fixed-seed episodes (same seeds for every
cell) and appends one row to sweep.csv. Cells already in the CSV are skipped,
so the script is resumable and new axis values can be added later.

Run: python sweep.py
"""

import csv
import itertools
import os
import time

import numpy as np

import sys

from aci import CALIBRATORS, make_calibrator
from eval import DEVICE, run
from policies import FlowPolicy
from predictor import ConstantVelocity
from sim import DT, HORIZON, N_PEDS

EPISODES = 30
SEEDS = range(1000, 1000 + EPISODES)
# --cal=dtaci|max|max+ sweeps that calibrator instead (own CSV, so the
# resume logic and existing aci results stay untouched). dtaci ignores gamma.
CAL = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--cal=")), "aci")
assert CAL in CALIBRATORS, f"--cal must be one of {list(CALIBRATORS)}"
OUT = "sweep.csv" if CAL == "aci" else f"sweep_{CAL.replace('+', 'plus')}.csv"

DEFAULTS = dict(alpha=0.1, window=100, gamma=0.01, replan_every=4,
                stale=0.5, tau=0.6, n_samples=4)
AXES = {
    "alpha": [0.05, 0.2, 0.3],       # disk size: guarantee target (Pareto axis)
    "window": [200, 300, 600],       # disk size: how long outlier errors linger
    "gamma": [0.003, 0.03],          # ACI adaptation speed
    "replan_every": [2, 8],          # plan commitment
    "stale": [0.75, 1.0],            # replan-on-divergence threshold
    "tau": [0.0, 0.3, 0.9],          # warm-start strength
    "n_samples": [2, 8],             # best-of-N candidates per batch
}
FACTORIAL = [{"alpha": a, "window": w}
             for a, w in itertools.product([0.2, 0.3], [100, 200])]


def cells():
    out = [dict(DEFAULTS)]
    for axis, values in AXES.items():
        out += [{**DEFAULTS, axis: v} for v in values]
    out += [{**DEFAULTS, **extra} for extra in FACTORIAL]
    return out


def evaluate(cfg):
    policy = FlowPolicy(replan_every=cfg["replan_every"], n_samples=cfg["n_samples"],
                        tau=cfg["tau"], stale=cfg["stale"], device=DEVICE)
    predictor = ConstantVelocity(DT, HORIZON)
    aci = make_calibrator(CAL, alpha=cfg["alpha"], gamma=cfg["gamma"], horizon=HORIZON,
                          n_peds=N_PEDS, window=cfg["window"])
    rows = [run(policy, seed, predictor, aci) for seed in SEEDS]
    ok = [r for r in rows if r["success"]] or rows
    return {"success": np.mean([r["success"] for r in rows]),
            "steps": np.mean([r["steps"] for r in ok]),
            "collisions": np.mean([r["collisions"] for r in rows]),
            "closest": np.mean([r["closest"] for r in rows]),
            "near": np.mean([r["near"] for r in rows]),
            "heading": np.mean([r["heading"] for r in rows]),
            "jerk": np.mean([r["jerk"] for r in rows])}


if __name__ == "__main__":
    fields = list(DEFAULTS) + ["success", "steps", "collisions", "closest",
                               "near", "heading", "jerk", "minutes"]
    seen = set()
    if os.path.exists(OUT):
        with open(OUT) as f:
            seen = {tuple(row[k] for k in DEFAULTS) for row in csv.DictReader(f)}
    else:
        with open(OUT, "w", newline="") as f:
            csv.DictWriter(f, fields).writeheader()

    for cfg in cells():
        key = tuple(str(cfg[k]) for k in DEFAULTS)
        if key in seen:
            continue
        seen.add(key)
        t0 = time.time()
        metrics = evaluate(cfg)
        metrics["minutes"] = round((time.time() - t0) / 60, 1)
        with open(OUT, "a", newline="") as f:
            csv.DictWriter(f, fields).writerow({**cfg, **metrics})
        print({k: v for k, v in cfg.items() if v != DEFAULTS[k]} or "defaults",
              "->", {k: round(v, 2) for k, v in metrics.items()})
