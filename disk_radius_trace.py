"""Disk radius traces: union-bound ACI vs the max family, identical streams.

The claim under test: max-over-horizon scoring gives markedly smaller disks
than the union bound for the same tube-level guarantee, because each disk is
calibrated at 1-alpha instead of 1-alpha/H.

Protocol mirrors hm3d_eval.warm_calibrator: parked-robot episodes, so streams
are never interrupted by an episode reset (a reset would score in-flight
predictions against a re-spawned crowd and poison the max-score buffers).
All calibrators are warmed on identical per-episode streams, with pending
predictions cleared at episode boundaries, then fed one identical fresh
stream while the displayed radii are logged each step.

Writes disk_radius_trace.csv (per-step curves, for the paper plot).

Run: python disk_radius_trace.py
"""

import csv
import time

import numpy as np

from aci import ACI, MaxDtACI, PartialMaxACI
from predictor import ConstantVelocity
from sim import DT, HORIZON, N_PEDS, Env

ALPHA = 0.1
WARM_SEEDS = range(2000, 2020)   # same range as compare_cp's calibration seeds
WARM_STEPS = 150                 # per episode, matches hm3d_eval.warm_calibrator
TRACE_SEED = 3000
TRACE_STEPS = 350


def parked_stream(seed, steps):
    """(prediction, actual) pairs with the robot parked — crowd undisturbed."""
    predictor = ConstantVelocity(DT, HORIZON)
    env = Env(seed=seed)
    obs, pairs = env.reset(), []
    for _ in range(steps):
        pred = predictor.predict(obs["peds"])
        obs, _, _ = env.step(np.zeros(2))
        pairs.append((pred, obs["peds"][:, :2]))
    return pairs


def warmed(cls, streams, **kw):
    calib = cls(alpha=ALPHA, horizon=HORIZON, n_peds=N_PEDS, **kw)
    for pairs in streams:
        for pred, actual in pairs:
            calib.update(pred, actual)
        calib.past = []  # pending predictions don't survive an episode boundary
    return calib


if __name__ == "__main__":
    t0 = time.time()
    streams = [parked_stream(s, WARM_STEPS) for s in WARM_SEEDS]
    print(f"calibration: {len(streams)} episodes x {WARM_STEPS} steps "
          f"in {time.time() - t0:.0f}s")

    cals = {"aci": warmed(ACI, streams, gamma=0.01),
            "max+": warmed(PartialMaxACI, streams, gamma=0.01),
            "maxdt+": warmed(MaxDtACI, streams)}

    rows = []
    for step, (pred, actual) in enumerate(parked_stream(TRACE_SEED, TRACE_STEPS)):
        row = {"step": step}
        for name, calib in cals.items():
            radii = calib.radii()  # (HORIZON, N_PEDS), the displayed disks
            finite = radii[np.isfinite(radii)]
            row[f"mean_{name}"] = finite.mean() if finite.size else np.nan
            row[f"k1_{name}"] = radii[0][np.isfinite(radii[0])].mean()
            row[f"kH_{name}"] = radii[-1][np.isfinite(radii[-1])].mean()
            calib.update(pred, actual)
        rows.append(row)

    with open("disk_radius_trace.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for name in cals:
        m = np.nanmean([r[f"mean_{name}"] for r in rows])
        print(f"{name:8s}: mean displayed radius {m:.3f} m "
              f"(k=1 {np.nanmean([r[f'k1_{name}'] for r in rows]):.3f}, "
              f"k={HORIZON} {np.nanmean([r[f'kH_{name}'] for r in rows]):.3f})")
    print(f"wrote disk_radius_trace.csv ({len(rows)} steps, {time.time() - t0:.0f}s total)")
