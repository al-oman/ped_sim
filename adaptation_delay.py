"""Adaptation delay: MaxACI vs PartialMaxACI after a mid-episode shift.

The claim under test: partial-max removes the horizon-step feedback lag, so
its disks recover coverage sooner after the world changes. Protocol is
shift_experiment's speed_mid condition (toy sim, crowd speed 1.2 -> 2.0 m/s
at step SHIFT_STEP), constrained flow policy, both calibrators warmed on
identical calibration data with error scales locked (sigma_frozen).

Delay metric, computed on the across-episode mean of the per-step all-k miss
rate, smoothed with a W-step rolling mean: the first post-shift step at which
smoothed coverage returns to (pre-shift baseline - TOL) and stays there for
SUSTAIN consecutive steps. Also reports the post-shift dip depth. Writes
adaptation_delay.csv (per-step mean curves, for the paper plot).

Run: python adaptation_delay.py
"""

import csv
import time

import numpy as np

from aci import MaxACI, PartialMaxACI
from compare_cp import ALPHA, DEVICE, calibration_data
from policies import FlowPolicy
from predictor import ConstantVelocity
from shift_experiment import DEFERENCE, MAX_STEPS, SHIFT_STEP, episode, speed_mid
from sim import DT, HORIZON, N_PEDS

N_EP = 30
W = 20          # rolling-mean window, steps
TOL = 0.01      # recovery threshold below the pre-shift baseline
SUSTAIN = 5     # must hold recovery this many consecutive steps
CURVE_LEN = 350


def run(cls, pairs):
    calib = cls(alpha=ALPHA, gamma=0.01, horizon=HORIZON, n_peds=N_PEDS)
    for pred, actual in pairs:
        calib.update(pred, actual)
    calib.sigma_frozen = True  # lock error scales: isolate the dial dynamics
    curves = np.full((N_EP, CURVE_LEN), np.nan)
    for i in range(N_EP):
        policy, predictor = FlowPolicy(device=DEVICE), ConstantVelocity(DT, HORIZON)
        _, miss_all, _ = episode(1000 + i, policy, predictor, calib, speed_mid)
        curves[i, :min(len(miss_all), CURVE_LEN)] = miss_all[:CURVE_LEN]
    return curves


def delay(curve_mean):
    """(delay steps, dip depth) from a per-step mean miss curve."""
    cov = 1 - curve_mean
    smooth = np.convolve(cov, np.ones(W) / W, mode="valid")  # smooth[t] ends at t+W-1
    base = np.nanmean(smooth[max(0, SHIFT_STEP - W - 60):SHIFT_STEP - W])
    post = smooth[SHIFT_STEP - W + 1:]  # windows containing post-shift steps
    dip = base - np.nanmin(post)
    ok = post >= base - TOL
    for t in range(len(ok) - SUSTAIN):
        if ok[t:t + SUSTAIN].all():
            return t + 1, dip  # steps after the shift entered the window
    return np.nan, dip


if __name__ == "__main__":
    t0 = time.time()
    pairs = calibration_data(deference=DEFERENCE)
    print(f"calibrated on {len(pairs)} steps in {time.time() - t0:.0f}s")

    results = {}
    for name, cls in [("max", MaxACI), ("max+", PartialMaxACI)]:
        t0 = time.time()
        curves = run(cls, pairs)
        mean = np.nanmean(curves, axis=0)
        d, dip = delay(mean)
        results[name] = (d, dip, mean)
        print(f"{name:5s}: adaptation delay {d:.0f} steps, coverage dip {dip:.4f} "
              f"[{time.time() - t0:.0f}s]", flush=True)

    if all(np.isfinite(results[k][0]) for k in results):
        x = 1 - results["max+"][0] / results["max"][0]
        print(f"partial-max reduces adaptation delay by {x:.0%} "
              f"({results['max'][0]:.0f} -> {results['max+'][0]:.0f} steps)")

    with open("adaptation_delay.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step"] + [f"miss_{k}" for k in results])
        for t in range(CURVE_LEN):
            w.writerow([t] + [results[k][2][t] for k in results])
    print("wrote adaptation_delay.csv")
