"""Coverage under distribution shift: split CP vs online CP vs ACI.

All calibrators are warmed on the same calibration data (unconstrained flow
episodes, normal pedestrians), then deployed with the constrained policy
under two shift conditions:

    speed_between — episodes 1..20 normal, 21..40 pedestrians at 2.0 m/s
    speed_mid     — every episode: speed jumps 1.2 -> 2.0 m/s at step 100

Faster pedestrians make constant-velocity predictions worse, so the frozen
split-CP radii calibrated on calm data under-cover after the shift, while
ACI adapts back to target. Coverage is tracked two ways: over the certified
disks (k <= replan_every, what the safety guarantee rides on — insensitive,
since 0.4 s of lookahead hides most of the shift) and over all lookaheads
k = 1..HORIZON (where the shift actually bites). Every disk shares the same
per-disk target 1 - alpha/HORIZON.

Writes shift_results.csv and shift_coverage.png.
"""

import csv

import numpy as np

from compare_cp import ALPHA, calibration_data, calibrators
from policies import FlowPolicy
from predictor import ConstantVelocity
from sim import DT, HORIZON, Env

EPISODES = 40      # per condition; the "between" condition shifts after episode 20
SHIFT_EP = 20
SHIFT_STEP = 100   # for the mid-episode condition
FAST = 2.0         # shifted pedestrian speed, m/s (normal is PED_SPEED = 1.2)
MAX_STEPS = 600
PLOT_STEPS = 350   # truncate mid-episode curves: beyond this only failed
                   # episodes are still running (survivor bias)
TARGET = 1 - ALPHA / HORIZON


def speed_now(env, step):
    if step == 0:
        env.crowd.set_speed(FAST)


def speed_mid(env, step):
    if step == SHIFT_STEP:
        env.crowd.set_speed(FAST)


CONDITIONS = {
    "speed_between": [None] * SHIFT_EP + [speed_now] * (EPISODES - SHIFT_EP),
    "speed_mid": [speed_mid] * EPISODES,
}


def episode(seed, policy, predictor, calib, shift):
    """One deployment episode. Returns per-step miss rates of the certified
    (k <= replan_every) disks and of all disks, and whether the robot made it."""
    env = Env(seed=seed)
    obs, done, steps = env.reset(), False, 0
    calib.past = []  # pending predictions don't survive a world reset
    miss_cert, miss_all = [], []
    while not done and steps < MAX_STEPS:
        if shift is not None:
            shift(env, steps)
        pred = predictor.predict(obs["peds"])
        obs, _, done = env.step(policy.act(obs, (pred, calib.radii())))
        missed = calib.update(pred, obs["peds"][:, :2])
        miss_cert.append(missed[:policy.replan_every].mean())
        miss_all.append(missed.mean())
        steps += 1
    return np.array(miss_cert), np.array(miss_all), done


if __name__ == "__main__":
    pairs = calibration_data()
    print(f"calibrated on {len(pairs)} steps; per-disk coverage target {TARGET:.4f}")

    results = {}  # (condition, calibrator) -> list of (miss_cert, miss_all, success)
    for cond, shifts in CONDITIONS.items():
        for name, calib in calibrators(pairs).items():
            policy, predictor = FlowPolicy(), ConstantVelocity(DT, HORIZON)
            eps = [episode(1000 + i, policy, predictor, calib, s)
                   for i, s in enumerate(shifts)]
            results[cond, name] = eps
            pre, post = eps[:SHIFT_EP], eps[SHIFT_EP:]

            def stat(rows, j):
                return np.mean([1 - m[j].mean() for m in rows])

            print(f"{cond:13s} {name}: certified pre {stat(pre, 0):.4f} post {stat(post, 0):.4f}, "
                  f"all-k pre {stat(pre, 1):.4f} post {stat(post, 1):.4f}, "
                  f"success pre {np.mean([m[2] for m in pre]):.0%} "
                  f"post {np.mean([m[2] for m in post]):.0%}")

    with open("shift_results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "calibrator", "episode", "coverage_certified",
                    "coverage_allk", "steps", "success"])
        for (cond, name), eps in results.items():
            for i, (cert, allk, success) in enumerate(eps):
                w.writerow([cond, name.strip(), i, 1 - cert.mean(), 1 - allk.mean(),
                            len(cert), int(success)])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex="col", sharey="row")
    for col, cond in enumerate(CONDITIONS):
        for row, metric in enumerate(("certified disks (k <= 4)", "all disks (k = 1..16)")):
            ax = axes[row, col]
            for name in [n for (c, n) in results if c == cond]:
                curves = [m[row] for m in results[cond, name]]
                if cond == "speed_mid":  # x = step within episode, mean over episodes
                    stack = np.full((len(curves), PLOT_STEPS), np.nan)
                    for i, m in enumerate(curves):
                        stack[i, :min(len(m), PLOT_STEPS)] = m[:PLOT_STEPS]
                    cov = 1 - np.nanmean(stack, axis=0)
                    cov = np.convolve(cov, np.ones(20) / 20, mode="valid")
                    ax.plot(cov, label=name.strip())
                    ax.axvline(SHIFT_STEP, color="gray", ls=":")
                else:  # x = episode index
                    ax.plot([1 - m.mean() for m in curves], label=name.strip())
                    ax.axvline(SHIFT_EP - 0.5, color="gray", ls=":")
            ax.axhline(TARGET, color="black", ls="--", lw=0.8, label="per-disk target (union)")
            ax.axhline(1 - ALPHA, color="gray", ls="--", lw=0.8, label="tube target (max)")
            if row == 0:
                ax.set_title(cond)
            if col == 0:
                ax.set_ylabel(f"coverage, {metric}")
        axes[1, col].set_xlabel("step within episode" if cond == "speed_mid"
                                else "deployment episode")
    axes[0, 0].legend()
    fig.tight_layout()
    fig.savefig("shift_coverage.png", dpi=150)
    print("wrote shift_results.csv and shift_coverage.png")
