"""Compares policies over a batch of episodes.

Task metrics: success rate, steps to goal (successful episodes), path length.
Safety metrics: collision steps, closest approach, fraction of steps near a
pedestrian. Smoothness metrics: heading change per step, acceleration, jerk.

The "flow + aci" row runs the flow model with ACI-constrained sampling (disks
around predicted pedestrians enforced during generation); its ACI calibrator
persists across episodes. Seeds are outside the training range (train.py uses
0..EPISODES) so the flow model is evaluated on crowds it has never seen."""

import numpy as np

from aci import ACI
from policies import FlowPolicy, OrcaExpert
from predictor import ConstantVelocity
from sim import DT, HEIGHT, HORIZON, N_PEDS, WALLS, WIDTH, Env

EPISODES = 20
MAX_STEPS = 600
NEAR = 0.6  # near-miss distance between centers, m (collision is < 0.5)


def run(policy, seed, predictor=None, aci=None):
    env = Env(seed=seed)
    obs, done, steps, collisions = env.reset(), False, 0, 0
    miss, miss_n = 0.0, 0
    path, dists = [obs["robot"]], []
    while not done and steps < MAX_STEPS:
        disks = None
        if aci is not None:
            pred = predictor.predict(obs["peds"])
            disks = (pred, aci.radii())
        obs, reward, done = env.step(policy.act(obs, disks))
        if aci is not None:
            # Realized coverage of the certified (executed-prefix) disks.
            missed = aci.update(pred, obs["peds"][:, :2])
            miss += missed[:getattr(policy, "replan_every", len(missed))].mean()
            miss_n += 1
        collisions += reward < 0
        path.append(obs["robot"])
        dists.append(np.linalg.norm(obs["peds"][:, :2] - obs["robot"], axis=1).min())
        steps += 1

    path, dists = np.array(path), np.array(dists)
    vel = np.diff(path, axis=0) / DT
    acc = np.diff(vel, axis=0) / DT
    jerk = np.diff(acc, axis=0) / DT
    moving = np.linalg.norm(vel, axis=1) > 1e-3
    u = vel[moving] / np.linalg.norm(vel[moving], axis=1, keepdims=True)
    heading = np.degrees(np.arccos(np.clip((u[:-1] * u[1:]).sum(axis=1), -1, 1)))
    return {"success": done,
            "steps": steps,
            "collisions": collisions,
            "coverage": 1 - miss / miss_n if miss_n else float("nan"),
            "closest": dists.min(),
            "near": (dists < NEAR).mean(),
            "path": np.linalg.norm(np.diff(path, axis=0), axis=1).sum(),
            "heading": heading.mean(),
            "accel": np.linalg.norm(acc, axis=1).mean(),
            "jerk": np.linalg.norm(jerk, axis=1).mean()}


if __name__ == "__main__":
    setups = {
        "orca expert": (OrcaExpert(goal=(WIDTH - 0.3, HEIGHT / 2), dt=DT, walls=WALLS), False),
        "flow": (FlowPolicy(), False),
        "flow + aci": (FlowPolicy(), True),
    }
    for name, (policy, constrained) in setups.items():
        predictor = ConstantVelocity(DT, HORIZON) if constrained else None
        aci = ACI(alpha=0.1, horizon=HORIZON, n_peds=N_PEDS) if constrained else None
        rows = [run(policy, seed, predictor, aci) for seed in range(1000, 1000 + EPISODES)]
        ok = [r for r in rows if r["success"]] or rows  # task metrics: successful eps only

        def mean(key, rs=rows):
            return np.mean([r[key] for r in rs])

        print(f"{name:12s}: success {mean('success'):.0%}, steps {mean('steps', ok):.0f}, "
              f"path {mean('path', ok):.1f} m, collision steps {mean('collisions'):.1f}, "
              f"closest {mean('closest'):.2f} m, near(<{NEAR}m) {mean('near'):.1%}")
        cov = f", coverage {mean('coverage'):.3f}" if constrained else ""
        print(f"{'':12s}  heading {mean('heading'):.1f} deg/step, "
              f"accel {mean('accel'):.2f} m/s^2, jerk {mean('jerk'):.0f} m/s^3{cov}")
