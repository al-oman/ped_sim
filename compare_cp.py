"""Compares disk calibrators under the ACI-constrained flow policy:

    split cp  — calibrated on unconstrained-flow episodes, then frozen
    online cp — same calibration, scores keep updating, fixed quantile level
    aci       — same calibration, scores update AND alpha_t adapts (ours)

All three are warmed on identical calibration data (unconstrained flow
rollouts, seeds disjoint from eval), then deployed under the constrained
policy. The question is realized coverage of the certified disks: deploying
the safety layer changes the robot's behavior — and with it the prediction
error distribution — which breaks the exchangeability that split CP's
guarantee needs. ACI's guarantee survives that shift by construction.
"""

import time

import numpy as np

from aci import ACI, DtACI, MaxACI, MaxDtACI, PartialMaxACI
from eval import DEVICE, EPISODES, MAX_STEPS, run
from policies import FlowPolicy
from predictor import ConstantVelocity
from sim import DT, HORIZON, N_PEDS, Env

CALIB_SEEDS = range(2000, 2020)
ALPHA = 0.1


def calibration_data(deference=0.4):
    """(prediction, actual) pairs from unconstrained flow rollouts."""
    policy, predictor = FlowPolicy(device=DEVICE), ConstantVelocity(DT, HORIZON)
    pairs = []
    for seed in CALIB_SEEDS:
        env = Env(seed=seed, deference=deference)
        obs, done, steps = env.reset(), False, 0
        while not done and steps < MAX_STEPS:
            pred = predictor.predict(obs["peds"])
            obs, _, done = env.step(policy.act(obs))  # no disks: unconstrained
            pairs.append((pred, obs["peds"][:, :2]))
            steps += 1
    return pairs


def warmed(pairs, gamma, window=100):
    """A calibrator with its buffers pre-filled from calibration pairs."""
    aci = ACI(alpha=ALPHA, gamma=gamma, horizon=HORIZON, n_peds=N_PEDS, window=window)
    for pred, actual in pairs:
        aci.update(pred, actual)
    return aci


def calibrators(pairs):
    """The comparison calibrators, warmed on identical data."""
    split = warmed(pairs, gamma=0.0, window=10 ** 6)  # split CP keeps all calibration scores
    split.freeze()
    dtaci = DtACI(alpha=ALPHA, horizon=HORIZON, n_peds=N_PEDS)
    maxdt = MaxDtACI(alpha=ALPHA, horizon=HORIZON, n_peds=N_PEDS)  # no gamma: self-tunes
    maxes = {}
    for name, cls in [("aci max  ", MaxACI), ("aci max+ ", PartialMaxACI)]:
        m = cls(alpha=ALPHA, gamma=0.01, horizon=HORIZON, n_peds=N_PEDS)
        maxes[name] = m
    for pred, actual in pairs:
        dtaci.update(pred, actual)
        maxdt.update(pred, actual)
        for m in maxes.values():
            m.update(pred, actual)
    for m in (*maxes.values(), maxdt):
        m.sigma_frozen = True  # lock the error scales once calibration ends
    return {"split cp ": split,
            "online cp": warmed(pairs, gamma=0.0),
            "aci      ": warmed(pairs, gamma=0.01),
            "dtaci    ": dtaci,
            **maxes,
            "aci maxdt": maxdt}


if __name__ == "__main__":
    t0 = time.time()
    pairs = calibration_data()
    print(f"calibrated on {len(pairs)} steps from {len(CALIB_SEEDS)} unconstrained episodes "
          f"in {time.time() - t0:.0f}s (device: {DEVICE})")
    setups = calibrators(pairs)

    print(f"per-disk coverage target {1 - ALPHA / HORIZON:.4f} (union methods), "
          f"tube coverage target {1 - ALPHA} (all methods)")
    for name, calib in setups.items():
        policy, predictor = FlowPolicy(device=DEVICE), ConstantVelocity(DT, HORIZON)
        t0 = time.time()
        rows = [run(policy, seed, predictor, calib) for seed in range(1000, 1000 + EPISODES)]
        ok = [r for r in rows if r["success"]] or rows
        print(f"{name}: coverage {np.mean([r['coverage'] for r in rows]):.4f}, "
              f"tube {np.mean([r['tube'] for r in rows]):.3f}, "
              f"success {np.mean([r['success'] for r in rows]):.0%}, "
              f"steps {np.mean([r['steps'] for r in ok]):.0f}, "
              f"collisions {np.mean([r['collisions'] for r in rows]):.2f}, "
              f"closest {np.mean([r['closest'] for r in rows]):.2f} m "
              f"[{time.time() - t0:.0f}s]")
