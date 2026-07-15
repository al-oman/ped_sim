"""Crowd deference sweep: how much pedestrians yield to the robot.

`deference` is the radius pedestrians assign to the robot in their ORCA sim.
At 0.4 (the usual setting) the crowd gives the robot a wide berth and covers
for even a reckless policy; as it shrinks toward 0.02 (essentially
non-yielding), collision avoidance becomes the robot's problem — the regime
where the safety layer has to earn its keep. Reports mean +/- 95% CI over
N_EP episodes per cell. Writes deference.csv.
"""

import csv
import time

import numpy as np

from aci import ACI, MaxACI
from eval import DEVICE, run
from policies import FlowPolicy
from predictor import ConstantVelocity
from sim import DT, HORIZON, N_PEDS

DEFERENCE = [0.4, 0.25, 0.1, 0.02]
N_EP = 100


def ci(vals):
    return 1.96 * np.std(vals) / np.sqrt(len(vals))


if __name__ == "__main__":
    out = []
    for d in DEFERENCE:
        for name, calib in [("raw      ", None),
                            ("union aci", ACI(alpha=0.1, horizon=HORIZON, n_peds=N_PEDS)),
                            ("max aci  ", MaxACI(alpha=0.1, horizon=HORIZON, n_peds=N_PEDS))]:
            policy = FlowPolicy(device=DEVICE)
            predictor = ConstantVelocity(DT, HORIZON) if calib else None
            t0 = time.time()
            rows = [run(policy, seed, predictor, calib, deference=d)
                    for seed in range(1000, 1000 + N_EP)]
            ok = [r for r in rows if r["success"]] or rows
            coll = [r["collisions"] for r in rows]
            line = dict(deference=d, policy=name.strip(),
                        success=np.mean([r["success"] for r in rows]),
                        steps=np.mean([r["steps"] for r in ok]),
                        collisions=np.mean(coll), collisions_ci=ci(coll),
                        closest=np.mean([r["closest"] for r in rows]),
                        tube=np.mean([r["tube"] for r in rows]),
                        tube_ci=ci([r["tube"] for r in rows]))
            out.append(line)
            tube = f", tube {line['tube']:.3f}±{line['tube_ci']:.3f}" if calib else ""
            print(f"deference {d}: {name}: success {line['success']:.0%}, "
                  f"steps {line['steps']:.0f}, "
                  f"collisions {line['collisions']:.2f}±{line['collisions_ci']:.2f}, "
                  f"closest {line['closest']:.2f} m{tube} [{time.time() - t0:.0f}s]")

    with open("deference.csv", "w", newline="") as f:
        w = csv.DictWriter(f, out[0].keys())
        w.writeheader()
        w.writerows(out)
    print("wrote deference.csv")
