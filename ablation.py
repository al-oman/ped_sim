"""Enforcement-mechanism ablation at deference 0.1 (assertive crowd, where
safety differences are visible), plus the replan_every 2-vs-4 decision.

    full             CBF brake + annealed projection (the system)
    projection only  kappa=0: no CBF brake, projection carries everything
    cbf only         project=False: no hard projection; violating plans
                     trigger the detect-and-brake fallback instead
    selection only   both off: disks only enter the best-of-N violation
                     penalty (+ detect-and-brake)
    full, replan 2   the system, replanning every 2 steps
    raw              no disks at all

All variants that see disks keep the best-of-N violation penalty and the
infeasibility brake, and use a fresh union-ACI calibrator. Writes
ablation.csv.
"""

import csv
import time

import numpy as np

from aci import ACI
from eval import DEVICE, run
from policies import FlowPolicy
from predictor import ConstantVelocity
from sim import DT, HORIZON, N_PEDS

DEF = 0.1
N_EP = 50
VARIANTS = [
    ("full (replan 4)", dict()),
    ("projection only", dict(kappa=0)),
    ("cbf only       ", dict(project=False)),
    ("selection only ", dict(kappa=0, project=False)),
    ("full, replan 2 ", dict(replan_every=2)),
    ("raw            ", None),
]

if __name__ == "__main__":
    out = []
    for name, kw in VARIANTS:
        policy = FlowPolicy(device=DEVICE, **(kw or {}))
        calib = ACI(alpha=0.1, horizon=HORIZON, n_peds=N_PEDS) if kw is not None else None
        predictor = ConstantVelocity(DT, HORIZON) if calib else None
        t0 = time.time()
        rows = [run(policy, seed, predictor, calib, deference=DEF)
                for seed in range(1000, 1000 + N_EP)]
        ok = [r for r in rows if r["success"]] or rows
        line = dict(variant=name.strip(),
                    success=np.mean([r["success"] for r in rows]),
                    steps=np.mean([r["steps"] for r in ok]),
                    collisions=np.mean([r["collisions"] for r in rows]),
                    closest=np.mean([r["closest"] for r in rows]),
                    heading=np.mean([r["heading"] for r in rows]),
                    tube=np.mean([r["tube"] for r in rows]))
        out.append(line)
        tube = f", tube {line['tube']:.3f}" if calib else ""
        print(f"{name}: success {line['success']:.0%}, steps {line['steps']:.0f}, "
              f"collisions {line['collisions']:.2f}, closest {line['closest']:.2f} m, "
              f"heading {line['heading']:.1f} deg{tube} [{time.time() - t0:.0f}s]")

    with open("ablation.csv", "w", newline="") as f:
        w = csv.DictWriter(f, out[0].keys())
        w.writeheader()
        w.writerows(out)
    print("wrote ablation.csv")
