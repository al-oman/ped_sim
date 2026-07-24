"""Builds the three Experiment-2 finding tables from hm3d_exp2.csv.
Run from the ped_sim root: python results/tables/_build_finding_tables.py
Kept in the folder so the tables are reproducible / refreshable from the raw sweep.
"""

import csv
import os

HERE = os.path.dirname(__file__)
SRC = os.path.join(HERE, "..", "..", "hm3d_exp2.csv")

REACT = ["0.0", "0.1", "0.3", "1.0"]


def load():
    rows = {}
    with open(SRC) as f:
        for r in csv.DictReader(f):
            rows[(r["reactive"], r["policy"])] = r
    return rows


def pct(x):
    return round(float(x) * 100, 1)


def main():
    d = load()
    n = d[("0.1", "flow")]["n"]

    # Finding 1: success rate (%) of every policy at every reactivity level.
    # flow+aci is best or tied-best in each column.
    with open(os.path.join(HERE, "finding1_success_rate_vs_reactivity.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([f"# Finding 1: flow+aci has best/tied-best success rate at every human->robot reactivity level (n={n}/cell)."])
        w.writerow(["# reactive=0 is Falcon (robot-blind); higher = crowd yields more to the robot. Values are success rate (%)."])
        w.writerow(["policy"] + [f"reactive_{r}" for r in REACT])
        for pol in ["astar", "rvo", "flow", "flow+aci", "flow+maxdt+"]:
            w.writerow([pol] + [pct(d[(r, pol)]["success"]) for r in REACT])

    # Finding 2: flow+aci collision rate FALLS and clearance GROWS as the crowd
    # shifts away from the training/warmup distribution -> online adaptation absorbs the shift.
    with open(os.path.join(HERE, "finding2_flow_aci_distribution_shift.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([f"# Finding 2: reactive>0 is a distribution shift (model + calibrator warmup assume robot-blind humans);"])
        w.writerow([f"# flow+aci's collision rate falls and median clearance grows as reactivity rises -> ACI adapts online (n={n}/cell)."])
        w.writerow(["reactive", "collision_rate_pct", "success_rate_pct", "closest_median_m"])
        for r in REACT:
            row = d[(r, "flow+aci")]
            w.writerow([r, pct(row["coll_rate"]), pct(row["success"]), round(float(row["closest_med"]), 2)])

    # Finding 3: baselines. A* recovers SR when humans dodge but keeps colliding;
    # the conservative RVO2 expert is left behind on success at every level.
    with open(os.path.join(HERE, "finding3_baselines.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([f"# Finding 3: A* recovers success when humans dodge but never stops colliding (>=16%);"])
        w.writerow([f"# the conservative RVO2 expert trails on success at every level; flow+aci beats both (n={n}/cell)."])
        w.writerow(["reactive",
                    "astar_success_pct", "astar_collision_pct",
                    "rvo_success_pct", "rvo_collision_pct",
                    "flow_aci_success_pct", "flow_aci_collision_pct"])
        for r in REACT:
            w.writerow([r,
                        pct(d[(r, "astar")]["success"]), pct(d[(r, "astar")]["coll_rate"]),
                        pct(d[(r, "rvo")]["success"]), pct(d[(r, "rvo")]["coll_rate"]),
                        pct(d[(r, "flow+aci")]["success"]), pct(d[(r, "flow+aci")]["coll_rate"])])

    print("wrote finding1/2/3 CSVs to", os.path.normpath(HERE))


if __name__ == "__main__":
    main()
