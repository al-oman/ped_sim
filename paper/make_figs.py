"""Paper figures from the experiment CSVs. Writes PDF (for LaTeX) + PNG
(for quick viewing) into paper/figs/.

    fig_adaptation   adaptation_delay.csv    coverage around the shift: max vs max+
    fig_reactivity   hm3d_exp2.csv           SR and collision rate vs crowd reactivity
    fig_shift        hm3d_shift.csv          tube coverage under calibration/deployment shift
    fig_radii        disk_radius_trace.csv   mean disk radius: union vs max family

Run from the repo root: python paper/make_figs.py
Missing CSVs are skipped with a note (run their producer script first).
"""

import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "paper", "figs")
os.makedirs(OUT, exist_ok=True)

SHIFT_STEP = 100   # shift_experiment.SHIFT_STEP (import-free: torch not needed)
ALPHA = 0.1
W = 20             # rolling-mean window, matches adaptation_delay.py

# One color per method, consistent across all figures.
C = {"astar": "#999999", "rvo": "#c98a1c", "flow": "#7fb2e5", "flow+aci": "#1f5fa8",
     "flow+maxdt+": "#7a3db8", "split-cp": "#999999", "aci": "#1f5fa8",
     "max": "#c98a1c", "max+": "#2a9d5c", "maxdt+": "#7a3db8"}


def read(name):
    path = os.path.join(ROOT, name)
    if not os.path.exists(path):
        print(f"skip: {name} not found (run its producer first)")
        return None
    return list(csv.DictReader(open(path)))


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote figs/{name}.pdf/.png")


def rolling(xs, w):
    out, acc = [], []
    for x in xs:
        acc.append(x)
        out.append(sum(acc[-w:]) / len(acc[-w:]))
    return out


def fig_adaptation(rows):
    """Coverage around a mid-episode crowd-speed shift (toy sim)."""
    steps = [int(r["step"]) for r in rows]
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    for key, label in [("miss_max", "Max"), ("miss_max+", "Partial-Max (ours)")]:
        cov = rolling([1 - float(r[key]) for r in rows], W)
        name = "max" if key == "miss_max" else "max+"
        ax.plot(steps, cov, color=C[name], label=label, lw=1.5)
    ax.axvline(SHIFT_STEP, color="gray", ls=":", lw=1)
    ax.axhline(1 - ALPHA, color="black", ls="--", lw=0.8)
    ax.set_xlim(40, 250)   # skip the cold-start transient; tails are few episodes
    ax.set_ylim(0.82, 1.005)
    ax.annotate("shift", (SHIFT_STEP + 2, 0.83), fontsize=8, color="gray")
    ax.set_xlabel("step")
    ax.set_ylabel(f"coverage ({W}-step mean)")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    save(fig, "adaptation")


def fig_reactivity(rows):
    """Success rate and collision rate vs crowd reactivity (Social-HM3D)."""
    policies = ["astar", "rvo", "flow", "flow+aci", "flow+maxdt+"]
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.4), sharex=True)
    for pol in policies:
        pts = sorted(((float(r["reactive"]), r) for r in rows if r["policy"] == pol))
        xs = [x for x, _ in pts]
        axes[0].plot(xs, [100 * float(r["success"]) for _, r in pts],
                     "o-", color=C[pol], label=pol, lw=1.5, ms=3)
        axes[1].plot(xs, [100 * float(r["coll_rate"]) for _, r in pts],
                     "o-", color=C[pol], lw=1.5, ms=3)
    axes[0].set_ylabel("success rate (%)")
    axes[1].set_ylabel("episodes with a collision (%)")
    for ax in axes:
        ax.set_xlabel("crowd reactivity")
    axes[0].legend(frameon=False, fontsize=8)
    save(fig, "reactivity")


def fig_shift(rows):
    """Tube coverage, calibration speed vs shifted deployment speed."""
    cals = ["split-cp", "aci", "max+", "maxdt+"]
    speeds = sorted({r["human_speed"] for r in rows}, key=float)
    by = {(r["calibrator"], r["human_speed"]): float(r["tube"]) for r in rows}
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    width = 0.8 / len(speeds)
    for j, sp in enumerate(speeds):
        xs = [i + (j - (len(speeds) - 1) / 2) * width for i in range(len(cals))]
        label = ("calibration speed" if float(sp) == 1.0
                 else f"deployment {float(sp):.1f}x")
        ax.bar(xs, [100 * by[c, sp] for c in cals], width=width * 0.9,
               color=[C[c] for c in cals], alpha=1.0 if j else 0.45, label=label)
    ax.axhline(100 * (1 - ALPHA), color="black", ls="--", lw=0.8)
    ax.text(len(cals) - 0.55, 100 * (1 - ALPHA) + 0.4, "target", fontsize=7)
    ax.set_xticks(range(len(cals)), cals)
    ax.set_ylabel("tube coverage (%)")
    ax.set_ylim(70, 100)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    save(fig, "shift_coverage")


def fig_radii(rows):
    """Mean displayed disk radius, identical streams: union vs max family."""
    steps = [int(r["step"]) for r in rows]
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    for name, label in [("aci", "Union-bound ACI"), ("max+", "Partial-Max"),
                        ("maxdt+", "MaxDtACI")]:
        ax.plot(steps, [float(r[f"mean_{name}"]) for r in rows],
                color=C[name], label=label, lw=1.5)
    ax.set_xlabel("step")
    ax.set_ylabel("mean disk radius (m)")
    ax.legend(frameon=False, fontsize=8)
    save(fig, "disk_radii")


if __name__ == "__main__":
    for fn, src in [(fig_adaptation, "adaptation_delay.csv"),
                    (fig_reactivity, "hm3d_exp2.csv"),
                    (fig_shift, "hm3d_shift.csv"),
                    (fig_radii, "disk_radius_trace.csv")]:
        rows = read(src)
        if rows:
            fn(rows)
