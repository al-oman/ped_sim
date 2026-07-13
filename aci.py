"""Adaptive conformal inference (Gibbs & Candes, 2021) over a prediction
horizon, one tracker per (lookahead, pedestrian) pair.

Each tracker maintains a radius such that a disk of that radius around the
k-step-ahead prediction of pedestrian n contains its true position roughly
(1 - alpha/horizon) of the time — with no assumptions on how pedestrians
behave. The trick is the alpha_t recursion: whenever a disk misses, its
alpha_t drops and its radius grows; whenever it covers, the radius slowly
shrinks back.

Targeting alpha/horizon per disk (instead of alpha) is the union bound: a
pedestrian's whole tube of horizon disks fails only if at least one disk
fails, so the tube misses at most horizon * (alpha/horizon) = alpha of the
time. That joint guarantee is what a planner using all disks at once needs.

The same class also implements the two classic conformal baselines:
gamma=0 gives non-adaptive online CP (scores stay fresh, quantile level is
fixed), and calling freeze() after a calibration phase gives frozen split CP
(radii never change again; update() only reports coverage).
"""

import numpy as np


class ACI:
    def __init__(self, alpha=0.1, gamma=0.01, horizon=1, n_peds=1, window=100):
        self.alpha = alpha             # target miscoverage of the whole tube
        self.target = alpha / horizon  # per-disk target, via union bound
        self.gamma = gamma             # adaptation step size
        self.horizon = horizon
        self.window = window
        self.alpha_t = np.full((horizon, n_peds), self.target)  # adapted online
        self.scores = [[[] for _ in range(n_peds)] for _ in range(horizon)]
        self.past = []                 # recent predictions, newest first
        self.frozen = False

    def freeze(self):
        """Stop adapting: radii stay exactly as they are (split CP mode)."""
        self.frozen = True

    def radii(self):
        """(horizon, n_peds) disk radius per lookahead step and pedestrian."""
        q = np.clip(1 - self.alpha_t, 0, 1)
        return np.array([[np.quantile(s, q[k, n]) if s else np.inf
                          for n, s in enumerate(row)]
                         for k, row in enumerate(self.scores)])

    def update(self, prediction, actual):
        """Call once per step with the (horizon, N, 2) prediction just made and
        the (N, 2) true positions just observed. Scores every past prediction
        whose target time is now: the prediction made j+1 steps ago aimed its
        row j at the current time. Returns (lookaheads, N) miss indicators."""
        self.past = [prediction] + self.past[:self.horizon - 1]
        radii = self.radii()
        missed = np.zeros((len(self.past), len(actual)))
        for k, pred in enumerate(self.past):
            errors = np.linalg.norm(actual - pred[k], axis=1)
            missed[k] = errors > radii[k]
            if not self.frozen:
                self.alpha_t[k] += self.gamma * (self.target - missed[k])
                for n, e in enumerate(errors):
                    self.scores[k][n] = (self.scores[k][n] + [e])[-self.window:]
        return missed
