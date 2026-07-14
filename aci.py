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


class MaxACI:
    """ACI with the max-over-horizon score (Cleaveland et al.): one tracker
    per pedestrian instead of one per (lookahead, pedestrian).

    Each lookahead k gets a typical error scale sigma_k (slow running mean).
    A prediction's score is max_k error_k / sigma_k, and one quantile q per
    pedestrian at level (1 - alpha_t) gives disks radius_k = q * sigma_k.
    If the max is covered, every lookahead is covered at once — so each
    pedestrian's whole tube holds at rate (1 - alpha) with NO union bound:
    per-disk quantiles sit at the 90th percentile instead of the 99.4th,
    giving much smaller disks for the same tube-level guarantee.

    The cost is delayed feedback: a prediction's max-score is only complete
    once its last lookahead resolves, horizon steps later, so adaptation
    lags the world by that long.

    The dial is graded on the operational event: "did the pedestrian escape
    any of this prediction's disks AS DISPLAYED at decision time?" — not on
    a recomputed after-the-fact score. That makes the coverage guarantee
    apply to the exact disks the robot planned around; the score buffer's
    only job is translating the dial position into radii."""

    def __init__(self, alpha=0.1, gamma=0.01, horizon=1, n_peds=1, window=100):
        self.alpha = alpha    # target miscoverage of the whole tube
        self.gamma = gamma
        self.horizon = horizon
        self.window = window
        self.alpha_t = np.full(n_peds, alpha)      # adapted online, per pedestrian
        self.scores = [[] for _ in range(n_peds)]  # max-scores, per pedestrian
        self.sigma = None                          # (horizon,) error scale per lookahead
        self.sigma_frozen = False                  # stop sigma drift (diagnostics)
        self.miss_log = []                         # internal tube misses, per finalization
        self.past = []                             # (prediction, error table), newest first
        self.frozen = False

    def freeze(self):
        """Stop adapting: radii stay exactly as they are (split CP mode)."""
        self.frozen = True

    def _quantiles(self):
        return np.array([np.quantile(s, np.clip(1 - a, 0, 1)) if s else np.inf
                         for s, a in zip(self.scores, self.alpha_t)])

    def radii(self):
        """(horizon, n_peds) disk radius per lookahead step and pedestrian."""
        if self.sigma is None:
            return np.full((self.horizon, len(self.alpha_t)), np.inf)
        return np.maximum(self.sigma, 1e-3)[:, None] * self._quantiles()[None, :]

    def update(self, prediction, actual):
        """Same interface as ACI.update: returns per-(lookahead, pedestrian)
        miss indicators. Internally, a prediction waits in self.past until
        all its lookaheads have resolved; its per-step hit/miss outcomes
        (against the radii displayed at each step) are collected alongside
        its errors, and on the last step it becomes one max-score for the
        buffer and one displayed-disks outcome for the dial."""
        self.past.insert(0, (prediction,
                             np.full((self.horizon, len(actual)), np.nan),
                             np.zeros((self.horizon, len(actual)), bool)))
        radii = self.radii()
        missed = np.zeros((len(self.past), len(actual)))
        for j, (pred, errs, outcomes) in enumerate(self.past):
            errs[j] = np.linalg.norm(actual - pred[j], axis=1)
            missed[j] = errs[j] > radii[j]
            outcomes[j] = missed[j]
        if len(self.past) == self.horizon:  # the oldest is now fully resolved
            _, errs, outcomes = self.past.pop()
            if not self.frozen:
                if not self.sigma_frozen:
                    mean_k = errs.mean(axis=1)
                    self.sigma = mean_k if self.sigma is None else 0.98 * self.sigma + 0.02 * mean_k
                score = (errs / np.maximum(self.sigma[:, None], 1e-3)).max(axis=0)
                miss = outcomes.any(axis=0)  # any displayed disk missed
                self.miss_log.append(miss.mean())
                self.alpha_t += self.gamma * (self.alpha - miss)
                for n, s in enumerate(score):
                    self.scores[n] = (self.scores[n] + [s])[-self.window:]
        return missed
