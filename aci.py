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

DtACI removes ACI's one tuning knob (gamma) by running a bank of dials and
weighting them online; PartialMaxACI removes MaxACI's horizon-step feedback
lag by scoring predictions incrementally as their lookaheads resolve.
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
        self.debt = np.zeros(n_peds)               # queued -gamma penalties (max+ variants)
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


class DtACI(ACI):
    """ACI with gamma chosen online (Gibbs & Candes, 2022) instead of fixed.

    Each (lookahead, pedestrian) tracker runs a small bank of ACI dials, one
    per candidate gamma, and plays their weighted average. After every
    observation each dial is scored by how far its quantile level sat from
    the level that would have *just barely* covered (pinball loss, so
    misses cost 1 - target and slack costs only target); weights shift
    exponentially toward dials that are tracking well, with a small
    fixed-share mix back to uniform so a written-off dial can return after
    a regime change.

    Same union-bound construction and guarantee as ACI — each tracker
    targets alpha/horizon — and the score buffers are shared across dials
    (a dial only owns a quantile level). The eta and fixed-share constants
    are the theory defaults from the paper for a `span`-step horizon of
    interest."""

    GAMMA_GRID = (0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.064, 0.128)

    def __init__(self, alpha=0.1, horizon=1, n_peds=1, window=100, span=100):
        super().__init__(alpha, 0.0, horizon, n_peds, window)
        self.gammas = np.array(self.GAMMA_GRID)
        K, t = len(self.gammas), self.target
        self.eta = np.sqrt(3 / span) * np.sqrt(
            (np.log(K * span) + 2) / (t ** 2 * (1 - t) ** 3 + (1 - t) ** 2 * t ** 3))
        self.share = 1 / (2 * span)
        self.alpha_t = np.full((K, horizon, n_peds), t)   # one dial bank per tracker
        self.weights = np.full((K, horizon, n_peds), 1 / K)

    def radii(self):
        """(horizon, n_peds) radius from each tracker's weighted-average dial."""
        q = np.clip(1 - (self.weights * self.alpha_t).sum(axis=0), 0, 1)
        return np.array([[np.quantile(s, q[k, n]) if s else np.inf
                          for n, s in enumerate(row)]
                         for k, row in enumerate(self.scores)])

    def update(self, prediction, actual):
        """Same interface and grading as ACI.update."""
        self.past = [prediction] + self.past[:self.horizon - 1]
        radii = self.radii()
        missed = np.zeros((len(self.past), len(actual)))
        for k, pred in enumerate(self.past):
            errors = np.linalg.norm(actual - pred[k], axis=1)
            missed[k] = errors > radii[k]
            if self.frozen:
                continue
            for n, e in enumerate(errors):
                s = self.scores[k][n]
                if s:
                    a = self.alpha_t[:, k, n]
                    beta = 1 - np.mean(np.array(s) <= e)  # level that would have just covered
                    miss = e > np.quantile(s, np.clip(1 - a, 0, 1))  # per-dial outcome
                    loss = self.target * (beta - a) + np.maximum(0, a - beta)
                    w = self.weights[:, k, n] * np.exp(-self.eta * loss)
                    w = (1 - self.share) * w + self.share * w.mean()
                    self.weights[:, k, n] = w / w.sum()
                    self.alpha_t[:, k, n] = a + self.gammas * (self.target - miss)
                self.scores[k][n] = (s + [e])[-self.window:]
        return missed


def make_calibrator(name, alpha=0.1, gamma=0.01, horizon=1, n_peds=1, window=100):
    """Build a calibrator from CALIBRATORS by name, with the shared knobs.
    DtACI has no gamma (it tunes its own); the argument is ignored there."""
    cls = CALIBRATORS[name]
    kw = dict(alpha=alpha, horizon=horizon, n_peds=n_peds, window=window)
    if cls not in (DtACI, MaxDtACI):  # these tune their own gamma
        kw["gamma"] = gamma
    return cls(**kw)


class PartialMaxACI(MaxACI):
    """MaxACI without the horizon-step feedback lag (ours).

    A prediction's max-score only grows as its lookaheads resolve, so
    nothing forces us to wait for the last one. Two changes:

    1. Quantiles: pending predictions contribute their *running* max over
       the lookaheads resolved so far — a lower bound that tightens to the
       true score — so fresh errors move the radii the step they occur.
    2. The dial update gamma * (alpha - miss) is delivered in two prompt
       pieces: +gamma*alpha when a prediction takes its first step, and
       -gamma at the exact step its tube is first violated. Summed over a
       prediction's life this equals MaxACI's single delayed update, so
       the guarantee's fixed point is unchanged; the dial just reacts
       immediately instead of up to horizon steps later.

    Penalties are rate-limited to one -gamma per step; a burst of tube
    violations (a sharp pedestrian turn can break up to horizon in-flight
    tubes within a few steps) queues as debt paid on later steps. Every
    ACI variant assumes at most one dial move per step — that premise is
    what bounds alpha_t near [0, 1] — and delivering a burst all at once
    drives the dial deeply negative, pinning radii at the buffer max for
    many steps. Queueing restores the one-move-per-step invariant; the
    total delivered is unchanged.

    The transient cost: running maxes underestimate final scores, so radii
    lean slightly small until pending predictions finalize; the dial's
    prompt miss penalty is what corrects for that."""

    def _pending(self):
        """(n_pending, n_peds) running max-scores of unresolved predictions."""
        if self.sigma is None:
            return np.empty((0, len(self.alpha_t)))
        sig = np.maximum(self.sigma, 1e-3)[:, None]
        rows = [np.nanmax(errs / sig, axis=0) for _, errs, _ in self.past
                if not np.isnan(errs).all()]
        return np.array(rows) if rows else np.empty((0, len(self.alpha_t)))

    def _quantiles(self):
        pending = self._pending()
        return np.array([np.quantile(s + list(pending[:, n]), np.clip(1 - a, 0, 1))
                         if s or pending.size else np.inf
                         for n, (s, a) in enumerate(zip(self.scores, self.alpha_t))])

    def update(self, prediction, actual):
        """Same interface and displayed-disks grading as MaxACI.update."""
        self.past.insert(0, (prediction,
                             np.full((self.horizon, len(actual)), np.nan),
                             np.zeros((self.horizon, len(actual)), bool)))
        radii = self.radii()
        missed = np.zeros((len(self.past), len(actual)))
        for j, (pred, errs, outcomes) in enumerate(self.past):
            prior = outcomes[:j].any(axis=0)  # tube already violated earlier
            errs[j] = np.linalg.norm(actual - pred[j], axis=1)
            missed[j] = errs[j] > radii[j]
            outcomes[j] = missed[j]
            if not self.frozen:
                if j == 0:
                    self.alpha_t += self.gamma * self.alpha  # budget, paid up front
                self.debt += (errs[j] > radii[j]) & ~prior
        if not self.frozen:
            pay = np.minimum(self.debt, 1.0)  # rate-limit: one -gamma per step
            self.alpha_t -= self.gamma * pay
            self.debt -= pay
        if len(self.past) == self.horizon:  # the oldest is now fully resolved
            _, errs, outcomes = self.past.pop()
            if not self.frozen:
                if not self.sigma_frozen:
                    mean_k = errs.mean(axis=1)
                    self.sigma = mean_k if self.sigma is None else 0.98 * self.sigma + 0.02 * mean_k
                self.miss_log.append(outcomes.any(axis=0).mean())
                score = (errs / np.maximum(self.sigma[:, None], 1e-3)).max(axis=0)
                for n, s in enumerate(score):
                    self.scores[n] = (self.scores[n] + [s])[-self.window:]
        return missed


class MaxDtACI(PartialMaxACI):
    """The lag-free max method (PartialMaxACI) with DtACI's self-tuning gamma —
    the two orthogonal fixes combined: small max-over-horizon disks, no
    horizon-step feedback lag, AND no gamma knob.

    Each pedestrian runs a bank of lag-free max dials, one per candidate gamma,
    and plays their weighted-average level; the weights shift toward the dials
    whose level best matches the realized max-scores (pinball loss), with a
    fixed-share mix back to uniform (Gibbs & Candes 2022, the same aggregation
    as DtACI).

    One deliberate design choice: all dials in a pedestrian's bank grade on the
    SAME operational displayed-disk event (as in MaxACI), not on their own
    hypothetical disks the way union DtACI does. That keeps the max family's
    property that the guarantee is about the exact disks the robot planned
    around — only the step size is being selected. The trade is that DtACI's
    expert-aggregation regret bound is proved for per-dial grading, so here the
    auto-tuning is principled-but-heuristic rather than bound-carrying; the
    per-dial fixed points are all the ACI target, so the blend still converges
    to displayed-tube miscoverage = alpha (see PartialMaxACI)."""

    GAMMA_GRID = DtACI.GAMMA_GRID

    def __init__(self, alpha=0.1, horizon=1, n_peds=1, window=100, span=100):
        super().__init__(alpha=alpha, gamma=0.0, horizon=horizon, n_peds=n_peds, window=window)
        self.gammas = np.array(self.GAMMA_GRID)
        K, t = len(self.gammas), alpha  # max targets alpha directly (no union split)
        self.eta = np.sqrt(3 / span) * np.sqrt(
            (np.log(K * span) + 2) / (t ** 2 * (1 - t) ** 3 + (1 - t) ** 2 * t ** 3))
        self.share = 1 / (2 * span)
        self.alpha_t = np.full((K, n_peds), alpha)   # a dial bank per pedestrian
        self.weights = np.full((K, n_peds), 1 / K)

    def _level(self):
        """Played miscoverage level per pedestrian: the weighted-average dial."""
        return (self.weights * self.alpha_t).sum(axis=0)

    def radii(self):
        if self.sigma is None:  # n_peds = len(scores); alpha_t is now (K, n_peds)
            return np.full((self.horizon, len(self.scores)), np.inf)
        return np.maximum(self.sigma, 1e-3)[:, None] * self._quantiles()[None, :]

    def _pending(self):
        if self.sigma is None:
            return np.empty((0, len(self.scores)))
        sig = np.maximum(self.sigma, 1e-3)[:, None]
        rows = [np.nanmax(errs / sig, axis=0) for _, errs, _ in self.past
                if not np.isnan(errs).all()]
        return np.array(rows) if rows else np.empty((0, len(self.scores)))

    def _quantiles(self):
        pending, level = self._pending(), self._level()
        return np.array([np.quantile(s + list(pending[:, n]), np.clip(1 - level[n], 0, 1))
                         if s or pending.size else np.inf
                         for n, s in enumerate(self.scores)])

    def update(self, prediction, actual):
        """Same interface and displayed-disks grading as PartialMaxACI.update;
        the dial bank replaces the single dial and gains the weight update.
        Penalties are debt-queued (one per step) exactly as in PartialMaxACI —
        the bank's large-gamma dials are what made burst delivery explosive."""
        self.past.insert(0, (prediction,
                             np.full((self.horizon, len(actual)), np.nan),
                             np.zeros((self.horizon, len(actual)), bool)))
        radii = self.radii()
        missed = np.zeros((len(self.past), len(actual)))
        for j, (pred, errs, outcomes) in enumerate(self.past):
            prior = outcomes[:j].any(axis=0)  # tube already violated earlier
            errs[j] = np.linalg.norm(actual - pred[j], axis=1)
            missed[j] = errs[j] > radii[j]
            outcomes[j] = missed[j]
            if not self.frozen:  # lag-free split, applied to every dial in the bank
                if j == 0:
                    self.alpha_t += self.gammas[:, None] * self.alpha  # budget, up front
                self.debt += (errs[j] > radii[j]) & ~prior
        if not self.frozen:
            pay = np.minimum(self.debt, 1.0)  # rate-limit: one -gamma_i per step
            self.alpha_t -= self.gammas[:, None] * pay[None, :]
            self.debt -= pay
        if len(self.past) == self.horizon:  # the oldest is now fully resolved
            _, errs, outcomes = self.past.pop()
            if not self.frozen:
                if not self.sigma_frozen:
                    mean_k = errs.mean(axis=1)
                    self.sigma = mean_k if self.sigma is None else 0.98 * self.sigma + 0.02 * mean_k
                self.miss_log.append(outcomes.any(axis=0).mean())
                score = (errs / np.maximum(self.sigma[:, None], 1e-3)).max(axis=0)
                # DtACI weight update, per pedestrian, on the resolved max-score
                for n, sc in enumerate(score):
                    s = self.scores[n]
                    if s:
                        a = self.alpha_t[:, n]
                        beta = 1 - np.mean(np.array(s) <= sc)  # level that would just cover
                        loss = self.alpha * (beta - a) + np.maximum(0, a - beta)
                        w = self.weights[:, n] * np.exp(-self.eta * loss)
                        w = (1 - self.share) * w + self.share * w.mean()
                        self.weights[:, n] = w / w.sum()
                    self.scores[n] = (s + [sc])[-self.window:]
        return missed


# Union-bound trackers (aci, dtaci) vs max-over-horizon aggregation (max, max+,
# maxdt+); fixed-gamma (aci, max, max+) vs self-tuning gamma (dtaci, maxdt+);
# max+ and maxdt+ also drop the horizon-step feedback lag.
CALIBRATORS = {"aci": ACI, "dtaci": DtACI, "max": MaxACI, "max+": PartialMaxACI,
               "maxdt+": MaxDtACI}
