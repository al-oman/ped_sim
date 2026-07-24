"""Human trajectory predictors. A predictor is anything with a
predict(peds) -> (horizon, N, 2) method giving each pedestrian's predicted
position 1..horizon steps ahead, where peds is the (N, 4) array of
[x, y, vx, vy] from the observation. Stateful predictors (SocialLSTM) also
have reset(), to be called at each episode start."""

import numpy as np


class ConstantVelocity:
    """Assumes each pedestrian keeps its current velocity."""

    def __init__(self, dt, horizon):
        self.dt, self.horizon = dt, horizon

    def predict(self, peds):
        steps = np.arange(1, self.horizon + 1)[:, None, None]  # (horizon, 1, 1)
        return peds[:, :2] + steps * self.dt * peds[:, 2:]


class SocialLSTM:
    """Social-LSTM (Alahi et al. 2016) rollout, wrapping the vendored
    quancore/social-lstm SocialModel (trained by scripts/train_social_lstm.py).

    Stateful: keeps the last obs_len observed positions per pedestrian —
    call reset() at each episode start (predict() also self-resets if the
    crowd size changes). Until obs_len frames have been seen, the missing
    history is back-extrapolated with the current velocity, so early
    predictions degrade gracefully toward constant-velocity.

    predict() returns the deterministic mean rollout (mux, muy fed back for
    horizon steps): ACI needs a point prediction, and it is exactly that
    prediction's residuals the disks calibrate. Coordinates follow the
    upstream training scheme: inputs vectorized per pedestrian (position
    minus that pedestrian's first history frame), social grids on absolute
    positions."""

    def __init__(self, dt, horizon, path="social_lstm.pt"):
        import copy

        import torch

        from vendor_social_lstm.model import SocialModel
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        self.args = copy.deepcopy(ckpt["args"])
        self.args.seq_length = 1  # frame-at-a-time inference (their infer mode)
        self.net = SocialModel(self.args)
        self.net.load_state_dict(ckpt["model"])
        self.net.eval()
        self.dt, self.horizon, self.obs_len = dt, horizon, ckpt["obs_len"]
        self.history = []

    def reset(self):
        self.history = []

    def _step(self, frame_abs, frame_vec, h, c, N):
        """One SocialModel step: absolute frame for the social grid,
        vectorized frame as the input."""
        import torch

        from vendor_social_lstm.grid import getGridMask
        grid = torch.from_numpy(getGridMask(
            torch.tensor(frame_abs, dtype=torch.float32), self.args.dims, N,
            self.args.neighborhood_size, self.args.grid_size)).float()
        x = torch.tensor(frame_vec, dtype=torch.float32).view(1, N, 2)
        return self.net(x, [grid], h, c, [list(range(N))], [N], None,
                        {i: i for i in range(N)})

    def predict(self, peds):
        import torch

        from vendor_social_lstm.helper import getCoef
        N = len(peds)
        if N == 0:
            return np.zeros((self.horizon, 0, 2))
        if self.history and len(self.history[0]) != N:
            self.reset()  # crowd size changed under us: stale history
        self.history = (self.history + [peds[:, :2].copy()])[-self.obs_len:]
        hist = list(self.history)
        while len(hist) < self.obs_len:  # back-extrapolate the missing past
            hist.insert(0, hist[0] - self.dt * peds[:, 2:])
        first = hist[0]  # per-ped origin of the vectorized coordinates
        h = torch.zeros(N, self.args.rnn_size)
        c = torch.zeros(N, self.args.rnn_size)
        with torch.no_grad():
            for frame in hist:  # warm the hidden states on the observed past
                out, h, c = self._step(frame, frame - first, h, c, N)
            preds, cur = [], None
            for _ in range(self.horizon):  # mean rollout, fed back
                mux, muy, *_ = getCoef(out)
                cur = torch.stack([mux[0], muy[0]], dim=1).numpy()
                preds.append(cur + first)
                out, h, c = self._step(cur + first, cur, h, c, N)
        return np.stack(preds)
