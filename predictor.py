"""Human trajectory predictors. A predictor is anything with a
predict(peds) -> (horizon, N, 2) method giving each pedestrian's predicted
position 1..horizon steps ahead, where peds is the (N, 4) array of
[x, y, vx, vy] from the observation."""

import numpy as np


class ConstantVelocity:
    """Assumes each pedestrian keeps its current velocity."""

    def __init__(self, dt, horizon):
        self.dt, self.horizon = dt, horizon

    def predict(self, peds):
        steps = np.arange(1, self.horizon + 1)[:, None, None]  # (horizon, 1, 1)
        return peds[:, :2] + steps * self.dt * peds[:, 2:]
