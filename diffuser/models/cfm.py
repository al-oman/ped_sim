"""Flow matching over the vendored Diffuser backbone — the counterpart of
SafeFlowMatcher's diffuser/models/cfm.py (CFM class), with two documented
deviations:

- Conditioning: the official implementations condition by inpainting values
  into the generated trajectory (helpers.apply_conditioning), which works
  because everything they condition on is itself a trajectory state. Our
  observation (robot position + pedestrian states) is not part of the
  generated trajectory, so it enters through the time embedding instead
  (_CondTimeMLP) — the single functional change to the vendored network,
  made from outside it.
- The flow-matching math is inlined instead of imported from torchcfm: with
  their sigma=0 it is exactly x_t = t x1 + (1-t) x0, target x1 - x0,
  uniform t, and inlining avoids torchcfm's dependency chain.

Safety corrections (the CBF brake and the annealed projection) sit inside
the Euler integration loop, in the same slot where SafeFlowMatcher's
p_sample_loop_ode_planning applies its CBF correction. Their NeuralODE /
diffusion-schedule compatibility buffers (marked "Not important for CFM" in
their code) are dropped.

Flow matching in one paragraph: pick noise x0 ~ N(0,1) and a real trajectory
x1, blend them as x_t = t x1 + (1-t) x0, and train the network v(x_t, t, cond)
to predict x1 - x0 — the straight-line velocity pointing from the noise to
the data. To generate, start at noise (t=0) and integrate v to t=1.
"""

import torch
import torch.nn as nn

from .helpers import Losses


class _CondTimeMLP(nn.Module):
    """Wraps the vendored network's time embedding so the observation vector
    joins it — the embedding side-door described above."""

    def __init__(self, time_mlp, cond_dim, dim):
        super().__init__()
        self.time_mlp = time_mlp
        self.cond_mlp = nn.Sequential(nn.Linear(cond_dim, dim * 4), nn.Mish(),
                                      nn.Linear(dim * 4, dim))
        self.cond = None

    def forward(self, t):
        return self.time_mlp(t) + self.cond_mlp(self.cond)


def _cbf(x, v, centers, radii, kappa, K):
    """CBF-style brake, applied to the velocity field: cap each waypoint's
    speed toward each disk the closer it gets (inside a disk, push outward).
    Edits v in place. x, v are (B, H, 2); disk (k, n) constrains waypoint k."""
    for n in range(centers.shape[1]):
        d = x[:, :K] - centers[:K, n]
        dist = d.norm(dim=2, keepdim=True).clamp(min=1e-6)
        h = dist - radii[:K, n, None]                   # signed distance to disk n
        toward = (v[:, :K] * (d / dist)).sum(2, keepdim=True)
        fix = (-kappa * h - toward).clamp(min=0)        # extra outward speed needed
        v[:, :K] += fix * d / dist


def _project(x, centers, radii, scale, K):
    """Push any waypoint inside a disk (scaled by `scale`) to its boundary."""
    for n in range(centers.shape[1]):
        d = x[:, :K] - centers[:K, n]
        dist = d.norm(dim=2, keepdim=True).clamp(min=1e-6)
        r = scale * radii[:K, n, None]
        x[:, :K] = torch.where(dist < r, centers[:K, n] + d / dist * r, x[:, :K])


class CFM(nn.Module):

    def __init__(self, model, horizon, observation_dim=2, action_dim=0, cond_dim=0,
                 n_timesteps=10, loss_type='l2', action_weight=1.0, loss_discount=1.0,
                 loss_weights=None):
        super().__init__()
        self.horizon = horizon
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.transition_dim = observation_dim + action_dim
        self.model = model
        if cond_dim:  # install the embedding side-door
            dim = model.time_mlp[0].dim  # SinusoidalPosEmb width
            model.time_mlp = _CondTimeMLP(model.time_mlp, cond_dim, dim)
        self.n_timesteps = int(n_timesteps)

        loss_weights = self.get_loss_weights(action_weight, loss_discount, loss_weights)
        self.loss_fn = Losses[loss_type](loss_weights, self.action_dim)

    def get_loss_weights(self, action_weight, discount, weights_dict):
        '''
            sets loss coefficients for trajectory

            action_weight   : float
                coefficient on first action loss
            discount   : float
                multiplies t^th timestep of trajectory loss by discount**t
            weights_dict    : dict
                { i: c } multiplies dimension i of observation loss by c
        '''
        self.action_weight = action_weight

        dim_weights = torch.ones(self.transition_dim, dtype=torch.float32)

        # set loss coefficients for dimensions of observation
        if weights_dict is None: weights_dict = {}
        for ind, w in weights_dict.items():
            dim_weights[self.action_dim + ind] *= w

        # decay loss with trajectory timestep: discount**t
        discounts = discount ** torch.arange(self.horizon, dtype=torch.float)
        discounts = discounts / discounts.mean()
        loss_weights = torch.einsum('h,t->ht', discounts, dim_weights)

        # manually set a0 weight
        loss_weights[0, :self.action_dim] = action_weight
        return loss_weights

    def model_forward(self, x, cond, t):
        """The vendored net's forward(x, cond, time) signature, with the
        observation routed through the side-door instead of the cond arg."""
        if isinstance(self.model.time_mlp, _CondTimeMLP):
            self.model.time_mlp.cond = cond
        return self.model(x, None, t)

    #------------------------------------------ sampling ------------------------------------------#

    @torch.no_grad()
    def conditional_sample(self, cond, disks=None, horizon=None, kappa=8.0, hard=None,
                           init=None, tau=0.0, project=True):
        """Euler-integrate the learned velocity field from noise to
        trajectories (B, horizon, 2) — uniform scheduling, as in
        SafeFlowMatcher's p_sample_loop_ode_planning, with the safety
        correction applied per step.

        disks: optional keep-out regions as (centers, radii), centers
        (K, N, 2) and radii (K, N), in the model's normalized robot-relative
        coordinates — waypoint k must end up outside disk (k, n). Enforced
        two ways: a CBF brake on the velocity while integrating (gentle,
        lets the model's own field route around the disks), and a projection
        whose disks anneal from zero to full size — the final full-size
        passes make every returned trajectory satisfy the constraints by
        construction. A disk with negative radius never binds (callers use
        -1 to disable one).

        hard: only constrain the first `hard` waypoints (default: all). In
        receding-horizon use only the executed prefix of the plan needs a
        certificate; constraining far waypoints against their huge
        uncertainty disks strangles the plan for no safety benefit.

        init/tau: warm start (Janner et al. 2022). Begin integration at flow
        time tau from the blend (1-tau)*noise + tau*init, where init is a
        previous plan in normalized coordinates. tau=0 is a cold start from
        pure noise; higher tau keeps the result closer to init — the
        consistency/reactivity knob for receding-horizon replanning. Also
        cheaper: only the remaining (1-tau) of the schedule is integrated."""
        horizon = horizon or self.horizon
        x = torch.randn(len(cond), horizon, self.transition_dim, device=self.device)
        first = 0
        if init is not None and tau > 0:
            x = (1 - tau) * x + tau * init
            first = min(int(tau * self.n_timesteps), self.n_timesteps - 1)
        if disks is not None:
            centers = torch.as_tensor(disks[0], dtype=torch.float32, device=self.device)
            radii = torch.as_tensor(disks[1], dtype=torch.float32, device=self.device)
            K = min(len(centers), horizon, hard or horizon)
        for i in range(first, self.n_timesteps):
            t = torch.full((len(cond),), i / self.n_timesteps, device=self.device)
            v = self.model_forward(x, cond, t)
            if disks is not None and kappa:  # kappa=0 disables the CBF brake (ablations)
                _cbf(x, v, centers, radii, kappa, K)
            x = x + v / self.n_timesteps
            if disks is not None and project:
                _project(x, centers, radii, (i + 1) / self.n_timesteps, K)
        if disks is not None and project:
            for _ in range(30):  # settle: leaving one disk can push a waypoint into another
                _project(x, centers, radii, 1.0, K)
        return x

    @property
    def device(self):
        return next(self.parameters()).device

    #------------------------------------------ training ------------------------------------------#

    def loss(self, x, cond):
        x, cond = x.to(self.device), cond.to(self.device)
        # torchcfm's ConditionalFlowMatcher(sigma=0), inlined (see header):
        x0 = torch.randn_like(x)
        t = torch.rand(len(x), device=x.device)
        xt = t[:, None, None] * x + (1 - t[:, None, None]) * x0
        vt = self.model_forward(xt, cond, t)
        loss, info = self.loss_fn(vt, x - x0)
        if self.action_dim == 0:
            info = {}  # their a0_loss diagnostic is empty (NaN) without actions
        return loss, info
