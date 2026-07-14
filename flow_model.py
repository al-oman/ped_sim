"""Conditional flow matching model that generates short robot trajectories.

A trajectory is a (2, TRAJ_LEN) array: x/y are channels, time runs along the
length. The model is a small 1D UNet conditioned on the flow time t and on
the current observation (robot position + pedestrian states).

Flow matching in one paragraph: pick noise z ~ N(0,1) and a real trajectory
x1, blend them as x_t = (1-t) z + t x1, and train the network v(x_t, t, cond)
to predict x1 - z — the straight-line velocity pointing from the noise to the
data. To generate, start at noise (t=0) and integrate v to t=1.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

TRAJ_LEN = 16  # generated trajectory length, steps


def condition(obs):
    """Flat conditioning vector: robot position + pedestrians relative to it."""
    peds = obs["peds"].copy()
    peds[:, :2] -= obs["robot"]
    return np.concatenate([obs["robot"], peds.ravel()])


def time_features(t):
    """Sinusoidal features of the flow time, so the net can resolve t sharply."""
    freqs = 2 ** torch.arange(8, device=t.device) * torch.pi
    return torch.cat([torch.sin(t * freqs), torch.cos(t * freqs)], dim=1)


class Block(nn.Module):
    """Two convs, with the (t, condition) embedding added in between."""

    def __init__(self, c_in, c_out, emb_dim):
        super().__init__()
        self.conv1 = nn.Conv1d(c_in, c_out, 3, padding=1)
        self.conv2 = nn.Conv1d(c_out, c_out, 3, padding=1)
        self.norm1 = nn.GroupNorm(4, c_out)
        self.norm2 = nn.GroupNorm(4, c_out)
        self.emb = nn.Linear(emb_dim, c_out)

    def forward(self, x, emb):
        h = F.silu(self.norm1(self.conv1(x)))
        h = h + self.emb(emb)[:, :, None]  # broadcast over the time axis
        return F.silu(self.norm2(self.conv2(h)))


class UNet1D(nn.Module):
    def __init__(self, cond_dim, ch=64, emb_dim=128):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(cond_dim + 16, emb_dim), nn.SiLU(),
                                   nn.Linear(emb_dim, emb_dim))
        self.enc1 = Block(2, ch, emb_dim)             # length 16
        self.enc2 = Block(ch, 2 * ch, emb_dim)        # length 8
        self.enc3 = Block(2 * ch, 4 * ch, emb_dim)    # length 4
        self.mid = Block(4 * ch, 4 * ch, emb_dim)     # length 2
        self.dec3 = Block(8 * ch, 2 * ch, emb_dim)    # length 4, sees enc3's output
        self.dec2 = Block(4 * ch, ch, emb_dim)        # length 8, sees enc2's output
        self.dec1 = Block(2 * ch, ch, emb_dim)        # length 16, sees enc1's output
        self.out = nn.Conv1d(ch, 2, 1)

    def forward(self, x, t, cond):
        emb = self.embed(torch.cat([time_features(t), cond], dim=1))
        h1 = self.enc1(x, emb)
        h2 = self.enc2(F.avg_pool1d(h1, 2), emb)
        h3 = self.enc3(F.avg_pool1d(h2, 2), emb)
        m = self.mid(F.avg_pool1d(h3, 2), emb)
        d3 = self.dec3(torch.cat([F.interpolate(m, scale_factor=2), h3], dim=1), emb)
        d2 = self.dec2(torch.cat([F.interpolate(d3, scale_factor=2), h2], dim=1), emb)
        d1 = self.dec1(torch.cat([F.interpolate(d2, scale_factor=2), h1], dim=1), emb)
        return self.out(d1)


class ResidualTemporalBlock(nn.Module):
    """Diffuser's building block: two kernel-5 convs (GroupNorm + Mish) with
    the (t, condition) embedding added in between, and a residual connection."""

    def __init__(self, c_in, c_out, emb_dim):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv1d(c_in, c_out, 5, padding=2),
                                   nn.GroupNorm(8, c_out), nn.Mish())
        self.conv2 = nn.Sequential(nn.Conv1d(c_out, c_out, 5, padding=2),
                                   nn.GroupNorm(8, c_out), nn.Mish())
        self.emb = nn.Sequential(nn.Mish(), nn.Linear(emb_dim, c_out))
        self.skip = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x, emb):
        h = self.conv1(x) + self.emb(emb)[:, :, None]
        return self.conv2(h) + self.skip(x)


class TemporalUnet(nn.Module):
    """The Diffuser backbone (Janner et al. 2022), as inherited by SafeDiffuser
    and SafeFlowMatcher: base dim 32, channel multipliers (1, 2, 4, 8),
    kernel-5 residual temporal blocks, Mish activations (~4M parameters).
    Same interface as UNet1D; the observation is folded into the flow-time
    embedding, as in the smaller model."""

    def __init__(self, cond_dim, dim=32, mults=(1, 2, 4, 8), emb_dim=128):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(cond_dim + 16, emb_dim), nn.Mish(),
                                   nn.Linear(emb_dim, emb_dim))
        dims = [2] + [dim * m for m in mults]  # [2, 32, 64, 128, 256]
        self.downs = nn.ModuleList()           # length 16 -> 8 -> 4 -> 2
        for i in range(len(mults)):
            self.downs.append(nn.ModuleList([
                ResidualTemporalBlock(dims[i], dims[i + 1], emb_dim),
                ResidualTemporalBlock(dims[i + 1], dims[i + 1], emb_dim)]))
        self.mid1 = ResidualTemporalBlock(dims[-1], dims[-1], emb_dim)
        self.mid2 = ResidualTemporalBlock(dims[-1], dims[-1], emb_dim)
        self.ups = nn.ModuleList()              # length 2 -> 4 -> 8 -> 16
        for i in reversed(range(len(mults) - 1)):
            self.ups.append(nn.ModuleList([
                ResidualTemporalBlock(dims[i + 2] + dims[i + 1], dims[i + 1], emb_dim),
                ResidualTemporalBlock(dims[i + 1], dims[i + 1], emb_dim)]))
        self.out = nn.Conv1d(dims[1], 2, 1)

    def forward(self, x, t, cond):
        emb = self.embed(torch.cat([time_features(t), cond], dim=1))
        skips = []
        for i, (block1, block2) in enumerate(self.downs):
            x = block2(block1(x, emb), emb)
            if i < len(self.downs) - 1:
                skips.append(x)
                x = F.avg_pool1d(x, 2)
        x = self.mid2(self.mid1(x, emb), emb)
        for block1, block2 in self.ups:
            x = torch.cat([F.interpolate(x, scale_factor=2), skips.pop()], dim=1)
            x = block2(block1(x, emb), emb)
        return self.out(x)


class _CondTimeMLP(nn.Module):
    """Wraps the vendored network's time embedding so the observation vector
    joins it — the single functional change to the baseline network, made
    from outside the vendored code."""

    def __init__(self, time_mlp, cond_dim, dim):
        super().__init__()
        self.time_mlp = time_mlp
        self.cond_mlp = nn.Sequential(nn.Linear(cond_dim, dim * 4), nn.Mish(),
                                      nn.Linear(dim * 4, dim))
        self.cond = None

    def forward(self, t):
        return self.time_mlp(t) + self.cond_mlp(self.cond)


class BaselineUnet(nn.Module):
    """The exact SafeFlowMatcher / SafeDiffuser / Diffuser backbone, vendored
    verbatim in vendor_diffuser/ (base dim 32, mults (1, 2, 4, 8), kernel-5
    residual blocks, sinusoidal time embedding on the raw flow time — all
    baseline behavior). Adapted only at the edges: trajectories are
    channel-first here vs channel-last there, and the observation vector is
    added into the time embedding (the baseline conditions by inpainting
    states it generates; our pedestrians aren't generated, so they enter
    through the embedding instead).

    Our flow_loss/sample already match SafeFlowMatcher's generative setup
    exactly: their cfm.py uses torchcfm's ConditionalFlowMatcher(sigma=0) —
    x_t = (1-t) noise + t data, target = data - noise, uniform t — integrated
    with uniform-grid Euler, with safety corrections applied to the state
    after each step."""

    def __init__(self, cond_dim, dim=32):
        super().__init__()
        from vendor_diffuser.temporal import TemporalUnet as DiffuserUnet
        self.net = DiffuserUnet(horizon=TRAJ_LEN, transition_dim=2, cond_dim=0, dim=dim)
        self.net.time_mlp = _CondTimeMLP(self.net.time_mlp, cond_dim, dim)

    def forward(self, x, t, cond):
        self.net.time_mlp.cond = cond
        return self.net(x.transpose(1, 2), None, t[:, 0]).transpose(1, 2)


def flow_loss(model, traj, cond):
    z = torch.randn_like(traj)
    t = torch.rand(len(traj), 1, device=traj.device)
    x_t = (1 - t[:, :, None]) * z + t[:, :, None] * traj
    return F.mse_loss(model(x_t, t, cond), traj - z)


def _cbf(x, v, centers, radii, kappa, K):
    """CBF-style brake, applied to the velocity field: cap each waypoint's
    speed toward each disk the closer it gets (inside a disk, push outward).
    Edits v in place. x, v are (B, 2, L); disk (k, n) constrains waypoint k."""
    xw, vw = x.transpose(1, 2), v.transpose(1, 2)  # (B, L, 2) views
    for n in range(centers.shape[1]):
        d = xw[:, :K] - centers[:K, n]
        dist = d.norm(dim=2, keepdim=True).clamp(min=1e-6)
        h = dist - radii[:K, n, None]                   # signed distance to disk n
        toward = (vw[:, :K] * (d / dist)).sum(2, keepdim=True)
        fix = (-kappa * h - toward).clamp(min=0)        # extra outward speed needed
        vw[:, :K] += fix * d / dist


def _project(x, centers, radii, scale, K):
    """Push any waypoint inside a disk (scaled by `scale`) to its boundary."""
    xw = x.transpose(1, 2)
    for n in range(centers.shape[1]):
        d = xw[:, :K] - centers[:K, n]
        dist = d.norm(dim=2, keepdim=True).clamp(min=1e-6)
        r = scale * radii[:K, n, None]
        xw[:, :K] = torch.where(dist < r, centers[:K, n] + d / dist * r, xw[:, :K])


@torch.no_grad()
def sample(model, cond, disks=None, steps=10, kappa=8.0, hard=None, init=None, tau=0.0):
    """Euler-integrate the learned velocity field from noise to trajectories.

    disks: optional keep-out regions as (centers, radii), centers (K, N, 2)
    and radii (K, N), in the model's normalized robot-relative coordinates —
    waypoint k must end up outside disk (k, n). Enforced two ways: a CBF
    brake on the velocity while integrating (gentle, lets the model's own
    field route around the disks), and a projection whose disks anneal from
    zero to full size — the final full-size passes make every returned
    trajectory satisfy the constraints by construction. A disk with negative
    radius never binds (callers use -1 to disable one).

    hard: only constrain the first `hard` waypoints (default: all). In
    receding-horizon use only the executed prefix of the plan needs a
    certificate; constraining far waypoints against their huge uncertainty
    disks strangles the plan for no safety benefit.

    init/tau: warm start (Janner et al. 2022). Begin integration at flow time
    tau from the blend (1-tau)*noise + tau*init, where init is a previous
    plan in normalized coordinates. tau=0 is a cold start from pure noise;
    higher tau keeps the result closer to init — the consistency/reactivity
    knob for receding-horizon replanning. Also cheaper: only the remaining
    (1-tau) of the schedule is integrated."""
    x = torch.randn(len(cond), 2, TRAJ_LEN, device=cond.device)
    first = 0
    if init is not None and tau > 0:
        x = (1 - tau) * x + tau * init
        first = min(int(tau * steps), steps - 1)
    if disks is not None:
        centers = torch.as_tensor(disks[0], dtype=torch.float32, device=cond.device)
        radii = torch.as_tensor(disks[1], dtype=torch.float32, device=cond.device)
        K = min(len(centers), TRAJ_LEN, hard or TRAJ_LEN)
    for i in range(first, steps):
        t = torch.full((len(cond), 1), i / steps, device=cond.device)
        v = model(x, t, cond)
        if disks is not None:
            _cbf(x, v, centers, radii, kappa, K)
        x = x + v / steps
        if disks is not None:
            _project(x, centers, radii, (i + 1) / steps, K)
    if disks is not None:
        for _ in range(30):  # settle: leaving one disk can push a waypoint into another
            _project(x, centers, radii, 1.0, K)
    return x
