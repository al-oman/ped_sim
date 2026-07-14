"""Collects episodes where the robot is steered by ORCA (the expert) and
trains the flow matching model to imitate it: given the current observation,
generate the robot's next TRAJ_LEN positions, relative to where it is now.

Saves weights and normalization stats to flow.pt, or:

    python train.py <out.pt> <expert_radius>   # e.g. train.py flow_bold.pt 0.25

to train an alternate model off a bolder/shyer expert without overwriting.
"""

import sys

import numpy as np
import torch

from flow_model import TRAJ_LEN, TemporalUnet, condition, flow_loss, sample
from policies import OrcaExpert
from sim import DT, HEIGHT, WALLS, WIDTH, Env, in_wall

EPISODES = 500
EPOCHS = 30
BATCH = 256


def random_start(rng):
    """Random wall-free robot start, so the data covers recovery situations."""
    while True:
        p = rng.uniform([0.5, 0.5], [WIDTH - 1.5, HEIGHT - 0.5])
        if not in_wall(p):
            return p


def collect(expert_radius=0.25):
    trajs, conds = [], []
    rng = np.random.default_rng(999)
    for ep in range(EPISODES):
        env = Env(seed=ep)
        expert = OrcaExpert(goal=(WIDTH - 0.3, HEIGHT / 2), dt=DT, walls=WALLS,
                            radius=expert_radius)
        # Half the episodes start from the usual spot, half from anywhere.
        obs = env.reset() if ep % 2 == 0 else env.reset(robot_start=random_start(rng))
        pos, cs, done = [obs["robot"]], [condition(obs)], False
        while not done:
            obs, _, done = env.step(expert.act(obs))
            pos.append(obs["robot"])
            cs.append(condition(obs))
        pos = np.array(pos)
        for t in range(len(pos) - TRAJ_LEN):  # one training window per step
            trajs.append((pos[t + 1:t + 1 + TRAJ_LEN] - pos[t]).T)  # (2, TRAJ_LEN)
            conds.append(cs[t])
    return np.array(trajs, np.float32), np.array(conds, np.float32)


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "flow.pt"
    expert_radius = float(sys.argv[2]) if len(sys.argv) > 2 else 0.25  # the bold expert
    trajs, conds = collect(expert_radius)
    print(f"collected {len(trajs)} windows from {EPISODES} episodes "
          f"(expert radius {expert_radius}) -> {out}")

    # Normalize: trajectories to unit scale, conditions to zero mean / unit std.
    traj_std = trajs.std()
    cond_mean, cond_std = conds.mean(axis=0), conds.std(axis=0) + 1e-6
    trajs = torch.tensor(trajs / traj_std)
    conds = torch.tensor((conds - cond_mean) / cond_std)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = TemporalUnet(cond_dim=conds.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(EPOCHS):
        losses = []
        for i in torch.randperm(len(trajs)).split(BATCH):
            loss = flow_loss(model, trajs[i].to(device), conds[i].to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        print(f"epoch {epoch + 1}/{EPOCHS}, loss {np.mean(losses):.4f}")

    torch.save({"model": model.state_dict(), "cond_dim": conds.shape[1], "arch": "temporal",
                "traj_std": traj_std, "cond_mean": cond_mean, "cond_std": cond_std,
                "provenance": {"expert_radius": expert_radius, "episodes": EPISODES,
                               "epochs": EPOCHS, "windows": len(trajs)}},
               out)
    print(f"saved {out}")

    # Smoke test: sample 5 trajectories for one training state and compare
    # the final displacement against the expert's.
    gen = sample(model, conds[:1].repeat(5, 1).to(device)).cpu().numpy() * traj_std
    print(f"expert displacement after {TRAJ_LEN} steps: {trajs[0, :, -1].numpy() * traj_std}")
    print(f"model  displacement (5 samples, mean):      {gen[:, :, -1].mean(axis=0)}")
