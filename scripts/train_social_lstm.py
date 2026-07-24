"""Trains the Social-LSTM pedestrian predictor on crowd trajectories from the
Social-HM3D scenes, using the vendored quancore/social-lstm model verbatim
(vendor_social_lstm/). Their hyperparameter defaults are kept (rnn 128,
embedding 64, grid 4, grad clip 10, Adagrad + weight decay 5e-4, dropout 0.5)
except: seq_length 24 = 8 observed + 16 predicted (their 8+12) to match the
flow model's horizon, and neighborhood_size — their 32 is in ETH/UCY pixel
units; with dims=[1, 1] the grid box spans neighborhood_size*2 metres, so 2.0
gives a 4 m social neighborhood.

One deliberate fix to their train.py: it grades the output of frame t against
the position AT frame t (a known target-alignment bug — their own inference
consumes outputs as next-step positions), which trains an identity map. We
grade outputs[:-1] against x_seq[1:], the standard Social-LSTM objective.

Data: WaypointCrowd rollouts (humans only — they are robot-blind, so the
robot's absence changes nothing), cut into overlapping 24-frame sequences.

    python scripts/train_social_lstm.py [out.pt] [n_scenes] [epochs]
"""

import os
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root

import numpy as np
import torch

from vendor_social_lstm.grid import getSequenceGridMask
from vendor_social_lstm.helper import Gaussian2DLikelihood, vectorize_seq
from vendor_social_lstm.model import SocialModel

OBS_LEN, PRED_LEN = 8, 16
DT = 0.1
STRIDE = 8          # window stride when cutting rollouts into sequences
ROLLOUT_STEPS = 400

args = SimpleNamespace(
    input_size=2, output_size=5, rnn_size=128, embedding_size=64,
    seq_length=OBS_LEN + PRED_LEN, grid_size=4, neighborhood_size=2.0,
    dims=[1, 1],  # unit dims -> neighborhood box spans 2*neighborhood_size m
    dropout=0.5, grad_clip=10., lambda_param=0.0005, maxNumPeds=27,
    use_cuda=False, gru=False)


def collect(n_scenes, seed=0):
    """(T, N, 2) crowd-only rollouts from the train split, one per episode."""
    from hm3d import WaypointCrowd
    from hm3d_eval import scene_episodes, scene_index

    rng = np.random.default_rng(seed)
    idx = scene_index("data", "/Users/alexoman/workspaces/diffusion/"
                              "socialnav_map_gen/pointnav/social-hm3d/train")
    scenes = list(idx.items())
    rng.shuffle(scenes)
    far = np.array([1e6, 1e6])  # robot parked out of existence (crowd is blind anyway)
    rollouts = []
    for scene, (maps, ep_files) in scenes[:n_scenes]:
        pairs = [(g, e) for g, e in scene_episodes(maps, ep_files)
                 if len(e["humans"]) >= 2]  # social pooling needs neighbors
        for i in rng.choice(len(pairs), min(2, len(pairs)), replace=False) if pairs else []:
            grid, ep = pairs[i]
            crowd = WaypointCrowd(ep["humans"], grid, DT)
            frames = [crowd.positions()]
            for _ in range(ROLLOUT_STEPS):
                crowd.step(far)
                frames.append(crowd.positions())
            rollouts.append(np.array(frames, np.float32))
    return rollouts


def sequences(rollouts):
    """Cut rollouts into (seq_length, N, 2) training sequences."""
    out = []
    for r in rollouts:
        for t in range(0, len(r) - args.seq_length, STRIDE):
            out.append(r[t:t + args.seq_length])
    return out


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "social_lstm.pt"
    n_scenes = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 5

    t0 = time.time()
    rollouts = collect(n_scenes)
    seqs = sequences(rollouts)
    rng = np.random.default_rng(1)
    rng.shuffle(seqs)
    n_val = max(1, len(seqs) // 20)
    val, seqs = seqs[:n_val], seqs[n_val:]
    print(f"{len(rollouts)} rollouts -> {len(seqs)} train / {len(val)} val sequences "
          f"in {time.time() - t0:.0f}s")

    net = SocialModel(args)
    optimizer = torch.optim.Adagrad(net.parameters(), weight_decay=args.lambda_param)

    def run_seq(seq, train=True):
        """Forward one sequence, return the (fixed-target) NLL loss."""
        N = seq.shape[1]
        pedlist = [list(range(N))] * args.seq_length
        look_up = {i: i for i in range(N)}
        grid_seq = getSequenceGridMask(torch.tensor(seq), args.dims, pedlist,
                                       args.neighborhood_size, args.grid_size, False)
        x_seq, _ = vectorize_seq(torch.tensor(seq), pedlist, look_up)
        h = torch.zeros(N, args.rnn_size)
        c = torch.zeros(N, args.rnn_size)
        outputs, _, _ = net(x_seq, grid_seq, h, c, pedlist, [N] * args.seq_length,
                            None, look_up)
        # outputs[t] is the model's guess for position t+1 (see module docstring)
        return Gaussian2DLikelihood(outputs[:-1], x_seq[1:], pedlist[1:], look_up)

    for epoch in range(epochs):
        t0, losses = time.time(), []
        net.train()
        for seq in seqs:
            optimizer.zero_grad()
            loss = run_seq(seq)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)
            optimizer.step()
            losses.append(loss.item())
        net.eval()
        with torch.no_grad():
            val_loss = np.mean([run_seq(s, train=False).item() for s in val])
        print(f"epoch {epoch + 1}/{epochs}: train NLL {np.mean(losses):.3f}, "
              f"val NLL {val_loss:.3f}  [{time.time() - t0:.0f}s]")

    torch.save({"model": net.state_dict(), "args": args, "obs_len": OBS_LEN,
                "provenance": {"n_scenes": n_scenes, "epochs": epochs,
                               "sequences": len(seqs)}}, out)
    print(f"saved {out}")
