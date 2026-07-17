"""Trains a flow model. Mirrors SafeFlowMatcher's scripts/train.py
(dataset -> model -> CFM -> Trainer), reading a plain config dict from
config/<name>.py instead of their Parser/Config pickle machinery.

Saves weights (raw + EMA) and normalization stats to <out.pt>:

    python scripts/train.py [out.pt] [--config=toy|hm3d] [--device=mps]

Extra key=value args override the config's loader_kwargs, e.g.
    python scripts/train.py flow_bold.pt expert_radius=0.5
"""

import importlib
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root

import numpy as np
import torch

import diffuser.datasets
from diffuser.models import CFM, TemporalUnet
from diffuser.utils import Trainer, device_arg

#-----------------------------------------------------------------------------#
#----------------------------------- setup -----------------------------------#
#-----------------------------------------------------------------------------#

config = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--config=")), "toy")
args = importlib.import_module(f"config.{config}").base['diffusion']
device = device_arg(args['device'])
positional = [a for a in sys.argv[1:] if not a.startswith("--") and "=" not in a]
out = positional[0] if positional else "flow.pt"
def _num(v):
    return int(v) if v.lstrip("-").isdigit() else float(v)

loader_kwargs = dict(args['loader_kwargs'],
                     **{k: _num(v) for k, v in
                        (a.split("=", 1) for a in sys.argv[1:] if "=" in a and not a.startswith("--"))})

#-----------------------------------------------------------------------------#
#---------------------------------- dataset ----------------------------------#
#-----------------------------------------------------------------------------#

t0 = time.time()
loader = getattr(diffuser.datasets, args['loader'].split(".")[-1])
dataset = loader(horizon=args['horizon'], **loader_kwargs)
print(f"collected {len(dataset)} windows ({args['loader']}, {loader_kwargs}) "
      f"in {time.time() - t0:.0f}s -> {out}")

#-----------------------------------------------------------------------------#
#------------------------------ model & trainer ------------------------------#
#-----------------------------------------------------------------------------#

model = TemporalUnet(horizon=args['horizon'], transition_dim=2, cond_dim=0,
                     dim=args['dim'], dim_mults=args['dim_mults'])

diffusion = CFM(model, horizon=args['horizon'], cond_dim=dataset.cond_dim,
                n_timesteps=args['n_diffusion_steps'], loss_type=args['loss_type'],
                action_weight=args['action_weight'], loss_discount=args['loss_discount'],
                loss_weights=args['loss_weights']).to(device)

trainer = Trainer(diffusion, dataset, train_batch_size=args['batch_size'],
                  train_lr=args['learning_rate'],
                  gradient_accumulate_every=args['gradient_accumulate_every'],
                  ema_decay=args['ema_decay'], log_freq=args['log_freq'])

#-----------------------------------------------------------------------------#
#--------------------------------- main loop ---------------------------------#
#-----------------------------------------------------------------------------#

t0 = time.time()
trainer.train(n_train_steps=args['n_train_steps'])
dt = time.time() - t0
print(f"trained {args['n_train_steps']} steps on {device} in {dt:.0f}s")

torch.save({"step": trainer.step,
            "model": trainer.model.state_dict(),
            "ema": trainer.ema_model.state_dict(),
            "arch": "cfm", "cond_dim": dataset.cond_dim, "horizon": args['horizon'],
            "traj_std": dataset.traj_normalizer.std,
            "cond_mean": dataset.cond_normalizer.means,
            "cond_std": dataset.cond_normalizer.stds,
            "provenance": {"config_name": config, "config": args,
                           "loader_kwargs": loader_kwargs, "windows": len(dataset)}},
           out)
print(f"saved {out}")

# Smoke test: sample 5 trajectories for one training state and compare
# the final displacement against the expert's (EMA weights, as at test time).
ema = trainer.ema_model.to(device)
cond = dataset.conditions[:1].repeat(5, 1).to(device)
gen = ema.conditional_sample(cond).cpu().numpy() * dataset.traj_normalizer.std
expert = dataset.trajectories[0, -1].numpy() * dataset.traj_normalizer.std
print(f"expert displacement after {args['horizon']} steps: {expert}")
print(f"model  displacement (5 samples, mean):      {gen[:, -1].mean(axis=0)}")
