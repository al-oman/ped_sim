"""HM3D flow model with futures conditioning: identical to config/hm3d.py
(same 120-scene / 100k-step budget, for a controlled A/B against the
state-conditioned hm3d_flow.pt) except the conditioning — each of the K=4
nearest pedestrians contributes its ConstantVelocity horizon x 2 forecast,
raw-flattened, instead of (pos, vel). See diffuser/datasets/hm3d.py.
"""

base = {

    'diffusion': {
        ## model
        'model': 'models.TemporalUnet',
        'diffusion': 'models.CFM',
        'horizon': 16,
        'n_diffusion_steps': 10,   # Euler steps at sample time
        'dim': 32,
        'dim_mults': (1, 2, 4, 8),
        'action_weight': 1,
        'loss_weights': None,
        'loss_discount': 1,
        'loss_type': 'l2',

        ## dataset
        'loader': 'datasets.HM3DSequenceDataset',
        'loader_kwargs': {'n_scenes': 120, 'eps_per_scene': 10, 'predictor': 'cv'},

        ## training
        'n_train_steps': 100000,
        'batch_size': 32,
        'learning_rate': 2e-4,
        'gradient_accumulate_every': 2,
        'ema_decay': 0.995,
        'log_freq': 1000,
        'device': 'cpu',
    },

}
