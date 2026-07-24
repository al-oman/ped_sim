"""HM3D flow model conditioned on Social-LSTM forecasts: identical to
config/hm3d_cvk.py (same 120-scene / 100k-step budget, same futures
conditioning shape) except the predictor — each of the K=4 nearest
pedestrians contributes the Social-LSTM horizon x 2 forecast instead of the
ConstantVelocity one. Requires social_lstm.pt (scripts/train_social_lstm.py).
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
        'loader_kwargs': {'n_scenes': 120, 'eps_per_scene': 10, 'predictor': 'slstm'},

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
