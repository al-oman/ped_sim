"""Hyperparameters for the HM3D flow model. Model/training keys match
config/toy.py (which in turn matches SafeFlowMatcher's maze2d cfm entry);
only the dataset loader differs — Social-HM3D episodes with the RvoPolicy
(A*-guided RVO2) expert, carrot goal-conditioning, K=4 nearest pedestrians.
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
        'loader_kwargs': {'n_scenes': 120, 'eps_per_scene': 10},

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
