"""Hyperparameters for the toy-sim flow model, in the style of
SafeFlowMatcher's config/maze2d.py `cfm` entry. Model keys match their
values wherever the tasks share a meaning (dim 32, dim_mults, loss settings,
lr 2e-4, batch 32, grad-accum 2, ema 0.995); horizon / integration steps /
data volume are ours (16-step plans in a small 2D world vs their 256-step
mazes). n_train_steps is set so the total samples seen matches the previous
pipeline's budget (30 epochs x ~50k windows at batch 256).
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
        'loader': 'datasets.SequenceDataset',
        'loader_kwargs': {'episodes': 500, 'expert_radius': 0.25},

        ## training
        'n_train_steps': 25000,
        'batch_size': 32,
        'learning_rate': 2e-4,
        'gradient_accumulate_every': 2,
        'ema_decay': 0.995,
        'log_freq': 1000,
        'device': 'cpu',
    },

}
