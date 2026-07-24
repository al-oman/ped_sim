"""MP3D flow model: identical to config/hm3d.py except the episode root —
Social-MP3D's train split (60 scenes with maps, so eps_per_scene is raised to
keep the window count comparable to HM3D's 120x10). Evaluation uses the val
split (data/mp3d has maps for all 11 val scenes) — a proper held-out split,
so no scene-index bookkeeping is needed. Only train here if zero-shot
transfer of the HM3D models proves insufficient.
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
        'loader': 'datasets.HM3DSequenceDataset',   # env-agnostic: ep_root decides
        'loader_kwargs': {'n_scenes': 60, 'eps_per_scene': 20,
                          'ep_root': '/Users/alexoman/workspaces/diffusion/'
                                     'socialnav_map_gen/pointnav/social-mp3d/train'},

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
