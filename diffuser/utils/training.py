"""Mirrors diffuser/utils/training.py (EMA + Trainer), minus the renderer,
cloud-bucket, and periodic-checkpoint machinery we don't use. One deliberate
difference from SafeFlowMatcher's copy: theirs aliases ema_model to the live
model (the deepcopy line is commented out), which silently disables EMA; we
keep the original Diffuser behavior — a real slow-moving copy, evaluated at
test time.
"""

import copy

import torch


def cycle(dl):
    while True:
        for data in dl:
            yield data


class EMA():
    '''
        empirical moving average
    '''
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        dataset,
        ema_decay=0.995,
        train_batch_size=32,
        train_lr=2e-4,
        gradient_accumulate_every=2,
        step_start_ema=2000,
        update_ema_every=10,
        log_freq=100,
    ):
        super().__init__()
        self.model = diffusion_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every
        self.step_start_ema = step_start_ema
        self.log_freq = log_freq

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.dataset = dataset
        self.dataloader = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=train_batch_size, num_workers=0, shuffle=True))
        self.optimizer = torch.optim.Adam(diffusion_model.parameters(), lr=train_lr)

        self.reset_parameters()
        self.step = 0

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    def train(self, n_train_steps):
        import time
        t0 = time.time()
        for _ in range(n_train_steps):
            for _ in range(self.gradient_accumulate_every):
                batch = next(self.dataloader)
                loss, infos = self.model.loss(*batch)
                loss = loss / self.gradient_accumulate_every
                loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()

            if self.step % self.update_ema_every == 0:
                self.step_ema()

            if self.step % self.log_freq == 0:
                infos_str = ' | '.join([f'{key}: {val:8.4f}' for key, val in infos.items()])
                print(f'{self.step}: {loss.item() * self.gradient_accumulate_every:8.4f} '
                      f'| {infos_str} | t: {time.time() - t0:8.1f}s')

            self.step += 1
