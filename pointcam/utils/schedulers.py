import math

from timm.scheduler import CosineLRScheduler
from torch.optim.lr_scheduler import LambdaLR


class TimmCosineLRScheduler(LambdaLR):
    def __init__(self, optimizer, **kwargs):
        self.init_lr = optimizer.param_groups[0]["lr"]
        self.scheduler = CosineLRScheduler(optimizer, **kwargs)
        super().__init__(optimizer, self)

    def __call__(self, step):
        desired_lr = self.scheduler._get_lr(step)[0]
        mult = desired_lr / self.init_lr
        return mult


class ConstantWithWarmupScheduler(LambdaLR):
    def __init__(self, optimizer, warmup_steps, last_epoch=-1):
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, self, last_epoch)

    def __call__(self, step):
        if step < self.warmup_steps:
            return step / self.warmup_steps
        else:
            return 1.0


class BaseScheduler:
    def __init__(
        self,
        initial_alpha: float,
        final_alpha: float,
        total_steps: int,
        flat_ratio: float = 0.1,
    ):
        self.total_steps = total_steps
        self.initial_alpha = initial_alpha
        self.final_alpha = final_alpha
        self.flat_ratio = flat_ratio

        self.current_step = 0

    def set_total_steps(self, total_steps):
        self.total_steps = total_steps
        self.flat_steps = int(self.total_steps * self.flat_ratio)
        self.change_steps = self.total_steps - self.flat_steps

    def set_current_step(self, current_step):
        self.current_step = current_step

    def get(self):
        if self.total_steps is None:
            return self.initial_alpha

    def step(self):
        self.current_step += 1
        return self


class CosineAnneal(BaseScheduler):
    def get(self):
        if self.total_steps is None:
            return self.initial_alpha

        if self.current_step <= self.flat_steps:
            return self.initial_alpha

        alpha_diff = (
            (self.final_alpha - self.initial_alpha)
            * (
                1.0
                + math.cos(
                    math.pi * (self.current_step - self.flat_steps) / self.change_steps
                )
            )
            / 2.0
        )

        return self.final_alpha - alpha_diff


class LinearAnneal(BaseScheduler):
    def get(self):
        if self.total_steps is None:
            return self.initial_alpha

        if self.current_step <= self.flat_steps:
            return self.initial_alpha

        alpha_diff = (
            (self.final_alpha - self.initial_alpha)
            * (self.current_step - self.flat_steps)
            / self.change_steps
        )

        return self.initial_alpha + alpha_diff
