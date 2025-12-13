import torch


class CosineAnnealingWithWarmupScheduler(torch.optim.lr_scheduler.LRScheduler):
    """Cosine annealing learning rate scheduler with warmup period.

    Args:
        optimizer: torch optimizer object.
        max_steps: Total number of training steps.
        warmup_fraction: Fraction of total steps for the warmup phase.
            [default: 0.05]
        min_lr: Minimum learning rate after decay.
            [default: 0]
        last_epoch: The index of last epoch. [default: -1]
    """

    def __init__(
        self,
        optimizer,
        max_steps,
        warmup_fraction=0.05,
        final_lr_ratio=0.0,
        last_epoch=-1,
    ):
        assert 0.0 <= warmup_fraction <= 1.0, "warmup_fraction must be in [0.0, 1.0]"
        assert 0.0 <= final_lr_ratio <= 1.0, "final_lr_ratio must be in [0.0, 1.0]"
        self.warmup_steps = int(max_steps * warmup_fraction)
        self.max_steps = max_steps
        self.final_lr_ratio = final_lr_ratio
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            return [
                base_lr * (self.last_epoch + 1) / self.warmup_steps
                for base_lr in self.base_lrs
            ]
        else:
            # Cosine annealing
            cosine_epoch = self.last_epoch - self.warmup_steps
            cosine_total_epochs = self.max_steps - self.warmup_steps
            # Spread cosine function from [0,pi] over the remaining epochs
            # LR goes from base_lr to final_lr_ratio * base_lr
            cosine_value = torch.cos(
                torch.tensor(torch.pi * cosine_epoch / cosine_total_epochs)
            )
            # rescale and shift cosine to [final_lr_ratio,1]
            relative_lr = (1 + cosine_value) / 2
            relative_lr = self.final_lr_ratio + (1 - self.final_lr_ratio) * relative_lr
            # rescale to [final_lr, base_lr]
            return [base_lr * relative_lr for base_lr in self.base_lrs]
