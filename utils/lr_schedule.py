from torch.optim.lr_scheduler import CosineAnnealingLR


class CosineAnnealingWithWarmupLR(CosineAnnealingLR):
    def __init__(self, optimizer, T_max, eta_min=0, last_epoch=-1, warmup_epochs=0):
        self.warmup_epochs = warmup_epochs
        super().__init__(optimizer, T_max, eta_min, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            return [
                base_lr * (self.last_epoch + 1) / self.warmup_epochs
                for base_lr in self.base_lrs
            ]
        else:
            return super().get_lr()
