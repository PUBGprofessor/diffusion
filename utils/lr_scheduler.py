from torch.optim.lr_scheduler import _LRScheduler

class lr_scheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_steps=500, decay_rate=0.99, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.decay_rate = decay_rate
        super(lr_scheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        step = max(1, self._step_count)
        if step <= self.warmup_steps:
            # 线性warmup：从0增长到初始学习率
            scale = step / self.warmup_steps
        else:
            # warmup之后进行指数衰减
            scale = (self.decay_rate ** (step - self.warmup_steps))
        return [base_lr * scale for base_lr in self.base_lrs]