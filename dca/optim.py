from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


class LRScheduleFun(object):
    def __init__(self, epochs: int = 200, start_decay: float = 0.5,
                 end_decay: float = 0.9,  end_scale: float = 0.01):
        assert 0.0 <= start_decay < end_decay <= 1.0
        super().__init__()
        self.epochs = epochs
        self.start_decay = start_decay
        self.end_decay = end_decay
        self.end_scale = end_scale

    def __call__(self, e):
        e = float(e) / self.epochs
        return max(min(1.0, 1.0 - (1.0-self.end_scale)*(e-self.start_decay)/(
                self.end_decay-self.start_decay)), self.end_scale)


# constant high lr, then linear decay to low lr, then constant low lr
def schedule_midway_linear_decay(
    optimizer: Optimizer, epochs: int = 200, start_decay: float = 0.5,
    end_decay: float = 0.9,  end_scale: float = 0.01
) -> LambdaLR:

    return LambdaLR(optimizer, LRScheduleFun(
        epochs, start_decay, end_decay, end_scale), last_epoch=-1)


def get_weightdecay(wd_normalized: float, datasize: int) -> float:
    return wd_normalized / float(datasize)
