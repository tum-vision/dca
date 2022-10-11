from typing import Tuple
from torch import nn, Tensor
from ..mcdropout import MCDropout
from .standard import GlobalAvgPool, ResNetHead


__all__ = ('preresnet20_mcdrop',)


class BaseBasic(nn.Module):
    def __init__(self, inchannel: int, outchannel: int, stride: int = 1,
                 activfunc=nn.ReLU, dropout=MCDropout, p: float = 0.05
                 ) -> None:
        super().__init__()
        self.activ_h = activfunc()
        self.bn_h = nn.BatchNorm2d(outchannel)
        self.drop_io = dropout(p)
        self.drop_ih = dropout(p)
        self.drop_ho = dropout(p)
        self.conv_io = nn.Conv2d(inchannel, outchannel, kernel_size=3,
                                 stride=stride, padding=1, bias=False)
        self.conv_ih = nn.Conv2d(inchannel, outchannel, kernel_size=3,
                                 stride=stride, padding=1, bias=False)
        self.conv_ho = nn.Conv2d(outchannel, outchannel, kernel_size=3,
                                 padding=1, bias=False)

    def forward(self, t: Tensor) -> Tensor:
        return self.conv_io(self.drop_io(t)) + self.conv_ho(self.drop_ho(
            self.activ_h(self.bn_h(self.conv_ih(self.drop_ih(t))))))


class RefineBasic(nn.Module):
    def __init__(self, channel: int, activfunc=nn.ReLU, dropout=MCDropout,
                 p: float = 0.05) -> None:
        super().__init__()
        self.activ_i = activfunc()
        self.bn_i = nn.BatchNorm2d(channel)
        self.activ_h = activfunc()
        self.bn_h = nn.BatchNorm2d(channel)
        self.drop_ih = dropout(p)
        self.drop_ho = dropout(p)
        self.conv_ih = nn.Conv2d(
            channel, channel, kernel_size=3, padding=1, bias=False)
        self.conv_ho = nn.Conv2d(
            channel, channel, kernel_size=3, padding=1, bias=False)

    def forward(self, t: Tensor) -> Tensor:
        return t + self.conv_ho(self.drop_ho(
            self.activ_h(self.bn_h(self.conv_ih(self.drop_ih(
                self.activ_i(self.bn_i(t))))))))


class BasicTrunk(nn.Sequential):
    def __init__(self, inchannel: int, outchannel: int, blocks: int,
                 stride: int = 1, activfunc=nn.ReLU, dropout=MCDropout,
                 p: float = 0.05):
        assert blocks >= 1
        super().__init__(
            BaseBasic(inchannel, outchannel, stride, activfunc, dropout, p),
            *[RefineBasic(outchannel, activfunc, dropout, p)
              for _ in range(blocks - 1)],
            nn.BatchNorm2d(outchannel),
            activfunc())


class ResNetClassifier(nn.Sequential):
    def __init__(self, inchannel: int, outclass: int, dropout=MCDropout,
                 p: float = 0.05):
        super().__init__(
            dropout(p),
            GlobalAvgPool(),
            nn.Flatten(),
            nn.Linear(inchannel, outclass))


# depth = 3 * sum(blocks) + 2
def _resnet(outclass: int, trunktype, blocks: Tuple[int, int, int],
            activfunc=nn.ReLU, dropout=MCDropout, p: float = 0.05
            ) -> nn.Sequential:
    c0, c1, c2, c3 = 16, 16, 32, 64
    return nn.Sequential(
        ResNetHead(c0, activfunc),
        trunktype(c0, c1, blocks[0], 1, activfunc, dropout, p),
        trunktype(c1, c2, blocks[1], 2, activfunc, dropout, p),
        trunktype(c2, c3, blocks[2], 2, activfunc, dropout, p),
        ResNetClassifier(c3, outclass, dropout, p))


def preresnet20_mcdrop(outclass: int, p: float = 0.05) -> nn.Sequential:
    return _resnet(outclass, BasicTrunk, (3, 3, 3), p=p)
