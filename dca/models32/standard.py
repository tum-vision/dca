from typing import Tuple
from torch import nn, Tensor
import torch.nn.functional as nnf


__all__ = ('preresnet20',)


class GlobalAvgPool(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, t: Tensor) -> Tensor:
        return nnf.avg_pool2d(t, t.size()[2:])


class BaseBasic(nn.Module):
    def __init__(self, inchannel: int, outchannel: int, stride: int = 1,
                 activfunc=nn.ReLU) -> None:
        super().__init__()
        self.activ_h = activfunc()
        self.bn_h = nn.BatchNorm2d(outchannel)
        self.conv_io = nn.Conv2d(inchannel, outchannel, kernel_size=3,
                                 stride=stride, padding=1, bias=False)
        self.conv_ih = nn.Conv2d(inchannel, outchannel, kernel_size=3,
                                 stride=stride, padding=1, bias=False)
        self.conv_ho = nn.Conv2d(outchannel, outchannel, kernel_size=3,
                                 padding=1, bias=False)

    def forward(self, t: Tensor) -> Tensor:
        return self.conv_io(t) + \
            self.conv_ho(self.activ_h(self.bn_h(self.conv_ih(t))))


class RefineBasic(nn.Module):
    def __init__(self, channel: int, activfunc=nn.ReLU) -> None:
        super().__init__()
        self.activ_i = activfunc()
        self.bn_i = nn.BatchNorm2d(channel)
        self.activ_h = activfunc()
        self.bn_h = nn.BatchNorm2d(channel)
        self.conv_ih = nn.Conv2d(
            channel, channel, kernel_size=3, padding=1, bias=False)
        self.conv_ho = nn.Conv2d(
            channel, channel, kernel_size=3, padding=1, bias=False)

    def forward(self, t: Tensor) -> Tensor:
        return t + self.conv_ho(self.activ_h(self.bn_h(self.conv_ih(
            self.activ_i(self.bn_i(t))))))


class ResNetHead(nn.Sequential):
    def __init__(self, outchannel: int, activfunc=nn.ReLU):
        super().__init__(
            nn.Conv2d(3, outchannel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            activfunc())


class BasicTrunk(nn.Sequential):
    def __init__(self, inchannel: int, outchannel: int, blocks: int,
                 stride: int = 1, activfunc=nn.ReLU):
        assert blocks >= 1
        super().__init__(
            BaseBasic(inchannel, outchannel, stride, activfunc),
            *[RefineBasic(outchannel, activfunc) for _ in range(blocks-1)],
            nn.BatchNorm2d(outchannel),
            activfunc())


class ResNetClassifier(nn.Sequential):
    def __init__(self, inchannel: int, outclass: int):
        super().__init__(
            GlobalAvgPool(),
            nn.Flatten(),
            nn.Linear(inchannel, outclass))


# depth = 3 * sum(blocks) + 2
def _resnet(outclass: int, trunktype, blocks: Tuple[int, int, int],
            activfunc=nn.ReLU) -> nn.Sequential:
    c0, c1, c2, c3 = 16, 16, 32, 64
    return nn.Sequential(
        ResNetHead(c0, activfunc),
        trunktype(c0, c1, blocks[0], 1, activfunc),
        trunktype(c1, c2, blocks[1], 2, activfunc),
        trunktype(c2, c3, blocks[2], 2, activfunc),
        ResNetClassifier(c3, outclass))


class ResNet(nn.Sequential):
    def __init__(
            self, outclass: int, trunktype, blocks: Tuple[int, int, int],
            activfunc=nn.ReLU):
        c0, c1, c2, c3 = 16, 16, 32, 64
        super().__init__(
            ResNetHead(c0, activfunc),
            trunktype(c0, c1, blocks[0], 1, activfunc),
            trunktype(c1, c2, blocks[1], 2, activfunc),
            trunktype(c2, c3, blocks[2], 2, activfunc),
            ResNetClassifier(c3, outclass)
        )


def preresnet20(outclass: int) -> nn.Sequential:
    return _resnet(outclass, BasicTrunk, (3, 3, 3))


def preresnet20_model(outclass: int) -> nn.Sequential:
    return ResNet(outclass, BasicTrunk, (3, 3, 3))
