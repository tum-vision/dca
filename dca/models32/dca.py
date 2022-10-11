from ..dca import wrapmodel, DCAModule, wrapmodelchannelwise
from .standard import preresnet20, ResNetHead, ResNetClassifier, BasicTrunk, \
    BaseBasic, RefineBasic, ResNet, preresnet20_model


__all__ = (
    'preresnet20_dca', 'preresnet20_dca_channel', 'preresnet20_dca_block',
    'preresnet20_dca_trunk', 'preresnet20_dca_model')


def preresnet20_dca(
        outclass: int, copies: int, randgen='IIDUniform', *args, **kwargs
) -> DCAModule:
    return wrapmodel(preresnet20(outclass), copies, randgen, *args, **kwargs)


def preresnet20_dca_channel(
        outclass: int, copies: int, randgen='IIDUniform', *args, **kwargs
) -> DCAModule:
    return wrapmodelchannelwise(
        preresnet20(outclass), copies, randgen, *args, **kwargs)


def preresnet20_dca_block(
        outclass: int, copies: int, randgen='IIDUniform', *args, **kwargs
) -> DCAModule:
    return wrapmodel(
        preresnet20(outclass), copies, randgen,
        leaftypes=(ResNetHead, BaseBasic, RefineBasic, ResNetClassifier),
        *args, **kwargs)


def preresnet20_dca_trunk(
        outclass: int, copies: int, randgen='IIDUniform', *args, **kwargs
) -> DCAModule:
    return wrapmodel(preresnet20(outclass), copies, randgen,
                     leaftypes=(ResNetHead, BasicTrunk, ResNetClassifier),
                     *args, **kwargs)


def preresnet20_dca_model(
        outclass: int, copies: int, randgen='IIDUniform', *args, **kwargs
) -> DCAModule:
    return wrapmodel(preresnet20_model(outclass), copies, randgen,
                     leaftypes=(ResNet,), *args, **kwargs)
