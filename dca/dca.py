from typing import Optional, Iterable, Dict, Tuple, Iterator
from collections import OrderedDict
from itertools import product
import abc
import copy
import torch
from torch import nn, LongTensor


# types of layer to duplicate for DCA
WEIGHT_LAYER_TYPES = (
    nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d,
    nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)


class RandIntGen(metaclass=abc.ABCMeta):
    choices: Dict[str, type] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.choices[cls.__name__] = cls

    def __init__(self, high: int, *args, **kwargs):
        self.high = high
        self.args = (args, kwargs)

    def __iter__(self):
        return self

    @abc.abstractmethod
    def __next__(self):
        raise NotImplementedError

    def getvector(self, size: int = 1, device=torch.device('cpu')
                  ) -> LongTensor:
        return torch.tensor([next(self) for _ in range(size)],
                            dtype=torch.long, device=device)

    def reset(self):
        pass


class IIDUniform(RandIntGen):
    def __init__(self, high: int):
        super().__init__(high)

    def __next__(self):
        return torch.randint(self.high, ()).item()

    def getvector(self, size: int = 1, device=torch.device('cpu')
                  ) -> LongTensor:
        return torch.randint(self.high, (size,),
                             dtype=torch.long, device=device)


# wraps vanilla modules for DCA
class DCAModule(nn.Module, metaclass=abc.ABCMeta):
    @classmethod
    def build(cls, m: nn.Module, randsource: Optional[RandIntGen] = None):
        if isinstance(m, DCAModule):
            raise ValueError(f'module is already a DCAModule')
        if cls.isleaf(m):  # only leaf DCAModules
            if randsource is None:
                raise ValueError('DCALeaf must specify randsource')
            else:
                return DCALeaf(m, randsource)
        else:
            if randsource is not None:
                raise ValueError('DCABranch requires no randsource')
            else:
                return DCABranch(m)

    @classmethod
    def buildchannelwise(
            cls, m: nn.Module, randsource: Optional[RandIntGen] = None):
        if isinstance(m, DCAModule):
            raise ValueError(f'module is already a DCAModule')
        if cls.isleaf(m):  # only leaf DCAModules
            if randsource is None:
                raise ValueError('ChannelDCALeaf must specify randsource')
            else:
                return ChannelDCALeaf(m, randsource)
        else:
            if randsource is not None:
                raise ValueError('DCABranch requires no randsource')
            else:
                return DCABranch(m)

    @staticmethod
    def isleaf(basemodule: nn.Module):
        for m in basemodule.modules():
            if isinstance(m, DCAModule):
                return False
        return True

    @property
    @abc.abstractmethod
    def randsource(self) -> Optional[RandIntGen]:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def rootrandsource(self) -> bool:
        raise NotImplementedError

    @rootrandsource.setter
    @abc.abstractmethod
    def rootrandsource(self, value: bool):
        raise NotImplementedError

    @abc.abstractmethod
    def getmodule(self) -> nn.Module:
        raise NotImplementedError

    @abc.abstractmethod
    def itermodules(self) -> Iterator[nn.Module]:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def copies(self) -> int:
        raise NotImplementedError

    def forward(self, *args, **kwargs):
        randsource = self.randsource
        if self.rootrandsource:  # rootrandsource should reset in each forward
            randsource.reset()
        return self.getmodule()(*args, **kwargs)

    # return a new copy of base Module with random combination
    @abc.abstractmethod
    @torch.no_grad()
    def samplemodule(self, sampleroot=True) -> nn.Module:
        raise NotImplementedError

    # return a new copy of weight averaged Module
    @abc.abstractmethod
    @torch.no_grad()
    def wamodule(self, waroot=True) -> nn.Module:
        raise NotImplementedError


# leaf DCA module
class ChannelDCALeaf(DCAModule):
    def __init__(self, m: nn.Module, randsource: RandIntGen):
        super().__init__()
        self._randsource = randsource
        self._rootrandsource = True
        self._mcopies = nn.ModuleList(
            [m] + [copy.deepcopy(m) for _ in range(1, randsource.high)])
        self._module, self._modulecfg = self._init_module()

    @property
    def randsource(self) -> RandIntGen:
        return self._randsource

    @property
    def rootrandsource(self) -> bool:
        return self._rootrandsource

    @rootrandsource.setter
    def rootrandsource(self, value: bool):
        self._rootrandsource = value

    @staticmethod
    def _get_channel(m: nn.Module) -> int:
        num_channel = set(
            p.size(0) for p in m._parameters.values() if p is not None)
        if len(num_channel) == 0:
            return 0
        elif len(num_channel) == 1:
            return next(iter(num_channel))
        else:
            raise ValueError(f'incompatible channel count: {num_channel}')

    def _init_module(self) -> Tuple[
            nn.Module, Tuple[Tuple[int, Tuple[str, ...]], ...]]:
        module = copy.deepcopy(self._mcopies[0])
        cfg = []
        for m in module.modules():
            # collect channel size
            c = self._get_channel(m)
            # collect all param names
            fullpnames = [(pname, p is None) for pname, p
                          in m._parameters.items()]
            # empty params
            m._parameters = OrderedDict()
            # create empty buffers
            for pname, _ in fullpnames:
                m.register_buffer(pname, None)
            # save config, keep only non-None param names
            cfg.append((c, tuple(
                pname for pname, isnone in fullpnames if not isnone)))
        return module, tuple(cfg)

    def getmodule(self) -> nn.Module:
        device = next(self._mcopies[0].parameters()).device
        for (c, pnames), subm, submcs in zip(
                self._modulecfg, self._module.modules(),
                zip(*[m.modules() for m in self._mcopies])):
            choices = self.randsource.getvector(c, device=device)
            indices = choices * c + torch.arange(
                c, dtype=torch.long, device=device)
            for pname in pnames:
                pcs = torch.cat([getattr(m, pname) for m in submcs])
                setattr(subm, pname, pcs[indices])
        return self._module

    def itermodules(self) -> Iterator[nn.Module]:
        raise NotImplementedError

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        self._module = None
        sd = super().state_dict(destination, prefix, keep_vars)
        self._module, self._modulecfg = self._init_module()
        return sd

    def load_state_dict(self, state_dict, strict: bool = True):
        self._module = None
        super().load_state_dict(state_dict, strict)
        self._module, self._modulecfg = self._init_module()

    @property
    def copies(self) -> int:
        return len(self._mcopies)

    @torch.no_grad()
    def samplemodule(self, sampleroot=True) -> nn.Module:
        return copy.deepcopy(self.getmodule())

    def wamodule(self, waroot=True) -> nn.Module:
        wamodel = copy.deepcopy(self._mcopies[0])
        # update all params in wamodel
        for mwa, mcs in zip(wamodel.modules(), zip(*[
                m.modules() for m in self._mcopies])):
            for pwa, pcs in zip(mwa.parameters(recurse=False), zip(*[
                    m.parameters(recurse=False) for m in mcs])):
                if pwa is not None:
                    pwa.data = torch.mean(torch.stack(pcs), 0)
        return wamodel


# leaf DCA module
class DCALeaf(DCAModule):
    def __init__(self, m: nn.Module, randsource: RandIntGen):
        super().__init__()
        self._randsource = randsource
        self._rootrandsource = True
        self._mcopies = nn.ModuleList(
            [m] + [copy.deepcopy(m) for _ in range(1, randsource.high)])

    @property
    def randsource(self) -> RandIntGen:
        return self._randsource

    @property
    def rootrandsource(self) -> bool:
        return self._rootrandsource

    @rootrandsource.setter
    def rootrandsource(self, value: bool):
        self._rootrandsource = value

    def getmodule(self) -> nn.Module:
        return self._mcopies[next(self._randsource)]

    def itermodules(self) -> Iterator[nn.Module]:
        for m in self._mcopies:
            yield copy.deepcopy(m)

    @property
    def copies(self) -> int:
        return len(self._mcopies)

    @torch.no_grad()
    def samplemodule(self, sampleroot=True) -> nn.Module:
        return copy.deepcopy(self.getmodule())

    def wamodule(self, waroot=True) -> nn.Module:
        wamodel = copy.deepcopy(self._mcopies[0])
        # update all params in wamodel
        for mwa, mcs in zip(wamodel.modules(), zip(*[
                m.modules() for m in self._mcopies])):
            for pwa, pcs in zip(mwa.parameters(recurse=False), zip(*[
                    m.parameters(recurse=False) for m in mcs])):
                if pwa is not None:
                    pwa.data = torch.mean(torch.stack(pcs), 0)
        return wamodel


# non-leaf DCA module
class DCABranch(DCAModule):
    def __init__(self, m: nn.Module):
        super().__init__()
        self._model = m
        self._randsource = self._inferrandsource(m)
        self._rootrandsource = self._updaterootrandsource()

    @staticmethod
    def _inferrandsource(m) -> Optional[RandIntGen]:
        source = None
        for subm in m.modules():
            if isinstance(subm, DCAModule):
                subsource = subm.randsource
                if subsource is None:  # already inconsistent
                    return None
                elif source is None:
                    source = subsource
                elif source is not subsource:  # inconsisntency found
                    return None
        if source is None:  # no source found at all, should not happen!
            raise RuntimeError('no randsource found in base module')
        return source

    def _updaterootrandsource(self) -> bool:
        if self.randsource is None:
            return False
        else:
            # set all submodules as non root randsource
            for m in self._model.modules():
                if isinstance(m, DCAModule):
                    m.rootrandsource = False
            return True

    @property
    def randsource(self) -> Optional[RandIntGen]:
        return self._randsource

    @property
    def rootrandsource(self) -> bool:
        return self._rootrandsource

    @rootrandsource.setter
    def rootrandsource(self, value: bool):
        self._rootrandsource = value

    def getmodule(self) -> nn.Module:
        return self._model

    def itermodules(self) -> Iterator[nn.Module]:
        self_model_modules = self._model._modules
        self._model._modules = OrderedDict()
        baremodule = copy.deepcopy(self._model)
        self._model._modules = self_model_modules

        nsubm = [n for n in self_model_modules.keys()]
        submiter = product(*[
            subm.itermodules() if isinstance(subm, DCAModule) else (subm,)
            for subm in self_model_modules.values()])
        for subms in submiter:
            modulecopy = copy.deepcopy(baremodule)
            modulecopy._modules = OrderedDict(zip(nsubm, subms))
            yield modulecopy

    @property
    def copies(self) -> int:
        return 1

    @torch.no_grad()
    def samplemodule(self, sampleroot=True) -> nn.Module:
        smodel = copy.deepcopy(self._model) if sampleroot else self._model
        for n, subm in smodel.named_children():
            if isinstance(subm, DCAModule):
                smodel._modules[n] = subm.samplemodule(False)
        return smodel

    def wamodule(self, waroot=True) -> nn.Module:
        wamodel = copy.deepcopy(self._model) if waroot else self._model
        for n, subm in wamodel.named_children():
            if isinstance(subm, DCAModule):
                wamodel._modules[n] = subm.wamodule(False)
        return wamodel


def wrapmodel(model: nn.Module, copies: int, randgen='IIDUniform',
              leaftypes: Iterable[type] = WEIGHT_LAYER_TYPES,
              *args, **kwargs) -> DCAModule:
    assert copies > 1
    assert DCAModule.isleaf(model)
    randsource = RandIntGen.choices[randgen](copies, *args, **kwargs)

    def _wrap_rec(model: nn.Module, randsource: RandIntGen) -> nn.Module:
        if type(model) in leaftypes:
            return DCAModule.build(model, randsource)
        else:
            mchildren = {n: m for n, m in model.named_children()}
            if len(mchildren) == 0:  # non DCA leaf
                return model
            else:  # DCABranch
                for n, m in mchildren.items():
                    model._modules[n] = _wrap_rec(m, randsource)
                return DCAModule.build(model)

    dcamodel = _wrap_rec(model, randsource)
    if isinstance(dcamodel, DCAModule):
        return dcamodel
    else:
        raise ValueError('model has no DCA wrappable component')


def wrapmodelchannelwise(
        model: nn.Module, copies: int, randgen='IIDUniform',
        leaftypes: Iterable[type] = WEIGHT_LAYER_TYPES, *args, **kwargs
) -> DCAModule:
    assert copies > 1
    assert DCAModule.isleaf(model)
    randsource = RandIntGen.choices[randgen](copies, *args, **kwargs)

    def _wrap_rec(model: nn.Module, randsource: RandIntGen) -> nn.Module:
        if type(model) in leaftypes:
            return DCAModule.buildchannelwise(model, randsource)
        else:
            mchildren = {n: m for n, m in model.named_children()}
            if len(mchildren) == 0:  # non DCA leaf
                return model
            else:  # DCABranch
                for n, m in mchildren.items():
                    model._modules[n] = _wrap_rec(m, randsource)
                return DCAModule.buildchannelwise(model)

    dcamodel = _wrap_rec(model, randsource)
    if isinstance(dcamodel, DCAModule):
        return dcamodel
    else:
        raise ValueError('model has no DCA wrappable component')
