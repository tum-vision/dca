from typing import Any, Iterable, Mapping
import torch
from torch import nn
from . import standard
from . import mcdrop
from . import swag
from . import dca
from .standard import *
from .mcdrop import *
from .swag import *
from .dca import *


MODELS = {n: globals()[n] for n in (
    *standard.__all__, *mcdrop.__all__, *swag.__all__, *dca.__all__)}


# standard save model function
def savemodel(to, modelname: str, modelargs: Iterable[Any],
              modelkwargs: Mapping[str, Any], model: nn.Module, **kwargs
              ) -> None:
    dic = {'modelname': modelname,
           'modelargs': tuple(modelargs),
           'modelkwargs': {k: modelkwargs[k] for k in modelkwargs},
           'modelstates': model.state_dict(),
           **kwargs}
    torch.save(dic, to)


# standard load model function
def loadmodel(fromfile, device=torch.device('cpu')):
    dic = torch.load(fromfile, map_location=device)
    model = globals()[dic['modelname']](
        *dic['modelargs'], **dic.get('modelkwargs', {})).to(device)
    model.load_state_dict(dic.pop('modelstates'))
    return model, dic
