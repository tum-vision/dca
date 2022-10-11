from typing import Mapping, Any, Iterable
import csv
from collections import OrderedDict
import statistics
import warnings
from os.path import exists
from os.path import join as pjoin
import torch
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
import dca.models32 as models
from dca.dataloaders import get_cifar10_train_loaders, \
    get_cifar10_test_loader
from dca.utils import autoinitcoroutine, cp, coro_npybatchgatherer
from dca.optim import schedule_midway_linear_decay
from dca.calibration import bins2acc, bins2ece, bins2conf


TRAINDATALOADERS = {'cifar10': get_cifar10_train_loaders}
TESTDATALOADER = {'cifar10': get_cifar10_test_loader}
ALLMODELS = models.MODELS
STANDARDMODELS = {
    k: v for k, v in ALLMODELS.items() if k in models.standard.__all__}
MCDROPMODELS = {
    k: v for k, v in ALLMODELS.items() if k in models.mcdrop.__all__}
SWAGMODELS = {k: v for k, v in ALLMODELS.items() if k in models.swag.__all__}
DCAMODELS = {k: v for k, v in ALLMODELS.items() if k in models.dca.__all__}
DCSWAGMODELS = {
    k: v for k, v in ALLMODELS.items() if k in models.dcswag.__all__}
# number of data
NTRAIN, NTEST = 50000, 10000
# number of classes
OUTCLASS = {'cifar10': 10}


def savecheckpoint(
        to, modelname: str, modelargs: Iterable[Any],
        modelkwargs: Mapping[str, Any], model: nn.Module, optimizer: SGD,
        scheduler: LambdaLR, **kwargs) -> None:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        models.savemodel(to, modelname, modelargs, modelkwargs, model, **{
            'optimargs': optimizer.defaults,
            'optimstates': optimizer.state_dict(),
            'schedulerstates': scheduler.state_dict()}, **kwargs)


def loadcheckpoint(fromfile, device=torch.device('cpu')):
    model, dic = models.loadmodel(fromfile, device)
    optimizer = SGD(model.parameters(), **dic.pop('optimargs'))
    optimizer.load_state_dict(dic.pop('optimstates'))
    scheduler = schedule_midway_linear_decay(optimizer)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        scheduler.load_state_dict(dic.pop('schedulerstates'))
    startepoch = scheduler.last_epoch
    return startepoch, model, optimizer, scheduler, dic


@autoinitcoroutine
def coro_trackbestloss(
        train_dir, modelname: str, modelargs, modelkwargs, finetuneepoch=0,
        predictions=None, initbest=float('inf')):
    best_epoch = None
    bestbins = None
    best_path = pjoin(train_dir, 'best_model.pt')
    try:
        epoch, model, bins, loss = (yield)
        while True:
            if finetuneepoch is not None and epoch == finetuneepoch:
                print('### Tracking best finetune result ... ###\n')
                initbest = float('inf')
            if loss < initbest:  # is better
                print(f'### BEST! epoch={epoch}, loss={loss:.6f} ###\n')
                # update best values
                initbest, best_epoch, bestbins = loss, epoch, bins
                # save new model
                models.savemodel(
                    best_path, modelname, modelargs, modelkwargs, model,
                    epoch=epoch, bins=bins, loss=loss)
                # save output predictions if exists
                if predictions and exists(pjoin(train_dir, predictions)):
                    cp(pjoin(train_dir, predictions),
                       pjoin(train_dir, f'best_{predictions}'))
            else:  # not better
                pass
            epoch, model, bins, loss = (yield)
    except StopIteration:
        print(f'### Best result: epoch={best_epoch}, loss={initbest}, '
              f'acc={bins2acc(bestbins)}, conf={bins2conf(bestbins)}, '
              f'ece={bins2ece(bestbins)}\n')
        print(f'### Model saved at: {best_path} ###\n')
        return initbest, best_epoch, bestbins, best_path


def get_outputsaver(save_dir, ndata, outclass, predictionfile):
    return coro_npybatchgatherer(
        pjoin(save_dir, predictionfile), ndata, (outclass,), True,
        str(torch.get_default_dtype())[6:])


def summarize_csv(csvfile):
    with open(csvfile, 'r') as csvfp:
        reader = csv.DictReader(csvfp)
        criteria = [k for k in reader.fieldnames if k != 'epoch']
        maxlen = max(len(k) for k in criteria)
        values = {k: [] for k in criteria}
        for row in reader:
            for k, v in row.items():
                if k != 'epoch':
                    values[k].append(float(v))
        for k, vals in values.items():
            print(f'{k:>{maxlen}}:\tmean {statistics.mean(vals):.4f}, '
                  f'std={statistics.stdev(vals):.4f}')


def average_state_dicts(state_dicts):
    avgsd = OrderedDict()
    for k, vs in zip(state_dicts[0].keys(),
                     zip(*[sd.values() for sd in state_dicts])):
        if torch.is_floating_point(vs[0]):
            avgsd[k] = torch.mean(torch.stack(vs, 0), 0)
        else:
            avgsd[k] = vs[0]
    return avgsd


def load_averaged_model(fromfiles, device=torch.device('cpu')):
    dics = [torch.load(f, map_location=device) for f in fromfiles]
    model = ALLMODELS[dics[0]['modelname']](*dics[0]['modelargs']).to(device)
    avgsd = average_state_dicts([d.pop('modelstates') for d in dics])
    model.load_state_dict(avgsd)
    return model, dics