from typing import Tuple, Optional, Iterable
from os.path import join as pjoin
import csv
import statistics
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from dca.utils import coro_npybatchgatherer, coro_trackavg_weighted, \
    coro_dict2csv, autoinitcoroutine
from dca.trainutils import cumentropy


def confidence_from_prediction_npy(npyfile: str) -> np.ndarray:
    probas = np.load(npyfile)
    return np.amax(probas, axis=1)


def cumconfidence(probas: Tensor) -> float:
    return torch.sum(torch.max(probas, dim=1)[0]).item()


def coro_epochlog(total: int, logfreq: int = 100, outputsaver=None):
    conftracker = coro_trackavg_weighted()
    enttracker = coro_trackavg_weighted()
    conf, ent = float('nan'), float('nan')
    try:
        yield
        while True:
            (outprobas, _, _), i = (yield)
            if outputsaver is not None:
                outputsaver.send(outprobas.cpu().numpy())
            bs = outprobas.size(0)
            ent = enttracker.send((cumentropy(outprobas), bs))
            conf = conftracker.send((cumconfidence(outprobas), bs))
            if i % logfreq == 0:
                print(f'  {i}/{total}: conf={conf:.4f}, entropy={ent:.4f}')
    except StopIteration:  # on manual stop, return final accumulations
        return conf, ent


@autoinitcoroutine
def coro_log(
        sw: Optional[SummaryWriter] = None, logfreq: int = 100, save_dir=''):
    ent, conf = float('nan'), float('nan')
    csvhead = ('epoch', 'confidence', 'entropy')
    csvcorologs = dict()
    try:
        epoch, prefix, total, outputsaver = (yield)
        while True:
            print(f"*** Epoch {epoch} {prefix} ***\n")
            conf, ent = yield from coro_epochlog(total, logfreq, outputsaver)
            print(f'\nEpoch {epoch}: conf={conf:.4f}, entropy={ent:.4f};\n')
            if save_dir:
                if prefix not in csvcorologs:
                    csvcorologs[prefix] = coro_dict2csv(
                        pjoin(save_dir, f'{prefix}.csv'), csvhead)
                csvcorologs[prefix].send({
                    'epoch': epoch, 'confidence': conf, 'entropy': ent})
            if sw is not None:
                sw.add_scalar(f'{prefix}/uncertainty', 1 - conf, epoch)
                sw.add_scalar(f'{prefix}/entropy', ent, epoch)
                sw.flush()
            epoch, prefix, total, outputsaver = yield (conf, ent)
    except StopIteration:
        return conf, ent


def dup_collate_fn(dups: int):

    def collate_fn(data):
        imgs, gts = tuple(zip(*data))
        t = torch.stack(imgs, dim=0)
        return t.repeat(dups, *(1,)*(t.ndim-1)), torch.as_tensor(gts)

    return collate_fn


def get_outputsaver(save_dir, ndata, outclass, predictionfile):
    return coro_npybatchgatherer(
        pjoin(save_dir, predictionfile), ndata, (outclass,), True,
        str(torch.get_default_dtype())[6:])


def mean_std(vals: Iterable[float]) -> Tuple[float, float]:
    return statistics.mean(vals), statistics.stdev(vals)


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
            mean, std = mean_std(vals)
            print(f'{k:>{maxlen}}:\tmean {mean:.4f}, std={std:.4f}')


class SVHNInfo:
    outclass = 10
    split = ('train', 'test', 'extra')
    count = {'train': 73257, 'test': 26032, 'extra': 531131}
    mean = (0.4376821, 0.4437697, 0.47280442)
    std = (0.19803012, 0.20101562, 0.19703614)


def get_svhn_loader(data_dir: str, workers: int, pin_memory: bool, batch: int,
                    split: str = 'test', dups: int = 1):
    assert split in SVHNInfo.split
    svhn_dir = pjoin(data_dir, 'svhn')

    normalize = transforms.Normalize(SVHNInfo.mean, SVHNInfo.std)

    dataset = datasets.SVHN(root=svhn_dir, split=split, download=True,
                            transform=transforms.Compose([
                             transforms.ToTensor(), normalize]))

    loader = DataLoader(
        dataset, batch_size=batch, num_workers=workers, pin_memory=pin_memory,
        collate_fn=dup_collate_fn(dups)
    ) if dups > 1 else DataLoader(dataset, batch_size=batch,
                                  num_workers=workers, pin_memory=pin_memory)

    return loader


def get_roc_curve_auc_score(
        indomain_confidence: np.ndarray, ood_confidence: np.ndarray
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    is_indomain = np.concatenate(
        (np.zeros_like(ood_confidence), np.ones_like(indomain_confidence)))
    confidence = np.concatenate((ood_confidence, indomain_confidence))
    fpr, tpr, thresh = roc_curve(is_indomain, confidence, pos_label=1)
    auc_score = roc_auc_score(is_indomain, confidence)
    return auc_score, fpr, tpr, thresh
