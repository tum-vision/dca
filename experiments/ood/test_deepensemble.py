from typing import List
import argparse
from os import listdir
from os.path import join as pjoin
import numpy as np
import torch
from dca.utils import coro_timer, mkdirp, npybatchiterator
from dca.trainutils import do_epoch, check_cuda, deteministic_run, \
    SummaryWriter
from dca.dataloaders import get_cifar10_test_loader
from utils import get_svhn_loader, get_roc_curve_auc_score, get_outputsaver, \
    summarize_csv, coro_log, SVHNInfo, confidence_from_prediction_npy, mean_std


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('rootdir', type=str,
                        help='path that collects all predictions.')
    parser.add_argument('-sp', '--svhn_split', default='test',
                        choices=SVHNInfo.split,
                        help='available split: ' + ' | '.join(SVHNInfo.split))
    parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                        help='number of data loading workers')
    parser.add_argument('-b', '--batch', default=128, type=int,
                        metavar='N', help='test mini-batch size')
    parser.add_argument('-ec', '--ensemblecount', default=5, type=int,
                        help='number of deep ensembles')
    parser.add_argument('-es', '--ensemblesize', default=5, type=int,
                        help='number of models in each deep ensemble')
    parser.add_argument('-pf', '--printfreq', default=100, type=int,
                        metavar='N', help='print frequency')
    parser.add_argument('-d', '--device', default='cpu', type=str,
                        metavar='DEV', help='run on cpu/cuda')
    parser.add_argument('-s', '--seed', type=int, default=0,
                        help='fixes seed for reproducibility')
    parser.add_argument('-sd', '--save_dir',
                        help='The directory used to save test results',
                        default='save_temp', type=str)
    parser.add_argument('-so', '--saveoutput', action='store_true',
                        help='save output probability')
    parser.add_argument('-dd', '--data_dir',
                        help='The directory to find/store dataset',
                        default='../data', type=str)
    parser.add_argument('-tbd', '--tensorboard_dir', default='', type=str,
                        help='if specified, record data for tensorboard.')

    return parser.parse_args()


def predfiles_per_model(
        args, indomain_prefix='indomain_test', ood_prefix='ood_test'):
    # collect all prediction files
    indomain_files = sorted([
        pjoin(args.rootdir, f) for f in listdir(args.rootdir)
        if f.endswith('.npy') and
        f.startswith(f'predictions_{indomain_prefix}')])
    ood_files = sorted([
        pjoin(args.rootdir, f) for f in listdir(args.rootdir)
        if f.endswith('.npy') and f.startswith(f'predictions_{ood_prefix}')])
    # deliver per model
    ec, es = args.ensemblecount, args.ensemblesize
    assert len(indomain_files) >= ec * es
    assert len(ood_files) >= ec * es
    for c in range(ec):
        yield (indomain_files[c*es:(c+1)*es], ood_files[c*es:(c+1)*es])


def get_indomain_loader(args, predfilenames: List[str]):
    device = torch.device(args.device)
    # load data
    data_loader = get_cifar10_test_loader(
        args.data_dir, args.workers, (device != torch.device('cpu')),
        args.batch)
    gts = (gt for _, gt in data_loader)
    prediters = [npybatchiterator(f, args.batch) for f in predfilenames]
    yield len(data_loader)
    yield from zip(*prediters, gts)


def get_ood_loader(args, predfilenames: List[str]):
    device = torch.device(args.device)
    # load data
    data_loader = get_svhn_loader(
        args.data_dir, args.workers, (device != torch.device('cpu')),
        args.batch, args.svhn_split)
    gts = (gt for _, gt in data_loader)
    prediters = [npybatchiterator(f, args.batch) for f in predfilenames]
    yield len(data_loader)
    yield from zip(*prediters, gts)


def do_devalbatch(batchinput):
    preds, gt = batchinput[:-1], batchinput[-1]
    meanpred = torch.from_numpy(np.mean(np.stack(preds, 0), 0)).to(gt.device)
    return meanpred, gt, 0.0


if __name__ == '__main__':
    timer = coro_timer()
    t_init = next(timer)
    print(f'>>> Test initiated at {t_init.isoformat()} <<<\n')

    args = get_args()
    print(args, end='\n\n')

    # if seed is specified, run deterministically
    if args.seed is not None:
        deteministic_run(seed=args.seed)

    # get device for this experiment
    device = torch.device(args.device)

    if device != torch.device('cpu'):
        check_cuda()

    # build train_dir for this experiment
    mkdirp(args.save_dir)

    # prep tensorboard if specified
    if args.tensorboard_dir:
        mkdirp(args.tensorboard_dir)
        sw = SummaryWriter(args.tensorboard_dir)
    else:
        sw = None

    # distinguish between runs on validation data and test data

    log_ece = coro_log(sw, args.printfreq, args.save_dir)
    outclass = 10
    indomain_prefix = 'indomain_test'
    ood_prefix = 'ood_test'
    aucroc_scores = []

    # iterate over saved predictions per each deep ensemble
    for modelid, (in_files, ood_files) in enumerate(predfiles_per_model(args)):
        print(f'ensembling from following {args.ensemblesize} files:')
        print('In-domain:')
        for f in in_files:
            print(f'- {f}')
        print('OOD:')
        for f in ood_files:
            print(f'- {f}')

        print(f'>>> Test starts at {next(timer)[0].isoformat()} <<<\n')

        # do in-domain deep ensemble evaluation
        predloader = get_indomain_loader(args, in_files)
        nbatch = next(predloader)

        if args.saveoutput:
            outputsaver = get_outputsaver(
                args.save_dir, 10000, outclass,
                f'predictions_{indomain_prefix}_{modelid}.npy')
        else:
            outputsaver = None

        log_ece.send((modelid, indomain_prefix, nbatch, outputsaver))
        with torch.no_grad():
            do_epoch(predloader, do_devalbatch, log_ece, device)

        log_ece.throw(StopIteration)

        if args.saveoutput:
            outputsaver.close()

        # do OOD deep ensemble evaluation
        predloader = get_ood_loader(args, ood_files)
        nbatch = next(predloader)

        if args.saveoutput:
            outputsaver = get_outputsaver(
                args.save_dir, SVHNInfo.count[args.svhn_split], outclass,
                f'predictions_{ood_prefix}_{modelid}.npy')
        else:
            outputsaver = None

        log_ece.send((modelid, ood_prefix, nbatch, outputsaver))
        with torch.no_grad():
            do_epoch(predloader, do_devalbatch, log_ece, device)

        log_ece.throw(StopIteration)

        if args.saveoutput:
            outputsaver.close()

        indomain_conf = confidence_from_prediction_npy(
            pjoin(args.save_dir,
                  f'predictions_{indomain_prefix}_{modelid}.npy'))
        ood_conf = confidence_from_prediction_npy(
            pjoin(args.save_dir,
                  f'predictions_{ood_prefix}_{modelid}.npy'))
        aucroc = get_roc_curve_auc_score(indomain_conf, ood_conf)[0]
        print(f'AUC-ROC score: {aucroc}')
        aucroc_scores.append(aucroc)

        print(f'>>> Time elapsed: {next(timer)[1]} <<<\n')

    print(f'{indomain_prefix}:')
    summarize_csv(pjoin(args.save_dir, f'{indomain_prefix}.csv'))
    print(f'\n{ood_prefix}:')
    summarize_csv(pjoin(args.save_dir, f'{ood_prefix}.csv'))
    mean, std = mean_std(aucroc_scores)
    print(f'\nAUC-ROC score:\tmean {mean:.4f}, std={std:.4f} \n')

    print(f'>>> Test completed at {next(timer)[0].isoformat()} <<<\n')

    log_ece.close()
