import argparse
from os import listdir
from os.path import join as pjoin, isdir, exists
import torch
from dca.models32 import loadmodel
from dca.utils import coro_timer, mkdirp
from dca.trainutils import do_epoch, do_evalbatch, check_cuda, \
    deteministic_run, SummaryWriter
from dca.dataloaders import get_cifar10_test_loader
from utils import get_svhn_loader, get_roc_curve_auc_score, get_outputsaver, \
    summarize_csv, coro_log, SVHNInfo, confidence_from_prediction_npy, mean_std


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('traindir', type=str,
                        help='path that collects all trained runs.')
    parser.add_argument('-sp', '--svhn_split', default='test',
                        choices=SVHNInfo.split,
                        help='available split: ' + ' | '.join(SVHNInfo.split))
    parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                        help='number of data loading workers')
    parser.add_argument('-b', '--batch', default=128, type=int,
                        metavar='N', help='test mini-batch size')
    parser.add_argument('-ts', '--testsamples', default=1, type=int,
                        help='create test samples via duplicating batch')
    parser.add_argument('-tr', '--testrepeat', default=1, type=int,
                        help='create test samples via process repeat')
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
                        help='The directory to find/store datasets',
                        default='../data', type=str)
    parser.add_argument('-tbd', '--tensorboard_dir', default='', type=str,
                        help='if specified, record data for tensorboard.')

    return parser.parse_args()


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

    log_ece = coro_log(sw, args.printfreq, args.save_dir)

    # iterate over all trained runs, assume model name best_model.pt
    indomain_prefix = 'indomain_test'
    indomain_loader = get_cifar10_test_loader(
        args.data_dir, args.workers, (device != torch.device('cpu')),
        args.batch, args.testsamples)
    ood_prefix = 'ood_test'
    ood_loader = get_svhn_loader(
        args.data_dir, args.workers, (device != torch.device('cpu')),
        args.batch, args.svhn_split, args.testsamples)
    aucroc_scores = []

    for runfolder in sorted([d for d in listdir(args.traindir)
                             if isdir(pjoin(args.traindir, d))]):

        model_path = pjoin(args.traindir, runfolder, 'best_model.pt')
        if not exists(model_path):
            print(f'skipping {pjoin(args.traindir, runfolder)}\n')
            continue

        # resume model
        print(f'loading model from {model_path} ...\n')
        model, dic = loadmodel(model_path, device)
        outclass = dic['modelargs'][0]

        print(f'>>> Test starts at {next(timer)[0].isoformat()} <<<\n')

        # In-domain test run
        if args.saveoutput:
            outputsaver = get_outputsaver(
                args.save_dir, 10000, outclass,
                f'predictions_{indomain_prefix}_{runfolder}.npy')
        else:
            outputsaver = None

        log_ece.send(
            (runfolder, indomain_prefix, len(indomain_loader), outputsaver))
        with torch.no_grad():
            model.eval()
            do_epoch(indomain_loader, do_evalbatch, log_ece, device,
                     model=model, dups=args.testsamples,
                     repeat=args.testrepeat)
        log_ece.throw(StopIteration)
        if args.saveoutput:
            outputsaver.close()

        # OOD test run
        if args.saveoutput:
            outputsaver = get_outputsaver(
                args.save_dir, SVHNInfo.count[args.svhn_split], outclass,
                f'predictions_{ood_prefix}_{runfolder}.npy')
        else:
            outputsaver = None

        log_ece.send((runfolder, ood_prefix, len(ood_loader), outputsaver))
        with torch.no_grad():
            model.eval()
            do_epoch(ood_loader, do_evalbatch, log_ece, device, model=model,
                     dups=args.testsamples, repeat=args.testrepeat)
        log_ece.throw(StopIteration)
        if args.saveoutput:
            outputsaver.close()
        del model

        indomain_conf = confidence_from_prediction_npy(
            pjoin(args.save_dir,
                  f'predictions_{indomain_prefix}_{runfolder}.npy'))
        ood_conf = confidence_from_prediction_npy(
            pjoin(args.save_dir,
                  f'predictions_{ood_prefix}_{runfolder}.npy'))
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
