import argparse
from os import listdir
from os.path import join as pjoin, isdir, exists
import torch
from dca.utils import coro_timer, mkdirp
from dca.models32 import loadmodel
from dca.trainutils import do_epoch, do_evalbatch, check_cuda, \
    deteministic_run, SummaryWriter, bn_update
from dca.dataloaders import get_cifar10_test_loader, \
    get_cifar10_train_loaders
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
    parser.add_argument('-tr', '--testrepeat', default=64, type=int,
                        help='create test samples via process repeat')
    parser.add_argument('-vd', '--valdata', action='store_true',
                        help='use validation instead of test data')
    parser.add_argument('--tvsplit', default=0.9, type=float,
                        metavar='RATIO',
                        help='ratio of data used for training')
    parser.add_argument('-pf', '--printfreq', default=10, type=int,
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
    parser.add_argument('-dbu', '--dca_bnupdate', action='store_true',
                        help='update BatchNorm for averaged model')

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

    # distinguish between runs on validation data and test data

    log_ece = coro_log(sw, args.printfreq, args.save_dir)
    bnupd_loader, _ = get_cifar10_train_loaders(
        args.data_dir, args.tvsplit, args.workers,
        (device != torch.device('cpu')), args.batch, args.batch)
    indomain_prefix = 'indomain_test'
    indomain_loader = get_cifar10_test_loader(
        args.data_dir, args.workers, (device != torch.device('cpu')),
        args.batch)
    ood_prefix = 'ood_test'
    ood_loader = get_svhn_loader(
        args.data_dir, args.workers, (device != torch.device('cpu')),
        args.batch, args.svhn_split)
    dca_aucroc_scores = []
    dcwa_aucroc_scores = []

    # iterate over all trained runs, assume model name best_model.pt
    for runfolder in sorted([d for d in listdir(args.traindir)
                             if isdir(pjoin(args.traindir, d))]):
        model_path = pjoin(args.traindir, runfolder, 'best_model.pt')
        if not exists(model_path):
            print(f'skipping {pjoin(args.traindir, runfolder)}\n')
            continue
        print(f'loading model from {model_path} ...\n')
        # resume model
        dcamodel, dic = loadmodel(model_path, device)
        outclass = dic['modelargs'][0]

        print(f'>>> Test starts at {next(timer)[0].isoformat()} <<<\n')

        # do in-domain DCA sample evaluation

        if args.saveoutput:
            outputsaver = get_outputsaver(
                args.save_dir, 10000, outclass,
                f'predictions_{indomain_prefix}_{runfolder}.npy')
        else:
            outputsaver = None

        log_ece.send(
            (runfolder, indomain_prefix, len(indomain_loader), outputsaver))
        with torch.no_grad():
            dcamodel.eval()
            do_epoch(indomain_loader, do_evalbatch, log_ece, device,
                     model=dcamodel, dups=args.testsamples,
                     repeat=args.testrepeat)

        log_ece.throw(StopIteration)

        if args.saveoutput:
            outputsaver.close()

        # do OOD DCA sample evaluation

        if args.saveoutput:
            outputsaver = get_outputsaver(
                args.save_dir, SVHNInfo.count[args.svhn_split], outclass,
                f'predictions_{ood_prefix}_{runfolder}.npy')
        else:
            outputsaver = None

        log_ece.send((runfolder, ood_prefix, len(ood_loader), outputsaver))
        with torch.no_grad():
            dcamodel.eval()
            do_epoch(ood_loader, do_evalbatch, log_ece, device,
                     model=dcamodel, dups=args.testsamples,
                     repeat=args.testrepeat)

        log_ece.throw(StopIteration)

        if args.saveoutput:
            outputsaver.close()

        indomain_conf = confidence_from_prediction_npy(
            pjoin(args.save_dir,
                  f'predictions_{indomain_prefix}_{runfolder}.npy'))
        ood_conf = confidence_from_prediction_npy(
            pjoin(args.save_dir,
                  f'predictions_{ood_prefix}_{runfolder}.npy'))
        aucroc = get_roc_curve_auc_score(indomain_conf, ood_conf)[0]
        print(f'AUC-ROC score: {aucroc}\n')
        dca_aucroc_scores.append(aucroc)

        print(f'>>> Time elapsed: {next(timer)[1]} <<<\n')

        with torch.no_grad():
            dcwamodel = dcamodel.wamodule()
            if args.dca_bnupdate:
                print('updating BatchNorm ...', end='')
                bn_update(bnupd_loader, dcwamodel, device=device)
                print(' Done.')
            dcwamodel.eval()

        # do in-domain DCWA evaluation
        prefix = 'dcwa_' + indomain_prefix

        if args.saveoutput:
            outputsaver = get_outputsaver(
                args.save_dir, 10000, outclass,
                f'predictions_{prefix}_{runfolder}.npy')
        else:
            outputsaver = None

        log_ece.send((runfolder, prefix, len(indomain_loader), outputsaver))

        with torch.no_grad():
            do_epoch(indomain_loader, do_evalbatch, log_ece, device,
                     model=dcwamodel)
        log_ece.throw(StopIteration)

        if args.saveoutput:
            outputsaver.close()

        # do OOD DCWA evaluation
        prefix = 'dcwa_' + ood_prefix

        if args.saveoutput:
            outputsaver = get_outputsaver(
                args.save_dir, SVHNInfo.count[args.svhn_split], outclass,
                f'predictions_{prefix}_{runfolder}.npy')
        else:
            outputsaver = None

        log_ece.send((runfolder, prefix, len(ood_loader), outputsaver))

        with torch.no_grad():
            do_epoch(ood_loader, do_evalbatch, log_ece, device,
                     model=dcwamodel)
        log_ece.throw(StopIteration)

        if args.saveoutput:
            outputsaver.close()

        del dcamodel
        del dcwamodel

        indomain_conf = confidence_from_prediction_npy(
            pjoin(args.save_dir,
                  f'predictions_dcwa_{indomain_prefix}_{runfolder}.npy'))
        ood_conf = confidence_from_prediction_npy(
            pjoin(args.save_dir,
                  f'predictions_dcwa_{ood_prefix}_{runfolder}.npy'))
        aucroc = get_roc_curve_auc_score(indomain_conf, ood_conf)[0]
        print(f'AUC-ROC score: {aucroc}\n')
        dcwa_aucroc_scores.append(aucroc)

        print(f'>>> Time elapsed: {next(timer)[1]} <<<\n')

    print('\n=== DCWA results ===\n')
    print(f'{indomain_prefix}:')
    summarize_csv(pjoin(args.save_dir, f'dcwa_{indomain_prefix}.csv'))
    print(f'\n{ood_prefix}:')
    summarize_csv(pjoin(args.save_dir, f'dcwa_{ood_prefix}.csv'))
    mean, std = mean_std(dcwa_aucroc_scores)
    print(f'\nAUC-ROC score:\tmean {mean:.4f}, std={std:.4f} \n')

    print('=== DCA results ===\n')
    print(f'{indomain_prefix}:')
    summarize_csv(pjoin(args.save_dir, f'{indomain_prefix}.csv'))
    print(f'\n{ood_prefix}:')
    summarize_csv(pjoin(args.save_dir, f'{ood_prefix}.csv'))
    mean, std = mean_std(dca_aucroc_scores)
    print(f'\nAUC-ROC score:\tmean {mean:.4f}, std={std:.4f} \n')

    print(f'>>> Test completed at {next(timer)[0].isoformat()} <<<\n')

    log_ece.close()
