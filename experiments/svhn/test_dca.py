import argparse
from os import listdir
from os.path import join as pjoin, isdir, exists
import torch
from dca.utils import coro_timer, mkdirp
from dca.models32 import loadmodel
from dca.calibration import bins2diagram
from dca.trainutils import coro_log, do_epoch, do_evalbatch, \
    check_cuda, deteministic_run, SummaryWriter, bn_update
from dca.dataloaders import SVHNInfo, get_svhn_train_loaders, \
    get_svhn_test_loader
from utils import get_outputsaver, summarize_csv


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('traindir', type=str,
                        help='path that collects all trained runs.')
    parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                        help='number of data loading workers')
    parser.add_argument('-b', '--batch', default=128, type=int,
                        metavar='N', help='test mini-batch size')
    parser.add_argument('-ts', '--testsamples', default=1, type=int,
                        help='create test samples via duplicating batch')
    parser.add_argument('-tr', '--testrepeat', default=64, type=int,
                        help='create test samples via process repeat')
    parser.add_argument('-sp', '--tvsplit', default=0.9, type=float,
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
    parser.add_argument('-nb', '--bins', default=20, type=int,
                        help='number of bins for ece & reliability diagram')
    parser.add_argument('-pd', '--plotdiagram', action='store_true',
                        help='plot reliability diagram for best val')
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
    ndata = SVHNInfo.counts['test']
    log_ece = coro_log(sw, args.printfreq, args.bins, args.save_dir)
    prefix = ''
    bnupd_loader, _ = get_svhn_train_loaders(
        args.data_dir, args.tvsplit, args.workers,
        (device != torch.device('cpu')), args.batch, args.batch)
    test_loader = get_svhn_test_loader(
        args.data_dir, args.workers, (device != torch.device('cpu')),
        args.batch)

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

        # do DCA sample evaluation

        prefix = 'test'

        if args.saveoutput:
            outputsaver = get_outputsaver(
                args.save_dir, ndata, outclass,
                f'predictions_{prefix}_{runfolder}.npy')
        else:
            outputsaver = None

        log_ece.send((runfolder, prefix, len(test_loader), outputsaver))
        with torch.no_grad():
            dcamodel.eval()
            do_epoch(test_loader, do_evalbatch, log_ece, device,
                     model=dcamodel, dups=args.testsamples,
                     repeat=args.testrepeat)

        bins, _, avgvloss = log_ece.throw(StopIteration)[:3]

        if args.saveoutput:
            outputsaver.close()

        if args.plotdiagram:
            bins2diagram(
                bins, False,
                pjoin(args.save_dir, f'calibration_{prefix}_{runfolder}.pdf'))

        print(f'>>> Time elapsed: {next(timer)[1]} <<<\n')

        # do DCWA evaluation

        prefix = 'dcwatest'

        if args.saveoutput:
            outputsaver = get_outputsaver(
                args.save_dir, ndata, outclass,
                f'predictions_{prefix}_{runfolder}.npy')
        else:
            outputsaver = None

        log_ece.send((runfolder, prefix, len(test_loader), outputsaver))
        with torch.no_grad():
            dcwamodel = dcamodel.wamodule()
            if args.dca_bnupdate:
                print('updating BatchNorm ...', end='')
                bn_update(bnupd_loader, dcwamodel, device=device)
                print(' Done.')
            dcwamodel.eval()
            do_epoch(test_loader, do_evalbatch, log_ece, device,
                     model=dcwamodel)
        bins, _, avgvloss = log_ece.throw(StopIteration)[:3]

        if args.saveoutput:
            outputsaver.close()

        del dcwamodel

        if args.plotdiagram:
            bins2diagram(
                bins, False,
                pjoin(args.save_dir, f'calibration_{prefix}_{runfolder}.pdf'))

        print(f'>>> Time elapsed: {next(timer)[1]} <<<\n')

    log_ece.close()

    if prefix:
        print('- DCWA results:')
        summarize_csv(pjoin(args.save_dir, f'{prefix}.csv'))
        print()
        print('- DCA results:')
        summarize_csv(pjoin(args.save_dir, f'{prefix[4:]}.csv'))
        print()

    print(f'>>> Test completed at {next(timer)[0].isoformat()} <<<\n')
