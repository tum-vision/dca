import argparse
from os.path import join as pjoin
import torch
import torch.nn.functional as nnf
from torch.optim import SGD
from dca.optim import schedule_midway_linear_decay, get_weightdecay
from dca.utils import coro_timer, mkdirp, rm
from dca.calibration import bins2diagram
from dca.trainutils import coro_log, do_epoch, check_cuda, bn_update, \
    do_evalbatch, SummaryWriter, deteministic_run, kldiv_logits
from dca.dataloaders import SVHNInfo, get_svhn_train_loaders
from utils import DCAMODELS, coro_trackbestloss, savecheckpoint, loadcheckpoint


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('arch', default='preresnet20_dca', choices=DCAMODELS,
                        help='model architecture: ' + ' | '.join(DCAMODELS))
    parser.add_argument('-dc', '--dcacopies', default=5, type=int,
                        metavar='N', help='number of DCA copies')
    parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                        help='number of data loading workers')
    parser.add_argument('-tb', '--tbatch', default=128, type=int,
                        metavar='N', help='train mini-batch size')
    parser.add_argument('-tr', '--trainrepeat', default=5, type=int,
                        help='repeat with DCA samples for each mini-batch')
    parser.add_argument('-vb', '--vbatch', default=128, type=int,
                        metavar='N', help='eval mini-batch size')
    parser.add_argument('-sp', '--tvsplit', default=0.9, type=float,
                        metavar='RATIO',
                        help='ratio of data used for training')
    parser.add_argument('-e', '--epochs', default=1000, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-lr', '--learning_rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('-lrr', '--lr_ratio', default=0.01, type=float,
                        metavar='LR', help='ratio of final / initial lr')
    parser.add_argument('-nwd', '--normalized_weightdecay', default=10.0,
                        type=float)
    parser.add_argument('-m', '--momentum', default=0.9, type=float,
                        metavar='M', help='momentum')
    parser.add_argument('-pf', '--printfreq', default=50, type=int,
                        metavar='N', help='print frequency')
    parser.add_argument('-r', '--resume', default='', type=str, metavar='PATH',
                        help='resume training from checkpoint')
    parser.add_argument('-d', '--device', default='cpu', type=str,
                        metavar='DEV', help='run on cpu/cuda')
    parser.add_argument('-s', '--seed', type=int,
                        help='if specified, fixes seed for reproducibility')
    parser.add_argument('-sd', '--save_dir',
                        help='The directory used to save the trained models',
                        default='save_temp', type=str)
    parser.add_argument('-dd', '--data_dir',
                        help='The directory to store dataset',
                        default='../data', type=str)
    parser.add_argument('-nb', '--bins', default=20, type=int,
                        help='number of bins for ece & reliability diagram')
    parser.add_argument('-pd', '--plotdiagram', action='store_true',
                        help='plot reliability diagram for best val')
    parser.add_argument('-tbd', '--tensorboard_dir', default='', type=str,
                        help='if specified, record data for tensorboard.')
    parser.add_argument('-dvf', '--dca_valfreq', type=int, default=5,
                        help='DCA wa model validation frequency')
    parser.add_argument('-dbu', '--dca_bnupdate', action='store_true',
                        help='update BatchNorm for averaged model')

    return parser.parse_args()


def do_trainbatch(batchinput, model, optimizer, repeat: int = 1):
    optimizer.zero_grad(set_to_none=True)
    inputs, gt = batchinput[:-1], batchinput[-1]
    cumloss = 0.0
    cumprob = torch.zeros([])
    with torch.no_grad():
        reflogits = model(*inputs)
    for _ in range(repeat):  # accumulate gradient during repeated runs
        logits = model(*inputs)
        loss = (
            nnf.cross_entropy(logits, gt) + kldiv_logits(reflogits, logits)
        ) / repeat
        loss.backward()
        cumloss += loss.item()
        prob = nnf.softmax(logits.detach(), 1)  # get likelihood
        cumprob = cumprob + prob / repeat
        reflogits = logits.detach()
    optimizer.step()
    return cumprob, gt, cumloss


if __name__ == '__main__':
    timer = coro_timer()
    t_init = next(timer)
    print(f'>>> Training initiated at {t_init.isoformat()} <<<\n')

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

    # compute weight decay
    weight_decay = get_weightdecay(
        args.normalized_weightdecay,
        int(SVHNInfo.counts['train'] * args.tvsplit))

    # resume or initialize
    if args.resume:
        startepoch, dcamodel, optimizer, scheduler, dic = loadcheckpoint(
            args.resume, device)
        modelargs, modelkwargs = dic['modelargs'], dic['modelkwargs']
        print(f'resumed from {args.resume}\n')
    else:
        startepoch = 0
        dcamodel = DCAMODELS[args.arch](
            SVHNInfo.outclass, args.dcacopies).to(args.device)
        modelargs, modelkwargs = (SVHNInfo.outclass, args.dcacopies), {}
        optimizer = SGD(dcamodel.parameters(), args.learning_rate,
                        momentum=args.momentum, weight_decay=weight_decay)
        scheduler = schedule_midway_linear_decay(optimizer, args.epochs,
                                                 end_scale=args.lr_ratio)

    # prep tensorboard if specified
    if args.tensorboard_dir:
        mkdirp(args.tensorboard_dir)
        sw = SummaryWriter(args.tensorboard_dir)
    else:
        sw = None

    # load data
    train_loader, val_loader = get_svhn_train_loaders(
        args.data_dir, args.tvsplit, args.workers,
        (device != torch.device('cpu')), args.tbatch, args.vbatch)

    # perform training
    log_ece = coro_log(sw, args.printfreq, args.bins, args.save_dir)
    trackbest = coro_trackbestloss(
        args.save_dir, args.arch, modelargs, modelkwargs, int(args.epochs*0.9))
    print(f'>>> Training starts at {next(timer)[0].isoformat()} <<<\n')

    for e in range(startepoch, args.epochs):
        # run training part
        log_ece.send((e, 'train', len(train_loader), None))
        dcamodel.train()
        do_epoch(train_loader, do_trainbatch, log_ece, device, model=dcamodel,
                 optimizer=optimizer, repeat=args.trainrepeat)
        log_ece.throw(StopIteration)
        # update lr scheduler and decay
        scheduler.step()
        # save checkpoint
        savecheckpoint(
            pjoin(args.save_dir, 'checkpoint.pt'), args.arch, modelargs,
            modelkwargs, dcamodel, optimizer, scheduler)

        print(f'>>> Time elapsed: {next(timer)[1]} <<<\n')

        # run evaluation part
        log_ece.send((e, 'val', len(val_loader), None))
        with torch.no_grad():
            dcamodel.eval()
            do_epoch(val_loader, do_evalbatch, log_ece, device, model=dcamodel)
        bins, _, avgvloss = log_ece.throw(StopIteration)[:3]
        # track best
        trackbest.send((e, dcamodel, bins, avgvloss))

        print(f'>>> Time elapsed: {next(timer)[1]} <<<\n')

        # run wa evaluation part
        if e % args.dca_valfreq == args.dca_valfreq - 1:
            log_ece.send((e, 'dcwaval', len(val_loader), None))
            with torch.no_grad():
                wamodel = dcamodel.wamodule()
                if args.dca_bnupdate:
                    print('updating BatchNorm ...', end='')
                    bn_update(train_loader, wamodel, device=device)
                    print(' Done.')
                wamodel.eval()
                do_epoch(val_loader, do_evalbatch, log_ece, device,
                         model=wamodel)
            log_ece.throw(StopIteration)

            print(f'>>> Time elapsed: {next(timer)[1]} <<<\n')

    log_ece.close()

    # visualize best eval results
    try:
        trackbest.throw(StopIteration)
    except StopIteration as e:
        _, _, bins, _ = e.value
        # plot diagrams if asked for
        if args.plotdiagram:
            bins2diagram(
                bins, False, pjoin(args.save_dir, 'calibration.pdf'))

    trackbest.close()

    # remove temporary files
    rm(pjoin(args.save_dir, 'checkpoint.pt'))

    print(f'>>> Training completed at {next(timer)[0].isoformat()} <<<\n')
