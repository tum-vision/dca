import argparse
from os import listdir
from os.path import join as pjoin, isdir, exists
import torch
import torch.nn.functional as nnf
from dca.swag import SWAG
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
    parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                        help='number of data loading workers')
    parser.add_argument('-sp', '--svhn_split', default='test',
                        choices=SVHNInfo.split,
                        help='available split: ' + ' | '.join(SVHNInfo.split))
    parser.add_argument('-b', '--batch', default=128, type=int,
                        metavar='N', help='test mini-batch size')
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
    parser.add_argument('-sms', '--swag_modelsamples', type=int, default=64,
                        help='number of swag model samples')
    parser.add_argument('-ssm', '--swag_samplemode', default='modelwise',
                        choices=SWAG.sample_mode,
                        help=f'specify at which level sampling will happen')
    parser.add_argument('-srr', '--swag_reducerank', type=int,
                        help='if specified, limit rank of off-diagonal part')
    parser.add_argument('-srs', '--swag_reducestep', type=int, default=1,
                        help='if reduce rank, step size for thinning')
    parser.add_argument('-sbu', '--swag_bnupdate', action='store_true',
                        help='update BatchNorm for averaged model')

    return parser.parse_args()


def do_swagevalbatch(batchinput, models):
    inputs, gt = batchinput[:-1], batchinput[-1]
    cumloss = 0.0
    cumprob = torch.zeros([])
    nmodel = len(models)
    for model in models:
        output = model(*inputs)
        loss = nnf.nll_loss(nnf.log_softmax(output, 1), gt) / nmodel
        cumloss += loss.item()
        cumprob = cumprob + nnf.softmax(output, 1) / nmodel
    return cumprob, gt, cumloss


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
    swag_aucroc_scores = []
    swa_aucroc_scores = []

    # iterate over all trained runs, assume model name best_model.pt
    for runfolder in sorted([d for d in listdir(args.traindir)
                             if isdir(pjoin(args.traindir, d))]):
        model_path = pjoin(args.traindir, runfolder, 'best_model.pt')
        if not exists(model_path):
            print(f'skipping {pjoin(args.traindir, runfolder)}\n')
            continue
        print(f'loading model from {model_path} ...\n')
        # resume model
        swagmodel, dic = loadmodel(model_path, device)
        if args.swag_reducerank is not None:
            swagmodel.reduce_rank(args.swag_reducerank, args.swag_reducestep)
        outclass = dic['modelargs'][0]

        print(f'>>> Test starts at {next(timer)[0].isoformat()} <<<\n')

        # sample swag models and do bn update if asked for
        with torch.no_grad():
            # sample models from swag
            sampledmodels = [swagmodel.sampled_model(mode=args.swag_samplemode)
                             for _ in range(args.swag_modelsamples)]
            # prepare them for evaluation
            for i, model in enumerate(sampledmodels):
                if args.swag_bnupdate:
                    print(f'updating BatchNorm for SWAG model sample '
                          f'{i+1}/{len(sampledmodels)} ...', end='')
                    bn_update(bnupd_loader, model, device=device)
                    print(' Done.')
                model.eval()
            print()

        # do in-domain SWAG sampled model evaluation

        if args.saveoutput:
            outputsaver = get_outputsaver(
                args.save_dir, 10000, outclass,
                f'predictions_{indomain_prefix}_{runfolder}.npy')
        else:
            outputsaver = None

        log_ece.send(
            (runfolder, indomain_prefix, len(indomain_loader), outputsaver))
        with torch.no_grad():
            do_epoch(indomain_loader, do_swagevalbatch, log_ece, device,
                     models=sampledmodels)

        log_ece.throw(StopIteration)

        if args.saveoutput:
            outputsaver.close()

        # do OOD SWAG sampled model evaluation

        if args.saveoutput:
            outputsaver = get_outputsaver(
                args.save_dir, SVHNInfo.count[args.svhn_split], outclass,
                f'predictions_{ood_prefix}_{runfolder}.npy')
        else:
            outputsaver = None

        log_ece.send((runfolder, ood_prefix, len(ood_loader), outputsaver))
        with torch.no_grad():
            do_epoch(ood_loader, do_swagevalbatch, log_ece, device,
                     models=sampledmodels)

        log_ece.throw(StopIteration)

        if args.saveoutput:
            outputsaver.close()

        del sampledmodels

        indomain_conf = confidence_from_prediction_npy(
            pjoin(args.save_dir,
                  f'predictions_{indomain_prefix}_{runfolder}.npy'))
        ood_conf = confidence_from_prediction_npy(
            pjoin(args.save_dir,
                  f'predictions_{ood_prefix}_{runfolder}.npy'))
        aucroc = get_roc_curve_auc_score(indomain_conf, ood_conf)[0]
        print(f'AUC-ROC score: {aucroc}')
        swag_aucroc_scores.append(aucroc)

        print(f'>>> Time elapsed: {next(timer)[1]} <<<\n')

        # load swa model and do bn update if required
        with torch.no_grad():
            swamodel = swagmodel.averaged_model()
            if args.swag_bnupdate:
                print('updating BatchNorm ...', end='')
                bn_update(bnupd_loader, swamodel, device=device)
                print(' Done.')
            swamodel.eval()

        # do in-domain SWA evaluation

        prefix = 'swa_' + indomain_prefix

        if args.saveoutput:
            outputsaver = get_outputsaver(
                args.save_dir, 10000, outclass,
                f'predictions_{prefix}_{runfolder}.npy')
        else:
            outputsaver = None

        log_ece.send((runfolder, prefix, len(indomain_loader), outputsaver))

        with torch.no_grad():
            do_epoch(indomain_loader, do_evalbatch, log_ece, device,
                     model=swamodel)
        log_ece.throw(StopIteration)

        if args.saveoutput:
            outputsaver.close()

        # do OOD SWA evaluation

        prefix = 'swa_' + ood_prefix

        if args.saveoutput:
            outputsaver = get_outputsaver(
                args.save_dir, SVHNInfo.count[args.svhn_split], outclass,
                f'predictions_{prefix}_{runfolder}.npy')
        else:
            outputsaver = None

        log_ece.send((runfolder, prefix, len(ood_loader), outputsaver))

        with torch.no_grad():
            do_epoch(ood_loader, do_evalbatch, log_ece, device,
                     model=swamodel)
        log_ece.throw(StopIteration)

        if args.saveoutput:
            outputsaver.close()

        del swamodel

        indomain_conf = confidence_from_prediction_npy(
            pjoin(args.save_dir,
                  f'predictions_swa_{indomain_prefix}_{runfolder}.npy'))
        ood_conf = confidence_from_prediction_npy(
            pjoin(args.save_dir,
                  f'predictions_swa_{ood_prefix}_{runfolder}.npy'))
        aucroc = get_roc_curve_auc_score(indomain_conf, ood_conf)[0]
        print(f'AUC-ROC score: {aucroc}')
        swa_aucroc_scores.append(aucroc)

        print(f'>>> Time elapsed: {next(timer)[1]} <<<\n')

    print('\n=== SWA results ===\n')
    print(f'{indomain_prefix}:')
    summarize_csv(pjoin(args.save_dir, f'swa_{indomain_prefix}.csv'))
    print(f'\n{ood_prefix}:')
    summarize_csv(pjoin(args.save_dir, f'swa_{ood_prefix}.csv'))
    mean, std = mean_std(swa_aucroc_scores)
    print(f'\nAUC-ROC score:\tmean {mean:.4f}, std={std:.4f} \n')

    print('=== SWAG results ===\n')
    print(f'{indomain_prefix}:')
    summarize_csv(pjoin(args.save_dir, f'{indomain_prefix}.csv'))
    print(f'\n{ood_prefix}:')
    summarize_csv(pjoin(args.save_dir, f'{ood_prefix}.csv'))
    mean, std = mean_std(swag_aucroc_scores)
    print(f'\nAUC-ROC score:\tmean {mean:.4f}, std={std:.4f} \n')

    print(f'>>> Test completed at {next(timer)[0].isoformat()} <<<\n')

    log_ece.close()
