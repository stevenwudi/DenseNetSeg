import json
import os
import shutil
import sys
import time

import torch
import torch.backends.cudnn as cudnn
from torch import nn

from config.configuration import Configuration
from models.DenseNetSeg import DenseSeg
from utils import data_transform_utils
from utils.data_loader import SegList, SegListMS
from utils.evaluation_utils import accuracy, AverageMeter
from utils.training_utils import adjust_learning_rate, save_checkpoint

os.environ['CUDA_VISIBLE_DEVICES'] = '3'


def validate(val_loader, model, criterion, eval_score=None, print_freq=10):
    batch_time = AverageMeter()
    losses = AverageMeter()
    score = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        if type(criterion) in [torch.nn.modules.loss.L1Loss,
                               torch.nn.modules.loss.MSELoss]:
            target = target.float()
        input = input.cuda()
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        # prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        if eval_score is not None:
            score.update(eval_score(output, target_var), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'NLLLoss2d {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Score {score.val:.3f} ({score.avg:.3f})'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses,
                score=score))

    print(' * Score {top1.avg:.3f}'.format(top1=score))

    return score.avg


def train(args, train_loader, model, criterion, optimizer, epoch, eval_score=None, print_freq=10):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    scores = AverageMeter()

    # switch to train mode
    model.train()
    end = time.time()

    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if type(criterion) in [torch.nn.modules.loss.L1Loss,
                               torch.nn.modules.loss.MSELoss]:
            target = target.float()

        input = input.cuda()
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        # prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        if eval_score is not None:
            scores.update(eval_score(output, target_var), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            sys.stdout.flush()
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DataLoading {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'NLLLoss2d {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Score {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=scores), flush=True)

            # ================= tensorboard logging ====================================#
            info = {'NLLLoss2d': losses.val, 'accuracy': scores.val}
            # (1) Log the scalar values
            args.logger_tf.step += 1

            for tag, value in info.items():
                args.logger_tf.scalar_summary(tag, value, args.logger_tf.step)

                # (2) Log values and gradients of the parameters (histogram)
                # for tag, value in model.named_parameters():
                #     tag = tag.replace('.', '/')
                #     logger_tf.histo_summary(tag, to_np(value), logger_tf.step)
                #     logger_tf.histo_summary(tag+'/grad', to_np(value.grad), logger_tf.step)


def test(eval_data_loader, model, output_dir='pred', save_vis=False):
    from torch.autograd import Variable
    import logging
    from utils.evaluation_utils import CITYSCAPE_PALETTE, save_output_images, save_colorful_images
    FORMAT = "[%(asctime)-15s %(filename)s:%(lineno)d %(funcName)s] %(message)s"
    logging.basicConfig(format=FORMAT)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    end = time.time()
    for iter, (image, label, name) in enumerate(eval_data_loader):
        data_time.update(time.time() - end)
        image_var = Variable(image, requires_grad=False, volatile=True)
        final = model(image_var)
        _, pred = torch.max(final, 1)
        pred = pred.cpu().data.numpy()
        batch_time.update(time.time() - end)
        if save_vis:
            save_output_images(pred, name, output_dir)
            save_colorful_images(pred, name, output_dir + '_color', CITYSCAPE_PALETTE)

        end = time.time()
        logger.info('Eval: [{0}/{1}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    .format(iter, len(eval_data_loader), batch_time=batch_time,
                            data_time=data_time))


def train_seg(args):
    single_model = DenseSeg(model_name=args.arch,
                            classes=args.num_class,
                            transition_layer=args.transition_layer,
                            conv_num_features=args.conv_num_features,
                            out_channels_num=args.out_channels_num,
                            ppl_out_channels_num=args.ppl_out_channels_num,
                            dilation=args.dilation,
                            pretrained=True)
    model = torch.nn.DataParallel(single_model).cuda()
    if args.pretrained:
        checkpoint = torch.load(args.pretrained)
        model.load_state_dict(checkpoint['state_dict'])
    criterion = nn.NLLLoss2d(ignore_index=255)
    criterion.cuda()

    # Data loading code
    info = json.load(open(os.path.join(args.data_dir, 'info.json'), 'r'))

    # data augmentation
    t = []
    normalize = data_transform_utils.Normalize(mean=info['mean'], std=info['std'])
    if args.random_rotate > 0:
        t.append(data_transform_utils.RandomRotate(args.random_rotate))
    if args.random_scale > 0:
        t.append(data_transform_utils.RandomScale(args.random_scale))
    t.extend([data_transform_utils.RandomCrop(args.crop_size),
              data_transform_utils.RandomHorizontalFlip(),
              data_transform_utils.ToTensor(),
              normalize])

    train_loader = torch.utils.data.DataLoader(
        SegList(args.data_dir, 'train', data_transform_utils.Compose(t)),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        pin_memory=True, drop_last=True
    )

    val_loader = torch.utils.data.DataLoader(
        SegList(args.data_dir, 'val',
                data_transform_utils.Compose([data_transform_utils.ToTensor(),
                                              normalize])),
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        pin_memory=True, drop_last=True
    )

    # define loss function (criterion) and pptimizer
    optimizer = torch.optim.SGD(single_model.optim_parameters(),
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    cudnn.benchmark = True
    best_prec1 = 0
    start_epoch = 0

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if args.evaluate:
        validate(val_loader, model, criterion, eval_score=accuracy)
        return

    for epoch in range(start_epoch, args.epochs):
        lr = adjust_learning_rate(args, optimizer, epoch)
        print('Epoch: [{0}]\tlr {1:.06f}'.format(epoch, lr))
        # train for one epoch
        train(args, train_loader, model, criterion, optimizer, epoch, eval_score=accuracy)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion, eval_score=accuracy)

        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        # Model saving
        checkpoint_path = './model_save_dir/checkpoint_latest' + args.model_save_suffix
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best, filename=checkpoint_path)

        if (epoch + 1) % 10 == 0:
            history_path = './model_save_dir/checkpoint_{:03d}_{:s}'.format(epoch + 1, args.model_save_suffix)
            shutil.copyfile(checkpoint_path, history_path)


def test_seg(args):
    batch_size = args.batch_size
    num_workers = args.num_workers

    for k, v in args.__dict__.items():
        print(k, ':', v)

    single_model = DenseSeg(model_name=args.arch,
                            classes=args.num_class,
                            transition_layer=args.transition_layer,
                            conv_num_features=args.conv_num_features,
                            out_channels_num=args.out_channels_num,
                            ppl_out_channels_num=args.ppl_out_channels_num,
                            dilation=args.dilation,
                            pretrained=False)
    model = torch.nn.DataParallel(single_model).cuda()
    if args.pretrained:
        checkpoint = torch.load(args.pretrained)
        model.load_state_dict(checkpoint['state_dict'])

    data_dir = args.data_dir
    info = json.load(open(os.path.join(data_dir, 'info.json'), 'r'))
    # data augmentation
    t = []
    normalize = data_transform_utils.Normalize(mean=info['mean'], std=info['std'])
    t.extend([data_transform_utils.ToTensor(), normalize])

    if args.ms:
        scales = [0.5, 0.75, 1.25, 1.5, 1.75]
        dataset = SegListMS(data_dir, 'test', data_transform_utils.Compose(t), scales)
    else:
        dataset = SegList(data_dir, 'test', data_transform_utils.Compose(t), out_name=True)

    test_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=False
    )

    cudnn.benchmark = True

    out_dir = '{}_{}'.format(args.exp_dir[:-22], 'test')
    if len(args.test_suffix) > 0:
        out_dir += '_' + args.test_suffix
    if args.ms:
        out_dir += '_ms'

    if args.ms:
        mAP = test_ms(test_loader, model, args.classes, save_vis=True,
                      has_gt=phase != 'test' or args.with_gt,
                      output_dir=out_dir,
                      scales=scales)
    else:
        mAP = test(test_loader, model, save_vis=True, output_dir=out_dir)


def main():
    config = Configuration()

    if config.config.cmd == 'train':
        train_seg(config.config)
    elif config.config.cmd == 'test':
        test_seg(config.config)


if __name__ == '__main__':
    main()
"""
### DenseNet:
Score 95.348

### DenseNet with dilation:
Score 95.243

Directly using DenseNet and with dilation (1,1,2,4)
Score 92.404
"""

