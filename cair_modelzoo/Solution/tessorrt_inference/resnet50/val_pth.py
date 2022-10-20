import os, sys
import time
import copy
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from utils.metric import MetricCollector
from collections import OrderedDict
import argparse

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def validate(val_loader, model, criterion, args, epoch=None):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    loss_columns=[]
    acc_columns=[]
    time_columns=[]
    iter_columns=[]

    model.eval()
    with torch.no_grad():
        end = time.time()
        total = time.time()
        infertime = []
        for i, (images, target) in enumerate(val_loader):
            if args.device == 'gpu':
                images = images.cuda(args.device_id, non_blocking=True)
                target = target.cuda(args.device_id, non_blocking=True)
            if args.device == 'mlu':
                images = images.to("mlu:{}".format(args.device_id), non_blocking=True)
                target = target.to("mlu:{}".format(args.device_id), non_blocking=True)
            t1 = time.time()
            output = model(images)
            t2 = time.time()
            infertime.append((t2-t1)/args.batch_size)
            loss = criterion(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if i % args.print_freq == 0:
                progress.display(i)

        # this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f} Infer Time {infertime}'
              .format(top1=top1, top5=top5, infertime=np.mean(infertime)))

        metric_collector = MetricCollector(enable_only_avglog=True)
        metric_collector.insert_metrics(net = args.arch,
                                        accuracy = [top1.avg.item(), top5.avg.item()])
        metric_collector.dump()

        loss_columns.append(loss.item())
        acc_columns.append(acc1[0].cpu().numpy())
        time_columns.append(time.time()-total)
        iter_columns.append(int(i))

    # csv_save=pd.DataFrame(columns=['iter','loss','acc','time'],data=np.transpose([iter_columns,loss_columns,acc_columns,time_columns]))
    # csv_save.to_csv('epoch_'+str(epoch)+'_val.csv')

    return OrderedDict([('loss', loss.item()), ('top1', top1.avg), ('top5', top5.avg)])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('-p', '--print-freq', default=1, type=int,
                        metavar='N', help='print frequency (default: 1)')
    parser.add_argument('-b', '--batch-size', default=16, type=int,
                        metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                             'batch size of all GPUs on the current node when '
                             'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--data', default="./imagenet",
                        type=str, metavar='DIR', help='path to dataset')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading works (default: 4)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--device', default='cpu', type=str,
                        help='Use cpu gpu or mlu device')
    parser.add_argument('--device_id', default=None, type=int,
                        help='Use specified device for training, useless in multiprocessing distributed training')

    model_names = sorted(name for name in models.__dict__
        if name.islower() and not name.startswith("__")
        and callable(models.__dict__[name]))

    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
    args = parser.parse_args()
    print(args.resume)
    
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
            ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers) #, pin_memory=True)
    
    model = models.__dict__[args.arch]()
    resume_point = torch.load(args.resume, map_location=torch.device('cpu'))
    resume_point_replace = {}
    for key in resume_point['state_dict'].keys():
        split_key = key.split('.')
        split_origin = copy.deepcopy(split_key)
        for item in split_origin:
            if item == "module":
                split_key.remove("module")
            elif item == "submodule":
                split_key.remove("submodule")
        resume_point_replace[".".join(split_key)] = resume_point['state_dict'][key]
    model.load_state_dict(resume_point_replace, strict=True if args.device=='gpu' else False)
    model.to(torch.device("cuda"))

    criterion = nn.CrossEntropyLoss()

    validate(val_loader, model, criterion, args)