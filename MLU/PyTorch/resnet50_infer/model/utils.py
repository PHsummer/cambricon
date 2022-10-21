import torch
import os   
import logging
import argparse
import json
import copy
import csv


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


def accuracy(output, target, last_batch = 0,topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0) if last_batch == 0 else last_batch

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        if last_batch > 0:
            _, pred = output[0:last_batch].topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target[0:last_batch].view(1, -1).expand_as(pred))

        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

def saveResult(imageNum,batch_size,top1,top5,meanAp,hardwaretime,endToEndTime,result_json):

    TIME=-1
    hardwareFps=-1
    hwLatencyTime = -1
    endToEndFps=-1
    e2eLatencyTime = -1
    if hardwaretime!=TIME:
        hardwareFps=imageNum/hardwaretime
        hwLatencyTime = hardwaretime / (imageNum / batch_size) * 1000
    if endToEndTime!=TIME:
        e2eLatencyTime = endToEndTime / (imageNum / batch_size) * 1000
        endToEndFps=imageNum/endToEndTime
    result={
            "Output":{
                "Accuracy":{
                    "top1":'%.2f'%top1,
                    "top5":'%.2f'%top5,
                    "meanAp":'%.2f'%meanAp
                    },
                "HostLatency(ms)":{
                    "average":'%.2f'%e2eLatencyTime,
                    "throughput(fps)":'%.2f'%endToEndFps,
                    },
                "HardwareLatency(ms)": {
                    "average":'%.2f'%hwLatencyTime,
                    "throughput(fps)":'%.2f'%hardwareFps,
                }
            }
    }

    if not os.path.exists(result_json):
        os.mknod(result_json)

    with open(result_json,"a") as outputfile:
        json.dump(result,outputfile,indent=4,sort_keys=True)
        outputfile.write('\n')
        outputfile.close()
