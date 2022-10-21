import argparse
import copy
import os
import re
import random
import shutil
import sys
import time
import warnings
import math
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
import pandas as pd
from collections import OrderedDict
from torchvision.datasets import FakeData


from utils import AverageMeter, ProgressMeter, accuracy, str2bool, saveResult

prec_map = {"float32": torch.float,
            "float16": torch.half,
            "int8":    torch.int8,
            "int16":   torch.int16}

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../../tools/utils/")

import argparse
parser = argparse.ArgumentParser(description='PyTorch ImageNet Infering')

parser.add_argument('-m', '--modeldir', type=str, default='torchvision', metavar='DIR',
                        help='path to dir of models and mlu operators, default is from torchvision')
parser.add_argument('--data', default="./imagenet",
                        type=str, metavar='DIR', help='path to dataset')
parser.add_argument('-b', '--batch-size', default=16, type=int,
                        metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                             'batch size of all GPUs on the current node when '
                             'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--device', default='cpu', type=str,
                        help='Use cpu gpu or mlu device')
parser.add_argument('--device_id', default=None, type=int,
                        help='Use specified device for infering')
parser.add_argument("--jit", default=True, type=str2bool,
                        help="if use jit trace")
parser.add_argument("--jit_fuse", default=True, type=str2bool,
                        help="if use jit fuse mode")
parser.add_argument("--input_data_type", default='float32', dest = 'input_data_type', type = str,
                        help = "the input data type, float32 or float16, default float32.")
parser.add_argument("--qint", default='no_quant', dest = 'qint', 
                        help = "the quantized data type for conv/linear ops, float or int8 or int16, default float.")
parser.add_argument("--quantized_iters", default = 5, dest = 'quantized_iters', type = int,
                    help = "Set image numbers to evaluate quantized params, default is 5.")
parser.add_argument('--iters', type=int, default=1000, metavar='N',
                    help='iters per epoch')
parser.add_argument('--dummy_test', dest='dummy_test', action='store_true',
                        help='use fake data to traing')
parser.add_argument("--save_result", default=True, type=str2bool,
                    help="if save result")


model_path = parser.parse_known_args()[0].modeldir
sys.path.append(model_path)

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')

args = parser.parse_args()
if args.device == 'mlu':
    import torch_mlu
    import torch_mlu.core.mlu_model as ct
    if args.qint == 'int8' or 'int16':
            import torch_mlu.core.mlu_quantize as mlu_quantize



def main():

    e2e_time = AverageMeter('E2e' , ':6.5f')
    hardware_time  = AverageMeter('Hardware' , ':6.5f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    total_e2e_time = 0
    total_hardware_time = 0

    resize = 256
    crop = 224
    data_scale = 1.0
    use_avg = False
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    valdir = os.path.join(args.data, 'val')


    if args.device_id is None:
        args.device_id = 0  # Default Device is 0

    # Create the network
    print("=> Create the network... ")
    if args.modeldir == "torchvision":
        print("initialize resnet50")
        net = models.resnet50(pretrained=True)
    else:
        net = models.resnet50(pretrained=False)
        dict = torch.load(args.modeldir)
        net.load_state_dict(dict['state_dict'])

    net.eval().float()

    # Loading dataset
    print ("=> loading dataset...")

    # fake_dataset = FakeData(size = 5000, transform = transforms.ToTensor())
    val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Resize(resize),
                transforms.CenterCrop(crop),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
                ])),
            batch_size=args.batch_size, shuffle=False,
            pin_memory=True)

    # val_loader = torch.utils.data.DataLoader(
    #                fake_dataset,
    #                batch_size=args.batch_size,
    #                shuffle=None,
    #                sampler=None)

    total_image_number = len(val_loader) * args.batch_size

    in_h = 224
    in_w = 224
    example_input = torch.randn(args.batch_size, 3, in_h, in_w, dtype=torch.float)

    if args.jit:
        print ("=> Infer in Jit Mode...")
        # jit_fuse
        ct._jit_override_can_fuse_on_mlu(args.jit_fuse)
        model = torch.jit.trace(net, example_input, check_trace=False)
        
        if args.device == "mlu" and args.jit_fuse is True:
            print ("  => Infer Data Tpye:{}...",format(args.input_data_type))
            print ("  => Infer In MLU:{}...",format(args.device_id))
            input_shape = [args.batch_size, 3, in_h, in_w]
            config0 = torch_mlu.Input(input_shape,
                                      dtype = prec_map[args.input_data_type],
                                      format = torch.contiguous_format)
            compile_spec = {"inputs": [config0],
                            "device": {"mlu_id" : args.device_id}}
            compile_spec["truncate_long_and_double"]=True

            if args.qint == "no_quant":
                print ("  => If Quant:No...")
                compile_spec["enabled_precisions"] = {prec_map[args.input_data_type]}

            else:
                print ("  => If Quant:{}...".format(args.qint))
                if args.dummy_test:
                    calibrator = torch_mlu.ptq.DataLoaderCalibrator(
                        val_loader, algo_type=torch_mlu.ptq.CalibrationAlgo.LINEAR_CALIBRATION,
                        max_calibration_samples=args.batch_size)
                else:
                    calibrator = torch_mlu.ptq.DataLoaderCalibrator(
                        val_loader, algo_type=torch_mlu.ptq.CalibrationAlgo.LINEAR_CALIBRATION)
                compile_spec["enabled_precisions"] = {prec_map[args.input_data_type],
                                                    prec_map[args.qint]}
                compile_spec["calibrator"] = calibrator
            model = torch_mlu.ts.compile(model, **compile_spec)

            if args.input_data_type == 'float16':
                example_input = example_input.half()
            example_input = example_input.to(ct.mlu_device(), non_blocking=True)
    else:
        print ("=> Infer in Eager Mode...")
        model = net
        print ("  => Infer Data Tpye:{}...".format(args.input_data_type))

        if args.input_data_type == 'float16':
            model.half()
            example_input = example_input.half()

        if args.device == "mlu":
            print ("  => Infer In MLU:{}...".format(args.device_id))
            # Set the MLU device id
            ct.set_device(args.device_id)
            model.to(ct.mlu_device())
            example_input = example_input.to(ct.mlu_device(), non_blocking=True)

    print("=> Warmuping Up...")
    output_warm = model(example_input)

    # Doing inference
    print ("=> Inferring Process...")

    progress = ProgressMeter(
            len(val_loader),
            [e2e_time, hardware_time, top1, top5],
            prefix='Test: ')

    for i, (images, target) in enumerate(val_loader):
        last_batch = 0

        if i == args.iters:
            total_image_number = args.iters * args.batch_size
            break

        e2e_time_start = time.perf_counter()

        if images.size(0) < args.batch_size:
            for index in range(images.size(0) + 1, args.batch_size + 1):
                images = torch.cat((images, images[0].unsqueeze(0)), 0)
                target = torch.cat((target, target[0].unsqueeze(0)), 0)

        if args.input_data_type == 'float16':
            images = images.half()

        if args.device == 'mlu':
            images = images.to(ct.mlu_device(), non_blocking=True)
            target = target.to(ct.mlu_device(), non_blocking=True)

        hardware_time_start = time.perf_counter()
        output = model(images)
        hardware_time_end = time.perf_counter()

        acc1, acc5 = accuracy(output.float().to(ct.mlu_device()), target, last_batch, topk=(1, 5))
        e2e_time_end = time.perf_counter()

        e2e_time.update(e2e_time_end - e2e_time_start)
        hardware_time.update(hardware_time_end - hardware_time_start)

        total_e2e_time += e2e_time_end - e2e_time_start
        total_hardware_time += hardware_time_end - hardware_time_start

        batch_size = images.size(0)
        top1.update(acc1[0], batch_size) # images.size(0))
        top5.update(acc5[0], batch_size) # images.size(0))

        # if i % args.print_freq == 0:
        progress.display(i)
        
    result_json = os.getcwd() + "/result.json"
    if args.save_result:
        saveResult(total_image_number,args.batch_size,top1.avg.item(),top5.avg.item(),-1,total_hardware_time,total_e2e_time,result_json)


if __name__ == '__main__':

    dataset_path = args.data
    assert dataset_path != "", "imagenet dataset should be provided."

    main()

