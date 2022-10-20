import argparse
from easydict import EasyDict as edict
import yaml
import os
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torch.distributed as dist

import backbone.crnn as crnn
import utils.utils_crnn as utils
from utils.dataloader import CRNNDataLoader
from utils import function
import config.alphabets as alphabets
from utils.utils_crnn import model_info

# import multiprocessing as mp

from tensorboardX import SummaryWriter


def parse_arg(args):
    with open(args.cfg, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        config = edict(config)

    config.DATASET.ALPHABETS = alphabets.alphabet
    config.MODEL.NUM_CLASSES = len(config.DATASET.ALPHABETS)

    return config

def main(args):

    # load config
    config = parse_arg(args)
    config['TRAIN']['BATCH_SIZE_PER_GPU']=args.batch_size
    config['DATASET']['ROOT']=args.data_path
    config['TRAIN']['END_EPOCH']=args.epochs

    # create output folder
    output_dict = utils.create_log_folder(config, phase='train')

    # cudnn
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    # writer dict
    writer_dict = {
        'writer': SummaryWriter(log_dir=output_dict['tb_dir']),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    # construct face related neural networks
    model = crnn.get_crnn(config).cuda()
    model = torch.nn.parallel.DistributedDataParallel(
        module=model, broadcast_buffers=False, device_ids=[args.local_rank], output_device=args.local_rank)

    # get device
    # if torch.cuda.is_available():
    device = torch.device("cuda:{}".format(args.local_rank))
    # else:
        # device = torch.device("cpu:0")

    # model = model.to(device)

    # define loss function
    criterion = torch.nn.CTCLoss()

    last_epoch = config.TRAIN.BEGIN_EPOCH
    optimizer = utils.get_optimizer(config, model)
    if isinstance(config.TRAIN.LR_STEP, list):
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, config.TRAIN.LR_STEP,
            config.TRAIN.LR_FACTOR, last_epoch-1
        )
    else:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, config.TRAIN.LR_STEP,
            config.TRAIN.LR_FACTOR, last_epoch - 1
        )

    if config.TRAIN.FINETUNE.IS_FINETUNE:
        model_state_file = config.TRAIN.FINETUNE.FINETUNE_CHECKPOINIT
        if model_state_file == '' and args.local_rank==0:
            print(" => no checkpoint found")
        checkpoint = torch.load(model_state_file, map_location='cpu')
        if 'state_dict' in checkpoint.keys():
            checkpoint = checkpoint['state_dict']

        from collections import OrderedDict
        model_dict = OrderedDict()
        for k, v in checkpoint.items():
            if 'cnn' in k:
                model_dict[k[4:]] = v
        model.cnn.load_state_dict(model_dict)
        if config.TRAIN.FINETUNE.FREEZE:
            for p in model.cnn.parameters():
                p.requires_grad = False

    elif config.TRAIN.RESUME.IS_RESUME:
        model_state_file = config.TRAIN.RESUME.FILE
        if model_state_file == '' and args.local_rank==0:
            print(" => no checkpoint found")
        checkpoint = torch.load(model_state_file, map_location='cpu')
        if 'state_dict' in checkpoint.keys():
            model.load_state_dict(checkpoint['state_dict'])
            last_epoch = checkpoint['epoch']
            # optimizer.load_state_dict(checkpoint['optimizer'])
            # lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        else:
            model.load_state_dict(checkpoint)

    model_info(model)
    train_dataset = CRNNDataLoader(config, is_train=True, local_rank=args.local_rank)
    valid_dataset = CRNNDataLoader(config, is_train=False, local_rank=args.local_rank)

    # DDP
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True, num_replicas=args.world_size, rank=args.rank)
    valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset, shuffle=False, num_replicas=args.world_size, rank=args.rank)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.TRAIN.BATCH_SIZE_PER_GPU,
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True,
        sampler=train_sampler)

    val_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=config.TEST.BATCH_SIZE_PER_GPU,
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True,
        sampler=valid_sampler)

    best_acc = 0.5
    converter = utils.strLabelConverter(config.DATASET.ALPHABETS)
    for epoch in range(last_epoch, config.TRAIN.END_EPOCH):

        function.train(config, train_loader, train_dataset, converter, model, criterion, optimizer, device, epoch, writer_dict, output_dict)
        lr_scheduler.step()

        acc = function.validate(config, val_loader, valid_dataset, converter, model, criterion, device, epoch, writer_dict, output_dict)

        is_best = acc > best_acc
        best_acc = max(acc, best_acc)
        if args.local_rank==0:
            print("is best:", is_best)
            print("best acc is:", best_acc)
            # save checkpoint
            # torch.save(
            #     {
            #         "state_dict": model.module.state_dict(),
            #         "epoch": epoch + 1,
            #         # "optimizer": optimizer.state_dict(),
            #         # "lr_scheduler": lr_scheduler.state_dict(),
            #         "best_acc": best_acc,
            #     },  os.path.join(output_dict['chs_dir'], "checkpoint_{}_acc_{:.4f}.pth".format(epoch, acc))
            # )
            torch.save(model.module.state_dict(),os.path.join(output_dict['chs_dir'], "checkpoint_{}_acc_{:.4f}.pth".format(epoch, acc)))

    writer_dict['writer'].close()

if __name__ == '__main__':
    # torch.multiprocessing.set_start_method('spawn')
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('--cfg',        type=str, default="./config/config_crnn.yaml",               help='experiment configuration filename')
    parser.add_argument('--batch_size', type=int, default=64,                                        help='total epoch')
    parser.add_argument('--data_path',  type=str, default="/workspace/LPDR/Database/DB_Recognition", help='train & test dataset path')
    parser.add_argument('--epochs',     type=int, default=100,                                       help='total epoch')
    parser.add_argument('--local_rank', type=int, default=0,                                         help='local_rank')
    args = parser.parse_args()

    try:
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.rank = int(os.environ['RANK'])
        dist.init_process_group('nccl')
    except KeyError:
        args.world_size = 1
        args.rank = 0
        dist.init_process_group(backend='nccl', init_method="tcp://127.0.0.1:12584", rank=args.rank, world_size=args.world_size)
    torch.cuda.set_device(args.local_rank)
    main(args)