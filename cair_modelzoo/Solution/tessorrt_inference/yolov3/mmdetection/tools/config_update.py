import os, sys
import argparse
from importlib import import_module

def file_modify(cur_path, module_name):
    sys.path.append(cur_path)
    config = import_module(os.path.splitext(module_name)[0])
    cont = open(os.path.join(cur_path, module_name)).readlines()
    for idx in range(len(cont)):
        try:
            if config.data_root in cont[idx]:
                cont[idx] = cont[idx].replace(config.data_root, args.data_path)
        except: pass

        try:
            if "samples_per_gpu" in cont[idx]:
                cont[idx] = cont[idx].replace(str(config.data["samples_per_gpu"]), str(args.batch_size))
        except: pass

        try:
            if "max_epochs" in cont[idx]:
                replace_num = config.runner["max_epochs"]
                if not args.epochs:
                    cont[idx] = cont[idx].replace("EpochBasedRunner", "IterBasedRunner")
                    cont[idx] = cont[idx].replace("max_epochs", "max_iters")
                    cont[idx] = cont[idx].replace(str(replace_num), str(args.iters))
                cont[idx] = cont[idx].replace(str(replace_num), str(args.epochs))
        except: pass

        try:
            if "max_iters" in cont[idx]:
                replace_num = config.runner["max_iters"]
                if not args.iters:
                    cont[idx] = cont[idx].replace("IterBasedRunner", "EpochBasedRunner")
                    cont[idx] = cont[idx].replace("max_iters", "max_epochs")
                    cont[idx] = cont[idx].replace(str(replace_num), str(args.epochs))
                cont[idx] = cont[idx].replace(str(replace_num), str(args.iters))
        except: pass

        try:
            if "interval" in cont[idx]:
                cont[idx] = cont[idx].replace(str(config.log_config["interval"]), str(args.interval))
        except: pass

    f = open(os.path.join(cur_path, module_name), "w")
    for line in cont: f.write(line)


def file_loop(base_path, config_name):
    sys.path.append(base_path)
    try:
        config_base = import_module(os.path.splitext(config_name)[0])
        config_base = config_base._base_
    except: return

    if type(config_base) is str:
        config_base = config_base.split(",")
    # config_base.append(config_name)
    file_modify(base_path, config_name)

    for config_file in config_base:
        module_path, module_name = os.path.split(config_file)
        cur_path = os.path.join(base_path, module_path)
        # print(cur_path)
        file_loop(cur_path, module_name)
        file_modify(cur_path, module_name)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./configs/yolo/yolov3_d53_mstrain-416_273e_coco.py')
    parser.add_argument('--data_path', type=str, default="/algo/modelzoo/datasets/COCO17/")
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=0)
    parser.add_argument('--iters', type=int, default=0)
    parser.add_argument('--interval', type=int, default=50)
    args = parser.parse_args()

    config_path, config_name = os.path.split(args.config)
    file_loop(config_path, config_name)
    print(args.config, "Modified.")