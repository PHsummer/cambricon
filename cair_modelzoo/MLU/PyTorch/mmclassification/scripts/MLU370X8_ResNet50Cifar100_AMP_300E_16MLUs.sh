#!/bin/bash
set -e

bash tools/dist_train.sh resnet50_cifar100_16*b496.py 16 --seed 42 
