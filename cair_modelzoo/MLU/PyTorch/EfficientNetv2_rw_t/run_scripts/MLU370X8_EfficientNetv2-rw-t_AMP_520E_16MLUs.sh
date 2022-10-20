export IMAGENET_TRAIN_DATASET=/data/pytorch/datasets/imagenet_training

python3 -m torch.distributed.launch --nproc_per_node=16 \
train.py ${IMAGENET_TRAIN_DATASET} \
--model efficientnetv2_rw_t -b 164 \
--sched step --epochs 520 --decay-epochs 2.4 \
--decay-rate .97 --opt rmsproptf --opt-eps .001 -j 8 \
--warmup-lr 1e-6 --weight-decay 1e-5 --drop 0.2 \
--drop-path 0.2 --model-ema --model-ema-decay 0.9999 \
--aa rand-m9-mstd0.5 --remode pixel --reprob 0.2 --lr .256 \
--amp --native-amp --seed 42 --pin-mem \
--color-jitter 0.4 --cutmix 1.0 \
--mixup 0.8 --mixup-mode batch --mixup-prob 1.  \
2>&1 | tee train.log

path=$(grep "./output/train" train.log | tail -n -1 | cut -d "/" -f 4)

python validate.py ${IMAGENET_TRAIN_DATASET} \
--model efficientnetv2_rw_t \
--checkpoint ./output/train/${path}/ \
--use-ema --amp --native-amp \
2>&1 | tee val.log
