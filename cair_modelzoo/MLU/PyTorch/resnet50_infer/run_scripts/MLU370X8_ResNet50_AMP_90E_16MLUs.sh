export IMAGENET_TRAIN_DATASET=/data/pytorch/datasets/imagenet_training
MLU_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 python model/classify_train.py -a resnet50 --iters -1 --batch-size 256 --lr 1.6 --device mlu --momentum 0.9  --wd 1e-4  --seed 42 --data $IMAGENET_TRAIN_DATASET --logdir resnet50_2_card_log --epochs 61 --save_ckp --ckpdir resnet50_2_card_ckps  --multiprocessing-distributed -j 8 --dist-backend cncl --world-size 1 --rank 0 --dist-url 'tcp://127.0.0.1:27756' --pyamp
sleep 20
python model/classify_train.py -a resnet50 -e --batch-size 64 --device mlu --logdir resnet50_2_card_log --seed 42 --data $IMAGENET_TRAIN_DATASET --iters -1 --resume resnet50_2_card_ckps/resnet50_61.pth
