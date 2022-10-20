export IMAGENET_TRAIN_DATASET=/data/pytorch/datasets/imagenet_training

python model/classify_infer.py -a resnet50 --data /home/Cambricon-Test/val/ -b 10 --device mlu --device_id 0 --input_data_type float16 --iters 10 --data $IMAGENET_TRAIN_DATASET
python model/classify_infer.py -a resnet50 --data /home/Cambricon-Test/val/ -b 32 --device mlu --device_id 0 --input_data_type float16 --iters 10 --data $IMAGENET_TRAIN_DATASET
python model/classify_infer.py -a resnet50 --data /home/Cambricon-Test/val/ -b 64 --device mlu --device_id 0 --input_data_type float16 --iters 10 --data $IMAGENET_TRAIN_DATASET
