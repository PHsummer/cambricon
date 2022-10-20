# clear all
ps aux|grep python|grep -v grep|cut -c 9-15|xargs kill -9 
rm -rf ./logs/*
rm -rf ./runs/*
rm -rf ./core*
rm -rf ./benchmark*

# train
batch_size=$1 # total batch_size, (batch_size_per_devide * devides)
epoch_num=$2
device_num=$3
export BENCHMARK_LOG=./benchmark_log
if [ ${device_num} != 1 ]; then
    python -m torch.distributed.run --nproc_per_node ${device_num} train.py --batch ${batch_size} --data coco.yaml --cfg yolov5m.yaml --device 'mlu' --weights "" --epochs ${epoch_num} --pyamp 2>&1 | tee ./logs/Yolov5m_v6.0_demo_train.log
else
    python train.py --batch ${batch_size} --data coco.yaml --cfg yolov5m.yaml --device 'mlu' --weights "" --epochs ${epoch_num} --pyamp 2>&1 | tee ./logs/Yolov5m_v6.0_demo_train.log
fi

# val
python val.py --data coco.yaml --img 640 --conf 0.001 --weights ./runs/train/exp/weights/best.pt --iou-thres 0.5 2>&1 | tee ./logs/Yolov5m_v6.0_demo_val.log

# output
precision=$(grep "IoU=0.50 " ./logs/Yolov5m_v6.0_demo_val.log|cut -d " " -f 22)
throughput=$(grep "throughput" ./benchmark_log|cut -d ":" -f 2|cut -d "," -f 1)
total_time=$(grep "epochs completed in" ./logs/Yolov5m_v6.0_demo_train.log|cut -d " " -f 5)

echo "Yolov5m(v6.0) best model mAP(iou=0.5): ${precision}"
echo "Yolov5m(v6.0) throughput per second:${throughput}"
echo "Yolov5m(v6.0) total training hours: ${total_time}"