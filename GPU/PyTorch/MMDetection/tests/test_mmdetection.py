
from mmdet.apis import init_detector, inference_detector
import os 
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
config_file = 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
checkpoint_file = 'checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
device = 'cuda:0'
model = init_detector(config_file, checkpoint_file, device=device)
result = inference_detector(model, 'demo/demo.jpg')
print(result)
