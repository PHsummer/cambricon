import cv2
import torch
from torchvision import transforms
import numpy as np
# from backbone.vgg import advancedEAST
# from utils.nms import nms
from utils.align import text_align


def color_shift(imdata):
    # imdata = cv2.imread(imPath)
    imdata = cv2.cvtColor(imdata, cv2.COLOR_BGR2HSV)
    imH, imS, imV = cv2.split(imdata)

    # yellow_floor = (11, 43, 46) 
    # yellow_ceil = (25, 255, 255)
    blue_floor = (100, 43, 46) 
    blue_ceil = (124, 255, 255)

    mask = cv2.inRange(imdata, blue_floor, blue_ceil)
    mask_base = np.array(mask/255, dtype=np.uint8)
    mask_base = mask_base^(mask_base&1==mask_base)
    mask_base = np.array(mask_base*255, dtype=np.uint8)
    imMask = imH*mask
    imBase = imH*mask_base

    #[(100, 43, 46), (124, 255, 255)] ==> [(11, 43, 46), (25, 255, 255)]
    imMask = np.array((imMask*7-568)/12, dtype=np.uint8)    
    imH = np.array(imMask+imBase, dtype=np.uint8)
    imdata = cv2.merge((imH, imS, imV))

    imdata = cv2.cvtColor(imdata, cv2.COLOR_HSV2BGR)
    return imdata


class prepros():
    def __init__(self, config, local_rank):
        self.local_rank = local_rank
        # self.east_detect = advancedEAST().cuda(local_rank)
        self.east_detect.load_state_dict(torch.load(config.DETECT_MODEL, map_location='cuda:{}'.format(local_rank)))
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(), 
            transforms.Resize((256, 256), interpolation=2),
            transforms.ToTensor()
            ])

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def imCrop(self, img_path):
        d_wight, d_height = 256, 256

        imdata = cv2.imread(img_path)
        imdata = imdata[:, :, ::-1]
        imQuad = imdata.copy()

        x = self.transform(imdata).cuda(self.local_rank)
        x = torch.unsqueeze(x, 0)  # 增加一个维度

        y = self.east_detect(x)
        y = torch.squeeze(y, 0)  # 减少一个维度
        y = y.cpu().detach().numpy()  # 7*64*64
        if y.shape[0] == 7:
            y = y.transpose((1, 2, 0))  # CHW->HWC

        y[:, :, :3] = self.sigmoid(y[:, :, :3])
        cond = np.greater_equal(y[:, :, 0], 0.9)
        activation_pixels = np.where(cond)
        quad_scores, quad_after_nms = nms(y, activation_pixels)
        
        scale_ratio_h = d_height / imQuad.shape[0]
        scale_ratio_w = d_wight / imQuad.shape[1]
        ratio = np.array([scale_ratio_w, scale_ratio_h])
        geo = quad_after_nms[np.argmax(np.sum(quad_scores, axis=1))]
        geo = np.array(geo/ratio, dtype=np.float32).astype(np.int32).reshape(-1, 4, 2)

        imAlign = text_align(imQuad[:, :, ::-1], quad=geo[0])
        
        return imAlign