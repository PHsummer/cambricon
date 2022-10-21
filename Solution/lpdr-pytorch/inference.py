import os
import cv2
import yaml
import time
import argparse
import numpy as np

import config.alphabets as alphabets
import utils.utils_crnn as utils
import backbone.crnn as crnn_model

from tqdm import tqdm
from PIL import Image, ImageFont, ImageDraw
from easydict import EasyDict as edict

import torch
from torch.autograd import Variable
# from torchvision import transforms
from backbone.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression_plate, scale_coords
from utils.datasets import letterbox
from utils.align import text_align
from utils.utils_yolo import eval_cal


val = eval_cal(0.5)

def sigmoid(x):
    """`y = 1 / (1 + exp(-x))`"""
    return 1 / (1 + np.exp(-x))

def load_model(weights, device):
    model = attempt_load(weights, map_location=device)  # load FP32 model
    return model

def coord_rectify(pts):
    pts = np.array(pts).reshape(4,2)
    # sort the points based on their x-coordinates
    xSorted = pts[np.argsort(pts[:, 0]), :]
    # grab the left-most and right-most points from the sorted
    # x-roodinate points
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]
    if leftMost[0, 1] != leftMost[1, 1]:
        leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    else:
        leftMost = leftMost[np.argsort(leftMost[:, 0])[::-1], :]
    (tl, bl) = leftMost
    if rightMost[0, 1] != rightMost[1, 1]:
        rightMost = rightMost[np.argsort(rightMost[:, 1]), :]
    else:
        rightMost = rightMost[np.argsort(rightMost[:, 0])[::-1], :]
    (tr, br) = rightMost
    # return the coordinates in top-left, top-right,
    # bottom-right, and bottom-left order
    return np.array([tl, tr, br, bl], dtype="float32")


def scale_coords_landmarks(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2, 4, 6]] -= pad[0]  # x padding
    coords[:, [1, 3, 5, 7]] -= pad[1]  # y padding
    coords[:, :8] /= gain
    #clip_coords(coords, img0_shape)
    coords[:, 0].clamp_(0, img0_shape[1])  # x1
    coords[:, 1].clamp_(0, img0_shape[0])  # y1
    coords[:, 2].clamp_(0, img0_shape[1])  # x2
    coords[:, 3].clamp_(0, img0_shape[0])  # y2
    coords[:, 4].clamp_(0, img0_shape[1])  # x3
    coords[:, 5].clamp_(0, img0_shape[0])  # y3
    coords[:, 6].clamp_(0, img0_shape[1])  # x4
    coords[:, 7].clamp_(0, img0_shape[0])  # y4

    return coords


def recognition(model, img):
    img_h, img_w = img.shape

    img = cv2.resize(img, (0,0), fx=config.MODEL.IMAGE_SIZE.W / img_w, fy=config.MODEL.IMAGE_SIZE.H / img_h, interpolation=cv2.INTER_CUBIC)
    img = np.reshape(img, (config.MODEL.IMAGE_SIZE.H, config.MODEL.IMAGE_SIZE.W, 1))

    img = img.astype(np.float32)
    img = (img/255. - config.DATASET.MEAN) / config.DATASET.STD
    img = img.transpose([2, 0, 1])
    img = torch.from_numpy(img).cuda()

    img = img.view(1, *img.size())
    model.eval()
    preds = model(img)
    _, preds = preds.max(2)
    preds = preds.transpose(1, 0).contiguous().view(-1)

    preds_size = Variable(torch.IntTensor([preds.size(0)]))
    sim_pred = converter.decode(preds.data, preds_size.data, raw=False)

    return sim_pred


def predict(args, detector, crnn_rec, img_path, gt):
    # Load model
    # img_size = 800
    conf_thres = 0.3
    iou_thres = 0.5

    imdata = cv2.imread(img_path)  # BGR
    h, w, _ = imdata.shape
    gt = np.array([int(float(x)*w) if idx%2==0 else int(float(x)*h) for idx, x in enumerate(gt)]).reshape(-1, 4, 2)

    img0 = imdata.copy()
    # imQuad = imdata.copy()
    assert imdata is not None, 'Image Not Found ' + img_path
    h0, w0 = imdata.shape[:2]  # orig hw
    r = args.img_size / max(h0, w0)  # resize image to img_size
    if r != 1:  # always resize down, only resize up if training with augmentation
        interp = cv2.INTER_AREA if r < 1  else cv2.INTER_LINEAR
        img0 = cv2.resize(img0, (int(w0 * r), int(h0 * r)), interpolation=interp)

    imgsz = check_img_size(args.img_size, s=detector.stride.max())  # check img_size
    img = letterbox(img0, new_shape=imgsz)[0]
    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1).copy()  # BGR to RGB, to 3x416x416

    # Run inference
    img = torch.from_numpy(img).cuda()#to(device)
    img = img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    pred = detector(img)[0]
    # Apply NMS
    pred = non_max_suppression_plate(pred, conf_thres, iou_thres)
    det = pred[0]
    # Process detections
    if len(det)==0:
        return imdata, None, None
    # Rescale boxes from img_size to im0 size
    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], imdata.shape).round()
    # Print results
    for c in det[:, -1].unique():
        n = (det[:, -1] == c).sum()  # detections per class
    det[:, 5:13] = scale_coords_landmarks(img.shape[2:], det[:, 5:13], imdata.shape).round()

    landmarks = (det[0, 5:13].view(1, 8)).view(-1).cpu().numpy().tolist()
    landmarks = coord_rectify(landmarks)
    landmarks = np.array(landmarks, dtype=np.float32).astype(np.int32).reshape(-1, 4, 2)
    val.add(landmarks, gt)

    imAlign = text_align(imdata, quad=landmarks[0])
    imQuad = cv2.polylines(imdata, landmarks, True, (0, 0, 255), thickness=2)

    # Recognition
    imGray = cv2.cvtColor(imAlign, cv2.COLOR_BGR2GRAY)
    lbPred = recognition(crnn_rec, imGray)
    # print(lbPred)
    
    pilImg=Image.fromarray(cv2.cvtColor(imQuad,cv2.COLOR_BGR2RGB))
    pilImg.size
    draw=ImageDraw.Draw(pilImg)
    font = ImageFont.truetype("./utils/simhei.ttf", 50, encoding="utf-8")
    draw.text((w-240,20), lbPred, (255, 0, 0), font=font)
    imPred = cv2.cvtColor(np.array(pilImg), cv2.COLOR_RGB2BGR)

    return imPred, imAlign, lbPred


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_size', '-i', type=int, default=800, help='')
    parser.add_argument('--det_model', '-m', type=str, default="./weights/model_yolo.pt", help='')
    parser.add_argument('--rec_model', '-r', type=str, default="./weights/model_crnn.pth", help='')
    parser.add_argument('--data_path', '-p', type=str, default='/workspace/LPDR/Database/LPR/test', help='image path')
    parser.add_argument('--save_path', '-s', type=str, default="./result/plr_val", help='')
    parser.add_argument('--visualize', '-v', type=bool, default=True, help='')
    args = parser.parse_args()

    imgs_path = args.data_path+"/images"
    labl_path = args.data_path+"/labels"
    os.makedirs(args.save_path, exist_ok=True)
    converter = utils.strLabelConverter(alphabets.alphabet)

    # Detection model load
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    yolo = load_model(args.det_model, device)

    # Recognition model load
    with open("./config/config_crnn.yaml", 'r') as f:
        config = yaml.safe_load(f)
        config = edict(config)
    config.DATASET.ALPHABETS = alphabets.alphabet
    config.MODEL.NUM_CLASSES = len(config.DATASET.ALPHABETS)
    crnn = crnn_model.get_crnn(config).cuda()
    crnn.load_state_dict(torch.load(args.rec_model, map_location='cuda:0'))
    print("Evaluating..")
    right=0
    undet=0
    false=0
    fpsList = []
    dataList = os.listdir(imgs_path)
    for img in tqdm(dataList):
        srt = time.time()
        imPath = os.path.join(imgs_path, img)
        lbPath = os.path.join(labl_path, img[:-4]+".txt")
        # label
        if os.path.exists(lbPath):
            with open(lbPath) as f:
                lines = f.readlines()[0].strip().split()
                bboxs = lines[1:5]
                coord = lines[5:13]
                label = lines[13]
        else: label = 'O'
        imQuad, imAlign, lbPred = predict(args, yolo, crnn, imPath, coord)
        if lbPred is None:
            savePath = os.path.join(args.save_path, "images/Undet")
            saveAlig = os.path.join(args.save_path, "align/Undet")
            lbPred="Undet"
            undet+=1
        elif lbPred==label:
            savePath = os.path.join(args.save_path, "images/True")
            saveAlig = os.path.join(args.save_path, "align/True")
            right+=1
        else:
            savePath = os.path.join(args.save_path, "images/False")
            saveAlig = os.path.join(args.save_path, "align/False")
            false+=1
        if args.visualize:
            quadSave = os.path.join(savePath, os.path.split(imPath)[-1])
            aligSave = os.path.join(saveAlig, os.path.split(imPath)[-1][:-4]+"_"+lbPred+".jpg")
            os.makedirs(savePath, exist_ok=True)
            os.makedirs(saveAlig, exist_ok=True)
            cv2.imwrite(quadSave, imQuad)
            cv2.imwrite(aligSave, imAlign)

        end = time.time()
        fps = 1/(end-srt)
        fpsList.append(fps)

    mPre, mRec, mF1_score, miou = val.val()
    # print('Detection \n  meanPrecision:\t{:.2f}% \n  meanRecall:\t{:.2f}% \n  meanF1-score:\t{:.2f}% \n  mIoU:\t{:.2f}'.format(mPre, mRec, mF1_score, miou))
    print('Recognition \n  Images:\t{} \n  Right:\t{} \n  Undetected:\t{} \n  False:\t{}'.format(len(dataList), right, undet, false))
    print('Total \n  Acc:\t{:.4f}%'.format(round(right/len(dataList)*100,6)))
