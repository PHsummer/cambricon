import os
import cv2
import sys
import random
import numpy as np
from tqdm import tqdm
sys.path.append("./")
from utils.align import text_align

INDEX_PROVINCE = {"京": 0, "沪": 1, "津": 2, "渝": 3, "冀": 4, "晋": 5, "蒙": 6, "辽": 7, "吉": 8, "黑": 9, "苏": 10,
                  "浙": 11, "皖": 12, "闽": 13, "赣": 14, "鲁": 15, "豫": 16, "鄂": 17, "湘": 18, "粤": 19, "桂": 20,
                  "琼": 21, "川": 22, "贵": 23, "云": 24, "藏": 25, "陕": 26, "甘": 27, "青": 28, "宁": 29, "新": 30}

STATS_PROVINCE={}
for key in INDEX_PROVINCE:
    STATS_PROVINCE[key]=[]


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
    return np.array([tl, tr, br, bl], dtype="float32").reshape(1,-1)[0]


def get_dict_key(dic, value):
    key = list(dic.keys())[list(dic.values()).index(value)]
    return key


def statistic(data_path):
    # statistic
    total = 0
    for idx in os.listdir(data_path):
        dataNum = len(os.listdir(os.path.join(data_path, idx)))
        label = get_dict_key(INDEX_PROVINCE, int(idx))
        print(label, dataNum)
        total += dataNum
    print("Total:", total)


def single_crop(imdata, coor, label, save_path):
    imCrop = text_align(imdata, coor[0])

    idx = INDEX_PROVINCE[label[0]]
    savePath = os.path.join(save_path, str(idx))
    os.makedirs(savePath, exist_ok=True)
    imSave = os.path.join(savePath, label+".jpg")
    idx = 0
    while True:
        if os.path.exists(imSave):
            idx+=1
            imSave = os.path.join(savePath, label+"_"+str(idx)+".jpg")
        else:
            break
    cv2.imwrite(imSave, imCrop)

def data_crop(args, data_path, save_path):
    # proc_num = 0
    for root, dirs, files in os.walk(data_path):
        for name in tqdm(files):
            imPath = os.path.join(root, name)
            if imPath.endswith(".jpg"):
                lbPath = imPath.replace("/images/", "/labels/").replace(".jpg", ".txt")
                imdata = cv2.imread(imPath)
                h, w, _ = imdata.shape
                with open(lbPath) as f:
                    for line in f.readlines():
                        bboxs = line.strip().split()[1:5]
                        coord = line.strip().split()[5:13]
                        label = line.strip().split()[-1]
                        
                        bboxs = np.array([int(float(x)*w) if idx%2==0 else int(float(x)*h) for idx, x in enumerate(bboxs)])
                        bboxs = [bboxs[0]-bboxs[2]/2,
                                bboxs[1]-bboxs[3]/2,
                                bboxs[0]+bboxs[2]/2,
                                bboxs[1]+bboxs[3]/2]
                        coord = np.array([int(float(x)*w) if idx%2==0 else int(float(x)*h) for idx, x in enumerate(coord)], dtype=np.float32).astype(np.int32).reshape(-1, 4, 2)
                        single_crop(imdata, coord, label, save_path)


def data_augment(args, data_path, save_path, bbox_aug=2500, coor_aug=2500, rand_pixel=5):
    proc_num = 0
    for root, dirs, files in os.walk(data_path):
        for name in files:
            imPath = os.path.join(root, name)
            if imPath.endswith(".jpg"):
                lbPath = imPath.replace("/images/", "/labels/").replace(".jpg", ".txt")
                with open(lbPath) as f:
                    for line in f.readlines():
                        label = line.strip().split()[-1]
                        STATS_PROVINCE[os.path.split(label)[-1][0]].append(imPath)
                proc_num+=1
                if proc_num % 50000==0:
                    print("prossed:", proc_num)
    print("prossed:", proc_num)

    for key, value in tqdm(STATS_PROVINCE.items()):
        if len(value)>=(args.bbox_aug+args.coor_aug):
            dataList = random.sample(value, int(args.bbox_aug+args.coor_aug))
            random.shuffle(dataList)
            dataBBox = dataList[:args.bbox_aug]
            dataCoor = dataList[args.bbox_aug:]
            for imPath in dataBBox:
                lbPath = imPath.replace("/images/", "/labels/").replace(".jpg", ".txt")
                imdata = cv2.imread(imPath)
                h, w, _ = imdata.shape
                with open(lbPath) as f:
                    for line in f.readlines():
                        bboxs = line.strip().split()[1:5]
                        label = line.strip().split()[13]
                        bboxs = np.array([int(float(x)*w) if idx%2==0 else int(float(x)*h) for idx, x in enumerate(bboxs)])
                        x1 = bboxs[0]-bboxs[2]/2
                        y1 = bboxs[1]-bboxs[3]/2
                        x3 = bboxs[0]+bboxs[2]/2
                        y3 = bboxs[1]+bboxs[3]/2
                        bboxs = [x1, y1, x3, y1, x3, y3, x1, y3]
                        # random augment
                        bboxs = np.array([int(x+random.randint(-args.rand_pixel,args.rand_pixel)) for x in bboxs], dtype=np.float32).astype(np.int32).reshape(-1, 4, 2)
                        single_crop(imdata, bboxs, label , save_path)

            for imPath in dataCoor:
                lbPath = imPath.replace("/images/", "/labels/").replace(".jpg", ".txt")
                imdata = cv2.imread(imPath)
                h, w, _ = imdata.shape
                with open(lbPath) as f:
                    for line in f.readlines():
                        coord = line.strip().split()[5:13]
                        label = line.strip().split()[-1]
                        coord = np.array([int(float(x)*w) if idx%2==0 else int(float(x)*h) for idx, x in enumerate(coord)])
                        # random augment
                        coord = np.array([int(x+random.randint(-args.rand_pixel,args.rand_pixel)) for x in coord], dtype=np.float32).astype(np.int32).reshape(-1, 4, 2)
                        single_crop(imdata, coord, label, save_path)
        else:
            if len(value)==0:
                continue
            # bounding box augment
            imList = []
            remain = args.bbox_aug
            while len(value) < remain:
                imList.extend(value)
                remain = int(remain-len(value))
            imList.extend(random.sample(value, remain))

            for imPath in imList:
                lbPath = imPath.replace("/images/", "/labels/").replace(".jpg", ".txt")
                imdata = cv2.imread(imPath)
                h, w, _ = imdata.shape
                with open(lbPath) as f:
                    for line in f.readlines():
                        bboxs = line.strip().split()[1:5]
                        label = line.strip().split()[-1]
                        bboxs = np.array([int(float(x)*w) if idx%2==0 else int(float(x)*h) for idx, x in enumerate(bboxs)])
                        x1 = bboxs[0]-bboxs[2]/2
                        y1 = bboxs[1]-bboxs[3]/2
                        x3 = bboxs[0]+bboxs[2]/2
                        y3 = bboxs[1]+bboxs[3]/2
                        bboxs = [x1, y1, x3, y1, x3, y3, x1, y3]
                        # random augment
                        bboxs = np.array([int(x+random.randint(-args.rand_pixel,args.rand_pixel)) for x in bboxs], dtype=np.float32).astype(np.int32).reshape(-1, 4, 2)
                        single_crop(imdata, bboxs, label, save_path)
            # poly coord augment
            imList = []
            remain = args.coor_aug
            while len(value) < remain:
                imList.extend(value)
                remain = int(remain-len(value))
            imList.extend(random.sample(value, remain))

            for imPath in imList:
                lbPath = imPath.replace("/images/", "/labels/").replace(".jpg", ".txt")
                imdata = cv2.imread(imPath)
                h, w, _ = imdata.shape
                with open(lbPath) as f:
                    for line in f.readlines():
                        coord = line.strip().split()[5:13]
                        label = line.strip().split()[-1]
                        coord = np.array([int(float(x)*w) if idx%2==0 else int(float(x)*h) for idx, x in enumerate(coord)])
                        # random augment
                        coord = np.array([int(x+random.randint(-args.rand_pixel,args.rand_pixel)) for x in coord], dtype=np.float32).astype(np.int32).reshape(-1, 4, 2)
                        single_crop(imdata, coord, label, save_path)
