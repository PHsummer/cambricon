import os
import cv2
import shutil
import random
import argparse
# import numpy as np
from tqdm import tqdm
from function import *


provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]
alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',
             'X', 'Y', 'Z', 'O']
ads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
       'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']
dataList = ["ccpd_base","ccpd_blur", "ccpd_challenge", "ccpd_db", "ccpd_fn", "ccpd_rotate", "ccpd_tilt", "ccpd_weather", "ccpd_green/train", "ccpd_green/test", "ccpd_green/val"]


def name_split(name):
    name = os.path.splitext(name)[0]
    area = name.split("-")[0]
    angl = name.split("-")[1]
    bbox = name.split("-")[2]   # x1,y1,x2,y2
    coor = name.split("-")[3]   # x1,y1,x2,y2,x3,y3,x4,y4
    nump = name.split("-")[4]
    brit = name.split("-")[5]
    blur = name.split("-")[6]
    return [area, angl, bbox, coor, nump, brit, blur]


def ele_split(cont):
    return [int(item) for sublist in [x.split("&") for x in cont.split("_")] for item in sublist]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--val_ratio', type=int, default=0.05, help='')
    parser.add_argument('--imgs_path', type=str, default='/media/darren/Data/Datasets/DB_CCPD/CCPD', help='')
    parser.add_argument('--det_save', type=str, default='/workspace/LPDR/Database/DB_Detection', help='')
    parser.add_argument('--rec_save', type=str, default='/workspace/LPDR/Database/DB_Recognition', help='')
    parser.add_argument('--bbox_aug', type=int, default=2500, help='')
    parser.add_argument('--coor_aug', type=int, default=2500, help='')
    parser.add_argument('--rand_pixel', type=int, default=10, help='')
    args = parser.parse_args()
    
    imageSave = args.det_save+"/train/images"
    labelSave = args.det_save+"/train/labels"
    imageTest = args.det_save+"/test/images"
    labelTest = args.det_save+"/test/labels"
    os.makedirs(imageSave, exist_ok=True)
    os.makedirs(labelSave, exist_ok=True)
    os.makedirs(imageTest, exist_ok=True)
    os.makedirs(labelTest, exist_ok=True)

    idx_count=0
    for data in tqdm(dataList):
        dataDir = os.path.join(args.imgs_path, data)
        for img in os.listdir(dataDir):
            imageNew = os.path.join(imageSave, "img_"+str(idx_count)+".jpg")
            labelNew = os.path.join(labelSave, "img_"+str(idx_count)+".txt")
            imPath = os.path.join(dataDir, img)
            shutil.copy(imPath, imageNew)

            imdata = cv2.imread(imPath)
            im_h, im_w, _ = imdata.shape
            nameList = name_split(img)
            numBBoxs = ele_split(nameList[2])
            numCoord = ele_split(nameList[3])
            numPlate = ele_split(nameList[4])
            x = (numBBoxs[0]+numBBoxs[2]) / 2 / im_w
            y = (numBBoxs[1]+numBBoxs[3]) / 2 / im_h
            w = (numBBoxs[2]-numBBoxs[0]) / im_w
            h = (numBBoxs[3]-numBBoxs[1]) / im_h
            numBBoxs = [str(x) for x in [x,y,w,h]]
            numCoord = [float(x) for x in coord_rectify(numCoord)]
            numCoord = [str(x/im_w) if idx%2==0 else str(x/im_h) for idx, x in enumerate(numCoord)]
            numBBoxs = ' '.join([str(x) for x in numBBoxs])
            numCoord = ' '.join([str(x) for x in numCoord])
            numPlateList = [provinces[numPlate[0]], alphabets[numPlate[1]]]
            numPlateList.extend([ads[x] for x in numPlate[2:]])
            label = ''.join(numPlateList)
            outline = "0  "+numBBoxs+" "+numCoord+" "+label
            with open(labelNew,"w") as f:
                f.write(outline+"\n")
            idx_count+=1

    # testset split
    dataList = os.listdir(imageSave)
    val_num = int(len(dataList)*args.val_ratio)
    testList = random.sample(dataList, val_num)
    for data in tqdm(testList):
        gtName = data.split(".")[0]+".txt"
        shutil.move(os.path.join(imageSave, data), imageTest)
        shutil.move(os.path.join(labelSave, gtName), labelTest)
    print("Testset splited {} images".format(val_num))

    # recognition dataset convert
    imRecSave = args.rec_save+"/train/images"
    imRecTest = args.rec_save+"/test/images"
    os.makedirs(imRecSave, exist_ok=True)
    os.makedirs(imRecTest, exist_ok=True)

    print("\nRecognition trainset converting...")
    data_augment(args, imageSave, imRecSave)
    statistic(imRecSave)

    print("\nRecognition testset converting...")
    data_crop(args, imageTest, imRecTest)
    statistic(imRecTest)