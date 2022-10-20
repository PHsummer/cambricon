import os,sys
import cv2
import json
import random
import shutil
import argparse
import numpy as np
from tqdm import tqdm
from function import *

def write_info(file_name, file_info):
    with open('{}.json'.format(file_name), 'w') as fp:
        json.dump(file_info, fp, indent=4, sort_keys=True, ensure_ascii=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--val_ratio', type=float, default=0.05, help='')
    parser.add_argument('--idx_count', type=int, default=400000, help='')
    parser.add_argument('--json_file', type=str, default="/home/darren/Database/DB_LPR_2k/batch1/label.json", help='')
    parser.add_argument('--imgs_path', type=str, default='/home/darren/Database/DB_LPR_2k/batch1/images', help='')
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

    print("Dataset converting into detection format...")
    report_data = open(args.json_file).read().strip()
    report_data = json.loads(report_data)
    
    # write_info('report', dict(report_data))
    # report_data = json.load(open("report.json"))
    
    nameDict = {}
    for data in report_data['image_info']:
        nameDict[data['image_id']]=data['image_name']

    for idx in tqdm(range(int(len(report_data['instance_info'])/2))):
        image_labl = []
        data = report_data['instance_info'][2*idx]
        image_indx = data['image_id']
        image_name = nameDict[image_indx]
        image_bbox = data['box'][0]

        imPath = os.path.join(args.imgs_path, image_name)
        imdata = cv2.imread(imPath)
        im_h, im_w, _ = imdata.shape

        image_bbox = [float(x) for x in [image_bbox[0], image_bbox[1], image_bbox[0]+image_bbox[2], image_bbox[1]+image_bbox[3]]]
        x = (image_bbox[0]+image_bbox[2]) / 2 / im_w
        y = (image_bbox[1]+image_bbox[3]) / 2 / im_h
        w = (image_bbox[2]-image_bbox[0]) / im_w
        h = (image_bbox[3]-image_bbox[1]) / im_h
        image_bbox = [str(x) for x in [x,y,w,h]]

        for s in data['property_info']:
            image_labl.extend(s['value'])
        image_labl = ''.join(image_labl)
        
        data = report_data['instance_info'][2*idx+1]

        image_poly = [str(x) for x in data["polygon"][0]]
        if image_poly==[] or len(image_poly)!=8:
            print("Error:", data)
            continue
        image_poly = [float(x) for x in coord_rectify(image_poly)]
        image_poly = [str(x/im_w) if idx%2==0 else str(x/im_h) for idx, x in enumerate(image_poly)]

        idx_final = str(int(image_indx)+args.idx_count)
        bbox_line = ' '.join(image_bbox)
        poly_line = ' '.join(image_poly)+" "+image_labl
        with open(os.path.join(labelSave,"img_"+idx_final+".txt"), "w") as f:
            f.write("0  "+bbox_line+" "+poly_line)

        imName = os.path.join(imageSave, "img_"+idx_final+".jpg")
        shutil.copy(imPath, imName)

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

