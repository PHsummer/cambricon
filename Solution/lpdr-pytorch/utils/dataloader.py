from __future__ import print_function, absolute_import
import torch.utils.data as data
import os
import numpy as np
import cv2
import random
# import torch
# from utils.preprocess import prepros


class CRNNDataLoader(data.Dataset):
    def __init__(self, config, is_train=True, local_rank=0):

        self.root = config.DATASET.ROOT
        self.is_train = is_train
        self.inp_h = config.MODEL.IMAGE_SIZE.H
        self.inp_w = config.MODEL.IMAGE_SIZE.W

        # self.dataset_name = config.DATASET.DATASET
        self.mean = np.array(config.DATASET.MEAN, dtype=np.float32)
        self.std = np.array(config.DATASET.STD, dtype=np.float32)

        # data list
        data_dir = self.root
        if is_train:
            self.origin_image_dir = os.path.join(data_dir, 'train')  # 'image_10000/'
            # self.origin_txt_dir = os.path.join(data_dir, 'train/labels')  # 'txt_10000/'
        else:
            self.origin_image_dir = os.path.join(data_dir, 'test')  # 'image_10000/'
            # self.origin_txt_dir = os.path.join(data_dir, 'test/labels')  # 'txt_10000/'
        print("Loading images...")
        self.imList = []
        self.listdir(self.imList, self.origin_image_dir)
        # self.imList = os.listdir(self.origin_image_dir)
        self.nSamples = len(self.imList)
        print("load {} images!".format(self.__len__()))

    def __len__(self):
        # return len(self.labels)
        return len(self.imList)

    def listdir(self, data_list, path):
        for file in os.listdir(path):  
            file_path = os.path.join(path, file)  
            if os.path.isdir(file_path):  
                self.listdir(data_list, file_path)  
            elif file_path.split(".")[-1] in ['jpg', 'png', 'bmp']:
                data_list.append(file_path)

    def __getitem__(self, idx):
        
        # img_name = self.imList[idx].strip()
        img_name = os.path.join(self.origin_image_dir, self.imList[idx])        
        lbl = os.path.split(img_name)[-1][:-4].split("_")[0]

        img = cv2.imread(img_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_h, img_w = img.shape

        img = cv2.resize(img, (0,0), fx=self.inp_w / img_w, fy=self.inp_h / img_h, interpolation=cv2.INTER_CUBIC)
        img = np.reshape(img, (self.inp_h, self.inp_w, 1))

        img = img.astype(np.float32)
        img = (img/255. - self.mean) / self.std
        img = img.transpose([2, 0, 1])

        return img, lbl








