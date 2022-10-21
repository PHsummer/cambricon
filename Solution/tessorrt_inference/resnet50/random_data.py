import os
import random
import shutil
import argparse
from tqdm import tqdm

# get all data path
def listdir(data_list, path):
    for file in os.listdir(path):  
        file_path = os.path.join(path, file)  
        if os.path.isdir(file_path):  
            listdir(data_list, file_path)  
        elif file_path.endswith("JPEG"):
            data_list.append(file_path)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--data_src', type=str, default="/workspace/dataset/private/imagenet_training/train")
    parser.add_argument('--data_dst', type=str, default="/workspace/dataset/private/imagenet_training/posttrain")
    parser.add_argument('--samples', type=int, default=20000)
    args = parser.parse_args()

    data_list = []
    listdir(data_list, args.data_src)
    imgs_list = random.sample(data_list, args.samples)

    for data in tqdm(imgs_list):
        save_dir = os.path.join(args.data_dst, data.split("/")[-2])
        os.makedirs(save_dir, exist_ok=True)
        save_img = os.path.join(save_dir, os.path.split(data)[-1])
        shutil.copy(data, save_img)