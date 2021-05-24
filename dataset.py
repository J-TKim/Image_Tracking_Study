import torch
import torch.utils.data as data
import xml.etree.ElementTree as ET
import numpy as np
import glob
import os
import cv2
from config import opt
from lib.augmentations import preproc_for_test, preproc_for_train


PERSON_LABELS = (
        'Person',
        'stroller',
        'blinder',
        'wheelchair',
        'child',
        'merchant',
    )


class CCTVDetection(data.Dataset):


    def __init__(self,is_train=True):

        self.is_train = is_train
        self.opt = opt
        self.ids = []
        root = 'dataset'
        if self.is_train:
            img_file = os.path.join(root,'train_img/*')
            ano_file = os.path.join(root,'train_label/*')
            file_list = glob.glob(img_file)
            file_list_img = [file for file in file_list if file.endswith(".jpg")]

            for i in file_list_img:
                # file_name = i[14:-4]
                file_name = os.path.split(i)[1].split('.')[0]
                img = f"{root}/train_img/{file_name}.jpg"
                ano = f"{root}/train_label/{file_name}.txt"
                # ano = os.path.join("dataset/train_label",file_name + '.txt')
                # img = os.path.join("dataset/train_img",file_name + '.jpg')
                
                if os.path.isfile(ano) and os.path.isfile(img):
                    self.ids.append((img, ano))
        else:
            img_file = os.path.join(root,'test_img/*')
            file_list = glob.glob(img_file)
            file_list_img = [file for file in file_list if file.endswith(".jpg")]
            for i in file_list_img:
                self.ids.append((i))
            
            
    def __getitem__(self, index):
        if self.is_train:
            img_path, ano_path = self.ids[index]
            image = cv2.imread(img_path, cv2.IMREAD_COLOR)
            boxes, labels = self.get_annotations(ano_path)

            if self.is_train:
                image, boxes, labels = preproc_for_train(image, boxes, labels, opt.min_size, opt.mean)
                image = torch.from_numpy(image)

            target = np.concatenate([boxes, labels.reshape(-1,1)], axis=1)

            return image, target
        else:
            img_path = self.ids[index]
            image = cv2.imread(img_path, cv2.IMREAD_COLOR)
            return image
        
    def get_annotations(self, path):
        
        f = open(path, 'r')
        det = f.readlines()
        boxes = []
        labels = []
        for d in det:
            obj = d.split(' ')
            label = int(obj[0])
            box = [float(obj[1]),float(obj[2]),float(obj[1])+float(obj[3]),float(obj[2])+float(obj[4])]
            boxes.append(box)
            labels.append(label)
        return np.array(boxes), np.array(labels)
            

        

    def __len__(self):
        return len(self.ids)
        
