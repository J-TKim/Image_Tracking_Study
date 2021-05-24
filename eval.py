import csv
from dataset import CCTVDetection
from dataset import PERSON_LABELS
from config import opt
import numpy as np
from lib.model import SSD
import torch
import torch.nn.functional as F
import os 
from lib.multibox_encoder import MultiBoxEncoder
from lib.ssd_loss import MultiBoxLoss
import cv2
from lib.utils import nms
from lib.augmentations import preproc_for_test
import matplotlib.pyplot as plt
from lib.utils import detect
import tqdm
import os
import argparse

def convert_to_xyxy(imgWidth, imgHeight, xywh):
    '''
    converts the coordinates of the bbox into a ratio and returns
    '''
    px = float(xywh[0])
    py = float(xywh[1])
    pw = float(xywh[2])
    ph = float(xywh[3])
    
    cpx = px + pw/2
    cpy = py + ph/2

    abspx = cpx / imgWidth
    abspy = cpy / imgHeight
    abspw = pw / imgWidth
    absph = ph / imgHeight  

    abspx = min(0.999999, abspx) # if abspx >= 1 else 
    abspy = min(0.999999, abspy) # 0.999999 if abspy >= 1 else abspy
    abspw = min(0.999999, abspw) # 0.999999 if abspw >= 1 else abspw
    absph = min(0.999999, absph) # 0.999999 if absph >= 1 else absph

    abspx = max(0.000001, abspx) # 0.000001 if abspx < 0 else abspx
    abspy = max(0.000001, abspy) # 0.000001 if abspy < 0 else abspy
    abspw = max(0.000001, abspw) # 0.000001 if abspw < 0 else abspw
    absph = max(0.000001, absph) # 0.000001 if absph < 0 else absph
            
    return abspx, abspy, abspw, absph

parser = argparse.ArgumentParser()

parser.add_argument('--model', 
                    default='weights/loss-1451.01.pth',
                    type=str,
                    help='model checkpoint used to eval CCTV dataset')
args = parser.parse_args()

checkpoint = args.model


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__': 
    
    print(f'using {device} to eval, use cpu may take an hour to complete !!')
    model = SSD(opt.num_classes, opt.anchor_num)
    
    # load model
    print(f'loading checkpoint from {checkpoint}')
    state_dict = torch.load(checkpoint, map_location=device) # None if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(state_dict)
    model.to(device)
    print('model loaded')
    
    multibox_encoder = MultiBoxEncoder(opt)
    test_dataset = CCTVDetection(False)

    print('start detect.........')
    
    f = open('pred.csv','w', newline='')
    wr = csv.writer(f)
    wr.writerow(["ImageID","LabelName","Conf","XMin","XMax","YMin","YMax"])
    
    for i in tqdm.tqdm(range(len(test_dataset))):

        src = test_dataset[i]
        img_name = os.path.basename(test_dataset.ids[i][19:-4])
        image = preproc_for_test(src, opt.min_size, opt.mean)
        image = torch.from_numpy(image).to(device)
        with torch.no_grad():
            loc, conf = model(image.unsqueeze(0))
        loc = loc[0]
        conf = conf[0]
        conf = F.softmax(conf, dim=1)
        conf = conf.cpu().numpy()
        loc = loc.cpu().numpy()

        decode_loc = multibox_encoder.decode(loc)
        gt_boxes, gt_confs, gt_labels = detect(decode_loc, conf, nms_threshold=0.5, gt_threshold=0.5)

        #no object detected
        if len(gt_boxes) == 0:
            continue

        h, w = src.shape[:2]
        gt_boxes[:, 0] = gt_boxes[:, 0] * w
        gt_boxes[:, 1] = gt_boxes[:, 1] * h
        gt_boxes[:, 2] = gt_boxes[:, 2] * w
        gt_boxes[:, 3] = gt_boxes[:, 3] * h
        
        for box, label, score in zip(gt_boxes, gt_labels, gt_confs):
            box[0],box[1],box[2],box[3] = convert_to_xyxy(w,h,[box[0],box[1],box[2],box[3]])
            wr.writerow([img_name,list(PERSON_LABELS)[label],f"{score:.3f}", f"{box[0]:.8f}",f"{box[1]:.8f}", f"{box2:.8f}", f"{box[3]:.8f}"])

    
    f.close()
