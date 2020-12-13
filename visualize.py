import json
from tqdm import tqdm
import os
import cv2
import numpy as np
import argparse
import random

color = [(0,51,204), #cam vao
          (64,0,128), #cam do xe
          (255,255,102), #cam re
          (0,204,0), #gioi han toc do
          (153, 51, 255), #mot so bien bao khac
          (179, 0, 0), #canh bao
          (0, 163, 204)] #hieu lenh

def visualize(test_dir, outdir, jsonfile):
    list_img = os.listdir(test_dir)
    outdir = outdir + '/'

    with open(jsonfile) as json_file:
        data_predict = json.load(json_file)

    if not os.path.exists(outdir):
        os.mkdir(outdir)

    dict_bbox_predict = {}
    for image in data_predict['annotations']:
        if image['image_id'] in dict_bbox_predict:
          dict_bbox_predict[image['image_id']].append([image['bbox'], image['category_id']])
        else:
          dict_bbox_predict[image['image_id']] = [[image['bbox'], image['category_id']]]

    for i in tqdm(range(0, len(list_img)-1)):
        index = int(os.path.splitext(list_img[i])[0])
        img = cv2.imread(test_dir + '/' + list_img[i])

        if index in dict_bbox_predict:
          for bbox in dict_bbox_predict[index]:
            x, y, w, h = bbox[0]
            cv2.rectangle(img, (int(x-2),int(y)-18), (int(x+w+5),int(y)), color[bbox[1]-1], -1)
            cv2.rectangle(img, (int(x),int(y)), (int(x+w),int(y+h)), color[bbox[1]-1], 2)
            if bbox[1] == 1:
              class_name = 'Cam vao'
            if bbox[1] == 2:
              class_name = 'Cam do xe'
            if bbox[1] == 3:
              class_name = 'Cam re'
            if bbox[1] == 4:
              class_name = 'Gioi han toc do'
            if bbox[1] == 5:
              class_name = 'Cac bien bao khac'
            if bbox[1] == 6:
              class_name = 'Canh bao'
            if bbox[1] == 7:
              class_name = 'Hieu lenh'
            cv2.putText(img, class_name , (int(x), int(y-5)), 0, 0.6, (255, 255, 255), 2)

        cv2.imwrite(outdir + list_img[i], img)

def get_args():
    parser = argparse.ArgumentParser('Train')
    parser.add_argument('--test_dir', type=str, default='./data',
                        help='Train dir', dest='test_dir')
    parser.add_argument('--outdir', type=str, default='./visualize',
                        help='Out dir', dest='outdir')
    parser.add_argument('--json', type=str, default='./result_image/submission.json',
                        help='Json dir', dest='jsonfile')
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = get_args()
    visualize(args.test_dir, args.outdir, args.jsonfile)
