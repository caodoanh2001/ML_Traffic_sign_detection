# Setup detectron2 logger
import json
from tqdm import tqdm
import numpy as np
import detectron2
from detectron2.utils.logger import setup_logger
import pickle
setup_logger()

# import some common libraries
import os
import numpy as np
import cv2
import random
from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
import argparse

from detectron2.config import get_cfg
from detectron2 import model_zoo

from skimage.feature import hog
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier

color = [(0,51,204), #cam vao
          (64,0,128), #cam do xe
          (255,255,102), #cam re
          (0,204,0), #gioi han toc do
          (153, 51, 255), #mot so bien bao khac
          (179, 0, 0), #canh bao
          (0, 163, 204)] #hieu lenh

def feature_hog (img):
  dim = (48, 48)
  ppc = 8
  img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
  fd, hog_image = hog(img, orientations=8, pixels_per_cell=(ppc,ppc),cells_per_block=(2, 2),block_norm= 'L2',visualize=True)
  return fd

def predict(video_path, test_dir, weight, threshold, config, model):
    
    if model == 'svm':
        with open('./model/classify/model_svm.pkl', 'rb') as file:
          pickle_model = pickle.load(file)
    if model == 'lr':
        with open('./model/classify/model_lr.pkl', 'rb') as file:
          pickle_model = pickle.load(file)
    if model == 'xgb':
        with open('./model/classify/model_xgb.pkl', 'rb') as file:
          pickle_model = pickle.load(file)

    cfg = get_cfg()
    cfg.merge_from_file("./configs/" + config)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
    cfg.MODEL.WEIGHTS = weight
    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.SOLVER.BASE_LR = 0.001
    cfg.SOLVER.WARMUP_ITERS = 1000
    cfg.SOLVER.MAX_ITER = 10000
    cfg.SOLVER.GAMMA = 0.05
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

    cfg.TEST.EVAL_PERIOD = 500
    predictor = DefaultPredictor(cfg)

    vid_path = os.path.basename(video_path)
    video_name = os.path.splitext(vid_path)[0]
    cap = cv2.VideoCapture(video_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(video_name +'_detect.mp4', fourcc, fps, (w, h))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_id = 0
    for frame_id in tqdm(range(frames)):   
      _, img = cap.read()
      outputs = predictor(img)
      output_pred_boxes = outputs["instances"].pred_boxes
      output_pred_scores = outputs['instances'].scores
      output_pred_classes = outputs['instances'].pred_classes
      detect = []
      for bbox, score, cls in zip(output_pred_boxes.__iter__(), output_pred_scores.__iter__(), output_pred_classes.__iter__()):
          detect.append([bbox.cpu().numpy(), float(score.cpu()), int(cls.cpu())])
      if detect:
        x1,y1,x2,y2 = detect[0][0]
        x = x1
        y = y1
        w = x2-x1
        h = y2-y1
        traffic_sign = img[int(y) : int(y + h) , int(x) : int(x + w), :]
        hog_im = feature_hog(traffic_sign)
        new_score = max(pickle_model.predict_proba(hog_im.reshape(1,-1))[0])
        new_category = pickle_model.predict(hog_im.reshape(1,-1))[0]

        cv2.rectangle(img, (int(x)-2,int(y)-18), (int(x+w)+5,int(y)), color[int(new_category)-1], -1)
        cv2.rectangle(img, (int(x),int(y)), (int(x+w),int(y+h)), color[int(new_category)-1], 2)

        new_category = int(new_category)

        if new_category == 1:
          class_name = 'Cam vao'
        if new_category == 2:
          class_name = 'Cam do xe'
        if new_category == 3:
          class_name = 'Cam re'
        if new_category == 4:
          class_name = 'Gioi han toc do'
        if new_category == 5:
          class_name = 'Cac bien bao khac'
        if new_category == 6:
          class_name = 'Canh bao'
        if new_category == 7:
          class_name = 'Hieu lenh'

        cv2.putText(img, class_name , (int(x), int(y-5)), 0, 0.6, (255, 255, 255), 2)
      out.write(img)
    
    cap.release()

def get_args():
    parser = argparse.ArgumentParser('Train')
    parser.add_argument('--video_path', type=str,
                        default='',
                        help='detect', dest='video_path')
    parser.add_argument('--test_dir', type=str,
                        default='./data',
                        help='test_dir', dest='test_dir')
    parser.add_argument('--weight', type=str,
                        default='./model/detect/model.pth',
                        help='weight', dest='weight')
    parser.add_argument('--threshold', type=int,
                        default=0.5,
                        help='weight', dest='threshold')
    parser.add_argument('--model', type=str,
                        default='svm',
                        help='', dest='model')
    parser.add_argument('--config', type=str,
                        default='COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml',
                        help='config', dest='config')
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = get_args()
    predict(args.video_path, args.test_dir, args.weight, args.threshold, args.config, args.model)