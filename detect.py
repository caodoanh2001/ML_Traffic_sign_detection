# Setup detectron2 logger
import json
from tqdm import tqdm
import numpy as np
import detectron2
from detectron2.utils.logger import setup_logger
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

def detect_npy(detect_dir, test_dir, weight, threshold, config):
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

    if not os.path.exists(detect_dir):
      os.mkdir(detect_dir)

    detect_dir = detect_dir + '/'
    #In detection ra npy
    list_test = os.listdir(test_dir)
    test_dir = test_dir + '/'
    predictor = DefaultPredictor(cfg)
    for i in tqdm(range(0,len(list_test)-1)):
      img = cv2.imread(test_dir + list_test[i])
      outputs = predictor(img)
      detect = []
      output_pred_boxes = outputs["instances"].pred_boxes
      output_pred_scores = outputs['instances'].scores
      output_pred_classes = outputs['instances'].pred_classes
      for bbox, score, cls in zip(output_pred_boxes.__iter__(), output_pred_scores.__iter__(), output_pred_classes.__iter__()):
          detect.append([bbox.cpu().numpy(), float(score.cpu()), int(cls.cpu())])
      np.save(detect_dir + os.path.splitext(list_test[i])[0] + '.npy', np.array(detect), allow_pickle=True)

def get_args():
    parser = argparse.ArgumentParser('Train')
    parser.add_argument('--detect_dir', type=str,
                        default='./detect',
                        help='detect', dest='detect_dir')
    parser.add_argument('--test_dir', type=str,
                        default='./data',
                        help='test_dir', dest='test_dir')
    parser.add_argument('--weight', type=str,
                        default='./model/detect/model.pth',
                        help='weight', dest='weight')
    parser.add_argument('--threshold', type=int,
                        default=0.5,
                        help='weight', dest='threshold')
    parser.add_argument('--config', type=str,
                        default='COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml',
                        help='config', dest='config')
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = get_args()
    detect_npy(args.detect_dir, args.test_dir, args.weight, args.threshold, args.config)