import argparse
# Setup detectron2 logger
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
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

def test_map(test_dir, json_dir, name_test_data, config, weight, threshold, num_classes):
    cfg = get_cfg()
    cfg.merge_from_file("./configs/" + config)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold  # set the testing threshold for this model

    try:
        register_coco_instances(name_test_data, {}, json_dir, test_dir) # Train data
    except ValueError:
        print("Data already registerd. Continue.")

    try:
        cfg.MODEL.WEIGHTS = weight # Train data
    except ValueError:
        print("Not found weight dir")

    predictor = DefaultPredictor(cfg)

    #Call the COCO Evaluator function and pass the Validation Dataset
    evaluator = COCOEvaluator(name_test_data, cfg, False, output_dir="output/")
    val_loader = build_detection_test_loader(cfg, name_test_data)

    #Use the created predicted model in the previous step
    inference_on_dataset(predictor.model, val_loader, evaluator)

def get_args():
    parser = argparse.ArgumentParser('Train')
    parser.add_argument('--test_dir', type=str, default='./datasets/ZALO_AI/train',
                        help='Test dir', dest='test_dir')
    parser.add_argument('--json_dir', type=str, default='./datasets/ZALO_AI/annotations/train_traffic_sign_dataset.json',
                        help='Json dir', dest='json_dir')
    parser.add_argument('--name', type=str,
                        default='ZALO_AI_test',
                        help='Name data', dest='name')
    parser.add_argument('--config', type=str,
                        default='Misc/scratch_mask_rcnn_R_50_FPN_3x_gn.yaml',
                        help='config name', dest='config')
    parser.add_argument('--weight', type=str,
                        default='./output/model_final.pth',
                        help='weight', dest='weight')
    parser.add_argument('--threshold', type=str,
                        default=0.5,
                        help='threshold', dest='threshold')
    parser.add_argument('--num_classes', type=int,
                        default=1, dest='num_classes')
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = get_args()
    test_map(args.test_dir, args.json_dir, args.name, args.config, args.weight, args.threshold, args.num_classes)