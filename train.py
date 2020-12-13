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

def train(train_dir, name_data, json_dir, config, resume_status, iteration, batch, lr):
    cfg = get_cfg()
    cfg.merge_from_file("./configs/" + config)
    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.SOLVER.BASE_LR = lr
    cfg.SOLVER.WARMUP_ITERS = 1200
    cfg.SOLVER.MAX_ITER = iteration
    cfg.SOLVER.GAMMA = 0.05
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = batch
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.DATASETS.TRAIN = (name_data,)
    cfg.DATASETS.TEST = ()

    try:
        register_coco_instances(name_data, {}, json_dir, train_dir) # Train data
    except ValueError:
        print("Data already registerd. Continue.")

    if resume_status:
        cfg.MODEL.WEIGHTS = os.path.join('output', 'model_final.pth')
    
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=resume_status)
    trainer.train()


def get_args():
    parser = argparse.ArgumentParser('Train')
    parser.add_argument('--train_dir', type=str, default='./datasets/ZALO_AI/train',
                        help='Train dir', dest='train_dir')
    parser.add_argument('--json_dir', type=str, default='./datasets/ZALO_AI/annotations/train_traffic_sign_dataset.json',
                        help='Json dir', dest='json_dir')
    parser.add_argument('--name', type=str,
                        default='ZALO_AI_train',
                        help='Name data', dest='name')
    parser.add_argument('--config', type=str,
                        default='COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml',
                        help='config name', dest='config')
    parser.add_argument('--resume', type=str,
                        default=0,
                        help='resume or not', dest='resume')
    parser.add_argument('--iter', type=int,
                        default=500,
                        help='Iteration', dest='iter')
    parser.add_argument('--batch', type=int,
                        default=128,
                        help='num of batch', dest='batch')
    parser.add_argument('--lr', type=float,
                        default=0.0125,
                        help='learning rate', dest='lr')
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = get_args()
    train(args.train_dir, args.name, args.json_dir, args.config, args.resume, args.iter, args.batch, args.lr)
