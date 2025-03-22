#imports
import sys
import warnings
warnings.filterwarnings("ignore")

import os
import ast
from skimage import io
import argparse
import datetime
import numpy as np
import pandas as pd
import time
import torch
from torchvision.models.feature_extraction import create_feature_extractor
import torchvision
# import torch.backends.cudnn as cudnn
import json
import uuid
import math

from monai.transforms import Compose, EnsureChannelFirst, RandRotate90, ResizeWithPadOrCrop, ScaleIntensity, RandFlip, RandAffine, BorderPad, Resize, adaptor, ScaleIntensityRange

from pathlib import Path

from timm.data import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma

from models.deit.datasets import build_dataset
from models.deit.engine import train_one_epoch, evaluate
from models.deit.losses import DistillationLoss
from models.deit.samplers import RASampler
from models.deit.augment import new_data_aug_generator

import models.deit.models as models
import models.deit.models_v2 as models_v2

import models.deit.utils as utils

from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score, jaccard_score
from sklearn.metrics.pairwise import cosine_similarity

import bitarray, os
from bitarray import util as butil

from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_
import torch
import torch.nn as nn
from functools import partial


seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

#paths
models_root_path = "../models/"
data_root_path = "../../../../../mnt/data/"

#functions
@torch.no_grad()
def model_test(model_name):    
    checkpoint = torch.load(os.path.join(models_root_path, model_name, "best_checkpoint.pth"), map_location='cpu')
    args = checkpoint["args"]
    device = "cpu"

    model = create_model(
        args.model,
        pretrained=False,
        num_classes=args.nb_classes, #1000 for pretrained
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        img_size=args.input_size
    )

    # model.load_state_dict(checkpoint['model_ema'])
    model.load_state_dict(checkpoint['model'])
    dataset_val, similarity_dataset, _ = build_dataset(is_train=False, is_test=True, args=args)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    data_loader = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=int(1.5 * args.batch_size),
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False
        )
    if args.bce_loss:
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Model Test:'

    # switch to evaluation mode
    model.eval()

    for batch in metric_logger.log_every(data_loader, 10, header):
        if args.data_set == 'WSS':
            images = batch["image"]
            target = batch["label"]
        
        if args.data_set == 'patch_cam' or args.data_set == 'mhist' or args.data_set == 'crc':
            images, target = batch
        
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            if args.data_set == 'crc':
                output = output
            else:
                output = output.squeeze(1) #not in crc
            loss = criterion(output, target)

        # f1, acc = accuracy(output, target, topk=(1, 5))
        if args.data_set == 'crc' or args.data_set == 'WSS':
            pred = output.softmax(dim=1)
            if args.data_set == 'crc':
                _, pred = torch.max(pred, 1)
            else:
                pred = (pred >= 0.5).int()
            pred = np.array(pred.detach().cpu(), dtype=int)
        else:
            pred = output.sigmoid()
            pred = np.array(output.detach().cpu() > 0.5, dtype=float)
        if args.data_set == 'WSS':
            f1 = f1_score(target.detach().cpu(), pred, average='samples')
            jaccard = jaccard_score(target.detach().cpu(), pred, average='samples')
        if args.data_set == 'crc':
            f1 = f1_score(y_true=np.array(target.detach().cpu()), y_pred=pred, average='weighted')
        if args.data_set == 'patch_cam' or args.data_set == 'mhist':
            f1 = f1_score(target.detach().cpu(), pred, average='binary')
        acc = accuracy_score(target.detach().cpu(), pred)
        if args.data_set == 'patch_cam' or args.data_set == 'mhist' or args.data_set == 'crc':
            bacc = balanced_accuracy_score(target.detach().cpu(), pred)

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['f1'].update(f1.item(), n=batch_size)
        metric_logger.meters['acc'].update(acc.item(), n=batch_size)
        if args.data_set == 'WSS':
            metric_logger.meters['jaccard'].update(jaccard.item(), n=batch_size)
        if args.data_set == 'patch_cam' or args.data_set == 'mhist' or args.data_set == 'crc':
            metric_logger.meters['bacc'].update(bacc.item(), n=batch_size)

    if args.data_set == 'WSS':
        print('* F1 {top1.global_avg:.4f} ACC {top5.global_avg:.4f} jaccard {top6.global_avg:.4f} loss {losses.global_avg:.4f}'
            .format(top1=metric_logger.f1, top5=metric_logger.acc, top6=metric_logger.jaccard, losses=metric_logger.loss))
    if args.data_set == 'patch_cam' or args.data_set == 'mhist' or args.data_set == 'crc':
        print('* F1 {top1.global_avg:.4f} ACC {top5.global_avg:.4f} BACC {top6.global_avg:.4f} loss {losses.global_avg:.4f}'
            .format(top1=metric_logger.f1, top5=metric_logger.acc, top6=metric_logger.bacc, losses=metric_logger.loss))

def get_args_parser():
    parser = argparse.ArgumentParser('Models test script', add_help=False)

    parser.add_argument('--model_name', default="wss", type=str)
    
    return parser

def main(args):
    print(args)

    sys.path[0] = "/home/ge26xaj/."

    model_test(args.model_name)

if __name__ == '__main__':


    parser = argparse.ArgumentParser('Similarity test script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)