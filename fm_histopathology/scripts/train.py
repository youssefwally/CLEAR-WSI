from __future__ import print_function

#imports
from arguments import arg_parse
from data import get_data
from dataloader import BasicDataset
from projects.fm_histopathology.scripts.patchcam_h5_dataloader import PatchcamH5Dataset
from lmdb_dataloader import get_ids
from functools import partial

from projects.fm_histopathology.scripts.models.vit import ViT
from projects.fm_histopathology.scripts.models.mocov3.moco.builder import MoCo_ViT
import projects.fm_histopathology.scripts.models.mocov3.vits as vits
from projects.fm_histopathology.scripts.loss_functions.contrastive_loss import ContrastiveLoss


import glob
from itertools import chain
import os
import random
import wandb
import zipfile

import torch
import torchvision
from torch import nn
from torchvision.ops import sigmoid_focal_loss
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import nibabel as nib
import SimpleITK as sitk
from monai.utils import set_determinism
from monai.transforms import Compose, EnsureChannelFirst, RandRotate90, ResizeWithPadOrCrop, ScaleIntensity, RandFlip, RandAffine, BorderPad, Resize, adaptor, ScaleIntensityRange
from monai.transforms import EnsureChannelFirstd, RandRotate90d, ScaleIntensityd, RandFlipd, RandAffined, BorderPadd, Resized, ResizeWithPadOrCropd, LoadImaged, Resized, ScaleIntensityRanged
from monai.data import LMDBDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sklearn
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score, balanced_accuracy_score

from PIL import Image
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from tqdm.notebook import tqdm

from functools import partial
from torch.distributions.normal import Normal

import argparse
#import cv2
import numpy as np
import torch

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn


##################################################################################
#checks

print(f"Torch: {torch.__version__}")
print(f"Cuda Available: {torch.cuda.is_available()}")

##################################################################################

#functions
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    set_determinism(seed=seed)

def NansToZeros():
    def nanstozeros(img):
        img = np.nan_to_num(np.array(img))
        return img
    return nanstozeros

##################################################################################

#Train function
def train(model, optimizer, dataloader, loader="h5", loss_fn = nn.BCEWithLogitsLoss()):
    """Train network on training dataset."""
    model.train()
    cumulative_loss = 0.0
    all_predicted_class_labels = np.asarray([])
    all_loss_labels = np.asarray([])
    for batch_data in dataloader:
        if(loader == "lmdb"):
            data = batch_data["image"]
            label = batch_data["label"]
            data = data.to(device)
            label = label.to(device)
        elif(loader == "h5"):   
            sample, label = batch_data
            data = sample 
            data = data.to(device)
            label = label.to(device)
        else:   
            data, label = batch_data 
            data = data.to(device)
            label = label.to(device)

        output = model(data)
            
        if(wandb.config.binary):
            loss = loss_fn(output.squeeze(1), label.float())
        else:
            loss = loss_fn(output.squeeze(1), label)
        
        # loss = sigmoid_focal_loss(output.squeeze(1), label, alpha=0.25, gamma=2, reduction='mean')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        if(wandb.config.binary):
            predicted_class_labels = output.squeeze(1)
            predicted_class_labels = torch.nn.Sigmoid()(predicted_class_labels)
            predicted_class_labels = torch.round(predicted_class_labels)
        else:
            predicted_class_scores = output.squeeze(1)
            predicted_class_probs = torch.nn.Softmax(dim=1)(predicted_class_scores)
            predicted_class_labels = torch.argmax(predicted_class_probs, dim=1)
                
        all_predicted_class_labels = np.append(all_predicted_class_labels, predicted_class_labels.detach().cpu())
        all_loss_labels = np.append(all_loss_labels, label.detach().cpu())
        
        cumulative_loss += loss.item()

    acc = balanced_accuracy_score(all_loss_labels.flatten(), all_predicted_class_labels.flatten(), adjusted=False)
    return cumulative_loss / len(dataloader), acc
    
############################################################################################

#Validation function
def calculate_val_loss(model, dataloader, loader="h5", loss_fn = nn.BCEWithLogitsLoss()):
    model.eval()
    cumulative_loss = 0.0
    all_predicted_class_labels = np.asarray([])
    all_loss_labels = np.asarray([])
    with torch.no_grad():
        for batch_data in dataloader:
            if(loader == "lmdb"):
                data = batch_data["image"]
                label = batch_data["label"]
                data = data.to(device)
                label = label.to(device)
            elif(loader == "h5"):   
                sample, label = batch_data
                data = sample 
                data = data.to(device)
                label = label.to(device)
            else:   
                data, label = batch_data 
                data = data.to(device)
                label = label.to(device)

            output = model(data)

            if(wandb.config.binary):
                loss = loss_fn(output.squeeze(1), label.float())
            else:
                loss = loss_fn(output.squeeze(1), label)

            if(wandb.config.binary):
                predicted_class_labels = output.squeeze(1)
                predicted_class_labels = torch.nn.Sigmoid()(predicted_class_labels)
                predicted_class_labels = torch.round(predicted_class_labels)
            else:
                predicted_class_scores = output.squeeze(1)
                predicted_class_probs = torch.nn.Softmax(dim=1)(predicted_class_scores)
                predicted_class_labels = torch.argmax(predicted_class_probs, dim=1)

            all_predicted_class_labels = np.append(all_predicted_class_labels, predicted_class_labels.detach().cpu())
            all_loss_labels = np.append(all_loss_labels, label.detach().cpu())

            cumulative_loss += loss.item()
        
    acc = balanced_accuracy_score(all_loss_labels.flatten(), all_predicted_class_labels.flatten(), adjusted=False)
    cm = confusion_matrix(all_loss_labels.flatten(), all_predicted_class_labels.flatten())
    return cumulative_loss / len(dataloader), acc, cm
    
############################################################################################

#Test function
def test(model, dataloader, loader="h5", loss_fn = nn.BCEWithLogitsLoss()):
    model.eval()
    cumulative_loss = 0.0
    all_predicted_class_labels = np.asarray([])
    all_loss_labels = np.asarray([])
    with torch.no_grad():
        for batch_data in dataloader:
            if(loader == "lmdb"):
                data = batch_data["image"]
                label = batch_data["label"]
                data = data.to(device)
                label = label.to(device)
            elif(loader == "h5"):   
                sample, label = batch_data
                data = sample
                data = data.to(device)
                label = label.to(device)
            else:   
                data, label = batch_data 
                data = data.to(device)
                label = label.to(device)

            output = model(data)
            
            if(wandb.config.binary):
                loss = loss_fn(output.squeeze(1), label.float())
            else:
                loss = loss_fn(output.squeeze(1), label)

            if(wandb.config.binary):
                predicted_class_labels = output.squeeze(1)
                predicted_class_labels = torch.nn.Sigmoid()(predicted_class_labels)
                predicted_class_labels = torch.round(predicted_class_labels)
            else:
                predicted_class_scores = output.squeeze(1)
                predicted_class_probs = torch.nn.Softmax(dim=1)(predicted_class_scores)
                predicted_class_labels = torch.argmax(predicted_class_probs, dim=1)

            all_predicted_class_labels = np.append(all_predicted_class_labels, predicted_class_labels.detach().cpu())
            all_loss_labels = np.append(all_loss_labels, label.detach().cpu())

            cumulative_loss += loss.item()
        
    acc = balanced_accuracy_score(all_loss_labels.flatten(), all_predicted_class_labels.flatten(), adjusted=False)
    cm = confusion_matrix(all_loss_labels.flatten(), all_predicted_class_labels.flatten())
    return cumulative_loss / len(dataloader), acc, cm
    
############################################################################################

if __name__ == '__main__':
    args = arg_parse()
    if(not args.lrz):
        f = open("output.txt", "a")

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')

    run = wandb.init(
        project="ad_prediction_multimodality_3d_minit",
        entity="yussufwaly",
        notes="multimodality_3d_minit",
        tags=[],
        config=args,
        mode=args.wandb_mode
        )
    
    seed_everything(wandb.config.seed)
    errors = []

    train_dataset, val_dataset, test_dataset = get_data(wandb.config.data_path, split=wandb.config.split)

    if(wandb.config.loader == "lmdb"):
        train_transforms = Compose([LoadImaged(keys=["image"], image_only=True), EnsureChannelFirstd(keys=["image"], channel_dim='no_channel'), adaptor(NansToZeros(), "image", {"image": "img"}), ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=2.0, b_min=0.0, b_max=1.0), 
                                         RandRotate90d(keys=["image"], prob=0.5), RandFlipd(keys=["image"], prob=0.5), RandAffined(keys=["image"], prob=0.5), ResizeWithPadOrCropd(keys=["image"], spatial_size = (128, 128, 128))])
        val_transforms   = Compose([LoadImaged(keys=["image"], image_only=True), EnsureChannelFirstd(keys=["image"], channel_dim='no_channel'), adaptor(NansToZeros(), "image", {"image": "img"}), ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=2.0, b_min=0.0, b_max=1.0), 
                                         RandRotate90d(keys=["image"], prob=0.5), RandFlipd(keys=["image"], prob=0.5), RandAffined(keys=["image"], prob=0.5), ResizeWithPadOrCropd(keys=["image"], spatial_size = (128, 128, 128))])
        test_transforms  = Compose([LoadImaged(keys=["image"], image_only=True), EnsureChannelFirstd(keys=["image"], channel_dim='no_channel'), adaptor(NansToZeros(), "image", {"image": "img"}), ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=2.0, b_min=0.0, b_max=1.0), 
                                         ResizeWithPadOrCropd(keys=["image"], spatial_size = (128, 128, 128))])
        
        lmdb_train_dataset_ids = get_ids(wandb.config.data_path, train_dataset)
        lmdb_val_dataset_ids = get_ids(wandb.config.data_path, val_dataset)
        lmdb_test_dataset_ids = get_ids(wandb.config.data_path, test_dataset)

        train_data = LMDBDataset(data=lmdb_train_dataset_ids, transform=train_transforms, lmdb_kwargs={"map_async": True})
        valid_data = LMDBDataset(data=lmdb_val_dataset_ids, transform=val_transforms, lmdb_kwargs={"map_async": True})
        test_data = LMDBDataset(data=lmdb_test_dataset_ids, transform=test_transforms, lmdb_kwargs={"map_async": True})

    elif(wandb.config.loader == "h5"):
        if(wandb.config.binary):
            train_data = PatchcamH5Dataset(f"{wandb.config.data_path}/{wandb.config.split}-train.h5", True, wandb.config.num_classes, True, True)
            valid_data = PatchcamH5Dataset(f"{wandb.config.data_path}/{wandb.config.split}-valid.h5", True, wandb.config.num_classes, True, True)
            test_data = PatchcamH5Dataset( f"{wandb.config.data_path}/{wandb.config.split}-test.h5" , False, wandb.config.num_classes, True, True)
        else:
            train_data = PatchcamH5Dataset(f"{wandb.config.data_path}/{wandb.config.split}-train.h5", True, wandb.config.num_classes, True, True)
            valid_data = PatchcamH5Dataset(f"{wandb.config.data_path}/{wandb.config.split}-valid.h5", True, wandb.config.num_classes, True, True)
            test_data = PatchcamH5Dataset( f"{wandb.config.data_path}/{wandb.config.split}-test.h5", False, wandb.config.num_classes, True, True)
    
    else:
        train_transforms = Compose([EnsureChannelFirst(channel_dim='no_channel'), RandRotate90(prob=0.5), RandFlip(prob=0.5), RandAffine(prob=0.5), ResizeWithPadOrCrop(spatial_size = (128, 128, 128)), ScaleIntensityRange(a_min=-10.0, a_max=10.0, b_min=0.0, b_max=1.0)])
        val_transforms   = Compose([EnsureChannelFirst(channel_dim='no_channel'), RandRotate90(prob=0.5), RandFlip(prob=0.5), RandAffine(prob=0.5), ResizeWithPadOrCrop(spatial_size = (128, 128, 128)), ScaleIntensityRange(a_min=-10.0, a_max=10.0, b_min=0.0, b_max=1.0)])
        test_transforms  = Compose([EnsureChannelFirst(channel_dim='no_channel'), RandRotate90(prob=0.5), RandFlip(prob=0.5), RandAffine(prob=0.5), ResizeWithPadOrCrop(spatial_size = (128, 128, 128)), ScaleIntensityRange(a_min=-10.0, a_max=10.0, b_min=0.0, b_max=1.0)])
        
        train_data = BasicDataset(train_dataset, wandb.config.data_path, wandb.config.label, transform=train_transforms)
        valid_data = BasicDataset(val_dataset,   wandb.config.data_path, wandb.config.label, transform=val_transforms)
        test_data = BasicDataset(test_dataset,   wandb.config.data_path, wandb.config.label, transform=test_transforms)
    
    train_loader = DataLoader(dataset = train_data, batch_size=wandb.config.batch_size, shuffle=True)
    valid_loader = DataLoader(dataset = valid_data, batch_size=wandb.config.batch_size, shuffle=True)
    test_loader = DataLoader(dataset = test_data, batch_size=wandb.config.batch_size, shuffle=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    wandb.config.update( {'device': device }, allow_val_change=True)

    if(wandb.config.model == "original_vit"):
        model = ViT(
                    image_size = wandb.config.img_size,
                    patch_size = wandb.config.patch_size,
                    num_classes = wandb.config.num_classes,
                    dim = wandb.config.dim,
                    depth = wandb.config.depth,
                    heads = wandb.config.heads,
                    mlp_dim = wandb.config.dim * 4,
                    dropout = wandb.config.dropout,
                    emb_dropout = wandb.config.emb_dropout,
                    binary = wandb.config.binary
                    )
    if(wandb.config.model == "moco_vit"):
        model = MoCo_ViT(
                    partial(
                        vits.__dict__[wandb.config.arch], 
                        stop_grad_conv1=wandb.config.stop_grad_conv1
                        ),
                    wandb.config.input_dim, wandb.config.moco_mlp_dim, wandb.config.moco_t
                    )
        
    if( wandb.config.lrz):
        print(model)
        pytorch_total_params = sum(p.numel() for p in model.parameters())
        print(pytorch_total_params)
        pytorch_total_train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(pytorch_total_train_params)
    else:    
        print(model, file=f)
        pytorch_total_params = sum(p.numel() for p in model.parameters())
        print(pytorch_total_params, file=f)
        pytorch_total_train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(pytorch_total_train_params, file=f)

    if(wandb.config.binary):
        loss_fn = nn.BCEWithLogitsLoss()
    else:
        loss_fn = nn.CrossEntropyLoss()
    # loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(5.0, device=device))

    if wandb.config.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(),
                              lr=wandb.config.lr, momentum=wandb.config.momentum, weight_decay=wandb.config.weight_decay)
    elif wandb.config.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(),
                               lr=wandb.config.lr)
    elif wandb.config.optimizer == "adamW":
        optimizer = torch.optim.AdamW(model.parameters(),
                               lr=wandb.config.lr, weight_decay=wandb.config.weight_decay)
        
    if wandb.config.scheduler == "StepLR":
        scheduler = StepLR(optimizer, step_size=wandb.config.step_size)
    elif wandb.config.scheduler == "ReduceLROnPlateau":
        scheduler = ReduceLROnPlateau(optimizer, 'min')
    elif wandb.config.scheduler == "CosineAnnealingLR":
        scheduler = CosineAnnealingLR(optimizer, T_max=150, eta_min=0.00001)

    model = model.to(device)

    if(args.lrz):
        print("Starting Training")
    else:   
        print("Starting Training", file=f)
    wandb.watch(model, log='all', log_freq=10)
    max_accuracy = 0.0
    max_epoch = 0
    for epoch in range(1, wandb.config.epochs + 1):
        loss, acc = train(model, optimizer, train_loader, wandb.config.loader, loss_fn)
        if(args.lrz):
            print(optimizer.param_groups[0]["lr"])
        else:    
            print(optimizer.param_groups[0]["lr"], file=f)
        
        if wandb.config.scheduler == "StepLR":
            scheduler.step()
        elif wandb.config.scheduler == "ReduceLROnPlateau":
            scheduler.step(loss)
        elif wandb.config.scheduler == "CosineAnnealingLR":
            scheduler.step()

        val_loss, val_acc, val_cm = calculate_val_loss(model, valid_loader, wandb.config.loader, loss_fn)
        
        wandb.log({'train_loss': loss, 'train_acc': acc, 'val_loss': val_loss, 'val_acc': val_acc, 'epoch': epoch})
        if(args.lrz):
            print(f"train_loss: {loss}, train_acc: {acc}, val_loss: {val_loss}, val_acc: {val_acc}, val_cm: {val_cm}, epoch: {epoch}")
        else:    
            print(f"train_loss: {loss}, train_acc: {acc}, val_loss: {val_loss}, val_acc: {val_acc}, val_cm: {val_cm}, epoch: {epoch}", file=f)

        if(wandb.config.test):
            test_loss, test_accuracy, cm = test(model, test_loader, wandb.config.loader, loss_fn)

            wandb.log({'test_loss': test_loss, 'test_accuracy': test_accuracy, 'cm': cm, 'epoch': epoch})
            if(args.lrz):
                print(f"test_loss: {test_loss}, test_accuracy: {test_accuracy}, cm: {cm}, epoch: {epoch}")
            else:    
                print(f"test_loss: {test_loss}, test_accuracy: {test_accuracy}, cm: {cm}, epoch: {epoch}", file=f)

            if(max_accuracy < val_acc):
                max_accuracy = val_acc
                max_epoch = epoch
                cors_test_accuracy = test_accuracy
                if(wandb.config.save):
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss
                        }, f'../../models/{wandb.config.model}_{wandb.config.batch_size}_{wandb.config.split}_{wandb.config.binary}_{wandb.config.patch_size}.pt')
        else:
            if(max_accuracy < val_acc):
                max_accuracy = val_acc
                max_epoch = epoch
                cors_test_accuracy = test_accuracy
                if(wandb.config.save):
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss
                        }, f'../../models/{wandb.config.model}_{wandb.config.batch_size}_{wandb.config.split}_{wandb.config.binary}_{wandb.config.patch_size}.pt')

    wandb.log({'max_bacc': max_accuracy, 'max_bacc_epoch': max_epoch, 'cors_test_accuracy': cors_test_accuracy})
    if(args.lrz):
        print(f"Max BACC: {max_accuracy} at epoch {max_epoch} cors_test_accuracy: {cors_test_accuracy}")
        print(f"****************************Done with split {wandb.config.split}****************************")
    else:
        print(f"Max BACC: {max_accuracy} at epoch {max_epoch} cors_test_accuracy: {cors_test_accuracy}", file=f)
        print(f"****************************Done with split {wandb.config.split}****************************", file=f)
    
    if(not args.lrz):
        f.close()