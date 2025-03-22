#imports
import sys
import warnings
warnings.filterwarnings("ignore")

import os
import ast
import h5py
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

from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score
from sklearn.metrics.pairwise import cosine_similarity

import bitarray, os
from bitarray import util as butil

from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_
import torch
import torch.nn as nn
from functools import partial

# from models.deit.model_attn_wrapper import AttentionMIL
from models.deit.model_attn_wrapper_w_red import AttentionMIL
from models.deit.simple_model import SimpleModel
from sklearn.decomposition import PCA
import torch.nn.functional as F
from tqdm import tqdm


seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

# python ./slide_matching.py --model-name camelyon_16_full_wattn_uni --data camelyon_16 --wsi --agg attn
# python ./slide_matching.py --model-name camelyon_16_full_wattn_uni --data camelyon_16 --wsi --agg attn --label_filter --filter-type mv
# python ./slide_matching.py --model-name camelyon_16_full_wattn_uni --data camelyon_16 --wsi --agg attn --label_filter --filter-type gt

#paths
models_root_path = "../models/"
data_root_path = "../../../../../mnt/data/"

#functions
def get_search_image_index(data, search_id, nested=True):
    search_id = str(search_id)
    for i, item in enumerate(data):
        if nested:
            if str(item[0]) == search_id:
                return i
        else:
            if str(item) == search_id:
                return i
            
def is_nested(lst):
    if all(t.numel() == 1 for t in lst):  # Check if all tensors have 1 element (like in y)
        return any(isinstance(i, list) for i in lst)
    elif all(t.dim() == 1 and t.numel() > 1 for t in lst):  # Check if all tensors have more than 1 element and 1 dimension (like in x)
        return True
            
def cos_mul(x, y):
    x_norm = F.normalize(x, p=2, dim=1)
    y_norm = F.normalize(y, p=2, dim=1)

    return torch.matmul(x_norm, y_norm.T)

def similarity_dataset(k, sim_matrix,labels, preds, ids, label_filter, filter_type = "pred"):
    total_metrics_1 = {"percision": 0.0, "recall": 0.0, "NDCG": 0.0, "f1": 0.0, "correct_majority_vote": 0.0}
    total_metrics_3 = {"percision": 0.0, "recall": 0.0, "NDCG": 0.0, "f1": 0.0, "correct_majority_vote": 0.0}
    total_metrics_5 = {"percision": 0.0, "recall": 0.0, "NDCG": 0.0, "f1": 0.0, "correct_majority_vote": 0.0}
    count = sim_matrix.shape[0]
    #count = 1
    # convert both sim_matrix and labels to cpu
    sim_matrix = sim_matrix.cpu()
    for i in tqdm(range(0, count)):
        # if i % 100 == 0:
        #     print(i)
        metrics_1, metrics_3, metrics_5 = similarity_single_patch(k, i,sim_matrix,labels, preds, ids, label_filter, filter_type)

        total_metrics_1["percision"] = total_metrics_1["percision"] + metrics_1["percision"]
        total_metrics_1["recall"] = total_metrics_1["recall"] + metrics_1["recall"]
        total_metrics_1["NDCG"] = total_metrics_1["NDCG"] + metrics_1["NDCG"]
        total_metrics_1["f1"] = total_metrics_1["f1"] + metrics_1["f1"]
        total_metrics_1["correct_majority_vote"] = total_metrics_1["correct_majority_vote"] + metrics_1["correct_majority_vote"]

        total_metrics_3["percision"] = total_metrics_3["percision"] + metrics_3["percision"]
        total_metrics_3["recall"] = total_metrics_3["recall"] + metrics_3["recall"]
        total_metrics_3["NDCG"] = total_metrics_3["NDCG"] + metrics_3["NDCG"]
        total_metrics_3["f1"] = total_metrics_3["f1"] + metrics_3["f1"]
        total_metrics_3["correct_majority_vote"] = total_metrics_3["correct_majority_vote"] + metrics_3["correct_majority_vote"]

        total_metrics_5["percision"] = total_metrics_5["percision"] + metrics_5["percision"]
        total_metrics_5["recall"] = total_metrics_5["recall"] + metrics_5["recall"]
        total_metrics_5["NDCG"] = total_metrics_5["NDCG"] + metrics_5["NDCG"]
        total_metrics_5["f1"] = total_metrics_5["f1"] + metrics_5["f1"]
        total_metrics_5["correct_majority_vote"] = total_metrics_5["correct_majority_vote"] + metrics_5["correct_majority_vote"]

    print("1:", {total_metrics_1['f1'] / count}, {total_metrics_1['percision'] / count},
          {total_metrics_1['recall'] / count}, {total_metrics_1['NDCG'] / count}, {total_metrics_1['correct_majority_vote'] / count})

    print("3:", {total_metrics_3['f1'] / count}, {total_metrics_3['percision'] / count},
          {total_metrics_3['recall'] / count}, {total_metrics_3['NDCG'] / count}, {total_metrics_3['correct_majority_vote'] / count})

    print("5:", {total_metrics_5['f1'] / count}, {total_metrics_5['percision'] / count},
          {total_metrics_5['recall'] / count}, {total_metrics_5['NDCG'] / count}, {total_metrics_5['correct_majority_vote'] / count})

    return None

@torch.no_grad()
def similarity_single_patch(k,search_index, sim_matrix,label_vector, preds_vector, ids_vector, label_filter, filter_type = "pred"):
    sim_vector = sim_matrix[search_index][:]
    all_targets = label_vector
    all_preds = preds_vector
    all_ids = ids_vector


    # remebmer the search_images attibutes
    search_image = [all_ids[search_index], all_targets[search_index], all_preds[search_index], sim_vector[search_index]]

    #exclude the idx with value search_index from all the all_ids, all_targets, sim_vector
    sim_vector = torch.cat([sim_vector[0:search_index], sim_vector[search_index + 1:]])
    all_targets = all_targets[0:search_index] + all_targets[search_index + 1:]
    all_ids = all_ids[0:search_index] + all_ids[search_index + 1:]
    all_preds = all_preds[0:search_index] + all_preds[search_index + 1:]

    if label_filter and (filter_type == "gt" or filter_type == "pred"):
        sim_vector_temp = []
        all_targets_temp = []
        all_ids_temp = []
        all_preds_temp = []
        if filter_type == "gt":
            for i in range(len(sim_vector)):
                if ((all_preds[i] == search_image[1]).all()):
                    sim_vector_temp.append(sim_vector[i])
                    all_targets_temp.append(all_targets[i])
                    all_ids_temp.append(all_ids[i])
                    all_preds_temp.append(all_preds[i])
        if filter_type == "pred":
            for i in range(len(sim_vector)):
                if ((all_preds[i] == search_image[2]).all()):
                    sim_vector_temp.append(sim_vector[i])
                    all_targets_temp.append(all_targets[i])
                    all_ids_temp.append(all_ids[i])
                    all_preds_temp.append(all_preds[i])

        sim_vector = torch.tensor(np.asarray(sim_vector_temp))
        all_targets = all_targets_temp
        all_ids = all_ids_temp
        all_preds = all_preds_temp

    #order the all_targets, all_ids according to sim_vector from highest to lowest
    sorted_indices = torch.argsort(sim_vector, descending=True)
    sorted_sim_vector = sim_vector[sorted_indices]
    sorted_all_targets = [all_targets[i] for i in sorted_indices]
    sorted_all_preds = [all_preds[i] for i in sorted_indices]
    sorted_all_ids = [all_ids[i] for i in sorted_indices]

    if label_filter and (filter_type == "mv"):
        sorted_sim_vector_1, sorted_all_targets_1, sorted_all_preds_1, sorted_all_ids_1 = get_majority_vote(1, sorted_sim_vector, sorted_all_targets, sorted_all_preds, sorted_all_ids)
        sorted_sim_vector_3, sorted_all_targets_3, sorted_all_preds_3, sorted_all_ids_3 = get_majority_vote(3, sorted_sim_vector, sorted_all_targets, sorted_all_preds, sorted_all_ids)
        sorted_sim_vector_5, sorted_all_targets_5, sorted_all_preds_5, sorted_all_ids_5 = get_majority_vote(5, sorted_sim_vector, sorted_all_targets, sorted_all_preds, sorted_all_ids)
    
        metrics_1 = get_topk_stats(search_image, sorted_sim_vector_1, sorted_all_targets_1, sorted_all_preds_1, sorted_all_ids_1, 1)
        metrics_3 = get_topk_stats(search_image, sorted_sim_vector_3, sorted_all_targets_3, sorted_all_preds_3, sorted_all_ids_3, 3)
        metrics_5 = get_topk_stats(search_image, sorted_sim_vector_5, sorted_all_targets_5, sorted_all_preds_5, sorted_all_ids_5, 5)
    else:
        metrics_1 = get_topk_stats(search_image, sorted_sim_vector, sorted_all_targets, sorted_all_preds, sorted_all_ids, 1)
        metrics_3 = get_topk_stats(search_image, sorted_sim_vector, sorted_all_targets, sorted_all_preds, sorted_all_ids, 3)
        metrics_5 = get_topk_stats(search_image, sorted_sim_vector, sorted_all_targets, sorted_all_preds, sorted_all_ids, 5)
    return metrics_1, metrics_3, metrics_5

def get_majority_vote(k, sorted_sim_vector, sorted_all_targets, sorted_all_preds, sorted_all_ids):
    sim_vector_temp = []
    all_targets_temp = []
    all_ids_temp = []
    all_preds_temp = []
    if is_nested(sorted_all_targets):
        votes = [tuple(tensor.tolist()) for tensor in sorted_all_targets[:k]]
        majority_vote = torch.tensor(max(set(votes), key=votes.count))
    else:
        votes = sorted_all_targets[:k]
        majority_vote = max(set(votes), key=votes.count)

    for i in range(len(sorted_sim_vector)):
        if ((sorted_all_preds[i] == majority_vote).all()):
            sim_vector_temp.append(sorted_sim_vector[i])
            all_targets_temp.append(sorted_all_targets[i])
            all_ids_temp.append(sorted_all_ids[i])
            all_preds_temp.append(sorted_all_preds[i])

    sorted_sim_vector = torch.tensor(np.asarray(sim_vector_temp))
    sorted_all_targets = all_targets_temp
    sorted_all_ids = all_ids_temp
    sorted_all_preds = all_preds_temp
    return sorted_sim_vector, sorted_all_targets, sorted_all_preds, sorted_all_ids
            
def get_topk_stats(search_image,sorted_sim_vector,sorted_all_targets, sorted_all_labels, sorted_all_ids, k):
    if len(sorted_all_targets) == 0:
        metrics = {"percision": 0.0, "recall": 0.0, "NDCG": 0.0, "f1": 0.0, "correct_majority_vote": 0.0}
        return metrics
    k_search = len(sorted_all_targets) if len(sorted_all_targets) < k else k
    relavent = 0.01
    dcg = 0.01
    idcg = 0.01
    correct_majority_vote = 0

    for i in range(k):
        idcg = idcg + 1 / math.log2((i + 1) + 1)
    for i in range(k_search):
        if (sorted_all_targets[i] == search_image[1]).all():
            relavent += 1
            dcg = dcg + 1 / (math.log2((i + 1) + 1))
        else:
            dcg = dcg + 0 / (math.log2((i + 1) + 1))

    percision = relavent / (k)
    recall = relavent / (k)
    ndcg = dcg / idcg
    similarity_f1_score = 2 * ((percision * recall) / (percision + recall))
    
    if is_nested(sorted_all_targets):
        votes = [tuple(tensor.tolist()) for tensor in sorted_all_targets[:k_search]]
        majority_vote = torch.tensor(max(set(votes), key=votes.count))
    else:
        votes = sorted_all_targets[:k_search]
        majority_vote = max(set(votes), key=votes.count)
        
    if (majority_vote == search_image[1]).all():
        correct_majority_vote = correct_majority_vote + 1


    metrics = {"percision": percision, "recall": recall, "NDCG": ndcg, "f1": similarity_f1_score, "correct_majority_vote": correct_majority_vote}

    return metrics

def get_wsi_embeddings(model_name, agg, uni, gigapath):
    embeddings = []
    return_embeddings = []
    all_ids = []
    all_targets = []
    all_preds = []

    checkpoint = torch.load(os.path.join(models_root_path, model_name, "best_checkpoint.pth"), map_location='cpu')
    args = checkpoint["args"]
    device = "cpu"
    
    dataset, _, _ = build_dataset(is_train=False, is_test=True, args=args)
    sampler = torch.utils.data.SequentialSampler(dataset)
    data_loader = torch.utils.data.DataLoader(
            dataset, sampler=sampler,
            batch_size=int(1.5 * args.batch_size),
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False
        )

    if agg == "attn":
        if uni:
            model = AttentionMIL(input_dim = 1024, num_classes=args.nb_classes).to(device)
            model.load_state_dict(checkpoint['model'])
        if gigapath:
            model = AttentionMIL(input_dim = 768, num_classes=args.nb_classes).to(device)
            model.load_state_dict(checkpoint['model'])
        if not (uni or gigapath):
            model = AttentionMIL(input_dim = checkpoint["model"]["norm.weight"].shape[0], num_classes=args.nb_classes).to(device)
            model.load_state_dict(checkpoint['model_out'])
        
        model.eval()
        activation = {}
        def get_activation(name):
            def hook(model, input, output):
                try:
                    activation[name] = output.detach()
                except:
                    activation[name] = output[1].detach()
            return hook
        
        for batch in data_loader:
            if uni or gigapath:
                samples, targets = batch 
            else:
                root_dir, samples, targets = batch 
                root_dir = root_dir[0][0]
            slides_level_features = []
            targets = targets.to(device, non_blocking=True)

            if uni or gigapath:
                slides_level_features = samples
                slides_level_features = slides_level_features.to(device)
            else:

                for sample in samples:
                    if args.data_set == 'camelyon17':
                        slide_output_file_path = os.path.join(args.data_path, "additional_features", "feature_vectors", "attn", os.path.basename(args.output_dir), f"{sample[:-3]}h5")
                    if args.data_set == 'camelyon16':
                        slide_output_file_path = os.path.join(args.data_path, "additional_features", "feature_vectors", "attn", os.path.basename(args.output_dir), f"{sample}.h5")
                    if args.data_set == 'BRACS':
                        slide_output_file_path = os.path.join(args.data_path, "waly_feats", "attn", os.path.basename(args.output_dir), f"{sample}.h5")

                    slide_output = []

                    with h5py.File(slide_output_file_path, 'r') as r_file:
                        slide_output = r_file['feature_vector'][:]

                    slide_output = torch.tensor(slide_output, dtype=torch.float32)
                    slide_output = slide_output.reshape(slide_output.shape[0], -1)
                    slide_output = slide_output.to(device)

                    slides_level_features.append(slide_output)

                slides_level_features = torch.stack(slides_level_features)
                slides_level_features = slides_level_features.to(device)

            with torch.cuda.amp.autocast():
                output = model(slides_level_features)
                output = output.flatten()
                model.head[0].register_forward_hook(get_activation("embed"))
                model(slides_level_features)

            pred = output.sigmoid()
            pred = np.array(pred.detach().cpu() > 0.5, dtype=float)

            top_patches = activation["embed"]
            embeddings.append(top_patches)
            all_ids.extend(samples)
            all_targets.extend(targets)
            all_preds.extend(pred)
        
        embeddings = torch.cat(embeddings, dim=0)
        embeddings = embeddings.view(embeddings.shape[0], -1)
        for i, _ in enumerate(embeddings):
            return_embeddings.append([all_ids[i], all_targets[i], embeddings[i], all_preds[i]])

    if agg == "pca":
        model = SimpleModel(input_dim= checkpoint["model"]["norm.weight"].shape[0], output_dim=args.nb_classes).to(device)
        model.load_state_dict(checkpoint['model_out'])
        model.eval()
        for batch in data_loader:
            root_dir, samples, targets = batch 
            root_dir = root_dir[0][0]
            slides_level_features = []
            targets = targets.to(device, non_blocking=True)

            for sample in samples:
                if args.data_set == 'camelyon17':
                    slide_output_file_path = os.path.join(args.data_path, "additional_features", "feature_vectors", "pca", os.path.basename(args.output_dir), f"{sample[:-3]}h5")
                if args.data_set == 'camelyon16':
                    slide_output_file_path = os.path.join(args.data_path, "additional_features", "feature_vectors", "pca", os.path.basename(args.output_dir), f"{sample}.h5")
                if args.data_set == 'BRACS':
                    slide_output_file_path = os.path.join(args.data_path, "waly_feats", "attn", os.path.basename(args.output_dir), f"{sample}.h5")

                slide_output = []

                with h5py.File(slide_output_file_path, 'r') as r_file:
                    slide_output = r_file['feature_vector'][:]
                # Step 3: Apply PCA
                regions_pca = PCA(n_components=1)
                regions_pca_output = regions_pca.fit_transform(slide_output.T)
                pca_output = regions_pca_output.T


                # Step 4: Convert the PCA output back to a PyTorch tensor
                pca_output_tensor = torch.tensor(pca_output, dtype=torch.float32)
                if len(slides_level_features) == 0:
                    slides_level_features = pca_output_tensor
                else:
                    slides_level_features = np.concatenate((slides_level_features, pca_output_tensor))

            slides_level_features = np.array(slides_level_features, dtype=np.float32)
            slides_level_features = torch.from_numpy(slides_level_features)
            slides_level_features = slides_level_features.to(device, non_blocking=True)

            # compute output
            with torch.cuda.amp.autocast():
                output = model(slides_level_features)
                output = output.flatten()
            pred = output.sigmoid()
            pred = np.array(pred.detach().cpu() > 0.5, dtype=float)

            embeddings.append(slides_level_features)
            all_ids.extend(samples)
            all_targets.extend(targets)
            all_preds.extend(pred)
            
            
        for i, _ in enumerate(embeddings[0]):
            return_embeddings.append([all_ids[i], all_targets[i], embeddings[0][i], all_preds[i]])
    
    return return_embeddings
            
@torch.no_grad()
def get_patch_embeddings(model_name, scratch=None, data=None):
    translate = {"deit_small_patch16_224":"pretrained_mocov3_tiny_checkpoint", "deit_tiny_patch16_224":"deit_tiny_patch16_224"}
    similarities = []
    embeddings = []
    return_embeddings = []
    all_ids = []
    all_targets = []
    all_preds = []
    device = "cuda"
    
    if scratch:
        checkpoint = torch.load(os.path.join(models_root_path, f"{data}_{translate[model_name]}", "best_checkpoint.pth"), map_location='cpu')
        args = checkpoint["args"]

        model = create_model(
            model_name,
            pretrained=False,
            num_classes=args.nb_classes, #1000 for pretrained
            drop_rate=0.0,
            drop_path_rate=0.1,
            drop_block_rate=None,
            img_size=224
        )
        if scratch.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                scratch, map_location='cpu', check_hash=True)
        else:
            if not scratch == "True":
                checkpoint = torch.load(scratch, map_location='cpu')
        
        if not model_name == "deit_small_patch16_224":
            del checkpoint["model"]['head.weight']
            del checkpoint["model"]['head.bias']
    else:
        checkpoint = torch.load(os.path.join(models_root_path, model_name, "best_checkpoint.pth"), map_location='cpu')
        args = checkpoint["args"]
        device = torch.device(device)
        print(device)
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

    # print(f"model: {model}")
    model = model.to(device)
    dataset, _, _ = build_dataset(is_train=False, is_test=True, args=args)
    sampler = torch.utils.data.SequentialSampler(dataset)
    data_loader = torch.utils.data.DataLoader(
            dataset, sampler=sampler,
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
    header = 'Similarity Test:'
    
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    model.pre_logits.register_forward_hook(get_activation("embed"))

    # switch to evaluation mode
    model.eval()
    for batch in metric_logger.log_every(data_loader, 10, header):
        if args.data_set == 'WSS':
            ids = batch["id"]
            images = batch["image"]
            target = batch["label"]
        else:
            images, target = batch
            ids = target
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        activation = {}
        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            if args.data_set == 'crc':
                output = output
            else:
                output = output.squeeze(1) #not in crc
            loss = criterion(output, target)
            model(images)
        embeddings.append(activation['embed'].detach().cpu())
        all_ids.extend(ids)
        all_targets.extend(target.detach().cpu())
        # f1, acc = accuracy(output, target, topk=(1, 5))
        if args.data_set == 'crc':
            pred = output.softmax(dim=1)
            _, pred = torch.max(pred, 1)
            pred = np.array(pred.detach().cpu(), dtype=int) #softmax!
        else:
            pred = output.sigmoid()
            pred = np.array(output.detach().cpu() > 0.5, dtype=float)
        all_preds.extend(pred)
        if args.data_set == 'WSS':
            f1 = f1_score(target.detach().cpu(), pred, average='samples')
        if args.data_set == 'crc':
            f1 = f1_score(y_true=np.array(target.detach().cpu()), y_pred=pred, average='weighted')
        if args.data_set == 'patch_cam' or args.data_set == 'mhist':
            f1 = f1_score(target.detach().cpu(), pred, average='binary')
        acc = accuracy_score(target.detach().cpu(), pred)
        if args.data_set == 'patch_cam' or args.data_set == 'mhist' or args.data_set == 'crc' or args.data_set == 'camelyon16':
            bacc = balanced_accuracy_score(target.detach().cpu(), pred)
        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['f1'].update(f1.item(), n=batch_size)
        metric_logger.meters['acc'].update(acc.item(), n=batch_size)
        if args.data_set == 'patch_cam' or args.data_set == 'mhist' or args.data_set == 'crc' or args.data_set == 'camelyon16':
            metric_logger.meters['bacc'].update(bacc.item(), n=batch_size)
    # if args.data_set == 'WSS':
    #     search_index = get_search_image_index(all_ids, image_id, False)
    # if args.data_set == 'patch_cam' or args.data_set == 'mhist' or args.data_set == 'crc' or args.data_set == 'camelyon16':
    
    embeddings = torch.cat(embeddings, dim=0)

    for i, _ in enumerate(embeddings):
        return_embeddings.append([all_ids[i], all_targets[i], embeddings[i], all_preds[i]])

    return return_embeddings

def get_args_parser():
    parser = argparse.ArgumentParser('Similarity test script', add_help=False)

    parser.add_argument('--model-name', default="crc_pretrained_mocov3_tiny_checkpoint", type=str)
    parser.add_argument('--k', default=5, type=int)
    parser.add_argument('--search-image-index', default=0, type=int)
    parser.add_argument('--scratch', default="", type=str)
    parser.add_argument('--data', default="wss", type=str)
    parser.add_argument('--label_filter', action='store_true')
    parser.add_argument('--filter-type', default="pred", type=str)
    parser.add_argument('--wsi', action='store_true')
    parser.add_argument('--agg', default="attn", type=str)
    parser.add_argument('--uni', action='store_true')
    parser.add_argument('--gigapath', action='store_true')
    
    return parser

def main(args):
    print(args)

    sys.path[0] = "/home/ge26xaj/./projects/fm_histopathology/scripts/"
    print("In module products sys.path[0], __package__ ==", sys.path[0], __package__)

    if(args.wsi):
        embeddings = get_wsi_embeddings(args.model_name, args.agg, args.uni, args.gigapath)
    else:
        embeddings = get_patch_embeddings(args.model_name, args.scratch, args.data) #1005919, 31,6,0, 1226

    count = len(embeddings)
    print(count)
    
    all_ids = [row[0] for row in embeddings]
    all_targets = [row[1] for row in embeddings]
    all_preds = [row[3] for row in embeddings]
    embeddings = [row[2] for row in embeddings]
 
    embeddings_tensor = torch.tensor(np.asarray(embeddings))
    sim_matrix = cos_mul(embeddings_tensor,embeddings_tensor) #tensor

    similarity_dataset(k=[1, 3, 5],sim_matrix=sim_matrix,labels=all_targets, preds=all_preds, ids=all_ids, label_filter=args.label_filter, filter_type=args.filter_type)
    # similarity_dataset(k=3,sim_matrix=sim_matrix,labels=all_targets, preds=all_preds, ids=all_ids, label_filter=args.label_filter, filter_type=args.filter_type)
    # similarity_dataset(k=5,sim_matrix=sim_matrix,labels=all_targets, preds=all_preds, ids=all_ids, label_filter=args.label_filter, filter_type=args.filter_type)

    # for i in range(0, count):
    #     if i%100 == 0:
    #         print(i)
    #     _, _, metrics = similarity_test(args.model_name, 5, i, embeddings, args.scratch, args.data, args.label_filter)
    #     total_metrics["percision"] = total_metrics["percision"] + metrics["percision"]
    #     total_metrics["recall"] = total_metrics["recall"] + metrics["recall"]
    #     total_metrics["NDCG"] = total_metrics["NDCG"] + metrics["NDCG"]
    #     total_metrics["f1"] = total_metrics["f1"] + metrics["f1"]

    # print({"F1 Score:", total_metrics['f1']/count}, "Percision:", {total_metrics['percision']/count}, 
    #       "Recall:", {total_metrics['recall']/count}, "NDCG:", {total_metrics['NDCG']/count})

if __name__ == '__main__':

    parser = argparse.ArgumentParser('Similarity test script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)