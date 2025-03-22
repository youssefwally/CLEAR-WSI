#Imports
import os
import sys
import PIL
import argparse
import numpy as np
from pathlib import Path
from huggingface_hub import login

import timm
from PIL import Image

import torch
from torchvision import transforms
from gigapath.pipeline import load_tile_slide_encoder, run_inference_with_tile_encoder, run_inference_with_slide_encoder

import warnings
warnings.filterwarnings("ignore")

seed = 42
torch.manual_seed(seed)

#Functions
def create_encoders():
    # Load the tile and slide encoder models
    # NOTE: The CLS token is not trained during the slide-level pretraining.
    # Here, we enable the use of global pooling for the output embeddings.
    tile_encoder, slide_encoder_model = load_tile_slide_encoder(global_pool=True)

    return tile_encoder, slide_encoder_model

def get_tile_embeddings(image_paths, tile_encoder):
    # run inference with the tile encoder
    tile_encoder_outputs = run_inference_with_tile_encoder(image_paths, tile_encoder)
    
    return tile_encoder_outputs

def get_slide_embeddings(slide_encoder_model, tile_encoder_outputs):
    # run inference with the slide encoder
    slide_embeds = run_inference_with_slide_encoder(slide_encoder_model=slide_encoder_model, **tile_encoder_outputs)

    return slide_embeds

def get_embeddings(tile_dir, tile_encoder, slide_encoder, save_path=None):
    tile_paths = [os.path.join(tile_dir, img) for img in os.listdir(tile_dir) if img.endswith('.png')]
    try:
        tile_encoder_outputs = get_tile_embeddings(tile_paths, tile_encoder)
        slide_embeds = get_slide_embeddings(slide_encoder, tile_encoder_outputs)
    except:
        print(f"*******ERROR with slide {tile_dir}*******")
        return False

    if(save_path):
        torch.save(slide_embeds, f"{save_path}_{os.path.splitext(os.path.basename(tile_dir))[0]}.pt")

    return slide_embeds

def get_args_parser():
    parser = argparse.ArgumentParser('Similarity test script', add_help=False)

    parser.add_argument('--tiles-dir', type=str)
    parser.add_argument('--save-dir', type=str)
    
    return parser

def main(args):
    print(args)

    sys.path[0] = "/home/ge26xaj/./projects/fm_histopathology/scripts/"
    print("In module products sys.path[0], __package__ ==", sys.path[0], __package__)

    tile_encoder, slide_encoder_model = create_encoders()
    tiles_paths = [os.path.join(args.tiles_dir, img) for img in os.listdir(args.tiles_dir)]

    for tile_path in tiles_paths:
        _ = get_embeddings(tile_path, tile_encoder, slide_encoder_model, args.save_dir)

    print("Done")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('create gigapath embeddings', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)



# python create_embeddings.py --tiles-dir ../../../../../../../../../mnt/nfs03-R6/CAMELYON16/waly/gigapath_tiles/output/ --save-dir ../../../../../../../../../mnt/nfs03-R6/CAMELYON16/waly/gigapath_embeddings/
# python create_embeddings.py --tiles-dir ../../../../../../../../../mnt/nfs03-R6/BRACS/gigapath_tiles/[set]/Group_/Type_/output/ --save-dir ../../../../../../../../../mnt/nfs03-R6/BRACS/gigapath_embeddings/[set]/Group_/Type_/output/