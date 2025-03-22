import os
import sys
import argparse
import warnings
warnings.filterwarnings("ignore")

from torchvision import transforms
from gigapath.pipeline import tile_one_slide

def create_transforms():
    transform = transforms.Compose(
        [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    
    return transform

def create_tile(slide_path, save_dir):
    print("NOTE: Prov-GigaPath is trained with 0.5 mpp preprocessed slides. Please make sure to use the appropriate level for the 0.5 MPP")
    tile_one_slide(slide_path, save_dir=save_dir, level=1)

    print("NOTE: tiling dependency libraries can be tricky to set up. Please double check the generated tile images.")

    return True

def get_args_parser():
    parser = argparse.ArgumentParser('Similarity test script', add_help=False)

    parser.add_argument('--images-dir', type=str)
    parser.add_argument('--save-dir', type=str)
    
    return parser

def main(args):
    print(args)
    
    # slide_paths = [os.path.join(args.images_dir, img) for img in os.listdir(args.images_dir) if img.endswith('.tif')] #Camelyon
    slide_paths = [os.path.join(args.images_dir, img) for img in os.listdir(args.images_dir) if img.endswith('.svs')] #BRACs
    for slide_path in slide_paths:
        create_tile(slide_path, args.save_dir)

    print("Done")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('create gigapath tiles', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
