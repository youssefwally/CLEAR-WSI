import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import h5py
import pandas as pd
import torchio as tio
import monai.transforms as montrans
from typing import List, Tuple
import logging

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

DIAGNOSIS_MAP = {}
DIAGNOSIS_MAP_binary = {}
LOG = logging.getLogger(__name__)

def get_image_transform(is_training: bool, input_size: int = 224):
    # Image sizes Dataset {113, 137, 113}

    img_transforms = [
        tio.RescaleIntensity(out_min_max=(0, 1)),
        # tio.CropOrPad((128, 128, 128))
    ]

    if is_training:
        randomAffineWithRot = tio.RandomAffine(
            # scales=0.2,
            degrees=90,  # +-8 degree in each dimension
            translation=20,  # +-8 pixels offset in each dimension.
            image_interpolation="linear",
            default_pad_value="otsu",
            p=0.5,
        )
        img_transforms.append(randomAffineWithRot)
    
    if input_size:
        Rescale = tio.Resize((input_size, input_size, 3))
        img_transforms.append(Rescale)
    
    img_transform = montrans.Compose(img_transforms)
    return img_transform

class PatchcamH5Dataset(Dataset):
    def __init__(self, 
                path: str, 
                is_training: bool, 
                out_class_num: int, 
                input_size: int = 224,
                ):
        self.path = path
        self.transforms = [
            get_image_transform(is_training, input_size)
        ]

        self.out_class_num = out_class_num

        self._load()

    def _load(self):
        image_data = []
        diagnosis = []

        with h5py.File(f"{self.path}_x.h5", mode='r') as file:
            dataset = file['x']
            for patch in dataset:

                image_data.append(
                        patch[np.newaxis]                      
                    )
        
        with h5py.File(f"{self.path}_y.h5", mode='r') as file:
            dataset = file['y']
            for label in dataset:

                diagnosis.append(label.item())  
                                            
        LOG.info("DATASET: %s", self.path)
        LOG.info("SAMPLES: %d", len(image_data))

        print("DATASET: ", self.path)
        print("SAMPLES: ", len(image_data))

        labels, counts = np.unique(diagnosis, return_counts=True)
        LOG.info("Classes: %s", pd.Series(counts, index=labels))

        print("Classes: ", pd.Series(counts, index=labels))
        
        self._image_data = image_data
        self._diagnosis = diagnosis

    def __len__(self) -> int:
        return len(self._image_data)

    def __getitem__(self, index: int):
        label = self._diagnosis[index]
        scans = self._image_data[index]

        assert len(scans) == len(self.transforms)
        sample = self.transforms[0](scans)
        sample = sample[0]
        sample = np.moveaxis(sample, -1, 0)

        return torch.from_numpy(sample), torch.tensor(label, dtype=torch.float32)


