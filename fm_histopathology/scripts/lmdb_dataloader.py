#imports
import os
import ast
import torch
from torch.utils.data import DataLoader, Dataset


def get_labels(root_path, nb_classes, is_train, features, chosen_label):
    dataset = []

    for index, row in features.iterrows():
        if(not row["original_val"]):
            path = f"{root_path}/1.training"
        else:
            path = f"{root_path}/2.validation/img"

        if(not row["original_val"]):
            img_path = f"{path}/{row.id}-{int(row.x_axis)}-{int(row.y_axis)}-{row.patch_level_multi_class_labels}.png"
        else:
            img_path = f"{path}/{row.id}.png"

        if nb_classes > 1:
            label = ast.literal_eval(row[chosen_label])
            label = [float(element) for element in label]
        else:
            label = row[chosen_label]
            label = float(label)

        label = torch.as_tensor(label)

        dataset.append({"id": row.id, 'image': img_path, 'label': label})

    return dataset
    

                    