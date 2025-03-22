#imports
import os
import pandas as pd
import numpy as np

############################################################################################

def get_data_with_grades(data, quality_data, grades = ["B", "B-"]):
    good_quality_data = quality_data[quality_data.grade.isin(grades) == True]
    good_quality_data = good_quality_data.merge(data, left_on="imageid", right_on="IMAGEUID", how='inner')
    
    return good_quality_data

############################################################################################

def get_data(path, is_train, split):
    if(is_train):
        training_features = pd.read_csv("../../../../../mnt/data/WSSS4LUAD/features/training_binary.csv", index_col=0)
        # training_features = training_features.iloc[0:2]
        training_features['id'] = training_features['id'].astype(str).str.zfill(2)
        print(f"Used Train Data: {len(training_features)}")
        return training_features
    else:
        valid_features = pd.read_csv("../../../../../mnt/data/WSSS4LUAD/features/validation_binary.csv", index_col=0) 
        # valid_features = valid_features.iloc[0:2]
        valid_features['id'] = valid_features['id'].astype(str).str.zfill(2)
        print(f"Validation Data: {len(valid_features)}")
        return valid_features




############################################################################################
