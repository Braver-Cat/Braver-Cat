import os

import pandas as pd

import numpy as np

DATASET_PATH = f"../data/BRATS2013_patches_65_unbalanced/0/test_labels_df.json"

df = pd.read_json(DATASET_PATH)

def label_to_one_hot(row):

  return np.eye(6)[row["patch_label"]].astype(int)

df['patch_label_one_hot'] = df.apply(label_to_one_hot, axis=1)

df.to_json(DATASET_PATH.replace(".json", "_one_hot.json"))