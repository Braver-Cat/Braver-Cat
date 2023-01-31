import pandas as pd

from rich import print

import numpy as np

import os

DATASET_NAME = "brats_2013"
PATCH_SIZE = 65
DATASET_SPLIT = "train"
IS_BALANCED = "unbalanced"

DF_PATH = f"../data/{DATASET_NAME}_patch_{PATCH_SIZE}_df_{DATASET_SPLIT}.json"

print("Loading original DataFrame")
df = pd.read_json(DF_PATH)
print("Original DataFrame loaded")

DATASET_SIZE = 50000

TRAIN_PERCENTAGE = 0.7
VAL_PERCENTAGE = 0.2

DATASET_SIZE_TRAIN = int(DATASET_SIZE * TRAIN_PERCENTAGE)
DATASET_SIZE_VAL = int(DATASET_SIZE * VAL_PERCENTAGE)
DATASET_SIZE_TEST = DATASET_SIZE - DATASET_SIZE_TRAIN - DATASET_SIZE_VAL

size_dict = {
  "train": DATASET_SIZE_TRAIN,
  "val": DATASET_SIZE_VAL,
  "test": DATASET_SIZE_TEST
}

print("Subsampling")
subsampled_indexes = np.random.choice(
  df.index, size=size_dict[DATASET_SPLIT], replace=False
) 
print("Subsampled")

df_to_export = df.iloc[subsampled_indexes]

DATASET_SUBSAMPLE_ID = 0

dataset_name = DATASET_NAME.replace("_", "").upper()
DATASET_SUBSAMPLE_PATH = f"../data/{dataset_name}_patches_{PATCH_SIZE}_{IS_BALANCED}/{DATASET_SUBSAMPLE_ID}"

if not os.path.exists(DATASET_SUBSAMPLE_PATH):
  os.makedirs(DATASET_SUBSAMPLE_PATH)

print("Exporting subsampled DataFrame")
df_to_export.to_json(f"{DATASET_SUBSAMPLE_PATH}/{DATASET_SPLIT}_df.json")
print("Subsampled DataFrame exported")
