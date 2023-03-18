import os
import pandas as pd

def make_dir_if_absent(dir):
  if not os.path.isdir(dir):
    os.makedirs(dir)

DATASET_NAME = "BRATS2013_balanced"
DATASET_PATH = f"../../data/{DATASET_NAME}/patch_metadata/test.json"

SUBSAMPLED_NUM_ELS = 640 * 2

dataset_df = pd.read_json(DATASET_PATH)

subsampled_df = dataset_df.sample(SUBSAMPLED_NUM_ELS)

SUBSAMPLED_PATH = DATASET_PATH.replace(DATASET_NAME, f"{DATASET_NAME}_subsampled/{DATASET_NAME}_subsampled_{SUBSAMPLED_NUM_ELS}")
make_dir_if_absent("/".join(SUBSAMPLED_PATH.split("/")[:-1]))

subsampled_df.to_json(SUBSAMPLED_PATH)
