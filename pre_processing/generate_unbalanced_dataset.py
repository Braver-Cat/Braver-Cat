import multiprocessing as mp
import pandas as pd
from rich import print
import multiprocessing as mp
import numpy as np

SLICE_PATCH_ID_LABEL_MAP_BASE_PATH = "../data/BRATS2013_slice_patch_id_label_map/slice_patch_id_label_map"
SPLIT = ["train", "val", "test"]

NUM_LABELS = 5

DATASET_SIZE = 50000
TRAIN_PERCENTAGE = 0.6
VAL_PERCENTAGE = 0.2

SAMPLED_DF_BASE_PATH = "../data/BRATS2013_unbalanced"
TO_EXCLUDE_DF_BASE_PATH = "../data/BRATS2013_balanced"

split_sizes = {
  "train": int(DATASET_SIZE * TRAIN_PERCENTAGE),
  "val": int(DATASET_SIZE * VAL_PERCENTAGE),
  "test": DATASET_SIZE - int(DATASET_SIZE * TRAIN_PERCENTAGE) - int(DATASET_SIZE * VAL_PERCENTAGE)
}

for split in SPLIT:

  dataset_df_full_path = f"{SLICE_PATCH_ID_LABEL_MAP_BASE_PATH}_{split}.json"
  # dataset_df_full_path = f"{SLICE_PATCH_ID_LABEL_MAP_BASE_PATH}_DUMMY.json"

  print(f"\[{split}] loading DF")
  dataset_df = pd.read_json(dataset_df_full_path)
  print(f"\[{split}] DF loaded")

  print(f"\[{split}] loading DF to be excluded")
  to_exclude_df = pd.read_json(f"{TO_EXCLUDE_DF_BASE_PATH}/{split}.json")
  print(f"\[{split}] DF to be excluded loaded")

  print(f"\[{split}] computing rows to be excluded")
  common_df = dataset_df.reset_index().merge(
    to_exclude_df, on=dataset_df.columns.tolist()
  ).set_index('index')
  print(f"\[{split}] found {len(common_df)} rows to be excluded")

  print(f"\[{split}] excluding rows")
  dataset_df = dataset_df.drop(common_df.index)

  print(f"\[{split}] getting unique labels")
  label_list = dataset_df["label"].unique().tolist()
  print(f"\[{split}] got unique labels: {label_list}")

  print(f"\[{split}] sampling DF")
  sampled_df = dataset_df.sample(n=split_sizes[split], replace=False)
  print(f"\[{split}] DF sampled")

  print(f"\[{split}] converting labels to one-hot")
  sampled_df["label_one_hot"] = sampled_df.apply(
    lambda x: np.eye(NUM_LABELS)[x["label"]], axis=1
  )
  print(f"\[{split}] labels converted to one-hot")

  print(f"\[{split}] exporting DF")
  sampled_df.to_json(f"{SAMPLED_DF_BASE_PATH}/{split}.json")
  print(f"\[{split}] DF exported")




  