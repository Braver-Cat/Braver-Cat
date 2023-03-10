import multiprocessing as mp
import pandas as pd
from rich import print
import multiprocessing as mp
import numpy as np


SLICE_PATCH_ID_LABEL_MAP_BASE_PATH = "../data/BRATS2013_slice_patch_id_label_map/slice_patch_id_label_map"
SPLIT = ["train", "val", "test"]
# SPLIT = ["val", "test"]

NUM_LABELS = 5

DATASET_SIZE = int(50000 / NUM_LABELS)
TRAIN_PERCENTAGE = 0.6
VAL_PERCENTAGE = 0.2

SAMPLED_DF_BASE_PATH = "../data/BRATS2013_balanced/slice_metadata"

split_sizes_label = {
  "train": int(DATASET_SIZE * TRAIN_PERCENTAGE),
  "val": int(DATASET_SIZE * VAL_PERCENTAGE),
  "test": DATASET_SIZE - int(DATASET_SIZE * TRAIN_PERCENTAGE) - int(DATASET_SIZE * VAL_PERCENTAGE)
}

def get_n_patches_with_label(dataset_df, n, label, split):

  print(f"\[{split}] filtering by label {label}")
  sampled_df = dataset_df.loc[dataset_df["label"] == label].sample(n=n, replace=False)

  print(f"\[{split}] converting label {label} to one-hot")
  sampled_df["label_one_hot"] = sampled_df.apply(
    lambda x: np.eye(NUM_LABELS)[x["label"]], axis=1
  )

  return sampled_df

for split in SPLIT:

  dataset_df_full_path = f"{SLICE_PATCH_ID_LABEL_MAP_BASE_PATH}_{split}.json"

  print(f"\[{split}] loading DF")
  dataset_df = pd.read_json(dataset_df_full_path)
  print(f"\[{split}] DF loaded")

  print(f"\[{split}] getting unique labels")
  label_list = dataset_df["label"].unique().tolist()
  label_list = label_list[ : NUM_LABELS]
  print(f"\[{split}] got unique labels")
  print(label_list)

  sampled_df_list = None

  n_processes = 2 if split == "train" else 3

  with mp.Pool(processes = n_processes) as p:
    
    sampled_df_list = p.starmap(
      get_n_patches_with_label, 
      
      zip(
        [dataset_df] * len(label_list),
        [split_sizes_label[split]] * len(label_list),
        label_list,
        [split] * len(label_list)
      )
    )

  print(f"\[{split}] concatenating sampled DFs")
  sampled_df = pd.concat(sampled_df_list)
  print(f"\[{split}] sampled DFs concatenated")

  print(f"\[{split}] exporting DF")
  sampled_df.to_json(f"{SAMPLED_DF_BASE_PATH}/{split}.json")
  print(f"\[{split}] DF exported")




  