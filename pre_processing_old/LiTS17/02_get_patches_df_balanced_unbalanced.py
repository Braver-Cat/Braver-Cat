import glob

import random

import pandas as pd

import numpy as np

from rich import print

PATCHES_DIR = "../../data/LiTS17/LITS17_patches_65/"

patch_65_label_0 = set(glob.glob(f"{PATCHES_DIR}/*_label_0*"))
patch_65_label_1 = set(glob.glob(f"{PATCHES_DIR}/*_label_1*"))
patch_65_label_2 = set(glob.glob(f"{PATCHES_DIR}/*_label_2*"))

DATASET_SIZE = 50000

TRAIN_PERCENTAGE = 0.7
VAL_PERCENTAGE = 0.2

SIZE = {
  "train": int(DATASET_SIZE * TRAIN_PERCENTAGE),
  "val": int(DATASET_SIZE * VAL_PERCENTAGE),
  "test": DATASET_SIZE - int(DATASET_SIZE * TRAIN_PERCENTAGE) - int(DATASET_SIZE * VAL_PERCENTAGE)
}

NUM_LABELS = 6

SIZE_PER_LABEL = {
  "train": int(SIZE["train"] / NUM_LABELS),
  "val": int(SIZE["val"] / NUM_LABELS),
  "test": int(SIZE["test"] / NUM_LABELS)
}

SPLIT_ID = 0

DF_BALANCED_EXPORT_BASE_PATH = "../../data/LiTS17/LITS17_patches_65_balanced" \
  f"/{SPLIT_ID}"

DF_UNBALANCED_EXPORT_BASE_PATH = "../../data/LiTS17/LITS17_patches_65_unbalanced" \
  f"/{SPLIT_ID}"

for stage in ["train", "val", "test"]:

  patch_65_label_0_sampled = set(
    random.sample(list(patch_65_label_0), SIZE_PER_LABEL[stage])
  )
  patch_65_label_0 = patch_65_label_0.difference(patch_65_label_0_sampled)

  patch_65_label_1_sampled = set(
    random.sample(list(patch_65_label_1), SIZE_PER_LABEL[stage])
  )
  patch_65_label_1 = patch_65_label_1.difference(patch_65_label_1_sampled)


  patch_65_label_2_sampled = set(
    random.sample(list(patch_65_label_2), SIZE_PER_LABEL[stage])
  )
  patch_65_label_2 = patch_65_label_2.difference(patch_65_label_2_sampled)

  df_rows = []

  for patch_65 in patch_65_label_0_sampled:

    patch_65 = patch_65[:-4]

    df_rows.append(
      {
        "patch_65_x_65_img_path": patch_65,
        "patch_label_one_hot": [1, 0, 0, 0, 0, 0]
      }
    )
  
  for patch_65 in patch_65_label_1_sampled:

    patch_65 = patch_65[:-4]

    df_rows.append(
      {
        "patch_65_x_65_img_path": patch_65,
        "patch_label_one_hot": [0, 1, 0, 0, 0, 0]
      }
    )
  
  for patch_65 in patch_65_label_2_sampled:

    patch_65 = patch_65[:-4]

    df_rows.append(
      {
        "patch_65_x_65_img_path": patch_65,
        "patch_label_one_hot": [0, 0, 1, 0, 0, 0]
      }
    )

  df = pd.DataFrame(df_rows)

  df_export_path = f"{DF_BALANCED_EXPORT_BASE_PATH}" \
    f"/{stage}_labels_df_one_hot.json"
  
  print(f"Exporting {stage} DF in {df_export_path}")
  df.to_json(path_or_buf=df_export_path)
  print(f"{stage} DF exported in {df_export_path}")


patch_65_all_labels = patch_65_label_0.union(
  patch_65_label_1
).union(patch_65_label_2)

for stage in ["train", "val", "test"]:

  df_rows = []

  patch_65_all_labels_sampled = set(
    random.sample(list(patch_65_all_labels), SIZE[stage])
  )
  patch_65_all_labels = patch_65_all_labels.difference(
    patch_65_all_labels_sampled
  )

  for patch_65 in patch_65_all_labels_sampled:

    label = int(patch_65[-5])
    label_one_hot = np.eye(NUM_LABELS)[label]

    patch_65 = patch_65[:-4]

    df_rows.append(
      {
        "patch_65_x_65_img_path": patch_65,
        "patch_label_one_hot": label_one_hot
      }
    )

  df = pd.DataFrame(df_rows)

  df_export_path = f"{DF_UNBALANCED_EXPORT_BASE_PATH}" \
    f"/{stage}_labels_df_one_hot.json"
  
  print(f"Exporting {stage} DF in {df_export_path}")
  df.to_json(path_or_buf=df_export_path)
  print(f"{stage} DF exported in {df_export_path}")