import pandas as pd

import numpy as np

import skimage

import os

from rich.progress import *

def crop_center(img,cropx,cropy):
  y, x = img.shape
  startx = x//2-(cropx//2)
  starty = y//2-(cropy//2)    
  return img[starty:starty+cropy,startx:startx+cropx]

DATASET_NAME = "BRATS2013"
PATCH_SIZE = 65
IS_BALANCED = "unbalanced"
SPLIT_ID = 0
SPLIT_NAME = "train"

DF_PATH = f"../data" \
  f"/{DATASET_NAME}_patches_{PATCH_SIZE}_{IS_BALANCED}" \
  f"/{SPLIT_ID}/{SPLIT_NAME}_df.json"

print("Loading original DataFrame")
df = pd.read_json(DF_PATH)
print("Original DataFrame loaded")

PATCH_SIZES = [65, 53, 33]
NUM_CLASSES = 6

new_df_row = {}

pbar = Progress(
  TextColumn(
    "[progress.description]{task.description}",
    justify="right"
  ),
  TextColumn(
    "[progress.percentage]{task.percentage:>3.0f}%",
    justify="right"
  ),
  BarColumn(), MofNCompleteColumn(), TextColumn("•"),
  TimeElapsedColumn(), TextColumn("•"), TimeRemainingColumn()
  
)

pbar.start()

df_row_task = pbar.add_task(
  description="DF rows progress...", total=len(df.index)
)

patch_size_task = pbar.add_task(
  description="Patch sizes...", total=len(PATCH_SIZES)
)

channel_task = pbar.add_task(description="Channel progress...", total=4)

for df_index, df_row in df.iterrows():

  new_df_list = []
  new_df_row = {}

  pbar.reset(patch_size_task)

  for patch_size in PATCH_SIZES:
    img_np = np.load(f"{df_row['img_path']}/img.npy")

    patch_np = np.empty((img_np.shape[0], patch_size, patch_size))

    pbar.reset(channel_task)

    for channel in range(img_np.shape[0]):
      
      patches = skimage.util.view_as_windows(
        arr_in=img_np[channel, ...], window_shape=(PATCH_SIZE, PATCH_SIZE), 
        step=1
      )

      patch = patches[df_row['patch_id'][0], df_row['patch_id'][1]]

      if patch_size != 65:

        patch = crop_center(patch, patch_size, patch_size)

      patch_np[channel, ...] = patch

      pbar.update(channel_task, advance=1)

    patch_export_name = df_row['img_path'].split("/")[-1].split(".")[0]
    
    patch_export_dir = f"../data/{DATASET_NAME}_patches_{patch_size}_{IS_BALANCED}/{SPLIT_ID}/{SPLIT_NAME}"
    if not os.path.exists(patch_export_dir):
      os.makedirs(patch_export_dir)

    patch_export_path = f"{patch_export_dir}/{patch_export_name}_{df_row['patch_id'][0]}_{df_row['patch_id'][1]}"
    np.save(patch_export_path, patch_np)

    new_df_row[f"patch_{patch_size}_x_{patch_size}_img_path"] = patch_export_path

    pbar.update(patch_size_task, advance=1)

  pbar.update(df_row_task, advance=1)
  
  new_df_row[f"patch_label"] = df_row["label"]

  new_df_list.append(new_df_row)

######

df = pd.DataFrame(new_df_list)

for patch_size in PATCH_SIZES:
  df_patch_size = df[
    [f"patch_{patch_size}_x_{patch_size}_img_path", "patch_label"]
  ]

  df_export_path = f"../data" \
    f"/{DATASET_NAME}_patches_{patch_size}_{IS_BALANCED}/{SPLIT_ID}" \
    f"/{SPLIT_NAME}_labels_df.json"
  
  df_patch_size.to_json(df_export_path)
  
