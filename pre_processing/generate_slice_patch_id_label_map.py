from rich import print
from glob import glob
import numpy as np
from sklearn.feature_extraction.image import extract_patches_2d
import pandas as pd
from rich import progress

DATASET_PATH = "../data/BRATS2013_unstacked_padded_slices_train_val_test_split"

DATASET_SIZE = 50000
TRAIN_PERCENTAGE = 0.6
VAL_PERCENTAGE = 0.2

PATCH_SIZE = 65
PATCH_SHAPE = (PATCH_SIZE, PATCH_SIZE)

SLICE_PATCH_ID_LABEL_MAP_BASE_PATH = "../data/BRATS2013_slice_patch_id_label_map"

split_sizes = {
  "train": int(DATASET_SIZE * TRAIN_PERCENTAGE),
  "val": int(DATASET_SIZE * VAL_PERCENTAGE),
  "test": DATASET_SIZE - int(DATASET_SIZE * TRAIN_PERCENTAGE) - int(DATASET_SIZE * VAL_PERCENTAGE)
}

pb = progress.Progress(
  progress.TextColumn("[progress.description]{task.description}"),
  progress.TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
  progress.BarColumn(),
  progress.MofNCompleteColumn(),
  progress.TextColumn("•"),
  progress.TimeElapsedColumn(),
  progress.TextColumn("•"),
  progress.TimeRemainingColumn()
)

pb_split_task = pb.add_task("Split", total=len(list(split_sizes.keys())))
pb_slice_task = pb.add_task("Slice")
pb_patch_task = pb.add_task("Patch")

pb.start()

for split in split_sizes.keys():

  split_path = f"{DATASET_PATH}/{split}"

  slice_paths = glob(f"{split_path}/*")

  pb.reset(pb_slice_task)
  pb.update(pb_slice_task, total=len(slice_paths))

  split_df_list = []

  for slice_path in slice_paths:

    label_np = np.squeeze(np.load(f"{slice_path}/label.npy"), 0)
    
    label_patches = extract_patches_2d(image=label_np, patch_size=PATCH_SHAPE)

    pb.reset(pb_patch_task)
    pb.update(pb_patch_task, total=label_patches.shape[0])

    for patch_id in range(label_patches.shape[0]):

      label = round(label_patches[patch_id, PATCH_SIZE//2, PATCH_SIZE//2])

      split_df_list.append(
        {
          "slice_path": slice_path,
          "patch_id": patch_id, 
          "label": label
        }
      )  

      pb.advance(pb_patch_task) 
  
    pb.advance(pb_slice_task)
    
  split_df = pd.DataFrame(split_df_list)

  split_df.to_json(
    f"{SLICE_PATCH_ID_LABEL_MAP_BASE_PATH}/slice_patch_id_label_map_{split}.json"
  )
  
  pb.advance(pb_split_task)
  
pb.stop()