import pandas as pd
from rich import print, progress
import multiprocessing as mp
import numpy as np
from sklearn.feature_extraction.image import extract_patches_2d
import os

SLICE_METADATA_DF_PATH = "../data/BRATS2013_balanced/slice_metadata/train.json"
DATASET_METADATA_DF_PATH = SLICE_METADATA_DF_PATH.replace("slice_metadata", "patch_metadata")

print(f"[bold #DA70D6]Working on {SLICE_METADATA_DF_PATH}")

GLOBAL_PATCH_SIZE = 65
GLOBAL_PATCH_SHAPE = (GLOBAL_PATCH_SIZE, GLOBAL_PATCH_SIZE)
LOCAL_PATCH_SIZE = 33

NUM_THREADS = 12

PATCH_EXPORT_BASE_PATH = os.path.join(
  "/".join(SLICE_METADATA_DF_PATH.split("/")[:-2]),
  f"patches"
)

metadata_df = pd.read_json(SLICE_METADATA_DF_PATH)

print("[bold #DA70D6]Preparing partial DFs for multi-threading...")
slice_paths = metadata_df.slice_path.unique()

slices_patches_dict = {
  slice_path: metadata_df[ metadata_df["slice_path"] == slice_path ] for slice_path in slice_paths
}

slices_patches_list = []

for slice_path in slices_patches_dict:

  slices_patches_list.append(
    {
      "slice_path": slice_path, 
      "df_rows": slices_patches_dict[slice_path]
    }
  )

dataset_df = []

print("[bold #DA70D6]Partial DFs for multi-threading created!")

# taken from https://stackoverflow.com/a/57247758
def crop_center(img, cropx, cropy):
  _, y, x = img.shape
  startx = x // 2 - (cropx // 2)
  starty = y // 2 - (cropy // 2)
  return img[:, starty:starty + cropy, startx:startx + cropx]

def make_patch(slices_patches_dict_entry):

  slice_path = f"{slices_patches_dict_entry['slice_path']}/img.npy"
  slice_id = slice_path.split('/')[-2]

  slice = np.load(slice_path)
  slice = np.transpose(slice, (1, 2, 0))

  global_patches = extract_patches_2d(
    image=slice, patch_size=GLOBAL_PATCH_SHAPE
  )

  slices_patches_df = []

  for df_row in slices_patches_dict_entry["df_rows"].iterrows():
    
    patch_id = df_row[1]["patch_id"]
    patch_label = df_row[1]["label"]
    patch_label_one_hot = df_row[1]["label_one_hot"]

    global_patch = global_patches[patch_id, ...]
    global_patch = np.transpose(global_patch, (2, 0, 1))

    local_patch = crop_center(global_patch, LOCAL_PATCH_SIZE, LOCAL_PATCH_SIZE)

    # encoding slice ID, patch ID, patch size and patch label into the patch 
    # file names

    global_patch_path = os.path.join(
      PATCH_EXPORT_BASE_PATH, 
      f"{slice_id}_{patch_id}_{GLOBAL_PATCH_SIZE}_{patch_label}.npy"
    )
    np.save(file=global_patch_path, arr=global_patch)
  
    local_patch_path = os.path.join(
      PATCH_EXPORT_BASE_PATH, 
      f"{slice_id}_{patch_id}_{LOCAL_PATCH_SIZE}_{patch_label}.npy"
    )
    np.save(file=local_patch_path, arr=local_patch)

    slices_patches_df.append(
      {
      "global_patch_path": global_patch_path,
      "local_patch_path": local_patch_path,
      "patch_label": patch_label,
      "patch_label_one_hot": patch_label_one_hot
      }
    )

  return pd.DataFrame(slices_patches_df)
 
print("[bold #DA70D6]Creating and storing patches...")

with mp.Pool(NUM_THREADS) as pool:
  dataset_rows = pool.map(
    make_patch, 
    slices_patches_list
  )
print("[bold #DA70D6]Patches stored and created!")

print("[bold #DA70D6]Concatenating partial DFs...")
dataset_df = pd.concat(dataset_rows, ignore_index=True)
print("[bold #DA70D6]Partial DFs concatenated!")

print("[bold #DA70D6]Exporting DF")
dataset_df.to_json(DATASET_METADATA_DF_PATH)
print("[bold #DA70D6]DF exported")

