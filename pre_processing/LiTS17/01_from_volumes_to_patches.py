import nibabel as nib

from glob import glob

from rich import print

import numpy as np

from rich.progress import *

import argparse

arg_parser = argparse.ArgumentParser()

import skimage

import pandas as pd

arg_parser.add_argument(
  "--start", type=int, required=True, help="Starting index"
)
arg_parser.add_argument(
  "--stop", type=int, required=True, help="Stopping index"
)
parsed_args = arg_parser.parse_args()


DATASET_FULL_PATH = "../../data/LiTS17/LITS17"
PATCH_SIZE = 65
PATCHES_DATASET_BASE_PATH = "../../data/LiTS17/LITS17_patches_65"

START = parsed_args.start
STOP = parsed_args.stop

PATCHES_DF_PATH = f"{PATCHES_DATASET_BASE_PATH}/LITS17_patches_65_{START}_{STOP}.json"

volume_files = list(
  sorted(glob(f"{DATASET_FULL_PATH}/volume*.nii"))
)[START:STOP]

segmentation_files = list(
  sorted(glob(f"{DATASET_FULL_PATH}/segmentation*.nii"))
)[START:STOP]

df_rows = []

pbar = Progress(
  TextColumn("[progress.description]{task.description}", justify="right"),
  TextColumn("[progress.percentage]{task.percentage:>3.0f}%", justify="right"),
  BarColumn(), MofNCompleteColumn(), TextColumn("•"), TimeElapsedColumn(),
  TextColumn("•"), TimeRemainingColumn(),
)

patient_task = pbar.add_task(
  description="Patient progress...", total=len(volume_files)
)

slice_task = pbar.add_task("Slice progress...", total=0)

patch_task = pbar.add_task("Patch progress...", total=0)

pbar.start()

label_set = set()

for pat_id, (vol_file, seg_file) in enumerate(zip(volume_files, segmentation_files)):
  
  vol = np.asarray(nib.load(vol_file).dataobj, dtype=np.uint8)
  seg = np.asarray(nib.load(seg_file).dataobj, dtype=np.uint8)

  if vol.shape != seg.shape:
    print("WARNING: shape of vol does NOT match seg.shape")
    
    continue
  
  
  pbar.update(slice_task, total=vol.shape[-1])
  pbar.reset(slice_task)

  for slice in range(vol.shape[-1]):

    vol_patches = skimage.util.view_as_windows(
      arr_in=vol[..., slice], window_shape=(PATCH_SIZE, PATCH_SIZE), step=1
    )

    seg_patches = skimage.util.view_as_windows(
      arr_in=seg[..., slice], window_shape=(PATCH_SIZE, PATCH_SIZE), step=1
    )

    vol_patches = np.reshape(vol_patches, (-1, PATCH_SIZE, PATCH_SIZE))

    seg_patches = np.reshape(seg_patches, (-1, PATCH_SIZE, PATCH_SIZE))

    pbar.update(patch_task, total=vol_patches.shape[0])
    pbar.reset(patch_task)

    for patch_id in range(vol_patches.shape[0]):
      
      if (not np.any(vol_patches[patch_id, ...])):
        continue

      label = seg_patches[patch_id, PATCH_SIZE//2, PATCH_SIZE//2]
      
      patch_path = f"{PATCHES_DATASET_BASE_PATH}" \
        f"/{str(pat_id).zfill(3)}_{str(patch_id).zfill(6)}_label_{label}.npy"
      
      np.save(patch_path, vol_patches[patch_id, ...])

      pbar.advance(patch_task)

    pbar.advance(slice_task)

  pbar.advance(patient_task)
