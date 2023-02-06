import nibabel as nib

import numpy as np

import matplotlib.pyplot as plt

import glob

LITS17_DIR = "../../data/LiTS17/LITS17/"
lits17_files = glob.glob(f"{LITS17_DIR}/segmentation*.nii")

from rich.progress import *


counter = 0

non_zero_list = []


pbar = Progress(
  TextColumn(
    "[progress.description]{task.description}",
    justify="right"
  ),
  TextColumn(
    "[progress.percentage]{task.percentage:>3.0f}%",
    justify="right"
  ),
  BarColumn(),
  MofNCompleteColumn(),
  TextColumn("•"),
  TimeElapsedColumn(),
  TextColumn("•"),
  TimeRemainingColumn(),
)

lits17_files_task = pbar.add_task(
  description="Volume progress...", total=len(lits17_files)
)
img_id_task = pbar.add_task(description="Image ID progress...", total=0)

pbar.start()

for lits17_file in lits17_files:
  img_batch = nib.load(lits17_file).dataobj

  pbar.update(img_id_task, total=img_batch.shape[-1])
  pbar.reset(img_id_task)

  for img_id in range(img_batch.shape[-1]):

    img = img_batch[..., img_id]

    if (np.any(img)):

      # fname = lits17_file.replace(
      #   "LITS17", "LITS17_NON_ZERO"
      # ).replace(".nii", f"_{counter}.png")

      # plt.imsave(fname=fname, arr=img, cmap="gray")

      # counter += 1

      non_zero_list.append(img)

    pbar.advance(img_id_task)
  
  pbar.advance(lits17_files_task)

non_zero_np = np.asarray(non_zero_list, dtype=np.uint8)

np.save(f"{LITS17_DIR}/non_zero.npy", non_zero_np)

