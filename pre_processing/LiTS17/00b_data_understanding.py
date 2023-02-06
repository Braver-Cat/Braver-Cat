import numpy as np

import matplotlib.pyplot as plt

from rich.progress import *

NON_ZERO_FULL_PATH = f"../../data/LiTS17/non_zero.npy"

non_zero_np = np.load(NON_ZERO_FULL_PATH)

vmin = np.min(non_zero_np)
vmax = np.max(non_zero_np)

IMG_SAVE_BASE_PATH = "../../data/LiTS17/LITS17_NON_ZERO"

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

# LIM_NUM_FILES = 3
LIM_NUM_FILES = non_zero_np.shape[0]

img_id_task = pbar.add_task(
  description="Image ID progress...", total=LIM_NUM_FILES
)

pbar.start()

for img_id in list(range(non_zero_np.shape[0]))[: LIM_NUM_FILES]:

  img = non_zero_np[img_id, ...]

  plt.imsave(
    fname=f"{IMG_SAVE_BASE_PATH}/{str(img_id).zfill(5)}.png", arr=img,
    vmin=vmin, vmax=vmax
  )

  pbar.advance(img_id_task)



