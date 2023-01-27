from BRATS2013DatasetPatch import BRATS2013DatasetPatch

from torch.utils.data import DataLoader

from TwoPathCNN import TwoPathCNN

from torch.optim import SGD

import torch

from rich.progress import *
from rich import print

DATASET_BASE_PATH = "../data/BRATS2013_patches_33_balanced/0"
PATCH_SIZE = 33
LOAD_DATA_IN_MEMORY = False

BATCH_SIZE = 16
NUM_WORKERS = 16
LIMIT_NUM_BATCHES_TRAIN = 55
LIMIT_NUM_BATCHES_VAL   = 0
LIMIT_TOTAL_NUM_BATCHES_TRAIN = LIMIT_NUM_BATCHES_TRAIN + LIMIT_NUM_BATCHES_VAL

NUM_INPUT_CHANNELS = 4
NUM_CLASSES = 6
DROPOUT = 0.0

LR = 0.0001

NUM_EPOCHS = 25
PROGRESS_BAR_EPOCHS_COLOR = "#8B008B"
PROGRESS_BAR_TRAIN_COLOR  = "#00BFFF"
PROGRESS_BAR_VAL_COLOR    = "#FF7F50"

dataset_train = BRATS2013DatasetPatch(
    patch_df_path=f"{DATASET_BASE_PATH}/train_labels_df_one_hot.json", 
    patch_size=PATCH_SIZE, 
    load_data_in_memory=LOAD_DATA_IN_MEMORY, stage="train"
)
dataset_val = BRATS2013DatasetPatch(
    patch_df_path=f"{DATASET_BASE_PATH}/val_labels_df_one_hot.json", 
    patch_size=PATCH_SIZE, 
    load_data_in_memory=LOAD_DATA_IN_MEMORY, stage="val"
)

dataloader_train = DataLoader(
  dataset=dataset_train, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
  shuffle=True
)
dataloader_val = DataLoader(
  dataset=dataset_val, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
  shuffle=True
)

net = TwoPathCNN(
  num_input_channels=NUM_INPUT_CHANNELS, num_classes=NUM_CLASSES, 
  dropout=DROPOUT
)

optim = SGD(net.parameters(), lr=LR)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with Progress(
  TextColumn("[progress.description]{task.description}", justify="right"),
  TextColumn("[progress.percentage]{task.percentage:>3.0f}%", justify="right"),
  BarColumn(), MofNCompleteColumn(), TextColumn("•"), TimeElapsedColumn(), 
  TextColumn("•"), TimeRemainingColumn()
) as progress_bar:
  
  epoch_pb = progress_bar.add_task(
    description=f"[bold {PROGRESS_BAR_EPOCHS_COLOR}] Epochs progress", 
    start=True, total=NUM_EPOCHS
  )
  train_pb = progress_bar.add_task(
    description=f"[bold {PROGRESS_BAR_TRAIN_COLOR}] Train step progress", 
    start=True, total=LIMIT_NUM_BATCHES_TRAIN
  )
  val_pb = progress_bar.add_task(
    description=f"[bold {PROGRESS_BAR_VAL_COLOR}] Val step progress", 
    start=False, total=LIMIT_NUM_BATCHES_VAL
  )

  net = net.to(device)

  for epoch in range(NUM_EPOCHS):

    running_loss_train = 0

    progress_bar.reset(task_id=train_pb)
    progress_bar.reset(task_id=val_pb)

    net.train()

    for batch_id, batch in enumerate(dataloader_train):

      if batch_id > LIMIT_NUM_BATCHES_TRAIN: 
        break

      optim.zero_grad()

      patch = batch["patch"].to(device)
      gt_label = batch["patch_label"].to(device)
      gt_label = gt_label.squeeze(1)

      pred_label = net(patch)
      pred_label = pred_label.squeeze(-1).transpose(1, 2)
      pred_label = pred_label.squeeze(1)

      loss = torch.nn.functional.cross_entropy(pred_label, gt_label)

      loss.backward()

      optim.step()

      running_loss_train += loss.item() * BATCH_SIZE

      progress_bar.update(
        task_id=epoch_pb, advance=1 / LIMIT_TOTAL_NUM_BATCHES_TRAIN
      )
      progress_bar.update(task_id=train_pb, advance=1)

    print(f"train loss: {running_loss_train}")
