from rich import print
from glob import glob
import random
import shutil 

# Starting with patient-level split in Train, Val and Test sets, in order to be 
# 100% sure of absence of label leakage

DATASET_PATH = "../data/BRATS2013_unstacked_padded_slices_train_val_test_split"

unstacked_slices_paths = glob(f"{DATASET_PATH}/*")

patient_ids = list(
  set(
    map(
      lambda slice_path: slice_path.split("/")[-1].split("_")[0], 
      unstacked_slices_paths
    )
  )
)

random.shuffle(patient_ids)

TRAIN_PERCENTAGE = 0.6
VAL_PERCENTAGE = 0.2

dataset_size = len(patient_ids)
train_size = int(dataset_size * TRAIN_PERCENTAGE)
val_size = int(dataset_size * VAL_PERCENTAGE)
test_size = dataset_size - train_size - val_size

patient_ids_dict = {
  "train": patient_ids[: train_size],
  "val": patient_ids[train_size : train_size + val_size],
  "test": patient_ids[train_size + val_size : ]
}

for split in patient_ids_dict.keys():

  for patient_id in patient_ids_dict[split]:

    src_list = glob(f"{DATASET_PATH}/{patient_id}_*")

    for src_dir in src_list:

      dst = src_dir.split("/")
      dst.insert(-1, split)
      dst = "/".join(dst)

      shutil.move(src_dir, dst)