import os, sys
sys.path.append("../")
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
import json
import matplotlib.pyplot as plt

from GlobalLocalScalePatchesDataset import GlobalLocalScalePatchesDataset

DF_PATH = "../data/BRATS2013_balanced/patch_metadata/train.json"

MEAN_STD_PATH = DF_PATH.replace(".json", "_mean_std.json")

mean_std_dict = json.load(open(MEAN_STD_PATH))

dataset = GlobalLocalScalePatchesDataset(
  df_path=DF_PATH,
  stage="train",
  load_data_in_memory=False,
  transforms=transforms.Normalize(
        mean=torch.tensor(mean_std_dict["mean"]), 
        std=torch.tensor(mean_std_dict["std"])
      )
)


a = [0] * 5
for x in dataset:
  a[x["patch_label"].item()] += 1
print(a)
