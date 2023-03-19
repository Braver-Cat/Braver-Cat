import sys
sys.path.append("../src")
from GlobalLocalScalePatchesDataset import GlobalLocalScalePatchesDataset
from torchvision import transforms
import torch
import os
import json

DF_PATH = "../data/BRATS2013_balanced/patch_metadata/train.json"
IS_OLD = False

ds = GlobalLocalScalePatchesDataset(
  df_path=DF_PATH, 
  load_data_in_memory=True,
  stage="train",
  transforms=transforms.Compose([
    transforms.Normalize(mean=0, std=1),
  ]),
  is_old=IS_OLD
)

entries = []

for entry in ds:
  entries.append(entry["patch_global_scale"])

std, mean = torch.std_mean(torch.stack(entries), dim=(0, 2, 3))


mean_std_export_dir = os.path.dirname(DF_PATH)
mean_std_export_file_name = DF_PATH.split("/")[-1].split(".")[0]
mean_std_export_path = os.path.join(
  mean_std_export_dir, f"{mean_std_export_file_name}_mean_std.json"
)

mean_std_dict = {
  "mean": mean.numpy().tolist(),
  "std": std.numpy().tolist()
}

with open(mean_std_export_path, 'w') as f: 
  json.dump(mean_std_dict, f)