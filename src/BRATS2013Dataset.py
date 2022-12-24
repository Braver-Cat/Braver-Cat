import torch
from torch.utils.data import Dataset

import numpy as np

class BRATS2013Dataset(Dataset):

  def __init__(self, obs_list):
    self.obs_list = obs_list

  def __len__(self):
    return len(self.obs_list)

  def __getitem__(self, idx):
    
    print(f"self.obs_list[idx]: {self.obs_list[idx]}")

    img_np = np.load(f"{self.obs_list[idx]}/img.npy")
    label_np = np.load(f"{self.obs_list[idx]}/label.npy")

    observation_id = self.obs_list[idx].split("/")[-1].split("_")[0]
    slice_id = self.obs_list[idx].split("/")[-1].split("_")[1]

    return {
      "img": img_np,
      "label": label_np,
      "observation_id": observation_id,
      "slice_id": slice_id,
      "full_path": self.obs_list[idx]
    }



