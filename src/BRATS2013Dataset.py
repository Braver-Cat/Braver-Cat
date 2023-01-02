import torch
from torch.utils.data import Dataset

import numpy as np

import json


class BRATS2013Dataset(Dataset):

  def __init__(self, obs_list, stage):
    
    self.obs_list = obs_list
    self.stage = stage

  def __len__(self):
    return len(self.obs_list)

  def __getitem__(self, idx):
    
    img = torch.tensor(np.load(f"{self.obs_list[idx]}/img.npy"))
    label = torch.tensor(np.load(f"{self.obs_list[idx]}/label.npy"))

    obs_folder_name = self.obs_list[idx].split("/")[-1].split("_")
    obs_id = obs_folder_name[0]
    slice_id = obs_folder_name[1]

    return {
      "full_path": self.obs_list[idx],

      "img": img,
      "label": label,
      
      "obs_id": obs_id,
      "slice_id": slice_id,
      
      "stage": self.stage
    }


