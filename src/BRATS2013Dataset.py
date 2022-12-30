import torch
from torch.utils.data import Dataset

import numpy as np

import json


class BRATS2013Dataset(Dataset):

  def __init__(self, obs_list, stage, patch_size):
    
    self.obs_list = obs_list
    self.stage = stage
    self.patch_size = patch_size

  def __len__(self):
    return len(self.obs_list)

  def __getitem__(self, idx):
    
    img = torch.tensor(np.load(f"{self.obs_list[idx]}/img.npy"))
    label = torch.tensor(np.load(f"{self.obs_list[idx]}/label.npy"))

    obs_folder_name = self.obs_list[idx].split("/")[-1].split("_")
    obs_id = obs_folder_name[0]
    slice_id = obs_folder_name[1]

    patch_metadata = self._load_json(
      f"{self.obs_list[idx]}/seg_label_to_patch_id_patch_size_{self.patch_size}.json"
    )

    return {
      "full_path": self.obs_list[idx],

      "img": img,
      "label": label,
      
      "obs_id": obs_id,
      "slice_id": slice_id,
      
      "patch_size": self.patch_size,
      "patch_metadata": patch_metadata,
      
      "stage": self.stage
    }

  def _load_json(self, json_path):
  
    json_file = open(json_path)
      
    json_data = json.load(json_file)
      
    json_file.close()

    return json_data
