import torch
from torch.utils.data import Dataset

import numpy as np

import json

from BRATS2013Dataset import BRATS2013Dataset


class BRATS2013DatasetPatchMetadata(BRATS2013Dataset):

  def __init__(self, obs_list, stage, patch_size):

    super().__init__(
      obs_list=obs_list, stage=stage
    )

    self.patch_size = patch_size

  def __len__(self):
    return len(super().obs_list)

  def __getitem__(self, idx):

    super_getitem = super().__getitem__(idx)

    patch_metadata = self._load_json(
      f"{self.obs_list[idx]}/seg_label_to_patch_id_patch_size_{self.patch_size}.json"
    )

    super_getitem["patch_metadata"] = patch_metadata
    super_getitem["patch_size"] = self.patch_size

    return super_getitem

  def _load_json(self, json_path):
  
    json_file = open(json_path)
      
    json_data = json.load(json_file)
      
    json_file.close()

    return json_data
