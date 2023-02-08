from rich import print

import torch
from torch.utils.data import Dataset

import pandas as pd

import numpy as np

from tqdm import tqdm

from dataset.CustomDatasetPatch import CustomDatasetPatch

class CustomDatasetLocalGlobalScalePatch(Dataset):

  def __init__(
      self, 
      local_scale_df_path, global_scale_df_path,
      local_scale_patch_size, global_scale_patch_size, 
      local_scale_mean, local_scale_std,
      global_scale_load_data_in_memory, local_scale_load_data_in_memory, 
      global_scale_mean, global_scale_std,
      stage
    ):

    self.local_scale_dataset = CustomDatasetPatch(
      patch_df_path=local_scale_df_path, patch_size=local_scale_patch_size,
      load_data_in_memory=local_scale_load_data_in_memory, stage=stage,
      mean=local_scale_mean, std=local_scale_std
    )
    
    self.global_scale_dataset = CustomDatasetPatch(
      patch_df_path=global_scale_df_path, patch_size=global_scale_patch_size,
      load_data_in_memory=global_scale_load_data_in_memory, stage=stage,
      mean=global_scale_mean, std=global_scale_std
    )

    self._check_datasets_lens()

  def __len__(self):
    # safe to use the len of one of the two datasets because of the 
    # _check-datasets_lens() control that is called in the constructor

    return len(self.local_scale_dataset)

  def __getitem__(self, idx):

    return {
      "global_scale": self.global_scale_dataset[idx],
      "local_scale": self.local_scale_dataset[idx],
    }
  
  def _check_datasets_lens(self):
    
    if len(self.local_scale_dataset) != len(self.global_scale_dataset):
      raise ValueError(
        "Local and global scale datasets should have the same length!"
      )
  