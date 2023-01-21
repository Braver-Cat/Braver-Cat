import torch
from torch.utils.data import Dataset

import pandas as pd

import numpy as np

class BRATS2013DatasetPatch(Dataset):

  def __init__(self, patch_df_path, patch_size, stage):
    
    self.patch_df = self._load_df(patch_df_path)
    self.patch_size = patch_size
    self.patch_column_name = f"patch_{self.patch_size}_x_{self.patch_size}_img_path"

    self.stage = stage

  def __len__(self):
    return len(self.patch_df.index)

  def __getitem__(self, idx):

    patch = np.load(
      f"{self.patch_df.iloc[idx][self.patch_column_name]}.npy"
    )
    patch_label = self.patch_df.iloc[idx]["patch_label"]

    return {
      "patch": patch,
      "patch_label": patch_label
    }
    
  
  def _load_df(self, df_path):
    return pd.read_json(df_path)
  