import torch
from torch.utils.data import Dataset

import pandas as pd

import numpy as np

from tqdm import tqdm

class BRATS2013DatasetPatch(Dataset):

  def __init__(
      self, patch_df_path, patch_size, load_in_memory, stage
    ):
    
    self.patch_df = pd.read_json(patch_df_path)
    self.patch_size = patch_size
    self.patch_column_name = f"patch_{self.patch_size}_x_{self.patch_size}_img_path"
    self.label_column_name = "patch_label"

    self.load_in_memory = load_in_memory
    self.data, self.labels = self._load_data_in_memory()

    self.stage = stage

  def __len__(self):
    return len(self.patch_df.index)

  def __getitem__(self, idx):

    if self.load_in_memory:
      patch = self.data[idx]
      patch_label = self.labels[idx]

    else:

      patch = np.load(
        f"{self.patch_df.iloc[idx][self.patch_column_name]}.npy"
      )
      patch_label = self.patch_df.iloc[idx]["patch_label"]

    return {
      "patch": patch,
      "patch_label": patch_label
    }
  
  def _load_data_in_memory(self):

    if not self.load_in_memory:
      return None, None
    
    else:

      data = []
      labels = []

      for _, row in tqdm(
        self.patch_df.iterrows(), colour="#a24fba", total=self.__len__(),
        desc="Dataset loading into memory progress"
      ):
        
        patch = np.load(f"{row[self.patch_column_name]}.npy")
        data.append(patch)
        
        labels.append(row[self.label_column_name])

      return data, labels
  