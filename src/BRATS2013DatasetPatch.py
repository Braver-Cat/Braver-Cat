import torch
from torch.utils.data import Dataset

import pandas as pd

import numpy as np

from tqdm import tqdm

from rich import print

DEFAULT_TENSOR_DTYPE = torch.float32

class BRATS2013DatasetPatch(Dataset):

  def __init__(
      self, patch_df_path, patch_size, load_data_in_memory, stage, 
      mean=0, std=1
    ):
    
    self.patch_df_path = patch_df_path
    
    self.patch_size = patch_size

    self._check_patch_size_consistency()

    self.patch_df = pd.read_json(self.patch_df_path)
    self.patch_column_name = f"patch_{self.patch_size}_x_{self.patch_size}_img_path"
    self.label_column_name = "patch_label_one_hot"

    self.stage = stage

    self.load_data_in_memory = load_data_in_memory
    self.data, self.labels = self._load_data_in_memory()

    self.mean, self.std = mean, std

  def _check_patch_size_consistency(self):

    if self.patch_size not in self.patch_df_path:

      raise ValueError(
        f"Patch size {self.patch_size} not present in patch dataframe path: {self.patch_df_path}"
      )


  def __len__(self):
    return len(self.patch_df.index)

  def __getitem__(self, idx):

    if self.load_data_in_memory:
      patch = self.data[idx]
      patch_label = self.labels[idx]

    else:

      patch = torch.tensor(
        data=np.load(
          f"{self.patch_df.iloc[idx][self.patch_column_name]}.npy"
        ),
        dtype=DEFAULT_TENSOR_DTYPE
      )

      patch_label = torch.tensor(
        data=self.patch_df.iloc[idx][self.label_column_name], 
        dtype=DEFAULT_TENSOR_DTYPE
      ).unsqueeze(dim=0)


    return {
      "patch": patch,
      "patch_label": patch_label,
      "mean": self.mean,
      "std": self.std
    }
  
  def _load_data_in_memory(self):

    if not self.load_data_in_memory:
      return None, None
    
    else:

      data = []
      labels = []

      for _, row in tqdm(
        self.patch_df.iterrows(), colour="#a24fba", total=self.__len__(),
        desc=f"Loading {self.stage.ljust(5)} dataset in memory"
      ):
        
        patch = np.load(f"{row[self.patch_column_name]}.npy")
        data.append(patch)
        
        labels.append(row[self.label_column_name])

      return (
        torch.tensor(data=data, dtype=DEFAULT_TENSOR_DTYPE), 
        torch.tensor(data=labels, dtype=DEFAULT_TENSOR_DTYPE).unsqueeze(1)
      )
  