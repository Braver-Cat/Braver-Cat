import torch
from torch.utils.data import Dataset

import polars as pls

import numpy as np

class BRATS2013DatasetPatch(Dataset):

  def __init__(self, patch_df, patch_size, stage):
    
    self.patch_df = patch_df
    self.patch_size = patch_size

    self.stage = stage

  def __len__(self):
    return len(self.patch_df)

  def __getitem__(self, idx):
    
    return -123


