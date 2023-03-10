import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from rich import print, progress

class GlobalLocalScalePatchesDataset(Dataset):
  
  def __init__(
    self, df_path_dict, stage, load_data_in_memory, mean=0, std=1
  ):
    
    self.df = pd.read_json(df_path_dict)
    self.stage = stage
    
    self.load_data_in_memory = load_data_in_memory
    self.data = self.load_data() if self.load_data_in_memory else None
    
    self.mean = mean
    self.std = std

  def __len__(self):
    return len(self.df.index)
  
  def __getitem__(self, idx):
    
    if self.load_data_in_memory:
      return self.data[idx]
    else:
      return {
        "patch_global_scale": torch.tensor(np.load(self.df.iloc[idx]["global_patch_path"])),
        "patch_local_scale": torch.tensor(np.load(self.df.iloc[idx]["local_patch_path"])),
        "patch_label_one_hot": torch.tensor(self.df.iloc[idx]["patch_label_one_hot"])
      }  

  def load_data(self): 

    data = []

    pb = progress.Progress(
      progress.TextColumn("[progress.description]{task.description}"),
      progress.TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
      progress.BarColumn(),
      progress.MofNCompleteColumn(),
      progress.TextColumn("•"),
      progress.TimeElapsedColumn(),
      progress.TextColumn("•"),
      progress.TimeRemainingColumn()
    )

    pb_load_data_task = pb.add_task(
      "[bold #008080]Loading data in memory...", total=self.__len__()
    ) 

    pb.start()
    
    for df_row in self.df.iterrows():
      data.append(
        {
          "patch_global_scale": torch.tensor(np.load(df_row[1]["global_patch_path"])),
          "patch_local_scale": torch.tensor(np.load(df_row[1]["local_patch_path"])),
          "patch_label_one_hot": torch.tensor(df_row[1]["patch_label_one_hot"])
        }  
      )

      pb.advance(pb_load_data_task)

    pb.stop()

    return data


def __main__():

  ds = GlobalLocalScalePatchesDataset(
    df_path_dict="../data/BRATS2013_balanced/patch_metadata/train.json", 
    load_data_in_memory=True,
    stage="train"
  )

  from torch.utils.data import DataLoader
  dl = DataLoader(ds, batch_size=64, shuffle=True, num_workers=16)


  pb = progress.Progress(
    progress.TextColumn("[progress.description]{task.description}"),
    progress.TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    progress.BarColumn(),
    progress.MofNCompleteColumn(),
    progress.TextColumn("•"),
    progress.TimeElapsedColumn(),
    progress.TextColumn("•"),
    progress.TimeRemainingColumn()
  )

  pb_dataset_benchmark_task = pb.add_task(
    "Benchmarking...", total=len(dl)
  )

  pb.start()
  for entry in dl:
    pb.advance(pb_dataset_benchmark_task)
  pb.stop()

# __main__()
