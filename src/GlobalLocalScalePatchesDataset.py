import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from rich import print, progress
from torchvision import transforms

class GlobalLocalScalePatchesDataset(Dataset):
  
  def __init__(
    self, df_path, stage, load_data_in_memory, transforms
  ):

    self.df = pd.read_json(df_path)
    self.stage = stage
    self.transforms = transforms
    self.load_data_in_memory = load_data_in_memory
    self.data = self.load_data() if self.load_data_in_memory else None
    

  def __len__(self):
    return len(self.df.index)
  
  def _create_item_dict(self, df_row):

    return {
      "patch_global_scale": self.transforms(np.load(df_row["global_patch_path"])),
      "patch_local_scale": self.transforms(np.load(df_row["local_patch_path"])),
      "patch_label_one_hot": torch.tensor(df_row["patch_label_one_hot"])
    }
  
  def __getitem__(self, idx):
    
    if self.load_data_in_memory: return self.data[idx]
    
    else: return self._create_item_dict(self.df.iloc[idx])

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
      
      data.append(self._create_item_dict(df_row[1]))

      pb.advance(pb_load_data_task)

    pb.stop()

    return data


def __main__():

  ds = GlobalLocalScalePatchesDataset(
    df_path="../data/BRATS2013_balanced/patch_metadata/train.json", 
    load_data_in_memory=True,
    stage="train",
    transforms=transforms.Compose(
      [
        transforms.ToTensor(),
        transforms.Normalize(mean=0, std=1),
      ]
    )
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


if __name__ == "__main__":
  __main__()
