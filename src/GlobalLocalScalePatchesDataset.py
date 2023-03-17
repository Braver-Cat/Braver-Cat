import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from rich import print, progress
from torchvision import transforms

class GlobalLocalScalePatchesDataset(Dataset):
  
  def __init__(
    self, df_path, stage, load_data_in_memory, transforms, is_old
  ):

    self.df = pd.read_json(df_path)
    self.stage = stage
    self.transforms = transforms
    self.load_data_in_memory = load_data_in_memory
    self.data = self.load_data() if self.load_data_in_memory else None
    self.is_old = is_old
    

  def __len__(self):
    return len(self.df.index)
  
  def _create_item_dict(self, df_row):
    if self.is_old:
      global_patch_path = df_row["patch_65_x_65_img_path"].replace("/data", "/data_old") + ".npy"
      local_patch_path = global_patch_path.replace("patches_65_", "patches_33_")
    else:
      global_patch_path = df_row["global_patch_path"]
      local_patch_path = df_row["local_patch_path"]

    return {
      "patch_global_scale": self.transforms(torch.tensor(np.load(global_patch_path), dtype=torch.float32)),
      "patch_local_scale": self.transforms(torch.tensor(np.load(local_patch_path), dtype=torch.float32)),
      "patch_label": torch.tensor(df_row["patch_label"], dtype=torch.float32),
      "patch_label_one_hot": torch.tensor(df_row["patch_label_one_hot"], dtype=torch.float32)
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
      f"[bold #008080]Loading data in memory ({self.stage} split)", 
      total=self.__len__()
    ) 

    pb.start()
    
    for df_row in self.df.iterrows():
      
      data.append(self._create_item_dict(df_row[1]))

      pb.advance(pb_load_data_task)

    pb.stop()

    return data


def __main__():

  import json
  
  DF_PATH = "../data/BRATS2013_balanced/patch_metadata/train.json"
  MEAN_STD_PATH = DF_PATH.replace(".json", "_mean_std.json")

  mean_std_dict = json.load(open(MEAN_STD_PATH))

  ds = GlobalLocalScalePatchesDataset(
    df_path=DF_PATH,
    load_data_in_memory=False,
    stage="train",
    transforms=transforms.Compose([
      transforms.Normalize(
        mean=torch.tensor(mean_std_dict["mean"]), 
        std=torch.tensor(mean_std_dict["std"])
      ),
    ])
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
