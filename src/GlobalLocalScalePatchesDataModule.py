import pytorch_lightning as pl
from GlobalLocalScalePatchesDataset import GlobalLocalScalePatchesDataset
from torch.utils.data import DataLoader


class GlobalLocalScalePatchesDataModule(pl.LightningDataModule):
  
  def __init__(
    self, df_path_dict: dict, load_data_in_memory: bool, 
    batch_size: int, num_workers: int,
    transforms, is_old
  ):
    
    super().__init__()
    self.df_path_dict = df_path_dict
    self.load_data_in_memory = load_data_in_memory

    self.num_workers = num_workers
    self.batch_size = batch_size

    self.transforms = transforms
    self.is_old = is_old

    # self.train, self.val, self.test, self.predict = None, None, None, None

  def _common_setup(self, stage: str):

    return GlobalLocalScalePatchesDataset(
      df_path=self.df_path_dict[stage],
      stage=stage,
      load_data_in_memory=self.load_data_in_memory,
      transforms=self.transforms, is_old=self.is_old
    )

  def setup(self, stage: str):

    if stage == "fit":
      self.train = self._common_setup(stage="train")
        
      self.val = self._common_setup(stage="val")
    
    if stage == "test":
      self.test = self._common_setup(stage="test")

    if stage == "predict":
      # TODO this could be used to store the four brains (2 from val and 2 from 
      # test) that we wanted to test entirely, not just on randomly-sampled
      # patches

      self.predict = None
      raise NotImplementedError("TODO implement \"predict\" stage")


  def train_dataloader(self):
    return DataLoader(
      self.train, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True
    )

  def val_dataloader(self):
    return DataLoader(
      self.val, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True
    )

  def test_dataloader(self):
    return DataLoader(
      self.test, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True
    )

  def predict_dataloader(self):
    return DataLoader(
      self.predict, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True
    )
  

def __main__():

  from torchvision import transforms
  import torch

  datamodule = GlobalLocalScalePatchesDataModule(
    df_path_dict={
      "train": "../data_old/BRATS2013_patches_65_balanced/0/train_labels_df_one_hot.json",
      "val": "../data_old/BRATS2013_patches_33_balanced/0/val_labels_df_one_hot.json",
      "test": "../data_old/BRATS2013_patches_33_balanced/0/test_labels_df_one_hot.json"
    },
    load_data_in_memory=False,
    batch_size=64,
    num_workers=16,
    transforms=transforms.Compose([
      transforms.Normalize(
        mean=torch.zeros((4,1,1)), 
        std=torch.ones((4,1,1)))
    ]),
    is_old=True
  )

  datamodule.setup(stage="fit")

  dl = datamodule.train_dataloader()

  for x in dl:
    pass


if __name__ == "__main__":
  __main__()