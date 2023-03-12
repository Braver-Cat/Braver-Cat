import pytorch_lightning as pl
from GlobalLocalScalePatchesDataset import GlobalLocalScalePatchesDataset
from torch.utils.data import DataLoader


class GlobalLocalScalePatchesDataModule(pl.LightningDataModule):
  
  def __init__(
    self, df_path_dict: dict, load_data_in_memory: bool, 
    batch_size: int, num_workers: int,
    transforms
  ):
    
    super().__init__()
    self.df_path_dict = df_path_dict
    self.load_data_in_memory = load_data_in_memory

    self.num_workers = num_workers
    self.batch_size = batch_size

    self.transforms = transforms

    # self.train, self.val, self.test, self.predict = None, None, None, None

  def _common_setup(self, stage: str):

    return GlobalLocalScalePatchesDataset(
      df_path=self.df_path_dict[stage],
      stage=stage,
      load_data_in_memory=self.load_data_in_memory,
      transforms=self.transforms 
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
      self.train, batch_size=self.batch_size, num_workers=self.num_workers
    )

  def val_dataloader(self):
    return DataLoader(
      self.val, batch_size=self.batch_size, num_workers=self.num_workers
    )

  def test_dataloader(self):
    return DataLoader(
      self.test, batch_size=self.batch_size, num_workers=self.num_workers
    )

  def predict_dataloader(self):
    return DataLoader(
      self.predict, batch_size=self.batch_size, num_workers=self.num_workers
    )
  

def __main__():

  from torchvision import transforms
  import torch

  datamodule = GlobalLocalScalePatchesDataModule(
    df_path_dict={
      "train": "../data/BRATS2013_balanced/patch_metadata/train.json",
      "val": "../data/BRATS2013_balanced/patch_metadata/val.json",
      "test": "../data/BRATS2013_balanced/patch_metadata/test.json",
    },
    load_data_in_memory=False,
    batch_size=64,
    num_workers=16,
    transforms=transforms.Compose([
      transforms.Normalize(
        mean=torch.zeros((4,1,1)), 
        std=torch.ones((4,1,1)))
    ])
  )

  datamodule.setup(stage="fit")

  dl = datamodule.train_dataloader()

  for x in dl:
    pass


if __name__ == "__main__":
  __main__()