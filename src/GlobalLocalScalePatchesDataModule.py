import pytorch_lightning as pl
from GlobalLocalScalePatchesDataset import GlobalLocalScalePatchesDataset
from torch.utils.data import DataLoader


class GlobalLocalScalePatchesDataModule(pl.LightningDataModule):
  
  def __init__(
    self, df_path_dict: dict, load_data_in_memory: bool, 
    batch_size: int, num_workers: int,
  ):
    
    super().__init__()
    self.df_path_dict = df_path_dict
    self.load_data_in_memory = load_data_in_memory

    self.num_workers = num_workers
    self.batch_size = batch_size

  def setup(self, stage: str):

    if stage == "fit":
      self.train = GlobalLocalScalePatchesDataset(
        df_path=self.df_path_dict["train"],
        stage="train",
        load_data_in_memory=self.load_data_in_memory,
        mean=self.mean, 
        std=self.std 
      )
        
      self.val = GlobalLocalScalePatchesDataset(
        df_path=self.df_path_dict["val"],
        stage="val",
        load_data_in_memory=self.load_data_in_memory,
        mean=self.mean, 
        std=self.std 
      )
    
    if stage == "test":
      self.test = GlobalLocalScalePatchesDataset(
        df_path=self.df_path_dict["test"],
        stage="test",
        load_data_in_memory=self.load_data_in_memory,
        mean=self.mean, 
        std=self.std 
      )

    if stage == "predict":
      # TODO this could be used to store the four brains (2 from val and 2 from 
      # test) that we wanted to test entirely, not just on randomly-sampled
      # patches
      self.predict = None


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