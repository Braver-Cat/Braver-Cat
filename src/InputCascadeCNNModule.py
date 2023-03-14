import pytorch_lightning as pl
import TwoPathCNN
import torch
from torch.nn import functional as F
from torch.optim import *
from torch.optim.lr_scheduler import *
from torchmetrics import Accuracy
import time

class InputCascadeCNNModule(pl.LightningModule):
    
  def __init__(
      self, global_scale_CNN: TwoPathCNN, local_scale_CNN: TwoPathCNN,
      optim_conf: dict, scheduler_conf: dict, num_classes: int,
      model_state_dict_path=None
    ):
    
    super().__init__()

    self.global_scale_CNN = global_scale_CNN
    self.local_scale_CNN = local_scale_CNN

    self.optim_conf = optim_conf
    self.scheduler_conf = scheduler_conf

    self.accuracy = Accuracy(task="multiclass", num_classes=num_classes)

    self.model_state_dict_path = model_state_dict_path

    self.train_start_time = None

  def forward(self, x_global, x_local) -> torch.tensor:

    x = self.global_scale_CNN.forward(x_global)

    x = torch.concat((x, x_local), dim=1)

    x = self.local_scale_CNN.forward(x)

    return x
  
  def _common_step(self, batch, stage):
    patch_global_scale = batch["patch_global_scale"]
    patch_local_scale = batch["patch_local_scale"] 
    label_one_hot = batch["patch_label_one_hot"]
    label = batch["patch_label"]
    
    pred_one_hot = self.forward(patch_global_scale, patch_local_scale)
    pred_one_hot = pred_one_hot.squeeze(-1).squeeze(-1)

    loss = F.cross_entropy(pred_one_hot, label_one_hot)
    acc = self.accuracy(pred_one_hot, label)

    if stage == "train" or stage == "val":
      self.log(f"loss/{stage}", loss, on_epoch=True, on_step=False, prog_bar=True)
      # needed in order to save weights_only checkpoint, since PL does not
      # allow to monitor two ModelCheckpoint with the same "monitor" and 
      # "metric" values.
      self.log(f"loss_{stage}", loss, on_epoch=True, on_step=False, prog_bar=True)
      self.log(f"acc/{stage}", acc, on_epoch=True, on_step=False, prog_bar=True)
    
    return loss

  
  def training_step(self, batch, batch_idx):
    return self._common_step(batch, "train")
  
  def validation_step(self, batch, batch_idx):
    return self._common_step(batch, "val")

  def test_step(self, batch, batch_idx):
    return self._common_step(batch, "test")

  def predict_step(self, batch, batch_idx):
    return self._common_step(batch, "prediction")

  def _configure_optimizers(self):

    if self.optim_conf["name"] == "SGD":
      
      return SGD(
        params=self.parameters(), lr=self.optim_conf["lr"], 
        momentum=self.optim_conf["momentum"], 
        weight_decay=self.optim_conf["weight_decay"]
      )
    
    else:
      raise ValueError(f"Optimizer {self.optim_conf['name']} not supported")
    
  def _configure_schedulers(self, optim):
    
    if self.scheduler_conf["name"] == "StepLR":
      return StepLR(
        optimizer=optim, step_size=self.scheduler_conf["step_size"], 
        gamma=self.scheduler_conf["gamma"]
      )

  def configure_optimizers(self):

    optim = self._configure_optimizers() 

    return {
      "optimizer": optim, 
      "lr_scheduler": self._configure_schedulers(optim)
    }
  
  def on_fit_start(self):

    if self.model_state_dict_path is not None:
      print(f"Loading weights_only checkpoint from {self.model_state_dict_path}")
      self.load_state_dict(torch.load(self.model_state_dict_path)["state_dict"])

    self.logger.watch(self, log_graph=False)

  def on_train_epoch_start(self):

    self.log("epoch/lr", self.lr_schedulers().get_last_lr()[0])

    self.train_start_time = time.time()

  def on_train_epoch_end(self):

    epoch_exec_time = time.time() - self.train_start_time
    
    self.log("epoch/exec_time_seconds", epoch_exec_time)

  

def __main__():

  from TwoPathCNN import TwoPathCNN 
  from GlobalPathCNN import GlobalPathCNN
  from LocalPathCNN import LocalPathCNN

  NUM_CLASSES = 5

  global_scale_CNN = TwoPathCNN(
    global_path_CNN=GlobalPathCNN(in_channels=4, dropout_p=0.3),
    local_path_CNN=LocalPathCNN(in_channels=4, dropout_p=0.1),
    out_channels=NUM_CLASSES
  )
  
  local_scale_CNN = TwoPathCNN(
    global_path_CNN=GlobalPathCNN(in_channels=4 + NUM_CLASSES, dropout_p=0.3),
    local_path_CNN=LocalPathCNN(in_channels=4 + NUM_CLASSES, dropout_p=0.1),
    out_channels=NUM_CLASSES
  )

  input_cascade_CNN = InputCascadeCNNModule(
    global_scale_CNN=global_scale_CNN,
    local_scale_CNN=local_scale_CNN
  )

  x_global = torch.rand((128, 4, 65, 65))
  x_local = torch.rand((128, 4, 33, 33))

  input_cascade_CNN.forward(x_global, x_local)

if __name__ == "__main__":
  __main__()
