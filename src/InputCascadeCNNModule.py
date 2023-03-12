import pytorch_lightning as pl
import TwoPathCNN
import torch
from torch.nn import functional as F

class InputCascadeCNNModule(pl.LightningModule):
    
  def __init__(
      self, global_scale_CNN: TwoPathCNN, local_scale_CNN: TwoPathCNN,
      optim_conf: dict
    ):
    
    super().__init__()

    self.global_scale_CNN = global_scale_CNN
    self.local_scale_CNN = local_scale_CNN

    self.optim_conf = optim_conf


  def forward(self, x_global, x_local) -> torch.tensor:

    x = self.global_scale_CNN.forward(x_global)

    x = torch.concat((x, x_local), dim=1)

    x = self.local_scale_CNN.forward(x)

    return x
  
  def _common_step(self, batch):
    patch_global_scale = batch["patch_global_scale"]
    patch_local_scale = batch["patch_local_scale"] 
    label_one_hot = batch["patch_label_one_hot"]
    
    pred_one_hot = self.forward(patch_global_scale, patch_local_scale)
    pred_one_hot = pred_one_hot.squeeze(-1).squeeze(-1)
    
    return F.cross_entropy(pred_one_hot, label_one_hot)

  
  def training_step(self, batch, batch_idx):
    return self._common_step(batch)
  
  def validation_step(self, batch, batch_idx):
    return self._common_step(batch)

  def test_step(self, batch, batch_idx):
    return self._common_step(batch)

  def predict_step(self, batch, batch_idx):
    return self._common_step(batch)

  
  def configure_optimizers(self):

    if self.optim_conf["name"] == "SGD":
      
      return torch.optim.SGD(
        params=self.parameters(), lr=self.optim_conf["lr"]
      )
    
    else:

      raise ValueError(
        f"Optimizer named {self.optim_conf['name']} not supported"
      )
  

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
