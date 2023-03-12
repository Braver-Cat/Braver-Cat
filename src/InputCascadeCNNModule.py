import pytorch_lightning as pl
import TwoPathCNN
import torch

class InputCascadeCNNModule(pl.LightningModule):
    
  def __init__(
      self, global_scale_CNN: TwoPathCNN, local_scale_CNN: TwoPathCNN
    ):
    
    super().__init__()

    self.global_scale_CNN = global_scale_CNN
    self.local_scale_CNN = local_scale_CNN


  def forward(self, x_global, x_local):

    x = self.global_scale_CNN.forward(x_global)

    x = torch.concat((x, x_local), dim=1)

    x = self.local_scale_CNN.forward(x)

    return x
  

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
