import pytorch_lightning as pl
from GlobalPathCNN import GlobalPathCNN
from LocalPathCNN import LocalPathCNN
import torch
from torch import nn
from rich import print

class TwoPathCNN(nn.Module):
    
  def __init__(
      self, global_path_CNN: GlobalPathCNN, local_path_CNN: LocalPathCNN,
      out_channels
    ):
    
    super().__init__()

    self.global_path_CNN = global_path_CNN
    
    self.local_path_CNN = local_path_CNN

    self.concat_conv = nn.Conv2d(
      in_channels=224, out_channels=out_channels, kernel_size=21
    )

  def forward(self, x):
    x_global = self.global_path_CNN.forward(x)
    
    x_local = self.local_path_CNN.forward(x)

    x = torch.concat((x_local, x_global), dim=1)

    x = self.concat_conv.forward(x)

    x = nn.functional.softmax(x, dim=1)

    return x



def __main__():

  NUM_CLASSES = 5

  global_path_CNN = GlobalPathCNN(
    in_channels=4, out_channels=160, dropout_p=0.2
  )
  
  local_path_CNN = LocalPathCNN(
    in_channels=4, out_channels=64, dropout_p=0.3
  )

  two_path_CNN = TwoPathCNN(
    global_path_CNN=global_path_CNN, local_path_CNN=local_path_CNN, 
    out_channels=NUM_CLASSES
  )

  x = torch.rand((128, 4, 33, 33))

  two_path_CNN.forward(x)


__main__()
    