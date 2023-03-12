from torch import nn
import torch

from MaxOutLayer import MaxOutLayer

class LocalPathCNN(nn.Module):
  
  def __init__(self, in_channels, dropout_p):

    super().__init__()

    self.maxout_layer_0 = MaxOutLayer(
      in_channels=in_channels, out_channels=64, kernel_size=7
    )
    
    self.max_pool_0 = nn.MaxPool2d(kernel_size=4, stride=1)
    
    self.maxout_layer_1 = MaxOutLayer(
      in_channels=64, out_channels=21, kernel_size=3
    )

    self.max_pool_1 = nn.MaxPool2d(kernel_size=2, stride=1)

    self.dropout = torch.nn.Dropout2d(p=dropout_p)
  
  def forward(self, x):

    x = self.maxout_layer_0(x)

    x = self.dropout(x)

    x = self.max_pool_0(x)

    x = self.maxout_layer_1(x)

    x = self.dropout(x)

    x = self.max_pool_1(x)

    return x