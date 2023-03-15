from torch import nn
import torch

from MaxOutLayer import MaxOutLayer

class GlobalPathCNN(nn.Module):
  
  def __init__(self, in_channels, dropout_p):

    super().__init__()

    self.maxout_layer = MaxOutLayer(
      in_channels=in_channels, out_channels=160, kernel_size=13
    )

    self.dropout = torch.nn.Dropout2d(p=dropout_p)

  
  def forward(self, x):

    x = self.maxout_layer(x)

    x = self.dropout(x)

    return x
  
  def freeze(self):
    for param in self.parameters():
      param.requires_grad = False