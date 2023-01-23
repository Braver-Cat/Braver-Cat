import torch
from torch import nn

from TwoPathCNN import TwoPathCNN

class InputCascadeCNN(nn.Module):
  def __init__(self, num_input_channels, num_classes, dropout):
    
    super().__init__()
    self.cascade_type = "input"
    
    self.num_input_channels = num_input_channels
    self.num_classes = num_classes

    self.dropout = dropout

    self.large_scale_CNN = TwoPathCNN(
      num_input_channels=self.num_input_channels,
      num_classes=self.num_classes, dropout=self.dropout
    )

    self.small_scale_CNN = TwoPathCNN(
      num_input_channels=self.num_input_channels + self.num_classes,
      num_classes=self.num_classes, dropout=self.dropout
    )

  def forward(self, x_small_scale, x_large_scale):
    x_large_scale = self.large_scale_CNN(x_large_scale)

    x_concat = torch.concat((x_large_scale, x_small_scale), dim=1)

    x_concat = self.small_scale_CNN(x_concat)
    print(x_concat.shape)

    return x_concat