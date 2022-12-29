import torch
from torch import nn

class TwoPathCNN(nn.Module):
  def __init__(self, num_input_channels):
    
    super().__init__()

    self.num_input_channels = num_input_channels

    self.local_path = nn.Sequential()
    self.global_path = nn.Sequential()

    self.local_path.add_module(
      name="local_conv_1", module=nn.Conv2d(
        in_channels=self.num_input_channels, kernel_size=7, padding_mode="valid"
      )
    )
    
    

  def forward(self, x):
    
    return x