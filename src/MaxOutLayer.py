from torch import nn
import torch

UNIFORM_INIT_LOWER_BOUND = -0.005
UNIFORM_INIT_UPPER_BOUND = +0.005

class MaxOutLayer(nn.Module):
  def __init__(
    self, in_channels, out_channels, kernel_size
  ):
    
    super().__init__()
    
    self.maxout_unit_0 = nn.Conv2d(
      in_channels=in_channels, out_channels=out_channels,
      kernel_size=kernel_size
    )
    nn.init.uniform_(
      self.maxout_unit_0.weight, UNIFORM_INIT_LOWER_BOUND, UNIFORM_INIT_UPPER_BOUND
    )

    self.maxout_unit_1 = nn.Conv2d(
      in_channels=in_channels, out_channels=out_channels,
      kernel_size=kernel_size
    )
    nn.init.uniform_(
      self.maxout_unit_1.weight, UNIFORM_INIT_LOWER_BOUND, UNIFORM_INIT_UPPER_BOUND
    )

  def forward(self, x):

    x_0 = self.maxout_unit_0(x)
    x_1 = self.maxout_unit_1(x)

    x = torch.stack((x_0, x_1), dim=0)

    x = torch.amax(x, dim=0)
    
    return x

    