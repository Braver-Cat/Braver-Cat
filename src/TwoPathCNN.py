import torch
from torch import nn

class TwoPathCNN(nn.Module):
  def __init__(self, num_input_channels):
    
    super().__init__()

    self.num_input_channels = num_input_channels

    ### BEGIN local path, conv_0 block

    self.local_conv_0_maxout_unit_0 = nn.Conv2d(
      in_channels=self.num_input_channels, kernel_size=7, 
      out_channels=64
    )

    self.local_conv_0_maxout_unit_1 = nn.Conv2d(
      in_channels=self.num_input_channels, kernel_size=7, 
      out_channels=64
    )

    self.local_pool_0 = nn.MaxPool2d(kernel_size=4, stride=1)

    ### END local path, conv_0 block
    
    ### BEGIN local path, conv_1 block
    
    self.local_conv_1_maxout_unit_0 = nn.Conv2d(
      in_channels=64, kernel_size=3, out_channels=64
    )

    self.local_conv_1_maxout_unit_1 = nn.Conv2d(
      in_channels=64, kernel_size=3, out_channels=64
    )

    self.local_pool_1 = nn.MaxPool2d(kernel_size=2, stride=1)

    ### END local path, conv_1 block
    
    

  def forward(self, x):

    x_local_path = x
    x_global_path = x

    ### BEGIN local path

    ## BEGIN conv 0

    x_maxout_unit_0 = self.local_conv_0_maxout_unit_0(x_local_path)
    print(x_maxout_unit_0.shape)
    x_maxout_unit_1 = self.local_conv_0_maxout_unit_1(x_local_path)
    print(x_maxout_unit_1.shape)
    x_maxout_units = torch.stack((x_maxout_unit_0, x_maxout_unit_1), dim=0)
    print(x_maxout_units.shape)

    x_local_path = x_maxout_units.amax(dim=0)
    print(x_local_path.shape)

    x_local_path = self.local_pool_0(x_local_path)
    print(x_local_path.shape)

    ## END conv 0

    ## BEGIN conv 1

    x_maxout_unit_0 = self.local_conv_1_maxout_unit_0(x_local_path)
    print(x_maxout_unit_0.shape)
    x_maxout_unit_1 = self.local_conv_1_maxout_unit_1(x_local_path)
    print(x_maxout_unit_1.shape)
    x_maxout_units = torch.stack((x_maxout_unit_0, x_maxout_unit_1), dim=0)
    print(x_maxout_units.shape)

    x_local_path = x_maxout_units.amax(dim=0)
    print(x_local_path.shape)

    x_local_path = self.local_pool_1(x_local_path)
    print(x_local_path.shape)

    ## END conv 1



    ### END local path



    return x