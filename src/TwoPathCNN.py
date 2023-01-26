import torch
from torch import nn

UNIFORM_INIT_LOWER_BOUND = -0.005
UNIFORM_INIT_UPPER_BOUND = +0.005

class TwoPathCNN(nn.Module):
  def __init__(self, num_input_channels, num_classes, dropout):
    
    super().__init__()

    self.num_input_channels = num_input_channels
    self.num_classes = num_classes

    ### BEGIN local path, conv_0 block

    self.local_conv_0_maxout_unit_0 = nn.Conv2d(
      in_channels=self.num_input_channels, kernel_size=7, 
      out_channels=64
    )
    self.local_conv_0_maxout_unit_0
    torch.nn.init.uniform_(
      self.local_conv_0_maxout_unit_0.weight, 
      UNIFORM_INIT_LOWER_BOUND, UNIFORM_INIT_UPPER_BOUND
    )

    self.local_conv_0_maxout_unit_1 = nn.Conv2d(
      in_channels=self.num_input_channels, kernel_size=7, 
      out_channels=64
    )
    torch.nn.init.uniform_(
      self.local_conv_0_maxout_unit_1.weight, 
      UNIFORM_INIT_LOWER_BOUND, UNIFORM_INIT_UPPER_BOUND
    )

    self.local_pool_0 = nn.MaxPool2d(kernel_size=4, stride=1)
    
    self.local_dropout_0 = torch.nn.Dropout2d(p=dropout)

    ### END local path, conv_0 block
    
    ### BEGIN local path, conv_1 block
    
    self.local_conv_1_maxout_unit_0 = nn.Conv2d(
      in_channels=64, kernel_size=3, out_channels=64
    )
    torch.nn.init.uniform_(
      self.local_conv_1_maxout_unit_0.weight, 
      UNIFORM_INIT_LOWER_BOUND, UNIFORM_INIT_UPPER_BOUND
    )

    self.local_conv_1_maxout_unit_1 = nn.Conv2d(
      in_channels=64, kernel_size=3, out_channels=64
    )
    torch.nn.init.uniform_(
      self.local_conv_1_maxout_unit_1.weight, 
      UNIFORM_INIT_LOWER_BOUND, UNIFORM_INIT_UPPER_BOUND
    )

    self.local_pool_1 = nn.MaxPool2d(kernel_size=2, stride=1)

    self.local_dropout_1 = torch.nn.Dropout2d(p=dropout)


    ### END local path, conv_1 block
    
    ### BEGIN global path, conv_0 block
    
    self.global_conv_0_maxout_unit_0 = nn.Conv2d(
      in_channels=self.num_input_channels, kernel_size=13, 
      out_channels=160
    )
    torch.nn.init.uniform_(
      self.global_conv_0_maxout_unit_0.weight, 
      UNIFORM_INIT_LOWER_BOUND, UNIFORM_INIT_UPPER_BOUND
    )

    self.global_conv_0_maxout_unit_1 = nn.Conv2d(
      in_channels=self.num_input_channels, kernel_size=13, 
      out_channels=160
    )
    torch.nn.init.uniform_(
      self.global_conv_0_maxout_unit_1.weight, 
      UNIFORM_INIT_LOWER_BOUND, UNIFORM_INIT_UPPER_BOUND
    )

    self.global_dropout_0 = torch.nn.Dropout2d(p=dropout)


    ### END global path, conv_0 block

    ### BEGIN concat path, conv_0 block
    
    self.concat_conv_0 = nn.Conv2d(
      in_channels=224, out_channels=self.num_classes, kernel_size=21
    )

    self.concat_dropout_0 = torch.nn.Dropout2d(p=dropout)

    ### END concat path, conv_0 block
    
    

  def forward(self, x):

    x_local_path = x
    x_global_path = x

    ### BEGIN local path

    ## BEGIN conv 0

    x_maxout_unit_0 = self.local_conv_0_maxout_unit_0(x_local_path)
    # print(x_maxout_unit_0.shape)
    x_maxout_unit_1 = self.local_conv_0_maxout_unit_1(x_local_path)
    # print(x_maxout_unit_1.shape)
    x_maxout_units = torch.stack((x_maxout_unit_0, x_maxout_unit_1), dim=0)
    # print(x_maxout_units.shape)

    x_local_path = x_maxout_units.amax(dim=0)
    # print(x_local_path.shape)

    x_local_path = self.local_dropout_0(x_local_path)

    x_local_path = self.local_pool_0(x_local_path)
    # print(x_local_path.shape)

    ## END conv 0

    ## BEGIN conv 1

    x_maxout_unit_0 = self.local_conv_1_maxout_unit_0(x_local_path)
    # print(x_maxout_unit_0.shape)
    x_maxout_unit_1 = self.local_conv_1_maxout_unit_1(x_local_path)
    # print(x_maxout_unit_1.shape)
    x_maxout_units = torch.stack((x_maxout_unit_0, x_maxout_unit_1), dim=0)
    # print(x_maxout_units.shape)

    x_local_path = x_maxout_units.amax(dim=0)
    # print(x_local_path.shape)

    x_local_path = self.local_dropout_1(x_local_path)

    x_local_path = self.local_pool_1(x_local_path)
    # print(x_local_path.shape)

    ## END conv 1

    ### END local path
    
    ### BEGIN global path
    
    ## BEGIN conv 0
    
    x_maxout_unit_0 = self.global_conv_0_maxout_unit_0(x_global_path)
    # print(x_maxout_unit_0.shape)
    x_maxout_unit_1 = self.global_conv_0_maxout_unit_1(x_global_path)
    # print(x_maxout_unit_1.shape)
    x_maxout_units = torch.stack((x_maxout_unit_0, x_maxout_unit_1), dim=0)
    # print(x_maxout_units.shape)

    x_global_path = x_maxout_units.amax(dim=0)
    # print(x_global_path.shape)

    x_global_path = self.global_dropout_0(x_global_path)

    ## END conv 0
    
    ### END global path

    ### BEGIN concatenated path
    
    x_concat = torch.concat((x_local_path, x_global_path), dim=1)
    
    x_concat = self.concat_conv_0(x_concat)
    # print(x_concat.shape)

    x_concat = nn.functional.softmax(x_concat, dim=1)
    # print(x_concat.shape)
    
    ### END concatenated path

    return x_concat