from rich import print

import torch
from torch import nn

UNIFORM_INIT_LOWER_BOUND = -0.005
UNIFORM_INIT_UPPER_BOUND = +0.005

class TwoPathCNN(nn.Module):
  def __init__(self, num_input_channels, num_classes, dropout, model_role):
    
    super().__init__()

    self.num_input_channels = num_input_channels
    self.num_classes = num_classes

    self.model_role = model_role

    ### BEGIN local path, conv_0 block

    self.local_conv_0_maxout_unit_0 = nn.Conv2d(
      in_channels=self.num_input_channels, kernel_size=7, 
      out_channels=64
    )
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
    
    self.LAST_LAYER_NAME = "concat_conv_0"
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
  
  def _get_layer_weights(self, layer):

    layer_weights = []

    for name, param in layer.named_parameters():
      if param.requires_grad:
          layer_weights.append(param.data)

    weights = layer_weights[0].flatten()
    biases = layer_weights[1].flatten()

    weights_and_biases = torch.cat((weights, biases))

    return weights_and_biases


  
  def get_model_weights(self, device):

    weights = torch.empty((1)).to(device)

    weights = torch.cat(
      ( weights, self._get_layer_weights(self.local_conv_0_maxout_unit_0) )
    )
    weights = torch.cat(
      ( weights, self._get_layer_weights(self.local_conv_0_maxout_unit_0) )
    )
    weights = torch.cat(
      ( weights, self._get_layer_weights(self.local_conv_0_maxout_unit_1) )
    )
    weights = torch.cat(
      ( weights, self._get_layer_weights(self.local_conv_1_maxout_unit_0) )
    )
    weights = torch.cat(
      ( weights, self._get_layer_weights(self.local_conv_1_maxout_unit_1) )
    )
    weights = torch.cat(
      ( weights, self._get_layer_weights(self.global_conv_0_maxout_unit_0) )
    )
    weights = torch.cat(
      ( weights, self._get_layer_weights(self.global_conv_0_maxout_unit_1) )
    )
    weights = torch.cat(
      ( weights, self._get_layer_weights(self.concat_conv_0) )
    )

    return weights
  
  def get_num_trainable_parameters(self):
    return sum(p.numel() for p in self.parameters() if p.requires_grad)
  
  def freeze_layer(self, layer):

    for param in layer.parameters():
      param.requires_grad = False
  
  def freeze_first_layers(self):
    
    for layer in [
      self.local_conv_0_maxout_unit_0, self.local_conv_0_maxout_unit_1,
      self.local_conv_1_maxout_unit_0, self.local_conv_1_maxout_unit_1,
      self.global_conv_0_maxout_unit_0, self.global_conv_0_maxout_unit_1
    ]:
      
      self.freeze_layer(layer=layer)

  def freeze_last_layer(self):

    self.freeze_layer(layer=self.concat_conv_0)
    
  def prepare_for_tl(self, new_in_channels):

    ### BEGIN 
    
    local_conv_0_maxout_unit_0_old_state_dict = self.local_conv_0_maxout_unit_0.state_dict()

    self.local_conv_0_maxout_unit_0 = nn.Conv2d(
      in_channels=new_in_channels, kernel_size=7, out_channels=64
    )

    local_conv_0_maxout_unit_0_old_state_dict["weight"] = local_conv_0_maxout_unit_0_old_state_dict["weight"][:, : new_in_channels, ...]
    # local_conv_0_maxout_unit_0_old_state_dict["weight"] = local_conv_0_maxout_unit_0_old_state_dict["weight"].unsqueeze(1)

    self.local_conv_0_maxout_unit_0.load_state_dict(
      local_conv_0_maxout_unit_0_old_state_dict
    )

    ### END
    
    ### BEGIN 
    
    local_conv_0_maxout_unit_1_old_state_dict = self.local_conv_0_maxout_unit_1.state_dict()

    self.local_conv_0_maxout_unit_1 = nn.Conv2d(
      in_channels=new_in_channels, kernel_size=7, out_channels=64
    )

    local_conv_0_maxout_unit_1_old_state_dict["weight"] = local_conv_0_maxout_unit_1_old_state_dict["weight"][:, : new_in_channels, ...]
    # local_conv_0_maxout_unit_1_old_state_dict["weight"] = local_conv_0_maxout_unit_1_old_state_dict["weight"].unsqueeze(1)

    self.local_conv_0_maxout_unit_1.load_state_dict(
      local_conv_0_maxout_unit_1_old_state_dict
    )

    # END

    ### BEGIN 
    
    global_conv_0_maxout_unit_0_old_state_dict = self.global_conv_0_maxout_unit_0.state_dict()

    self.global_conv_0_maxout_unit_0 = nn.Conv2d(
      in_channels=new_in_channels, kernel_size=13, out_channels=160
    )

    global_conv_0_maxout_unit_0_old_state_dict["weight"] = global_conv_0_maxout_unit_0_old_state_dict["weight"][:, : new_in_channels, ...]

    self.global_conv_0_maxout_unit_0.load_state_dict(
      global_conv_0_maxout_unit_0_old_state_dict
    )

    ### END

    ### BEGIN 
    
    global_conv_0_maxout_unit_1_old_state_dict = self.global_conv_0_maxout_unit_1.state_dict()

    self.global_conv_0_maxout_unit_1 = nn.Conv2d(
      in_channels=new_in_channels, kernel_size=13, out_channels=160
    )

    global_conv_0_maxout_unit_1_old_state_dict["weight"] = global_conv_0_maxout_unit_1_old_state_dict["weight"][:, : new_in_channels, ...]

    self.global_conv_0_maxout_unit_1.load_state_dict(
      global_conv_0_maxout_unit_1_old_state_dict
    )

    ### END