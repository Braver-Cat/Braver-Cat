import torch
from torch import nn

from model.TwoPathCNN import TwoPathCNN

class InputCascadeCNN(nn.Module):
  def __init__(self, num_input_channels, num_classes, dropout):
    
    super().__init__()
    self.cascade_type = "input"
    
    self.num_input_channels = num_input_channels
    self.num_classes = num_classes

    self.dropout = dropout

    self.global_scale_CNN: TwoPathCNN = TwoPathCNN(
      num_input_channels=self.num_input_channels,
      num_classes=self.num_classes, dropout=self.dropout, 
      model_role="global_scale"
    )

    self.local_scale_CNN: TwoPathCNN = TwoPathCNN(
      num_input_channels=self.num_input_channels + self.num_classes,
      num_classes=self.num_classes, dropout=self.dropout,
      model_role="local_scale"
    )

  def forward(self, x_local_scale, x_global_scale):

    x_global_scale = self.global_scale_CNN(x_global_scale)

    x_concat = torch.concat((x_global_scale, x_local_scale), dim=1)

    x_concat = self.local_scale_CNN(x_concat)
    # print(x_concat.shape)

    return x_concat
  
  def get_model_weights(self, device):

    return torch.cat(
      (
        self.local_scale_CNN.get_model_weights(device=device), 
        self.global_scale_CNN.get_model_weights(device=device)
      )
    )
  
  def get_num_trainable_parameters(self):
    
    return {
      "local_scale_CNN": self.local_scale_CNN.get_num_trainable_parameters(), 
      "global_scale_CNN": self.global_scale_CNN.get_num_trainable_parameters(), 
    }
  
  def switch_local_global_scale_layers(self, layers_to_switch):
    local_state_dict = self.local_scale_CNN.state_dict()
    global_state_dict = self.global_scale_CNN.state_dict()

    for local_layer_name, global_layer_name in zip(
      local_state_dict.keys(), global_state_dict.keys()
    ):
      
      if local_layer_name.split(".")[0] in layers_to_switch:
        
        temp = local_state_dict[local_layer_name]
        local_state_dict[local_layer_name] = global_state_dict[
          local_layer_name
        ]
        global_state_dict[local_layer_name] = temp
    
    self.local_scale_CNN.load_state_dict(local_state_dict)
    self.global_scale_CNN.load_state_dict(global_state_dict)
  
  def turn_off_layers(self, layers_to_turn_off, model):
    for module_name, module in model.named_modules():

      if module_name in layers_to_turn_off:

        for parameter_name, parameter in module.named_parameters():

          if "bias" in parameter_name:
            parameter = torch.zeros_like(parameter)
          
          if "weight" in parameter_name:
            torch.nn.init.dirac_(parameter)


  def copy_layers(self, layers_to_copy, copy_mode):
    local_state_dict = self.local_scale_CNN.state_dict()
    global_state_dict = self.global_scale_CNN.state_dict()

    for local_layer_name, global_layer_name in zip(
      local_state_dict.keys(), global_state_dict.keys()
    ):
      
      if local_layer_name.split(".")[0] in layers_to_copy:
        
        if copy_mode == "local_to_global":
          global_state_dict[local_layer_name] = local_state_dict[
            local_layer_name
          ]
        elif copy_mode == "global_to_local":
          local_state_dict[local_layer_name] = global_state_dict[
            local_layer_name
          ]
        else:
          raise ValueError(
            f"{copy_mode} copy mode does not exist. Supported values: local_to_global, global_to_local"
          )
    
    self.local_scale_CNN.load_state_dict(local_state_dict)
    self.global_scale_CNN.load_state_dict(global_state_dict)


  def change_num_in_channels(self, new_in_channels, num_classes):

    self.local_scale_CNN.change_num_in_channels(
      new_in_channels=new_in_channels + num_classes
    )
    self.global_scale_CNN.change_num_in_channels(new_in_channels=new_in_channels)