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

    self.global_scale_CNN = TwoPathCNN(
      num_input_channels=self.num_input_channels,
      num_classes=self.num_classes, dropout=self.dropout
    )

    self.local_scale_CNN = TwoPathCNN(
      num_input_channels=self.num_input_channels + self.num_classes,
      num_classes=self.num_classes, dropout=self.dropout
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
    old_local_state_dict = self.local_scale_CNN.state_dict()
    old_global_state_dict = self.global_scale_CNN.state_dict()

    for local_layer_name, global_layer_name in zip(
      old_local_state_dict.keys(), old_global_state_dict.keys()
    ):
      
      if local_layer_name.split(".")[0] in layers_to_switch:
        
        temp = old_local_state_dict[local_layer_name]
        old_local_state_dict[local_layer_name] = old_global_state_dict[
          local_layer_name
        ]
        old_global_state_dict[local_layer_name] = temp
    
    self.local_scale_CNN.load_state_dict(old_local_state_dict)
    self.global_scale_CNN.load_state_dict(old_global_state_dict)
  
  def copy_local_layers_to_global(self):
    
    old_local_state_dict = self.local_scale_CNN.state_dict()
    old_global_state_dict = self.global_scale_CNN.state_dict()

    for local_layer_name, global_layer_name in zip(
      old_local_state_dict.keys(), old_global_state_dict.keys()
    ):
      
      if local_layer_name.split(".")[0] in [
        "local_conv_1_maxout_unit_0", "local_conv_1_maxout_unit_1",
        "concat_conv_0"
      ]:
        temp = old_local_state_dict[local_layer_name]
        old_local_state_dict[local_layer_name] = old_global_state_dict[
          local_layer_name
        ]
        old_global_state_dict[local_layer_name] = temp
    
    self.local_scale_CNN.load_state_dict(old_local_state_dict)
    self.global_scale_CNN.load_state_dict(old_global_state_dict)