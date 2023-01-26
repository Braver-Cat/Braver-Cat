import torch

class CrossEntropyLossElasticNet(torch.nn.Module):
  
  def __init__(self, delta_1, delta_2):
    super(CrossEntropyLossElasticNet, self).__init__()

    self.delta_1 = delta_1
    self.delta_2 = delta_2

    self.loss_fn = torch.nn.CrossEntropyLoss()

  def _l1_loss(self, w):
      return torch.abs(w).sum()
  
  def _l2_loss(self, w):
      return torch.square(w).sum()

  def forward(self, prediction, label, model):

    model_weights = model.get_model_weights()
     
    return self.loss_fn(
      data=prediction, target=label
    ) + self.delta_1 * self._l1_loss(
      model_weights
    ) + self.delta_2 * self._l2_loss(model_weights)



  