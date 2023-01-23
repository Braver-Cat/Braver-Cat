import torch
from torch import nn

class InputCascadeCNNModelTrainer():
  
  def __init__(
    self, model, optimizer, learning_rate_scheduler, batch_size,
    dl_train, dl_val, dl_test, delta_1, delta_2
  ):
    
    self.model=model
    self.optimizer=optimizer 
    self.learning_rate_scheduler=learning_rate_scheduler
    self.batch_size=batch_size
    self.dl_train=dl_train 
    self.dl_val=dl_val 
    self.dl_test=dl_test
    self.delta_1 = delta_1
    self.delta_2 = delta_2

    




    # TODO 
    # Elastic-net regularization 
    # https://github.com/christianversloot/machine-learning-articles/blob/main/how-to-use-l1-l2-and-elastic-net-regularization-with-pytorch.md