import torch

from tqdm import tqdm
PBAR_EPOCHS_COLOR = "#483c46"
PBAR_BATCHES_TRAIN_COLOR = "#70ae6e"
PBAR_BATCHES_VAL_COLOR = "#3c6e71"
PBAR_BATCHES_TEST_COLOR = "#beee62"

import time

class InputCascadeCNNModelTrainer():
  
  def __init__(
    self, device, model, num_epochs, optimizer, learning_rate_scheduler, 
    batch_size, num_batches, dl_train, dl_val, dl_test, delta_1, delta_2
  ):
    
    self.device = device
    self.model = model
    self.num_epochs = num_epochs
    self.optimizer = optimizer 
    self.learning_rate_scheduler = learning_rate_scheduler
    self.batch_size = batch_size
    self.num_batches = num_batches,
    self.dl_train = dl_train 
    self.dl_val = dl_val 
    self.dl_test = dl_test
    self.delta_1 = delta_1
    self.delta_2 = delta_2

    self.pbar_epochs = None
    self.pbar_batches_train = None
    self.pbar_batches_val = None
    self.pbar_batches_test = None
  
  def _train(self):
    
    self.pbar_epochs = tqdm(
      iterable=range(self.num_epochs), colour=PBAR_EPOCHS_COLOR, position=0,
    )
    self.pbar_batches_train = tqdm(
      iterable=self.dl_train, colour=PBAR_BATCHES_TRAIN_COLOR, leave=False, 
      position=1
    )
    self.pbar_batches_val = tqdm(
      iterable=self.dl_val, colour=PBAR_BATCHES_VAL_COLOR, leave=False, 
      position=2
    )

    for epoch in self.pbar_epochs:

      self.pbar_batches_train.reset()
      self.pbar_batches_val.reset()
      
      self.pbar_epochs.set_description(f"epoch {epoch + 1}")
      self.pbar_batches_train.set_description(f"epoch {epoch + 1}")
      self.pbar_batches_val.set_description(f"epoch {epoch + 1}")

      for batch_train in self.dl_train:

        self.pbar_batches_train.update(1)

      
      with torch.no_grad():

        for batch_val in self.dl_val:

          self.pbar_batches_val.update(1)


      self.pbar_epochs.update(1)

    return 0
  
  def _test(self):

    ### BEGIN test

    self.pbar_batches_test = tqdm(
      iterable=self.dl_test, colour=PBAR_BATCHES_TEST_COLOR, leave=False, 
      position=3
    )
    self.pbar_epochs.set_description(f"epoch {self.num_epochs}")

    with torch.no_grad():

      for batch_test in self.dl_test:

        self.pbar_batches_test.update(1)
    

    ### END test


  def train(self):

    self._train()

    self._test()

    return 0

    




    # TODO 
    # Elastic-net regularization 
    # https://github.com/christianversloot/machine-learning-articles/blob/main/how-to-use-l1-l2-and-elastic-net-regularization-with-pytorch.md