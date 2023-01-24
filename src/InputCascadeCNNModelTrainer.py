import torch

from rich.progress import *

PBAR_EPOCHS_COLOR = "#830a48"
PBAR_TRAIN_COLOR = "#2a9d8f"
PBAR_VAL_COLOR = "#065a82"
PBAR_TEST_COLOR = "#00008B"

import time

import numpy as np

class InputCascadeCNNModelTrainer():
  
  def __init__(
    self, device, model, num_epochs, optimizer, learning_rate_scheduler, 
    batch_size, num_batches, dl_train, dl_val, dl_test, delta_1, delta_2,
    checkpoint_full_path, checkpoint_step
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
    self.checkpoint_full_path = checkpoint_full_path
    self.checkpoint_step_size = checkpoint_step

    self.pbar_epochs = None
    self.pbar_train = None
    self.pbar_val = None
    self.pbar_test = None

    self.best_epoch_val_acc = 0
    self.best_epoch_val_loss = 0

    self.best_val_acc = 0
    self.best_val_loss = np.inf

  def _set_pbars(self):
    self.pbar_epochs = self.pbar.add_task(
      f"[bold {PBAR_EPOCHS_COLOR}] Epochs", start=True, total=self.num_epochs,
    )
    self.pbar_train = self.pbar.add_task(
      f"[bold {PBAR_TRAIN_COLOR}] Train", start=True, total=len(self.dl_train),
    )
    self.pbar_val = self.pbar.add_task(
      f"[bold {PBAR_VAL_COLOR}] Validation", start=True, total=len(self.dl_val),
    )
    self.pbar_test = self.pbar.add_task(
      description=f"[bold {PBAR_TEST_COLOR}] Test", start=True, total=len(self.dl_test),
    )

  def _store_checkpoint(self, checkpoint_path_suffix):
    return 0
    
  def _handle_checkpoint(self, current_epoch, current_val_acc, current_val_loss):

    if current_epoch == self.num_epochs or current_epoch % self.checkpoint_step_size:
      self._store_checkpoint(
        checkpoint_path_suffix=f"_epoch_{current_epoch}"
      )
    
    if current_val_loss < self.best_val_loss:
      self._store_checkpoint(
        checkpoint_path_suffix=f"_epoch_{current_epoch}_best_val_loss"
      )

      self.best_epoch_val_loss = current_epoch

    if current_val_acc > self.best_val_acc:
      self._store_checkpoint(
        checkpoint_path_suffix=f"_epoch_{current_epoch}_best_val_acc"
      )

      self.best_epoch_val_acc = current_epoch

    return
  
  def _train(self):

    self.model = self.model.to(self.device)

    acc_train, acc_val, acc_test = 0, 0, 0
    loss_train, loss_val = 0, 0

    for epoch in range(self.num_epochs):

      self.pbar.reset(self.pbar_train) 
      self.pbar.reset(self.pbar_val)

      for batch_train in self.dl_train:

        patch = batch_train["patch"].to(self.device)
        patch_label = batch_train["patch_label"].to(self.device)

        self.pbar.update(task_id=self.pbar_train, advance=1)

      
      with torch.no_grad():

        for batch_val in self.dl_val:

          patch = batch_val["patch"].to(self.device)
          patch_label = batch_val["patch_label"].to(self.device)

          self.pbar.update(task_id=self.pbar_val, advance=1)

      self._handle_checkpoint(
        current_epoch=epoch, current_val_acc=acc_val, current_val_loss=loss_val,
      )

      self.pbar.update(task_id=self.pbar_epochs, advance=1)

    return 0
  
  def _test(self):

    with torch.no_grad():

      for batch_test in self.dl_test:

        patch = batch_test["patch"].to(self.device)
        patch_label = batch_test["patch_label"].to(self.device)

        self.pbar.update(task_id=self.pbar_test, advance=1)

    return 0
    


  def train(self):

    with Progress(
      TextColumn("[progress.description]{task.description}"),
      TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
      BarColumn(),
      MofNCompleteColumn(),
      TextColumn("•"),
      TimeElapsedColumn(),
      TextColumn("•"),
      TimeRemainingColumn(),
    ) as progress:

      print()

      self.pbar = progress
      self._set_pbars()

      self._train()

      self._test()
    
    print()

    return 0

    




    # TODO 
    # Elastic-net regularization 
    # https://github.com/christianversloot/machine-learning-articles/blob/main/how-to-use-l1-l2-and-elastic-net-regularization-with-pytorch.md