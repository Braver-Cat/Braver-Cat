import imp
import torch

from rich.progress import *

from rich.live import Live
from rich.table import Table
from CustomProgress import CustomProgress

PBAR_EPOCHS_COLOR = "#830a48"
PBAR_TRAIN_COLOR = "#2a9d8f"
PBAR_VAL_COLOR = "#065a82"
PBAR_TEST_COLOR = "#00008B"

import time

import numpy as np

from CrossEntropyLossElasticNet import CrossEntropyLossElasticNet

class InputCascadeCNNModelTrainer():
  
  def __init__(
    self, device, model, num_epochs, optimizer, learning_rate_scheduler, 
    batch_size, percentage_num_batches, dl_train, dl_val, dl_test, delta_1, delta_2,
    checkpoint_full_path, checkpoint_step
  ):
    
    self.device = device

    self.model = model

    self.num_epochs = num_epochs

    self.optimizer = optimizer 
    self.learning_rate_scheduler = learning_rate_scheduler
    self.delta_1 = delta_1
    self.delta_2 = delta_2
    
    self.loss_fn_train = CrossEntropyLossElasticNet(
      delta_1=self.delta_1, delta_2=self.delta_2, device=self.device
    )
    self.loss_fn_val = CrossEntropyLossElasticNet(
      delta_1=self.delta_1, delta_2=self.delta_2, device=self.device
    )
    
    self.batch_size = batch_size
    self.percentage_num_batches = percentage_num_batches,
    
    self.dl_train = dl_train 
    self.num_batches_train = int(len(dl_train) * percentage_num_batches/100)
    self.num_batches_train += self.num_batches_train == 0

    self.dl_val = dl_val
    self.num_batches_val = int(len(dl_val) * percentage_num_batches/100)
    self.num_batches_val += self.num_batches_val == 0
    
    self.dl_test = dl_test
    self.num_batches_test = int(len(dl_test) * percentage_num_batches/100)
    self.num_batches_test += self.num_batches_test == 0

    self.num_batches_tot_train = self.num_batches_train + self.num_batches_val
    
    self.checkpoint_full_path = checkpoint_full_path
    self.checkpoint_step_size = checkpoint_step

    self.pbar = None
    self.pbar_epochs = None
    self.pbar_train = None
    self.pbar_val = None
    self.pbar_test = None

    self.best_epoch_train_acc = 0
    self.best_epoch_train_loss = 0

    self.best_train_acc = 0
    self.best_train_loss = np.inf
    
    self.best_val_acc = 0
    self.best_val_loss = np.inf

    self.best_epoch_val_acc = 0
    self.best_epoch_val_loss = 0

    self.current_train_acc = 0
    self.current_train_loss = np.inf
    
    self.current_val_acc = 0
    self.current_val_loss = np.inf

  def _set_pbars(self):
    self.pbar_epochs = self.pbar.add_task(
      f"[bold {PBAR_EPOCHS_COLOR}] Epochs", start=True, total=self.num_epochs,
    )
    self.pbar_train = self.pbar.add_task(
      f"[bold {PBAR_TRAIN_COLOR}] Train", start=True, 
      total=self.num_batches_train,
    )
    self.pbar_val = self.pbar.add_task(
      f"[bold {PBAR_VAL_COLOR}] Validation", start=True, 
      total=self.num_batches_val,
    )
    self.pbar_test = self.pbar.add_task(
      description=f"[bold {PBAR_TEST_COLOR}] Test", start=True, 
      total=self.num_batches_test,
    )

  def _store_checkpoint(self, checkpoint_path_suffix, checkpoint_epoch):
    
    torch.save(
      {
        "current_epoch": checkpoint_epoch,
        
        "model_state_dict": self.model.state_dict(),
        "optimizer_state_dict": self.optimizer.state_dict(),
        
        "best_epoch_val_acc": self.best_epoch_val_acc,
        "best_epoch_val_loss": self.best_epoch_val_loss,
        
        "best_val_acc": self.best_val_acc,
        "best_val_loss": self.best_val_loss,
      }, 
      f"{self.checkpoint_full_path}{checkpoint_path_suffix}"
    )
    
    return 
    
  def _handle_checkpoint(self, current_epoch, current_val_acc, current_val_loss):

    if current_epoch == self.num_epochs or current_epoch % self.checkpoint_step_size:
      self._store_checkpoint(
        checkpoint_path_suffix=f"_epoch_{current_epoch}",
        checkpoint_epoch=current_epoch
      )
    
    if current_val_loss < self.best_val_loss:
      self._store_checkpoint(
        checkpoint_path_suffix=f"_epoch_{current_epoch}_best_val_loss",
        checkpoint_epoch=current_epoch
      )

      self.best_epoch_val_loss = current_epoch

    if current_val_acc > self.best_val_acc:
      self._store_checkpoint(
        checkpoint_path_suffix=f"_epoch_{current_epoch}_best_val_acc",
        checkpoint_epoch=current_epoch
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

      self.optimizer.zero_grad()

      self.model.train()

      for batch_idx, batch_train in enumerate(self.dl_train):

        if (batch_idx > self.num_batches_train):
          break

        patch_local_scale = batch_train["local_scale"]["patch"].to(self.device)
        label_local_scale = batch_train["local_scale"]["patch_label"].to(self.device)
        
        patch_global_scale = batch_train["global_scale"]["patch"].to(self.device)
        label_global_scale = batch_train["global_scale"]["patch_label"].to(self.device)

        prediction = self.model(
          x_local_scale=patch_local_scale, 
          x_global_scale=patch_global_scale
        )
        prediction = prediction.squeeze(-1).transpose(dim0=1, dim1=2)

        # loss = self.loss_fn_train(
        #   prediction=prediction, label=label_global_scale, model=self.model
        # )

        loss = torch.nn.functional.cross_entropy(prediction, label_global_scale)

        loss.backward()

        self.optimizer.step()

        self.current_train_loss = loss.item()

        if self.current_train_loss < self.best_train_loss:
          self.best_train_loss = self.current_train_loss
          self.best_epoch_train_loss = epoch

        self.pbar.update(task_id=self.pbar_train, advance=1)
        self.pbar.update(
          task_id=self.pbar_epochs, 
          advance=( 1/(self.num_batches_tot_train) )
        )

      
      with torch.no_grad():

        self.model.eval()

        for batch_idx, batch_val in enumerate(self.dl_val):

          if (batch_idx > self.num_batches_val):
            break

          patch_local_scale = batch_val["local_scale"]["patch"].to(self.device)
          label_local_scale = batch_val["local_scale"]["patch_label"].to(self.device)
          
          patch_global_scale = batch_val["global_scale"]["patch"].to(self.device)
          label_global_scale = batch_val["global_scale"]["patch_label"].to(self.device)

          prediction = self.model(
            x_local_scale=patch_local_scale, 
            x_global_scale=patch_global_scale
          )
          prediction = prediction.squeeze(-1).transpose(dim0=1, dim1=2)

          # loss = self.loss_fn_val(
          #   prediction=prediction, label=label_global_scale, model=self.model
          # )
          loss = torch.nn.functional.cross_entropy(prediction, label_global_scale)

          self.current_val_loss = loss.item()

          if self.current_val_loss < self.best_val_loss:
            self.best_val_loss = self.current_val_loss
            self.best_epoch_val_loss = epoch

          self.pbar.update(task_id=self.pbar_val, advance=1)
          self.pbar.update(
            task_id=self.pbar_epochs, 
            advance=( 1/(self.num_batches_tot_train) )
          )
      
      self.pbar.update_table(
        current_train_loss=self.current_train_loss,
        current_val_loss=self.current_val_loss,
        best_train_loss=self.best_train_loss,
        best_val_loss=self.best_val_loss,
        best_epoch_train_loss=self.best_epoch_train_loss,
        best_epoch_val_loss=self.best_epoch_val_loss,
        current_train_acc=self.current_train_acc,
        current_val_acc=self.current_val_acc,
        best_train_acc=self.best_train_acc,
        best_val_acc=self.best_val_acc,
        best_epoch_train_acc=self.best_epoch_train_acc,
        best_epoch_val_acc=self.best_epoch_val_acc
      )


      self._handle_checkpoint(
        current_epoch=epoch, current_val_acc=acc_val, current_val_loss=loss_val,
      )

    return 0
  
  def _test(self):

    with torch.no_grad():

      self.model.eval()

      for batch_idx, batch_test in enumerate(self.dl_test):

        if (batch_idx > self.num_batches_test):
          break

        patch_local_scale = batch_test["local_scale"]["patch"].to(self.device)
        label_local_scale = batch_test["local_scale"]["patch_label"].to(self.device)
        
        patch_global_scale = batch_test["global_scale"]["patch"].to(self.device)
        label_global_scale = batch_test["global_scale"]["patch_label"].to(self.device)

        prediction = self.model(
          x_local_scale=patch_local_scale, 
          x_global_scale=patch_global_scale
        )

        self.pbar.update(task_id=self.pbar_test, advance=1)

    return 0
    


  def train(self):

    with CustomProgress(
      TextColumn(
        "[progress.description]{task.description}",
        justify="right"
      ),
      TextColumn(
        "[progress.percentage]{task.percentage:>3.0f}%",
        justify="right"
      ),
      BarColumn(),
      MofNCompleteColumn(),
      TextColumn("•"),
      TimeElapsedColumn(),
      TextColumn("•"),
      TimeRemainingColumn()
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