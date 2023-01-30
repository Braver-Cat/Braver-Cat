import imp
import torch

from rich.progress import *
from rich import print

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
    batch_size, num_batches_train, num_batches_val, num_batches_test, 
    dl_train, dl_val, dl_test, 
    delta_1, delta_2,
    checkpoint_full_path, checkpoint_step, train_id, resumed_from_checkpoint,
    starting_epoch, wandb_helper
  ):
    
    self.device = device

    self.model = model

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
    
    self.dl_train = dl_train 
    self.num_batches_train = num_batches_train

    self.dl_val = dl_val
    self.num_batches_val = num_batches_val
    
    self.dl_test = dl_test
    self.num_batches_test = num_batches_test

    self.num_batches_tot_train = self.num_batches_train + self.num_batches_val
    
    self.checkpoint_full_path = checkpoint_full_path
    self.checkpoint_step_size = checkpoint_step
    self.train_id = train_id
    self.resumed_from_checkpoint = resumed_from_checkpoint
    self.starting_epoch = starting_epoch
    
    self.num_epochs = num_epochs
    self.last_epoch = self.starting_epoch + self.num_epochs

    self.wandb_helper = wandb_helper

    self.pbar = None
    self.pbar_epochs = None
    self.pbar_train = None
    self.pbar_val = None
    self.pbar_test = None

    self.best_epoch_train_acc = -1
    self.best_epoch_train_loss = -1

    self.best_train_acc = 0
    self.best_train_loss = np.inf
    self.delta_train_loss = 0
    
    self.best_val_acc = 0
    self.best_val_loss = np.inf
    self.delta_val_loss = 0

    self.best_epoch_val_acc = -1
    self.best_epoch_val_loss = -1

    self.running_train_acc = 0
    self.running_train_loss = np.inf
    
    self.running_val_acc = 0
    self.running_val_loss = np.inf


  def _set_pbars(self):
    self.pbar_epochs = self.pbar.add_task(
      f"[bold {PBAR_EPOCHS_COLOR}] Epochs", start=True, 
      total=self.last_epoch + 1, completed=self.starting_epoch
    )
    self.pbar_train = self.pbar.add_task(
      f"[bold {PBAR_TRAIN_COLOR}] Train", start=True, 
      total=self.num_batches_train + 1,
    )
    self.pbar_val = self.pbar.add_task(
      f"[bold {PBAR_VAL_COLOR}] Validation", start=True, 
      total=self.num_batches_val + 1,
    )
    self.pbar_test = self.pbar.add_task(
      description=f"[bold {PBAR_TEST_COLOR}] Test", start=True, 
      total=self.num_batches_test + 1,
    )

  def _store_checkpoint(self, checkpoint_path_suffix, checkpoint_epoch):

    export_full_path = f"{self.checkpoint_full_path}/checkpoint{checkpoint_path_suffix}.pth"
    
    torch.save(
      obj={
      
        "train_id": self.train_id,
        "resumed_from_checkpoint": self.resumed_from_checkpoint,

        "checkpoint_epoch": checkpoint_epoch,
        
        "model_state_dict": self.model.state_dict(),
        "optimizer_state_dict": self.optimizer.state_dict(),
        "learning_rate_scheduler_state_dict": self.learning_rate_scheduler.state_dict(),

        "best_epoch_train_acc": self.best_epoch_train_acc,
        "best_epoch_train_loss": self.best_epoch_train_loss,

        "best_train_acc": self.best_train_acc,
        "best_train_loss": self.best_train_loss,
        
        "best_val_acc": self.best_val_acc,
        "best_val_loss": self.best_val_loss,

        "best_epoch_val_acc": self.best_epoch_val_acc,
        "best_epoch_val_loss": self.best_epoch_val_loss,
        
        "best_epoch_val_acc": self.best_epoch_val_acc,
        "best_epoch_val_loss": self.best_epoch_val_loss,
        
        "best_val_acc": self.best_val_acc,
        "best_val_loss": self.best_val_loss,
      }, 
      f=export_full_path
    )
    
    return 
    
  def _handle_checkpoint(
      self, current_epoch, trigger_export_by_acc, trigger_export_by_loss
    ):

    if current_epoch != 0 and (
      current_epoch == self.last_epoch - 1 or 
      current_epoch % self.checkpoint_step_size == 0
    ):
    
      self._store_checkpoint(
        checkpoint_path_suffix=f"_epoch_{current_epoch}",
        checkpoint_epoch=current_epoch
      )
    
    if trigger_export_by_loss:
      self._store_checkpoint(
        checkpoint_path_suffix=f"_epoch_{current_epoch}_best_val_loss",
        checkpoint_epoch=current_epoch
      )

      self.best_epoch_val_loss = current_epoch

    if trigger_export_by_acc:
      self._store_checkpoint(
        checkpoint_path_suffix=f"_epoch_{current_epoch}_best_val_acc",
        checkpoint_epoch=current_epoch
      )

      self.best_epoch_val_acc = current_epoch

    return
  
  def get_num_correct_preds(self, outputs, labels):
    
    output_pred_ind = torch.argmax(outputs, dim=1)
    labels_ind = torch.argmax(labels, dim=1)
    
    matching_mask = (output_pred_ind == labels_ind).float()
    
    num_correct_preds = matching_mask.sum()
    
    return num_correct_preds
  
  def get_accuracy(self, outputs, labels, total_num_preds):

    return self._get_num_correct_preds(outputs, labels) / total_num_preds
  

  def get_wandb_config_update(self):
    
    return {
      "best_epoch_train_acc" : self.best_epoch_train_acc, 
      "best_epoch_train_loss" : self.best_epoch_train_loss,

      "best_train_acc" : self.best_train_acc,
      "best_train_loss" : self.best_train_loss,
      
      "best_val_acc" : self.best_val_acc,
      "best_val_loss" : self.best_val_loss,

      "best_epoch_val_acc" : self.best_epoch_val_acc,
      "best_epoch_val_loss" : self.best_epoch_val_loss,

      "running_train_acc" : self.running_train_acc,
      "running_train_loss" : self.running_train_loss,
      
      "running_val_acc" : self.running_val_acc,
      "running_val_loss" : self.running_val_loss
    }

  def _train(self):

    self.model = self.model.to(self.device)
    
    for epoch in range(self.starting_epoch, self.last_epoch):

      self.pbar.reset(self.pbar_train) 
      self.pbar.reset(self.pbar_val)

      self.running_train_loss = 0
      self.running_val_loss = 0

      self.running_train_acc = 0
      self.running_val_acc = 0

      self.model.train()

      for batch_idx, batch_train in enumerate(self.dl_train):

        if (batch_idx > self.num_batches_train):
          break

        self.optimizer.zero_grad()

        patch_local_scale = batch_train["local_scale"]["patch"].to(self.device)
        label_local_scale = batch_train["local_scale"]["patch_label"].to(self.device)
        
        patch_global_scale = batch_train["global_scale"]["patch"].to(self.device)
        label_global_scale = batch_train["global_scale"]["patch_label"].to(self.device)
        label_global_scale = label_global_scale.squeeze(1)

        local_scale_mean = batch_train["local_scale"]["mean"].to(self.device)
        local_scale_mean = local_scale_mean.unsqueeze(-1).unsqueeze(-1)
        local_scale_std = batch_train["local_scale"]["std"].to(self.device)
        local_scale_std = local_scale_std.unsqueeze(-1).unsqueeze(-1)
        
        global_scale_mean = batch_train["global_scale"]["mean"].to(self.device)
        global_scale_mean = global_scale_mean.unsqueeze(-1).unsqueeze(-1)
        global_scale_std = batch_train["global_scale"]["std"].to(self.device)
        global_scale_std = global_scale_std.unsqueeze(-1).unsqueeze(-1)

        patch_local_scale = (patch_local_scale - local_scale_mean) / local_scale_std
        patch_global_scale = (patch_global_scale - global_scale_mean) / global_scale_std

        prediction = self.model(
          x_local_scale=patch_local_scale, 
          x_global_scale=patch_global_scale
        )
        prediction = prediction.squeeze(-1).squeeze(-1)

        # loss = self.loss_fn_train(
        #   prediction=prediction, label=label_global_scale, model=self.model
        # )

        loss = torch.nn.functional.cross_entropy(prediction, label_global_scale)

        loss.backward()

        self.optimizer.step()

        self.running_train_loss += loss.item() * self.batch_size

        # accumulating the number of correct predictions to divide them by
        # the total number of preds later to get the accuracy
        self.running_train_acc += self.get_num_correct_preds(
          prediction, label_global_scale
        )

        self.pbar.update(task_id=self.pbar_train, advance=1)
      
        self.pbar.update(
          task_id=self.pbar_epochs, 
          advance=( 1/(self.num_batches_tot_train + 2) )
        )

      if self.running_train_loss < self.best_train_loss:
        self.delta_train_loss = self.running_train_loss - self.best_train_loss
        self.best_train_loss = self.running_train_loss
        self.best_epoch_train_loss = epoch

      self.running_train_acc /= self.batch_size * self.num_batches_train

      if self.best_train_acc < self.running_train_acc:
        self.best_train_acc = self.running_train_acc
        self.best_epoch_train_acc = epoch

      self.learning_rate_scheduler.step()
      
      with torch.no_grad():

        self.model.eval()

        trigger_export_by_acc = False
        trigger_export_by_loss = False

        for batch_idx, batch_val in enumerate(self.dl_val):

          if (batch_idx > self.num_batches_val):
            break

          patch_local_scale = batch_val["local_scale"]["patch"].to(self.device)
          label_local_scale = batch_val["local_scale"]["patch_label"].to(self.device)
          
          patch_global_scale = batch_val["global_scale"]["patch"].to(self.device)
          label_global_scale = batch_val["global_scale"]["patch_label"].to(self.device)
          label_global_scale = label_global_scale.squeeze(1)

          local_scale_mean = batch_val["local_scale"]["mean"].to(self.device)
          local_scale_mean = local_scale_mean.unsqueeze(-1).unsqueeze(-1)
          local_scale_std = batch_val["local_scale"]["std"].to(self.device)
          local_scale_std = local_scale_std.unsqueeze(-1).unsqueeze(-1)
          
          global_scale_mean = batch_val["global_scale"]["mean"].to(self.device)
          global_scale_mean = global_scale_mean.unsqueeze(-1).unsqueeze(-1)
          global_scale_std = batch_val["global_scale"]["std"].to(self.device)
          global_scale_std = global_scale_std.unsqueeze(-1).unsqueeze(-1)

          patch_local_scale = (patch_local_scale - local_scale_mean) / local_scale_std
          patch_global_scale = (patch_global_scale - global_scale_mean) / global_scale_std

          prediction = self.model(
            x_local_scale=patch_local_scale, 
            x_global_scale=patch_global_scale
          )
          prediction = prediction.squeeze(-1).squeeze(-1)

          # loss = self.loss_fn_val(
          #   prediction=prediction, label=label_global_scale, model=self.model
          # )
          loss = torch.nn.functional.cross_entropy(prediction, label_global_scale)

          self.running_val_loss += loss.item() * self.batch_size

          # accumulating the number of correct predictions to divide them by
          # the total number of preds later to get the accuracy
          self.running_val_acc += self.get_num_correct_preds(
            prediction, label_global_scale
          )

          self.pbar.update(task_id=self.pbar_val, advance=1)
          self.pbar.update(
            task_id=self.pbar_epochs, 
            advance=( 1/(self.num_batches_tot_train + 2) )
          )

      if self.running_val_loss < self.best_val_loss:
        self.delta_val_loss = self.running_val_loss - self.best_val_loss
        self.best_val_loss = self.running_val_loss
        self.best_epoch_val_loss = epoch
        trigger_export_by_loss = True
      
      self.running_val_acc /= self.batch_size * self.num_batches_val

      if self.best_val_acc < self.running_val_acc:
        self.best_val_acc = self.running_val_acc
        self.best_epoch_val_acc = epoch
        trigger_export_by_acc = True

      self.pbar.update_table(
        running_train_loss=self.running_train_loss,
        running_val_loss=self.running_val_loss,
        best_train_loss=self.best_train_loss,
        best_val_loss=self.best_val_loss,
        delta_train_loss=self.delta_train_loss,
        delta_val_loss=self.delta_val_loss,
        best_epoch_train_loss=self.best_epoch_train_loss,
        best_epoch_val_loss=self.best_epoch_val_loss,
        running_train_acc=self.running_train_acc,
        running_val_acc=self.running_val_acc,
        best_train_acc=self.best_train_acc,
        best_val_acc=self.best_val_acc,
        best_epoch_train_acc=self.best_epoch_train_acc,
        best_epoch_val_acc=self.best_epoch_val_acc
      )

      if self.wandb_helper is not None:
        self.wandb_helper.log(
          epoch=epoch, 
          running_loss_train = self.running_train_loss, 
          running_loss_val=self.running_val_loss,
          running_train_acc=self.running_train_acc,
          running_val_acc=self.running_val_acc,
          learning_rate=self.learning_rate_scheduler.get_last_lr()[0]
        )

      self._handle_checkpoint(
        current_epoch=epoch, 
        trigger_export_by_acc=trigger_export_by_acc,
        trigger_export_by_loss=trigger_export_by_loss
      )

      

    self.pbar.update(task_id=self.pbar_epochs, completed=self.last_epoch + 1)

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

        local_scale_mean = batch_test["local_scale"]["mean"].to(self.device)
        local_scale_mean = local_scale_mean.unsqueeze(-1).unsqueeze(-1)
        local_scale_std = batch_test["local_scale"]["std"].to(self.device)
        local_scale_std = local_scale_std.unsqueeze(-1).unsqueeze(-1)
        
        global_scale_mean = batch_test["global_scale"]["mean"].to(self.device)
        global_scale_mean = global_scale_mean.unsqueeze(-1).unsqueeze(-1)
        global_scale_std = batch_test["global_scale"]["std"].to(self.device)
        global_scale_std = global_scale_std.unsqueeze(-1).unsqueeze(-1)

        patch_local_scale = (patch_local_scale - local_scale_mean) / local_scale_std
        patch_global_scale = (patch_global_scale - global_scale_mean) / global_scale_std

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
      TimeRemainingColumn(),
      train_color = PBAR_TRAIN_COLOR, 
      val_color = PBAR_VAL_COLOR
      
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