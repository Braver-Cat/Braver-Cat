import argparse

import sys
sys.path.append("../")

from dataset.CustomDatasetLocalGlobalScalePatch import CustomDatasetLocalGlobalScalePatch


from torch.utils.data import DataLoader


from model.TwoPathCNN import TwoPathCNN
from model.InputCascadeCNN import InputCascadeCNN

from model.InputCascadeCNNModelTrainer import InputCascadeCNNModelTrainer

import torch.optim 

from utils.WandBHelper import WandBHelper

from datetime import datetime

import os

from rich import print

import json
from jsmin import jsmin

import numpy as np

DEFAULT_NUM_INPUT_CHANNELS = 4
DEFAULT_NUM_CLASSES = 6

WARNING_COLOR = "#DAA520"
METHOD_COLOR = "#191970"

GPU_NAME = "GeForce"

wandb_project_name = "Braver-Cat"
WANDB_ENTITY_NAME = "Braver-Cat"

TRANSFER_LEARNING_STRING = "_tl"
END_TO_END_LEARNING_STRING = "_e2e"

def parse_cli_args():
  
  arg_parser = argparse.ArgumentParser()

  arg_parser.add_argument(
    "--config-file", action="store", type=str, required=True,
    help="Path to the configuration file"
  )

  parsed_args = arg_parser.parse_args()

  with open(parsed_args.config_file) as js_file:
    minified = jsmin(js_file.read())
  
  parsed_args  = json.loads(minified)

  if parsed_args["checkpoint_to_load_path"] is not None \
    and not parsed_args["resume_from_checkpoint_statistics"] \
    and END_TO_END_LEARNING_STRING in parsed_args["job_type"]:
    
    print(
      f"[bold {METHOD_COLOR}]main: [/bold {METHOD_COLOR}]" + 
      f"[{WARNING_COLOR}]Selected to load a checkpoint without resuming " \
      f"from its statistics![/{WARNING_COLOR}]"
    )

  if TRANSFER_LEARNING_STRING in parsed_args["job_type"] and \
    parsed_args["checkpoint_to_load_path"] is None:

    raise AttributeError(
      "Transfer Learning mode requires a checkpoint to be specified, "
      "so as weights can be loaded from it"
    )
  
  parsed_args["num_input_channels"] = 4 if "BRATS2013" in parsed_args["dataset_local_scale_df_path"] else 1

  return parsed_args
  

def load_mean_std(mean_std_json_path):
  f = open(mean_std_json_path)

  mean_std_json = json.load(f)

  return torch.tensor(mean_std_json["mean"]), torch.tensor(mean_std_json["std"])


def get_datasets(parsed_args):

  global_scale_train_path = f"{parsed_args['dataset_global_scale_df_path']}/train_labels_df_one_hot.json"
  global_scale_val_path = f"{parsed_args['dataset_global_scale_df_path']}/val_labels_df_one_hot.json"
  global_scale_test_path = f"{parsed_args['dataset_global_scale_df_path']}/test_labels_df_one_hot.json"

  local_scale_train_path = f"{parsed_args['dataset_local_scale_df_path']}/train_labels_df_one_hot.json"
  local_scale_val_path = f"{parsed_args['dataset_local_scale_df_path']}/val_labels_df_one_hot.json"
  local_scale_test_path = f"{parsed_args['dataset_local_scale_df_path']}/test_labels_df_one_hot.json"

  local_scale_mean = torch.tensor(0)
  local_scale_std = torch.tensor(1)
  
  global_scale_mean = torch.tensor(0)
  global_scale_std = torch.tensor(1)

  if parsed_args["standardize"]:

    local_scale_mean, local_scale_std = load_mean_std(
      parsed_args["local_scale_mean_std_path"]
    )
    
    global_scale_mean, global_scale_std = load_mean_std(
      parsed_args["global_scale_mean_std_path"]
    )
    
  dataset_train = CustomDatasetLocalGlobalScalePatch(
    local_scale_df_path=local_scale_train_path,
    local_scale_patch_size=parsed_args["patch_size_local_scale"],
    local_scale_load_data_in_memory=parsed_args["load_data_in_memory"],
    local_scale_mean=local_scale_mean, local_scale_std=local_scale_std,

    global_scale_df_path=global_scale_train_path,
    global_scale_patch_size=parsed_args["patch_size_global_scale"],
    global_scale_load_data_in_memory=parsed_args["load_data_in_memory"],
    global_scale_mean=global_scale_mean, global_scale_std=global_scale_std,


    stage="train"

  )
  
  dataset_val = CustomDatasetLocalGlobalScalePatch(
    local_scale_df_path=local_scale_val_path,
    local_scale_patch_size=parsed_args["patch_size_local_scale"],
    local_scale_load_data_in_memory=parsed_args["load_data_in_memory"],
    local_scale_mean=local_scale_mean, local_scale_std=local_scale_std,

    global_scale_df_path=global_scale_val_path,
    global_scale_patch_size=parsed_args["patch_size_global_scale"],
    global_scale_load_data_in_memory=parsed_args["load_data_in_memory"],
    global_scale_mean=global_scale_mean, global_scale_std=global_scale_std,

    stage="val"

  )
  
  dataset_test = CustomDatasetLocalGlobalScalePatch(
    local_scale_df_path=local_scale_test_path,
    local_scale_patch_size=parsed_args["patch_size_local_scale"],
    local_scale_load_data_in_memory=parsed_args["load_data_in_memory"],
    local_scale_mean=local_scale_mean, local_scale_std=local_scale_std,

    global_scale_df_path=global_scale_test_path,
    global_scale_patch_size=parsed_args["patch_size_global_scale"],
    global_scale_load_data_in_memory=parsed_args["load_data_in_memory"],
    global_scale_mean=global_scale_mean, global_scale_std=global_scale_std,

    stage="test"

  )
  
  return dataset_train, dataset_val, dataset_test 

def get_dataloaders(dataset_train, dataset_val, dataset_test, parsed_args):
  dl_train = DataLoader(
    dataset=dataset_train, batch_size=parsed_args["batch_size"], 
    shuffle=True, num_workers=16
  )
  dl_val = DataLoader(
    dataset=dataset_val, batch_size=parsed_args["batch_size"], 
    shuffle=True, num_workers=16
  )
  dl_test = DataLoader(
    dataset=dataset_test, batch_size=parsed_args["batch_size"], 
    shuffle=True, num_workers=16
  )

  return dl_train, dl_val, dl_test

def get_model(cascade_type, num_input_channels, num_classes, dropout, device):

  if cascade_type == "input":
    return InputCascadeCNN(
      num_input_channels=num_input_channels, num_classes=num_classes, 
      dropout=dropout
    ).to(device)
  
def get_device():
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  current_device = torch.cuda.current_device()
  current_device_name = torch.cuda.get_device_name(current_device)
  

  if GPU_NAME not in current_device_name:
    print(
      f"[bold {METHOD_COLOR}] main: [{WARNING_COLOR}] [WARNING] Unable to use GPU as PyTorch device!"
    )


  return device
  
def get_model_trainer(
    device, model, num_epochs, optimizer, learning_rate_scheduler, 
    batch_size, num_batches_train, num_batches_val, num_batches_test,
    dl_train, dl_val, dl_test, 
    delta_1, delta_2,
    checkpoint_full_path, checkpoint_step, train_id, resumed_from_checkpoint,
    starting_epoch, wandb_helper,
    best_epoch_train_acc,
    best_epoch_train_loss,
    best_train_acc,
    best_train_loss,
    delta_train_loss,
    best_val_acc,
    best_val_loss,
    delta_val_loss,
    best_epoch_val_acc,
    best_epoch_val_loss,
    disable_dashboard, terminal_theme,
    layers_to_switch, layers_to_turn_off, layers_to_copy, 
    layers_to_copy_copy_mode
  ):

  if model.cascade_type == "input":
    
    return InputCascadeCNNModelTrainer(
      device=device,
      model=model,
      num_epochs=num_epochs,
      optimizer=optimizer, 
      learning_rate_scheduler=learning_rate_scheduler,
      batch_size=batch_size,
      num_batches_train=num_batches_train,
      num_batches_val=num_batches_val,
      num_batches_test=num_batches_test,
      dl_train=dl_train, 
      dl_val=dl_val, 
      dl_test=dl_test,
      delta_1=delta_1,
      delta_2=delta_2,
      checkpoint_full_path=checkpoint_full_path,
      checkpoint_step=checkpoint_step,
      train_id=train_id,
      resumed_from_checkpoint=resumed_from_checkpoint,
      starting_epoch=starting_epoch,
      wandb_helper=wandb_helper,
      best_epoch_train_acc=best_epoch_train_acc,
      best_epoch_train_loss=best_epoch_train_loss,
      best_train_acc=best_train_acc,
      best_train_loss=best_train_loss,
      delta_train_loss=delta_train_loss,
      best_val_acc=best_val_acc,
      best_val_loss=best_val_loss,
      delta_val_loss=delta_val_loss,
      best_epoch_val_acc=best_epoch_val_acc,
      best_epoch_val_loss=best_epoch_val_loss,
      disable_dashboard=disable_dashboard, terminal_theme=terminal_theme,
      layers_to_switch=layers_to_switch,
      layers_to_turn_off=layers_to_turn_off,
      layers_to_copy=layers_to_copy,
      layers_to_copy_copy_mode=layers_to_copy_copy_mode
    )
  
def get_optimizer(
  model, optimizer_name, learning_rate, learning_rate_decay_factor, 
  learning_rate_decay_step_size, momentum
):
  
  if optimizer_name == "SGD":

    optimizer = torch.optim.SGD(
      params=model.parameters(), lr=learning_rate, momentum=momentum
    )

    learning_rate_scheduler = torch.optim.lr_scheduler.StepLR(
      optimizer=optimizer, step_size=learning_rate_decay_step_size,
      gamma=learning_rate_decay_factor
    )

    return optimizer, learning_rate_scheduler
  
def get_train_id():
  return datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

def make_dir_if_absent(dir):
  if not os.path.exists(dir):
      os.makedirs(dir)

def get_checkpoint_full_path(base_path, train_id):
  checkpoint_full_path = f"{base_path}/{train_id}"

  make_dir_if_absent(checkpoint_full_path)
  
  return checkpoint_full_path

def load_checkpoint_from_disk(checkpoint_to_load_path, parsed_args):
  
  checkpoint_from_disk = None 
  resumed_from_checkpoint = None 
  checkpoint_epoch = 0

  job_type = parsed_args["job_type"]
  
  if checkpoint_to_load_path is not None:
    
    checkpoint_from_disk = torch.load(checkpoint_to_load_path)

    resumed_from_checkpoint = checkpoint_to_load_path

    if END_TO_END_LEARNING_STRING in job_type or \
      (
        TRANSFER_LEARNING_STRING in job_type and parsed_args[
          job_type
        ]["resume_optim_lr_scheduler_from_checkpoint"]
      ):

      checkpoint_epoch = checkpoint_from_disk["checkpoint_epoch"]

  return checkpoint_from_disk, resumed_from_checkpoint, checkpoint_epoch

def populate_state_dicts(
  checkpoint_from_disk, resumed_from_checkpoint, model, optimizer, 
  learning_rate_scheduler, parsed_args
):
  
  # look into config file for more information about the various if-else clauses
    
  if checkpoint_from_disk is not None:

    print(
      f"[bold {METHOD_COLOR}] main: [/bold {METHOD_COLOR}]"
      f"[#00000] loading checkpoint {resumed_from_checkpoint}"
    )

    model.load_state_dict(state_dict=checkpoint_from_disk["model_state_dict"])

    job_type = parsed_args["job_type"]
    
    if END_TO_END_LEARNING_STRING in job_type or \
      (
        TRANSFER_LEARNING_STRING in job_type and \
        parsed_args[job_type]["resume_optim_lr_scheduler_from_checkpoint"]
      ):
    
      optimizer.load_state_dict(
        state_dict=checkpoint_from_disk["optimizer_state_dict"]
      )

      learning_rate_scheduler.load_state_dict(
        state_dict=checkpoint_from_disk["learning_rate_scheduler_state_dict"]
      )

  return model, optimizer, learning_rate_scheduler

def populate_statistics_dict(checkpoint_from_disk, parsed_args):
  
  statistics_dict = {

    "best_epoch_train_acc": -1,
    "best_epoch_train_loss": -1,

    "best_train_acc": 0,
    "best_train_loss": np.inf,
    "delta_train_loss": 0,
    
    "best_val_acc": 0,
    "best_val_loss": np.inf,
    "delta_val_loss": 0,

    "best_epoch_val_acc": -1,
    "best_epoch_val_loss": -1

  }

  if checkpoint_from_disk is not None and \
    parsed_args["resume_from_checkpoint_statistics"]:
    
    statistics_dict = {

      "best_epoch_train_acc": checkpoint_from_disk["best_epoch_train_acc"],
      "best_epoch_train_loss": checkpoint_from_disk["best_epoch_train_loss"],

      "best_train_acc": checkpoint_from_disk["best_train_acc"],
      "best_train_loss": checkpoint_from_disk["best_train_loss"],
      "delta_train_loss": checkpoint_from_disk["delta_train_loss"],
      
      "best_val_acc": checkpoint_from_disk["best_val_acc"],
      "best_val_loss": checkpoint_from_disk["best_val_loss"],
      "delta_val_loss": checkpoint_from_disk["delta_val_loss"],

      "best_epoch_val_acc": checkpoint_from_disk["best_epoch_val_acc"],
      "best_epoch_val_loss": checkpoint_from_disk["best_epoch_val_loss"]

    }

  return statistics_dict

def handle_transfer_learning(parsed_args, model: InputCascadeCNN):

  job_type = parsed_args["job_type"]

  if TRANSFER_LEARNING_STRING in job_type:

    if parsed_args[job_type]["local_scale_freeze_first_layers"]:
      model.local_scale_CNN.freeze_first_layers()
    
    if parsed_args[job_type]["local_scale_freeze_last_layer"]:
      model.local_scale_CNN.freeze_last_layer()
    
    if parsed_args[job_type]["global_scale_freeze_first_layers"]:
      model.global_scale_CNN.freeze_first_layers()
    
    if parsed_args[job_type]["global_scale_freeze_last_layer"]:
      model.global_scale_CNN.freeze_last_layer()

  return model

def main():
  global wandb_project_name

  parsed_args = parse_cli_args()
  other_args = dict()

  dataset_train, dataset_val, dataset_test = get_datasets(parsed_args) 

  dl_train, dl_val, dl_test = get_dataloaders(
    dataset_train, dataset_val, dataset_test, parsed_args
  )

  if parsed_args["num_batches_train"] == None:
    parsed_args["num_batches_train"] = len(dl_train)
  
  if parsed_args["num_batches_val"] == None:
    parsed_args["num_batches_val"] = len(dl_val)
  
  if parsed_args["num_batches_test"] == None:
    parsed_args["num_batches_test"] = len(dl_test)

  ( 
    checkpoint_from_disk, 
    other_args["resumed_from_checkpoint"], 
    other_args["starting_epoch"]
  ) = load_checkpoint_from_disk(
    parsed_args["checkpoint_to_load_path"], parsed_args
  )

  device = get_device()

  model = get_model(
    cascade_type=parsed_args["cascade_type"], 
    # always init with the default num of in channels (4)
    # modify num in channels in handle_transfer_learning method, if needed
    num_input_channels=DEFAULT_NUM_INPUT_CHANNELS, 
    num_classes=parsed_args["num_classes"],
    dropout=parsed_args["dropout"], 
    device=device
  )

  optimizer, learning_rate_scheduler = get_optimizer(
    model=model, 
    optimizer_name=parsed_args["optimizer_name"],
    learning_rate=parsed_args["learning_rate"],
    learning_rate_decay_factor=parsed_args["learning_rate_decay_factor"],
    learning_rate_decay_step_size=parsed_args["learning_rate_decay_step_size"],
    momentum=parsed_args["momentum"], 
  )

  model, optimizer, learning_rate_scheduler = populate_state_dicts(
    checkpoint_from_disk, other_args["resumed_from_checkpoint"], model, 
    optimizer, learning_rate_scheduler, parsed_args
  )

  if parsed_args["num_input_channels"] != DEFAULT_NUM_INPUT_CHANNELS:
    model.change_num_in_channels(
      new_in_channels=parsed_args["num_input_channels"], 
      num_classes=DEFAULT_NUM_CLASSES
    )

  model = handle_transfer_learning(parsed_args=parsed_args, model=model)

  other_args["train_id"] = get_train_id()
  
  checkpoint_full_path = get_checkpoint_full_path(
    base_path=parsed_args["checkpoint_base_path"], train_id=other_args["train_id"]
  )

  if parsed_args["disable_wandb"]:
    wandb_helper = None
  else:

    wandb_project_name += "-Transfer-Learning" if TRANSFER_LEARNING_STRING in parsed_args["job_type"] else "-End-to-End"
    
    wandb_helper = WandBHelper(
      project=wandb_project_name, entity=WANDB_ENTITY_NAME,
      parsed_args=parsed_args, other_args=other_args, model=model
    )

  statistics_dict = populate_statistics_dict(
    checkpoint_from_disk=checkpoint_from_disk, parsed_args=parsed_args
  )

  model_trainer = get_model_trainer(
    device=device,
    model=model,
    num_epochs=parsed_args["num_epochs"],
    optimizer=optimizer, learning_rate_scheduler=learning_rate_scheduler,
    batch_size=parsed_args["batch_size"], 
    num_batches_train=parsed_args["num_batches_train"],
    num_batches_val=parsed_args["num_batches_val"],
    num_batches_test=parsed_args["num_batches_test"],
    dl_train=dl_train, dl_val=dl_val, dl_test=dl_test,
    delta_1=parsed_args["delta_1"], delta_2=parsed_args["delta_2"],
    checkpoint_full_path=checkpoint_full_path,
    checkpoint_step=parsed_args["checkpoint_step"],
    train_id=other_args["train_id"],
    resumed_from_checkpoint=other_args["resumed_from_checkpoint"],
    starting_epoch=other_args["starting_epoch"],
    wandb_helper=wandb_helper,
    best_epoch_train_acc = statistics_dict["best_epoch_train_acc"], 
    best_epoch_train_loss = statistics_dict["best_epoch_train_loss"],
    best_train_acc = statistics_dict["best_train_acc"],
    best_train_loss = statistics_dict["best_train_loss"],
    delta_train_loss = statistics_dict["delta_train_loss"],
    best_val_acc = statistics_dict["best_val_acc"],
    best_val_loss = statistics_dict["best_val_loss"],
    delta_val_loss = statistics_dict["delta_val_loss"],
    best_epoch_val_acc = statistics_dict["best_epoch_val_acc"],
    best_epoch_val_loss = statistics_dict["best_epoch_val_loss"],
    disable_dashboard=parsed_args["disable_dashboard"],
    terminal_theme=parsed_args["terminal_theme"],
    layers_to_switch = parsed_args["layers_to_switch"], 
    layers_to_turn_off = parsed_args["layers_to_turn_off"], 
    layers_to_copy = parsed_args["layers_to_copy"], 
    layers_to_copy_copy_mode = parsed_args["layers_to_copy_copy_mode"], 
  )

  if not parsed_args["disable_wandb"]:  
    wandb_helper.init_run()

    wandb_helper.watch()

  model_trainer.train()


  if not parsed_args["disable_wandb"]:
    wandb_helper.update_config(
      config_update=model_trainer.get_wandb_config_update()
    )



    wandb_helper.run.finish()






if __name__ == "__main__":
  main()