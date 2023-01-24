import argparse

from BRATS2013DatasetPatch import BRATS2013DatasetPatch

from tqdm import tqdm

from torch.utils.data import DataLoader

from TwoPathCNN import TwoPathCNN
from InputCascadeCNN import InputCascadeCNN

from InputCascadeCNNModelTrainer import InputCascadeCNNModelTrainer

import torch.optim 

from WandBHelper import WandBHelper

from datetime import datetime

import os

DEFAULT_NUM_INPUT_CHANNELS = 4
DEFAULT_NUM_CLASSES = 6

WARNING_COLOR ='\033[91m'
METHOD_COLOR = '\033[94m'
BOLD = '\033[1m'

GPU_NAME = "GeForce"

WANDB_PROJECT_NAME = "Braver-Cat-End-to-End"
WANDB_ENTITY_NAME = "Braver-Cat"

def parse_cli_args():
  
  arg_parser = argparse.ArgumentParser()

  arg_parser.add_argument(
    "-i", "--dataset-df-path", action="store", dest="dataset_df_path",
    type=str, required=True,
    help="Path of DataFrame storing the input-output tuples"
  )
  arg_parser.add_argument(
    "-s", "--patch-size", action="store", dest="patch_size",
    type=str, required=True,
    help="Size of the patch to center around the pixel that must be classified for the segmentation"
  )
  arg_parser.add_argument(
    "-l", "--load-data-in-memory", action="store_true", dest="load_data_in_memory",
    help="Whether to load the entire dataset in memory, rather than loading batch elements when needed in the forward pass"
  )
  arg_parser.add_argument(
    "-b", "--batch-size", action="store", dest="batch_size",
    type=int, required=True,
    help="Batch size"
  )
  arg_parser.add_argument(
    "-d", "--deterministic", action="store_true", dest="deterministic",
    help="Whether to use a deterministic behaviour, wherever possible"
  )
  arg_parser.add_argument(
    "--cascade-type", action="store", dest="cascade_type", type=str, 
    required=True, choices=["input", "local", "mfc"], 
    help="The kind of local and global concatenation to use.\nSee paper for more information https://arxiv.org/pdf/1505.03540.pdf"
  )
  arg_parser.add_argument(
    "--num-input-channels", action="store", dest="num_input_channels",
    type=int, required=False, default=DEFAULT_NUM_INPUT_CHANNELS,
    help="Number of channels in the input image.\nDefaults to 4, which corresponds to the four modalities of MRI."
  )
  arg_parser.add_argument(
    "--num-classes", action="store", dest="num_classes",
    type=int, required=False, default=DEFAULT_NUM_CLASSES,
    help="Number of channels in the input image.\nDefaults to 6, which corresponds to the number of classes considered in the original paper."
  )
  arg_parser.add_argument(
    "--optimizer", action="store", dest="optimizer_name", type=str, 
    choices=["SGD"], default="SGD", required=False,
    help="What optimizer to use during training.\nDefaults to SGD, as per paper setup\nSee paper for more information https://arxiv.org/pdf/1505.03540.pdf"
  )
  arg_parser.add_argument(
    "--momentum", action="store", dest="momentum", type=float, 
    default="0.7", required=False, 
    help="What momentum to use during training\nDefaults to 0.7, as per paper setup"
  )
  arg_parser.add_argument(
    "--learning-rate", action="store", dest="learning_rate", type=float, 
    default=0.005, required=False,
    help="What learning rate to use during training.\nDefaults to 0.005, as per paper setup\nPlease note that learning rate will be updated by the scheduler, if one is selected."
  )
  arg_parser.add_argument(
    "--learning-rate-scheduler-decay-factor", action="store", 
    dest="learning_rate_decay_factor", type=float, default=0.1, required=False, 
    help="What learning rate decay factor to use during training via the learning rate scheduler.\nDefaults to 0.1, as per paper setup"
  )
  arg_parser.add_argument(
    "--learning-rate-scheduler-decay-step-size", action="store", 
    dest="learning_rate_decay_step_size", type=int, default=1, required=False, 
    help="What learning rate decay factor to use during training via the learning rate scheduler.\nDefaults to 1, as per paper setup"
  )
  arg_parser.add_argument(
    "--num-batches", action="store", dest="num_batches",
    type=int, required=False, default=-1,
    help="Sets the number of batches to train on.\nUseful in debug.\nDefaults to -1, which ignores the limit."
  )
  arg_parser.add_argument(
    "--elastic-net-delta-1", action="store", 
    dest="delta_1", type=float, required=True, 
    help="Delta_1 value of elastic-net regularization"
  )
  arg_parser.add_argument(
    "--elastic-net-delta-2", action="store", 
    dest="delta_2", type=float, required=True, 
    help="Delta_2 value of elastic-net regularization"
  )
  arg_parser.add_argument(
    "--dropout", action="store", dest="dropout", type=float, 
    default=0.2, required=False, 
    help="Dropout value to use in training\nDefaults to 0.2\nPass 0.0 to avoid using Dropout, as per PyTorch implementation."
  )
  arg_parser.add_argument(
    "--num-epochs", action="store", dest="num_epochs", type=int, required=True, 
    help="Number of epochs to train for."
  )
  arg_parser.add_argument(
    "--job-type", action="store", dest="job_type", required=True, 
    choices=["train-e2e", "train-tl", "hyperparams-tuning-e2e", "hyperparams-tuning-tl"], 
    help="The kind of job that will be launched.\n" + 
      "train-e2e: train the selected model end-to-end, according to given configuration\n" + 
      "train-tl: train the selected model using transfer learning, starting from a pre-trained model specified in PRE_TRAINED_PATH_PLACEHOLDER\n" + 
      "hyperparams-tuning-e2e: perform hyperparameter tuning on a model trained end-to-end" + 
      "hyperparams-tuning-tl: perform hyperparameter tuning on a model trained via transfer learning"
  )
  arg_parser.add_argument(
    "--checkpoint-path", action="store", dest="checkpoint_base_path", type=str, 
    required=True, help="Base path to store model checkpoints in."
  )
  arg_parser.add_argument(
    "--checkpoint-step", action="store", dest="checkpoint_step", type=int, 
    default=-1, required=False, help="How often to store a checkpoint.\nDefaults to -1, which amounts to no checkpoint saved.\nModel after final epoch is always saved."
  )
  

  parsed_args = arg_parser.parse_args()

  return parsed_args

def check_patch_size_consistency(patch_size, dataset_df_path):

  if patch_size not in dataset_df_path:
    print(
      f"Patch size {patch_size} not present in patch dataframe path: {dataset_df_path}"
    )

    return False
  
  return True

def get_datasets(parsed_args):
  dataset_train_path = f"{parsed_args.dataset_df_path}/train_labels_df_one_hot.json"
  dataset_val_path = f"{parsed_args.dataset_df_path}/val_labels_df_one_hot.json"
  dataset_test_path = f"{parsed_args.dataset_df_path}/test_labels_df_one_hot.json"

  dataset_train = BRATS2013DatasetPatch(
    patch_df_path=dataset_train_path, patch_size=parsed_args.patch_size, 
    load_in_memory=parsed_args.load_data_in_memory, stage="train"
  )
  dataset_val = BRATS2013DatasetPatch(
    patch_df_path=dataset_val_path, patch_size=parsed_args.patch_size, 
    load_in_memory=parsed_args.load_data_in_memory, stage="val"
  )
  dataset_test = BRATS2013DatasetPatch(
    patch_df_path=dataset_test_path, patch_size=parsed_args.patch_size, 
    load_in_memory=parsed_args.load_data_in_memory, stage="test"
  )

  return dataset_train, dataset_val, dataset_test

def get_dataloaders(dataset_train, dataset_val, dataset_test, parsed_args):
  dl_train = DataLoader(
    dataset=dataset_train, batch_size=parsed_args.batch_size, 
    shuffle=True, num_workers=16
  )
  dl_val = DataLoader(
    dataset=dataset_val, batch_size=parsed_args.batch_size, 
    shuffle=True, num_workers=16
  )
  dl_test = DataLoader(
    dataset=dataset_test, batch_size=parsed_args.batch_size, 
    shuffle=True, num_workers=16
  )

  return dl_train, dl_val, dl_test

def get_model(cascade_type, num_input_channels, num_classes, dropout):

  if cascade_type == "input":
    return InputCascadeCNN(
      num_input_channels=num_input_channels, num_classes=num_classes, 
      dropout=dropout
    )
  
def get_device():
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  current_device = torch.cuda.current_device()
  current_device_name = torch.cuda.get_device_name(current_device)
  

  if GPU_NAME not in current_device_name:
    print(f"{METHOD_COLOR}{BOLD}main: {BOLD}" + f"{WARNING_COLOR}[WARNING] Unable to use GPU as PyTorch device!")


  return device
  
def get_model_trainer(
    device, model, num_epochs, optimizer, learning_rate_scheduler, batch_size, 
    num_batches, dl_train, dl_val, dl_test, delta_1, delta_2,
    checkpoint_full_path, checkpoint_step
  ):

  if model.cascade_type == "input":
    
    return InputCascadeCNNModelTrainer(
      device=device,
      model=model,
      num_epochs=num_epochs,
      optimizer=optimizer, 
      learning_rate_scheduler=learning_rate_scheduler,
      batch_size=batch_size,
      num_batches=num_batches,
      dl_train=dl_train, 
      dl_val=dl_val, 
      dl_test=dl_test,
      delta_1=delta_1,
      delta_2=delta_2,
      checkpoint_full_path=checkpoint_full_path,
      checkpoint_step=checkpoint_step,
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


def main():
  parsed_args = parse_cli_args()
  other_args = dict()

  patch_size_consistency_result = check_patch_size_consistency(
    patch_size=parsed_args.patch_size, 
    dataset_df_path=parsed_args.dataset_df_path
  )

  if not patch_size_consistency_result:
    exit(patch_size_consistency_result)

  dataset_train, dataset_val, dataset_test = get_datasets(parsed_args)

  dl_train, dl_val, dl_test = get_dataloaders(
    dataset_train, dataset_val, dataset_test, parsed_args
  )

  model = get_model(
    cascade_type=parsed_args.cascade_type, 
    num_input_channels=parsed_args.num_input_channels, 
    num_classes=parsed_args.num_classes,
    dropout=parsed_args.dropout
  )

  optimizer, learning_rate_scheduler = get_optimizer(
    model=model, 
    optimizer_name=parsed_args.optimizer_name,
    learning_rate=parsed_args.learning_rate,
    learning_rate_decay_factor=parsed_args.learning_rate_decay_factor,
    learning_rate_decay_step_size=parsed_args.learning_rate_decay_step_size,
    momentum=parsed_args.momentum, 
  )


  other_args["train_id"] = get_train_id()
  
  checkpoint_full_path = get_checkpoint_full_path(
    base_path=parsed_args.checkpoint_base_path, train_id=other_args["train_id"]
  )
  
  device = get_device()

  model_trainer = get_model_trainer(
    device=device,
    model=model,
    num_epochs=parsed_args.num_epochs,
    optimizer=optimizer, learning_rate_scheduler=learning_rate_scheduler,
    batch_size=parsed_args.batch_size, 
    num_batches=parsed_args.num_batches,
    dl_train=dl_train, dl_val=dl_val, dl_test=dl_test,
    delta_1=parsed_args.delta_1, delta_2=parsed_args.delta_2,
    checkpoint_full_path=checkpoint_full_path,
    checkpoint_step=parsed_args.checkpoint_step,
  )

  wandb_helper = WandBHelper(
    project=WANDB_PROJECT_NAME, entity=WANDB_ENTITY_NAME,
    parsed_args=parsed_args, other_args=other_args
  )

  wandb_helper.init_run()

  model_trainer.train()

  wandb_helper.update_config(
    config_update={
      "best_epoch_val_loss": model_trainer.best_epoch_val_loss,
      "best_epoch_val_acc": model_trainer.best_epoch_val_acc,
      "best_val_acc": model_trainer.best_val_acc,
      "best_val_loss": model_trainer.best_val_loss,
    }
  )

  
  
  
  
  
  
  
  
  
  
  
  
  wandb_helper.run.finish()












  # model = TwoPathCNN(num_input_channels=4, num_classes=6)
  # x = torch.randint(10, 99, (16, 4, 33, 33)).float()

  # model = InputCascadeCNN(num_input_channels=4, num_classes=6)
  
  # x_small_scale = torch.randint(10, 99, (16, 4, 33, 33)).float()
  # x_large_scale = torch.randint(10, 99, (16, 4, int(parsed_args.patch_size), int(parsed_args.patch_size))).float()
  # model(x_small_scale=x_small_scale, x_large_scale=x_large_scale)

  
  
  
  
  # for x in tqdm(iter(dl_train), total=len(dl_train)):
  #   print(x["patch"].shape)
  #   print(x["patch_label"].shape)
  #   print(x["patch_label"])
  #   return






if __name__ == "__main__":
  main()