
import argparse
import pyjson5
from warnings import warn
from TwoPathCNN import TwoPathCNN 
from GlobalPathCNN import GlobalPathCNN
from LocalPathCNN import LocalPathCNN
from InputCascadeCNNModule import InputCascadeCNNModule
from GlobalLocalScalePatchesDataModule import GlobalLocalScalePatchesDataModule
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import torch
from pytorch_lightning.callbacks import RichProgressBar, ModelCheckpoint
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme
import os
from rich import print

E2E_WRONG_CKPT_EX_MSG = "Resuming E2E training from checkpoint requires the " \
  "checkpoint to have Optimizer and Scheduler state dicts, but it "\
  "has the following keys: \n"

E2E_WRONG_CKPT_WARN_MSG = "Trying to resume E2E " \
  "starting from a checkpoint that contains \"weights_only\" in " \
  "its name.\n" \
  "Make sure to have selected the right checkpoint for TL from scratch!" 

TL_WRONG_CKPT_WARN_MSG = "Trying to perform Transfer Learning from scratch " \
  "starting from a checkpoint that does not contain \"weights_only\" in " \
  "its name.\n" \
  "Make sure to have selected the right checkpoint for TL from scratch!" 

TL_NO_CKPT_EX_MSG = "Performing Transfer Learning requires a checkpoint, " \
  "but None has been given."

TL_WRONG_CKPT_EX_MSG = "Performing Transfer Learning from scratch starting " \
  "from a checkpoint that stores Optimizer and Scheduler state dicts NOT " \
  "allowed.\nSelect a \"weights_only\" checkpoint"

def get_conf():
  
  arg_parser = argparse.ArgumentParser()

  arg_parser.add_argument(
    "--conf-file", required=True, help="Path to the configuration file"
  )

  return pyjson5.load(open(arg_parser.parse_args().conf_file))

def _validate_ckpt(conf):

  notable_keys = ['optimizer_states', 'lr_schedulers']

  if conf["train_mode"] == "e2e" and conf["e2e"]["from_ckpt"] is not None:
    
    ckpt = torch.load(conf[conf["train_mode"]]["from_ckpt"])
    notable_keys_in_ckpt = [x in ckpt.keys() for x in notable_keys]
    
    if "weights_only" in conf["e2e"]["from_ckpt"]:
      warn(E2E_WRONG_CKPT_WARN_MSG)
    
    if not all(notable_keys_in_ckpt):
      raise Exception(E2E_WRONG_CKPT_EX_MSG + str(ckpt.keys()))

  if conf["train_mode"] == "tl":

    if conf["tl"]["from_ckpt"] == None:
      raise Exception(TL_NO_CKPT_EX_MSG)
    
    ckpt = torch.load(conf[conf["train_mode"]]["from_ckpt"])
    notable_keys_in_ckpt = [x in ckpt.keys() for x in notable_keys]
    
    if conf["tl"]["from_scratch"] == True:

      if "weights_only" not in conf["tl"]["from_ckpt"]:
        warn(TL_WRONG_CKPT_WARN_MSG)

      if any(notable_keys_in_ckpt):
        raise Exception(TL_WRONG_CKPT_EX_MSG)

def validate_conf(conf):
  _validate_ckpt(conf)

  if conf[conf["train_mode"]]["from_ckpt"] is not None:
    ckpt = torch.load(conf[conf["train_mode"]]["from_ckpt"])

    conf["max_epochs"] += ckpt["epoch"]
  
  return conf

def load_mean_std(mean_std_path):
  mean_std_dict = pyjson5.load(open(mean_std_path))

  mean = torch.tensor(mean_std_dict["mean"]).unsqueeze(-1).unsqueeze(-1)
  std = torch.tensor(mean_std_dict["std"]).unsqueeze(-1).unsqueeze(-1)

  return mean, std

def get_progress_bar():
  
  return RichProgressBar(
    theme=RichProgressBarTheme(
      description="#0000CD",
      progress_bar="#8B008B",
      progress_bar_finished="#006400",
      progress_bar_pulse="#6206E0",
      batch_progress="#4169E1",
      time="#9370DB",
      processing_speed="#9370DB",
      metrics="#9370DB",
    )
  )

def make_dir_if_absent(dir):
  if not os.path.exists(dir):
    os.makedirs(dir)

def get_ckpt_callback(conf):

  checkpoint_path = os.path.join(conf["save_ckpt"]["path"], conf["run_id"])
  make_dir_if_absent(checkpoint_path)

  return [ 
    ModelCheckpoint(
      dirpath=checkpoint_path,
      filename="ckp_{epoch:03d}",
      save_last=True, save_top_k=conf["save_ckpt"]["top_k"],
      monitor="loss/val",
      mode="min"
    ),
    
    ModelCheckpoint(
      dirpath=checkpoint_path,
      filename="ckp_{epoch:03d}_weights_only",
      save_last=True, save_top_k=conf["save_ckpt"]["top_k"],
      monitor="loss_val",
      mode="min",
      save_weights_only=True
    ),
  ]


def __main__():

  conf = get_conf()
  conf = validate_conf(conf)
  print(conf["max_epochs"])

  global_scale_CNN = TwoPathCNN(
    global_path_CNN=GlobalPathCNN(
      in_channels=conf["data"]["in_channels"], dropout_p=conf["dropout_p"]
    ),
    local_path_CNN=LocalPathCNN(
      in_channels=conf["data"]["in_channels"], dropout_p=conf["dropout_p"]
    ),
    out_channels=conf["data"]["num_classes"]
  )
  
  local_scale_CNN = TwoPathCNN(
    global_path_CNN=GlobalPathCNN(
      in_channels=conf["data"]["in_channels"]+conf["data"]["num_classes"], 
      dropout_p=conf["dropout_p"]
    ),
    local_path_CNN=LocalPathCNN(
      in_channels=conf["data"]["in_channels"]+conf["data"]["num_classes"], 
      dropout_p=conf["dropout_p"]
    ),
    out_channels=conf["data"]["num_classes"]
  )

  input_cascade_CNN = InputCascadeCNNModule(
    global_scale_CNN=global_scale_CNN,
    local_scale_CNN=local_scale_CNN,
    optim_conf=conf["optim_conf"],
    scheduler_conf=conf["scheduler_conf"],
    num_classes=conf["data"]["num_classes"]
  )

  mean, std = load_mean_std(conf["data"]["mean_std_path"])

  datamodule = GlobalLocalScalePatchesDataModule(
    df_path_dict=conf["data"]["df_path_dict"],
    load_data_in_memory=False,
    batch_size=conf["batch_size"],
    num_workers=conf["num_workers"],
    transforms=transforms.Compose( [transforms.Normalize(mean=mean, std=std)] )
  )

  wandb_logger = WandbLogger(
    offline=conf["offline"],
    project=conf["project_name"]
  )
  conf["run_id"] = wandb_logger.experiment.name

  wandb_logger.log_hyperparams(conf)
  
  trainer = pl.Trainer(
    accelerator="gpu", devices="1",
    logger=wandb_logger,
    max_epochs=conf["max_epochs"],
    precision=conf["precision"],
    limit_train_batches=conf["limit_train_batches"],
    limit_val_batches=conf["limit_val_batches"],
    limit_test_batches=conf["limit_test_batches"],
    callbacks=[get_progress_bar(), *get_ckpt_callback(conf)],
    log_every_n_steps=1,
  )

  trainer.fit(
    model=input_cascade_CNN, datamodule=datamodule,
    ckpt_path=conf[conf["train_mode"]]["from_ckpt"]
  )

if __name__ == "__main__":
  __main__()