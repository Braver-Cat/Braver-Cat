
import argparse
import pyjson5
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

def get_conf():
  
  arg_parser = argparse.ArgumentParser()

  arg_parser.add_argument(
    "--conf-file", required=True, help="Path to the configuration file"
  )

  return pyjson5.load(open(arg_parser.parse_args().conf_file))

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

def get_checkpoint_callback(conf):

  checkpoint_path = os.path.join(conf["checkpoint"]["path"], conf["run_id"])
  make_dir_if_absent(checkpoint_path)

  return ModelCheckpoint(
    dirpath=checkpoint_path,
    filename="ckp_{epoch:03d}",
    save_last=True, save_top_k=conf["checkpoint"]["top_k"],
    monitor="loss/val",
    mode="min"
  )


def __main__():

  conf = get_conf()

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
    callbacks=[get_progress_bar(), get_checkpoint_callback(conf)],
    log_every_n_steps=1

  )

  trainer.fit(model=input_cascade_CNN, datamodule=datamodule)

if __name__ == "__main__":
  __main__()