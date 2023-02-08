import argparse

import sys
sys.path.append("../src")
from dataset.CustomDatasetLocalGlobalScalePatch import BRATS2013DatasetLocalGlobalScalePatch

import torch

from tqdm import tqdm

from torch.utils.data import DataLoader

import os

from rich import print

import json

WARNING_COLOR = "#DAA520"
METHOD_COLOR = "#191970"

def parse_cli_args():
  
  arg_parser = argparse.ArgumentParser()

  arg_parser.add_argument(
    "--local-scale-dataset-df-path", action="store", dest="dataset_local_scale_df_path",
    type=str, required=True,
    help="Path of DataFrame storing the input-output tuples for the local scale patches"
  )
  arg_parser.add_argument(
    "--global-scale-dataset-df-path", action="store", dest="dataset_global_scale_df_path",
    type=str, required=True,
    help="Path of DataFrame storing the input-output tuples for the global scale patches"
  )
  arg_parser.add_argument(
    "--local-scale-patch-size", action="store", dest="patch_size_local_scale",
    type=str, required=True,
    help="Size of the patch to center around the pixel that must be classified for the segmentation in the local scale"
  )
  arg_parser.add_argument(
    "--global-scale-patch-size", action="store", dest="patch_size_global_scale",
    type=str, required=True,
    help="Size of the patch to center around the pixel that must be classified for the segmentation in the global scale"
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

  parsed_args = arg_parser.parse_args()

  return parsed_args

def get_datasets(parsed_args):

  global_scale_train_path = f"{parsed_args.dataset_global_scale_df_path}/train_labels_df_one_hot.json"
  global_scale_val_path = f"{parsed_args.dataset_global_scale_df_path}/val_labels_df_one_hot.json"
  global_scale_test_path = f"{parsed_args.dataset_global_scale_df_path}/test_labels_df_one_hot.json"

  local_scale_train_path = f"{parsed_args.dataset_local_scale_df_path}/train_labels_df_one_hot.json"
  local_scale_val_path = f"{parsed_args.dataset_local_scale_df_path}/val_labels_df_one_hot.json"
  local_scale_test_path = f"{parsed_args.dataset_local_scale_df_path}/test_labels_df_one_hot.json"

  dataset_train = BRATS2013DatasetLocalGlobalScalePatch(
    local_scale_df_path=local_scale_train_path,
    local_scale_patch_size=parsed_args.patch_size_local_scale,
    local_scale_load_data_in_memory=parsed_args.load_data_in_memory,

    global_scale_df_path=global_scale_train_path,
    global_scale_patch_size=parsed_args.patch_size_global_scale,
    global_scale_load_data_in_memory=parsed_args.load_data_in_memory,

    stage="train"

  )
  
  dataset_val = BRATS2013DatasetLocalGlobalScalePatch(
    local_scale_df_path=local_scale_val_path,
    local_scale_patch_size=parsed_args.patch_size_local_scale,
    local_scale_load_data_in_memory=parsed_args.load_data_in_memory,
  
    global_scale_df_path=global_scale_val_path,
    global_scale_patch_size=parsed_args.patch_size_global_scale,
    global_scale_load_data_in_memory=parsed_args.load_data_in_memory,

    stage="val"

  )
  
  dataset_test = BRATS2013DatasetLocalGlobalScalePatch(
    local_scale_df_path=local_scale_test_path,
    local_scale_patch_size=parsed_args.patch_size_local_scale,
    local_scale_load_data_in_memory=parsed_args.load_data_in_memory,

    global_scale_df_path=global_scale_test_path,
    global_scale_patch_size=parsed_args.patch_size_global_scale,
    global_scale_load_data_in_memory=parsed_args.load_data_in_memory,

    stage="test"

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
  
def make_dir_if_absent(dir):
  if not os.path.exists(dir):
      os.makedirs(dir)

def get_mean_std(loader, scale):
  # var[X] = E[X**2] - E[X]**2
  channels_sum, channels_sqrd_sum, num_batches = 0, 0, 0

  for local_global_scale in tqdm(loader):

      data = local_global_scale[scale]["patch"]

      channels_sum += torch.mean(data, dim=[0, 2, 3])
      channels_sqrd_sum += torch.mean(data**2, dim=[0, 2, 3])
      num_batches += 1

  mean = channels_sum / num_batches
  std = (channels_sqrd_sum / num_batches - mean**2) ** 0.5

  return mean, std

def save_mean_std(save_path, mean, std):
  
  json_dict = {
    "mean": mean.cpu().numpy().tolist(),
    "std": std.cpu().numpy().tolist(),
  }

  json.dump(json_dict, open(save_path, 'w'))

def main():
  parsed_args = parse_cli_args()
  other_args = dict()

  dataset_train, dataset_val, dataset_test = get_datasets(parsed_args) 

  dl_train, dl_val, dl_test = get_dataloaders(
    dataset_train, dataset_val, dataset_test, parsed_args
  )

  local_scale_mean, local_scale_std = get_mean_std(
    loader=dl_train, scale="local_scale"
  )
  
  global_scale_mean, global_scale_std = get_mean_std(
    loader=dl_train, scale="global_scale"
  )

  LOCAL_SCALE_MEAN_STD_EXPORT_PATH = \
    f"{parsed_args.dataset_local_scale_df_path}/mean_std.json"
  
  save_mean_std(
    save_path=LOCAL_SCALE_MEAN_STD_EXPORT_PATH, 
    mean=local_scale_mean, std=local_scale_std
  )
  
  GLOBAL_SCALE_MEAN_STD_EXPORT_PATH = \
    f"{parsed_args.dataset_global_scale_df_path}/mean_std.json"
  
  save_mean_std(
    save_path=GLOBAL_SCALE_MEAN_STD_EXPORT_PATH, 
    mean=global_scale_mean, std=global_scale_std
  )









if __name__ == "__main__":
  main()