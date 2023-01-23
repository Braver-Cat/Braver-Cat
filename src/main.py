import argparse

from BRATS2013DatasetPatch import BRATS2013DatasetPatch

from tqdm import tqdm

from torch.utils.data import DataLoader

from TwoPathCNN import TwoPathCNN
from InputCascadeCNN import InputCascadeCNN

import torch

DEFAULT_NUM_INPUT_CHANNELS = 4
DEFAULT_NUM_CLASSES = 6

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
    help="The kind of local and global concatenation to use.\nPossible values: \"input\", \"local\", \"mfc\"\nSee paper for more information https://arxiv.org/pdf/1505.03540.pdf"
  )
  arg_parser.add_argument(
    "--num-input-channels", action="store", dest="num_input_channels",
    type=int, required=False, default=DEFAULT_NUM_INPUT_CHANNELS,
    help="Number of channels in the input image.\nDefaults to 4, which corresponds to the four modalities of MRI."
  )
  arg_parser.add_argument(
    "--num-classes", action="store", dest="num_classes",
    type=int, required=False, default=DEFAULT_NUM_CLASSES,
    help="Number of channels in the input image.\nDefaults to 4, which corresponds to the four modalities of MRI."
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

def get_model(cascade_type, num_input_channels, num_classes):

  if cascade_type == "input":
    return InputCascadeCNN(
      num_input_channels=num_input_channels, 
      num_classes=num_classes
    )



def main():
  parsed_args = parse_cli_args()

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
    num_classes=parsed_args.num_classes
  )







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