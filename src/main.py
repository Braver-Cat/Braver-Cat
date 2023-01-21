import torch

import argparse

from BRATS2013DatasetPatch import BRATS2013DatasetPatch

from tqdm import tqdm

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

  parsed_args = arg_parser.parse_args()

  return parsed_args

def check_patch_size_consistency(patch_size, dataset_df_path):

  if patch_size not in dataset_df_path:
    print(
      f"Patch size {patch_size} not present in patch dataframe path: {dataset_df_path}"
    )

    return False
  
  return True


def main():
  parsed_args = parse_cli_args()

  patch_size_consistency_result = check_patch_size_consistency(
    patch_size=parsed_args.patch_size, 
    dataset_df_path=parsed_args.dataset_df_path
  )

  if not patch_size_consistency_result:
    exit(patch_size_consistency_result)

  dataset_train_path = f"{parsed_args.dataset_df_path}/train_labels_df.json"
  dataset_val_path = f"{parsed_args.dataset_df_path}/val_labels_df.json"
  dataset_test_path = f"{parsed_args.dataset_df_path}/test_labels_df.json"

  dataset_train = BRATS2013DatasetPatch(
    patch_df_path=dataset_train_path, patch_size= parsed_args.patch_size, 
    stage="train"
  )

  for x in tqdm(dataset_train):
    pass





if __name__ == "__main__":
  main()