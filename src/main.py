import argparse

from BRATS2013DatasetPatch import BRATS2013DatasetPatch

from tqdm import tqdm

from torch.utils.data import DataLoader

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
  dataset_train_path = f"{parsed_args.dataset_df_path}/train_labels_df.json"
  dataset_val_path = f"{parsed_args.dataset_df_path}/val_labels_df.json"
  dataset_test_path = f"{parsed_args.dataset_df_path}/test_labels_df.json"

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






if __name__ == "__main__":
  main()