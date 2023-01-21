import torch

import argparse

from BRATS2013DatasetPatch import BRATS2013DatasetPatch

def parse_cli_args():
  
  arg_parser = argparse.ArgumentParser()

  arg_parser.add_argument(
    "-i", "--dataset-df-path", action="store", dest="dataset_df_path",
    type=str, required=True,
    help="Path of DataFrame storing the input-output tuples"
  )

  parsed_args = arg_parser.parse_args()

  return parsed_args


def main():
  parsed_args = parse_cli_args()


if __name__ == "__main__":
  main()