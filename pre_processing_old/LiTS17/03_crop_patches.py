import pandas as pd

import numpy as np

def crop_center(img,cropx,cropy):
  y, x = img.shape
  startx = x//2-(cropx//2)
  starty = y//2-(cropy//2)    
  return img[starty:starty+cropy,startx:startx+cropx]

SPLIT_ID = 0

IS_BALANCED = "unbalanced"

DF_PATCHES_65_BASE_PATH = f"../../data/LiTS17/LITS17_patches_65_{IS_BALANCED}/{SPLIT_ID}"

CROP_SIZE = 33

for stage in ["train", "val", "test"]:
  
  df_full_path = f"{DF_PATCHES_65_BASE_PATH}/{stage}_labels_df_one_hot.json"

  df = pd.read_json(path_or_buf=df_full_path)

  df_cropped_rows = []

  for row_idx, row in df.iterrows():

    patch_65 = np.load(row["patch_65_x_65_img_path"] + ".npy")

    patch_cropped = crop_center(patch_65, CROP_SIZE, CROP_SIZE)

    patch_cropped_export_path = row["patch_65_x_65_img_path"].replace(
      "_patches_65/", f"_patches_{str(CROP_SIZE)}/"
    )

    np.save(file=patch_cropped_export_path, arr=patch_cropped)

    df_cropped_rows.append(
      {
        f"patch_{CROP_SIZE}_x_{CROP_SIZE}_img_path": patch_cropped_export_path,
        "patch_label_one_hot": row["patch_label_one_hot"]
      }
    )

  df_cropped = pd.DataFrame(df_cropped_rows)

  df_cropped_export_path = f"{DF_PATCHES_65_BASE_PATH.replace('_patches_65_', f'_patches_{str(CROP_SIZE)}_')}" \
    f"/{stage}_labels_df_one_hot.json"
  
  df_cropped.to_json(path_or_buf=df_cropped_export_path)










