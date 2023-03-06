import os
import SimpleITK as sitk
import numpy as np
from tqdm import tqdm

DATASET_NAME = "BRATS2013"
DATASET_PATH = f"../data/{DATASET_NAME}/Image_Data/HG"

obs_id_list = os.listdir(DATASET_PATH)

obs_id_pbar = tqdm(obs_id_list, colour="#9400D3")
slice_pbar = tqdm(range(216), colour="#008080", leave=False)

for obs_id in obs_id_list:
  
  modalities_og = []
  label_og = []
  
  obs_full_path = f"{DATASET_PATH}/{obs_id}"

  obs_modalities_list = os.listdir(obs_full_path)

  for obs_modality in obs_modalities_list:

    img_path = f"{obs_full_path}/{obs_modality}"

    img_name = list(
      filter(lambda name: "_N4ITK.mha" in name, os.listdir(img_path))
    )[0]
    
    img = sitk.ReadImage(f"{img_path}/{img_name}")
    img_arr = sitk.GetArrayFromImage(img)

    if (".XX.XX.OT" not in obs_modality): 
      modalities_og.append(img_arr)
    
    else: 
      label_og.append(img_arr)

  modalities_np = np.asarray(modalities_og)
  modalities_np_reshaped = np.transpose(modalities_np, (2, 0, 1, 3))

  label_np = np.asarray(label_og)
  label_np_reshaped = np.transpose(label_np, (2, 0, 1, 3))

  slice_pbar.reset()

  for slice in range(modalities_np_reshaped.shape[0]):
    
    slice_img = modalities_np_reshaped[slice, ...]
    label_img = label_np_reshaped[slice, ...]

    slice_img = np.pad(
      slice_img, 
      ((0,0), ((240-slice_img.shape[1]),0),(230-slice_img.shape[2],0)), 
      'constant'
    )

    if int(np.sum(slice_img)) == 0:
      continue
    
    label_img = np.pad(
      label_img, 
      ((0,0), ((240-label_img.shape[1]),0),(230-label_img.shape[2],0)), 
      'constant'
    )

    export_base_path = "/".join(obs_full_path.split("/")[:3])
    export_base_path = export_base_path.replace(
      f"{DATASET_NAME}", f"{DATASET_NAME}_unstacked_padded_slices"
    )

    export_dir_name = f"{obs_id}_{str(slice).zfill(3)}"
    
    export_full_path = f"{export_base_path}/{export_dir_name}"
    
    if not os.path.exists(export_full_path):
      os.makedirs(export_full_path)

    np.save(file=f"{export_full_path}/img.npy", arr=slice_img)
    np.save(file=f"{export_full_path}/label.npy", arr=label_img)

    slice_pbar.update(1)

  obs_id_pbar.update(1)