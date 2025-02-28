import os
import h5py
import numpy as np
import imageio
from utils import *

def h5_to_tif(h5_path, tif_path):
    """
    Convert an RGB image stored under the 'GT' key in an .h5 file to a .tif image.
    
    Args:
        h5_path (str): Path to the input .h5 file.
        tif_path (str): Path to save the output .tif file.
        apply_tonemap (bool): Whether to apply tonemapping (default: True).
        gamma (float): Gamma correction value (default: 2.24).
    """
    with h5py.File(h5_path, 'r') as h5_file:
        if 'GT' not in h5_file:
            print(f"[WARNING] Skipping {h5_path}: 'GT' key not found.")
            return
        img = np.array(h5_file['IN'])  # Load GT image

    img = img.astype(np.float32)
    print(f"img.shape {img.shape}")
   
    # ✅ separate images
    for i in range(6):
        img_sep = img[i*3:(i+1)*3, :, :] 
        img_sep = np.transpose(img_sep, (2, 1, 0))
        print(f"img_sep.shape {img_sep.shape}")

        # ✅ Save as .tif
        tif_path = os.path.splitext(h5_path)[0] + f"_in{i}.tif"  # Same path, change extension to .tif
        imageio.imwrite(tif_path, img_sep, format='tiff')
        print(f"[INFO] Converted {h5_path} → {tif_path}")


def h5_to_tif_batch(directory):
    """
    Convert all .h5 files in a directory to .tif format.
    
    Args:
        directory (str): Path to the directory containing .h5 files.
        apply_tonemap (bool): Apply tonemapping (default: True).
        gamma (float): Gamma correction value (default: 2.24).
    """
    if not os.path.isdir(directory):
        print(f"[ERROR] Directory not found: {directory}")
        return

    h5_files = [f for f in os.listdir(directory) if f.endswith(".h5")]

    if not h5_files:
        print(f"[WARNING] No .h5 files found in {directory}")
        return

    print(f"[INFO] Found {len(h5_files)} .h5 files in {directory}")

    for h5_file in h5_files:
        h5_path = os.path.join(directory, h5_file)
        tif_path = os.path.splitext(h5_path)[0]
        h5_to_tif(h5_path, tif_path)

    print(f"[INFO] All .h5 files in {directory} have been converted to .tif")


# ✅ 실행 예시
#h5_to_tif_batch("./GenerH5Data/Result/Test")
h5_path = "./dataset_sice/test/13.h5"
tif_path = os.path.splitext(h5_path)[0]
h5_to_tif(h5_path, tif_path)
