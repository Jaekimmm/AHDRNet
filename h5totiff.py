import os
import h5py
import numpy as np
import imageio
from utils import *

def h5_to_tif(h5_path, tif_path, apply_tonemap=True, gamma=2.24):
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
        img = np.array(h5_file['GT'])  # Load GT image

    img = img.astype(np.float32)

    # ✅ Ensure it's in (H, W, C) format
    if img.shape[0] in [1, 3]:  
        img = np.transpose(img, (2, 1, 0))

    # ✅ Convert to RGB format (ignore alpha channel if exists)
    if img.shape[-1] == 4:  
        img = img[:, :, :3]

    # ✅ Apply Tonemapping if enabled
    if apply_tonemap:
        img = img ** gamma  # Apply gamma correction
        norm_perc = np.percentile(img, 99)  # Normalize
        img = tanh_norm_mu_tonemap(img, norm_perc)
        #img = img / norm_perc  

    # ✅ Save as .tif
    imageio.imwrite(tif_path, img, format='tiff')
    print(f"[INFO] Converted {h5_path} → {tif_path}")


def h5_to_tif_batch(directory, apply_tonemap=True, gamma=2.24):
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
        if apply_tonemap:
            tif_path = os.path.splitext(h5_path)[0] + "_target.tif"
        else:
            tif_path = os.path.splitext(h5_path)[0] + "_target_wotm.tif"  # Same path, change extension to .tif
        h5_to_tif(h5_path, tif_path, apply_tonemap, gamma)

    print(f"[INFO] All .h5 files in {directory} have been converted to .tif")


# ✅ 실행 예시
h5_to_tif_batch("./GenerH5Data/Result/Test", apply_tonemap=True)
