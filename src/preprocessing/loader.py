# src/preprocessing/loader.py
import numpy as np
import tifffile

def load_xyz_tiff(path):
    """
    Charge un fichier .tiff MVTec 3D-AD.
    Retourne un tableau (N, 3) de points 3D valides.
    """
    xyz_map = tifffile.imread(str(path))
    mask    = np.any(xyz_map != 0, axis=-1)
    return xyz_map[mask].astype(np.float32)


def load_gt_mask(path):
    """
    Charge un masque ground truth.
    Retourne un tableau binaire (H, W) : 1 = anomalie.
    """
    import matplotlib.pyplot as plt
    mask = plt.imread(str(path))
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    return (mask > 0.5).astype(np.float32)
