# src/preprocessing/normalize.py
import numpy as np


def normalize_pointcloud(points):
    """
    Normalisation standard d'un nuage de points :
    1. Recentrage sur le barycentre
    2. Mise à l'échelle dans une sphère unitaire (max dist = 1)

    Args:
        points : (N, 3) float32
    Returns:
        points_norm : (N, 3) normalisé
        centroid    : (3,) barycentre original
        max_dist    : float facteur d'échelle
    """
    centroid = points.mean(axis=0)
    points   = points - centroid
    max_dist = np.max(np.linalg.norm(points, axis=1))
    points   = (points / max_dist).astype(np.float32)
    return points, centroid, max_dist


def random_sample(points, n_points=2048):
    """
    Sous-échantillonnage aléatoire uniforme à n_points fixes.
    """
    n       = len(points)
    replace = n < n_points
    idx     = np.random.choice(n, n_points, replace=replace)
    return points[idx]


def augment_pointcloud(points):
    """
    Augmentations pour l'entraînement uniquement :
    - Rotation aléatoire autour de Z
    - Jitter gaussien (sigma=0.005)
    - Mise à l'échelle aléatoire ±10%
    """
    theta = np.random.uniform(0, 2 * np.pi)
    c, s  = np.cos(theta), np.sin(theta)
    R     = np.array([[c, -s, 0],
                      [s,  c, 0],
                      [0,  0, 1]], dtype=np.float32)
    points = points @ R.T
    points = points + np.random.normal(0, 0.005, points.shape).astype(np.float32)
    points = (points * np.random.uniform(0.9, 1.1)).astype(np.float32)
    return points
