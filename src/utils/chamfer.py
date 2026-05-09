# src/utils/chamfer.py
import torch


def chamfer_distance(pc1, pc2):
    """
    Chamfer Distance entre deux nuages de points.

    Pour chaque point de pc1, cherche le plus proche dans pc2 et vice-versa.
    CD = moyenne des deux distances minimales.

    Args:
        pc1, pc2 : tenseurs (B, N, 3)
    Returns:
        cd_loss       : scalaire — loss pour l'entraînement
        cd_per_sample : (B,) — score d'anomalie par pièce
    """
    diff          = pc1.unsqueeze(2) - pc2.unsqueeze(1)   # (B, N, M, 3)
    dist          = (diff ** 2).sum(dim=-1)                # (B, N, M)
    dist1         = dist.min(dim=2)[0]                     # (B, N)
    dist2         = dist.min(dim=1)[0]                     # (B, M)
    cd_per_sample = dist1.mean(dim=1) + dist2.mean(dim=1)  # (B,)
    return cd_per_sample.mean(), cd_per_sample
