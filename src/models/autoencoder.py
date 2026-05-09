# src/models/autoencoder.py
import torch
import torch.nn as nn


class PointNetEncoder(nn.Module):
    """
    Encoder PointNet : MLP partagés point par point + MaxPool global.
    Produit un vecteur latent invariant à l'ordre des points.
    """
    def __init__(self, latent_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv1d(3,   64,  1), nn.BatchNorm1d(64),  nn.ReLU(),
            nn.Conv1d(64,  128, 1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 256, 1), nn.BatchNorm1d(256), nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(256, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU()
        )

    def forward(self, x):
        # x : (B, N, 3)
        x = x.transpose(2, 1)   # (B, 3, N)
        x = self.mlp(x)          # (B, 256, N)
        x = x.max(dim=2)[0]      # (B, 256) MaxPool global
        return self.fc(x)        # (B, latent_dim)


class PointNetDecoder(nn.Module):
    """Decoder MLP : vecteur latent → nuage de points reconstruit."""
    def __init__(self, latent_dim=128, n_points=2048):
        super().__init__()
        self.n_points = n_points
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, 256),  nn.ReLU(),
            nn.Linear(256,        512),  nn.ReLU(),
            nn.Linear(512,        1024), nn.ReLU(),
            nn.Linear(1024, n_points * 3)
        )

    def forward(self, z):
        return self.mlp(z).view(-1, self.n_points, 3)


class PointNetAutoEncoder(nn.Module):
    """
    Autoencodeur complet pour la détection d'anomalies 3D.
    Score d'anomalie = Chamfer Distance(input, reconstruction).
    """
    def __init__(self, latent_dim=128, n_points=2048):
        super().__init__()
        self.encoder = PointNetEncoder(latent_dim)
        self.decoder = PointNetDecoder(latent_dim, n_points)

    def forward(self, x):
        z     = self.encoder(x)
        recon = self.decoder(z)
        return recon, z
