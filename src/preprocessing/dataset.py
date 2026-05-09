# src/preprocessing/dataset.py
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from .loader import load_xyz_tiff
from .normalize import normalize_pointcloud, random_sample, augment_pointcloud


def collect_samples(cat_path, split):
    """Collecte tous les chemins de fichiers pour un split donné."""
    samples    = []
    split_path = Path(cat_path) / split
    if not split_path.exists():
        return samples
    for defect_dir in sorted(split_path.iterdir()):
        if not defect_dir.is_dir():
            continue
        is_good = (defect_dir.name == 'good')
        xyz_dir = defect_dir / 'xyz'
        gt_dir  = defect_dir / 'gt'
        if not xyz_dir.exists():
            continue
        for xyz_file in sorted(xyz_dir.glob('*.tiff')):
            gt_file = None
            if not is_good and gt_dir.exists():
                cands = list(gt_dir.glob(f'{xyz_file.stem}*.png'))
                if cands:
                    gt_file = cands[0]
            samples.append({
                'xyz':         xyz_file,
                'label':       0 if is_good else 1,
                'defect_type': defect_dir.name,
                'gt':          gt_file
            })
    return samples


class BagelDataset(Dataset):
    """
    Dataset PyTorch pour MVTec 3D-AD.
    Pipeline : chargement → normalisation → sous-échantillonnage → augmentation.
    """
    def __init__(self, samples, n_points=2048, training=False, augment=False):
        self.samples  = samples
        self.n_points = n_points
        self.training = training
        self.augment  = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s          = self.samples[idx]
        pts, _, _  = normalize_pointcloud(load_xyz_tiff(s['xyz']))
        pts        = random_sample(pts, self.n_points)
        if self.augment and self.training:
            pts = augment_pointcloud(pts)
        return (
            torch.from_numpy(pts).float(),
            torch.tensor(s['label']).long(),
            s['defect_type']
        )


def build_dataloaders(cat_path, n_points=2048, batch_size=16):
    """Construit les 3 DataLoaders train/val/test."""
    train_s = collect_samples(cat_path, 'train')
    val_s   = collect_samples(cat_path, 'validation')
    test_s  = collect_samples(cat_path, 'test')

    train_loader = DataLoader(
        BagelDataset(train_s, n_points, training=True, augment=True),
        batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True
    )
    val_loader = DataLoader(
        BagelDataset(val_s, n_points, training=False),
        batch_size=batch_size, shuffle=False, num_workers=2
    )
    test_loader = DataLoader(
        BagelDataset(test_s, n_points, training=False),
        batch_size=batch_size, shuffle=False, num_workers=2
    )
    return train_loader, val_loader, test_loader, train_s, val_s, test_s
