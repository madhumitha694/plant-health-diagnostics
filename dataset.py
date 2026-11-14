"""Dataset + DataLoader setup.

Expected folder layout (one sub-folder per class; folder name = class label):

    data/
      healthy/              *.jpg
      nitrogen_deficiency/  *.jpg
      phosphorus_deficiency/*.jpg
      potassium_deficiency/ *.jpg
      dead_necrotic/        *.jpg

Any additional folder (e.g. magnesium_deficiency/) is picked up automatically.
"""
import random
from collections import defaultdict

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from config import IMG_SIZE, MEAN, STD


def get_transforms(train: bool) -> transforms.Compose:
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),          # leaves have no canonical "up"
            transforms.RandomRotation(15),
            # Keep hue shifts tiny: yellowing/purpling IS the diagnostic signal,
            # so aggressive hue jitter would corrupt the labels.
            transforms.ColorJitter(brightness=0.25, contrast=0.25,
                                   saturation=0.2, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def stratified_split(targets, val_fraction, seed):
    """Per-class split so every class appears in both train and val."""
    by_class = defaultdict(list)
    for idx, t in enumerate(targets):
        by_class[t].append(idx)
    rng = random.Random(seed)
    train_idx, val_idx = [], []
    for idxs in by_class.values():
        rng.shuffle(idxs)
        n_val = max(1, int(len(idxs) * val_fraction))
        val_idx += idxs[:n_val]
        train_idx += idxs[n_val:]
    return train_idx, val_idx


def build_dataloaders(data_dir, batch_size=32, val_fraction=0.2,
                      num_workers=2, seed=42):
    # Two ImageFolder views of the same files: one with train aug, one without.
    train_full = datasets.ImageFolder(data_dir, transform=get_transforms(True))
    val_full = datasets.ImageFolder(data_dir, transform=get_transforms(False))

    train_idx, val_idx = stratified_split(train_full.targets, val_fraction, seed)
    train_ds, val_ds = Subset(train_full, train_idx), Subset(val_full, val_idx)

    # Inverse-frequency class weights to counter imbalance
    # (dead plants / rare deficiencies are usually under-represented).
    counts = torch.zeros(len(train_full.classes))
    for i in train_idx:
        counts[train_full.targets[i]] += 1
    class_weights = counts.sum() / (len(counts) * counts.clamp(min=1))

    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin)
    return train_loader, val_loader, train_full.classes, class_weights
