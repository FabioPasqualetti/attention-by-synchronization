"""Process-level cache of the KWS dataset to avoid re-indexing the 5.5 GB
SPEECHCOMMANDS cache on every run.

cached_loaders(seed) is numerically identical to
kws_dataset.get_loaders(batch_size, smoke_test=False, seed): same
train/val Subsets, same per-feature mean/std, same seeded train shuffle generator.
Only the expensive indexing + stats pass is amortized across runs in one process.
"""
from . import paths
paths.ensure_paths()

import torch
from torch.utils.data import DataLoader

kws_ds = paths.load_kws_dataset()  # training/kws_dataset.py, loaded under a unique name

_CACHE = {}


def cached_loaders(seed, batch_size=64):
    if "base" not in _CACHE:
        train_ds = kws_ds._load_split("training")
        val_ds = kws_ds._load_split("validation")
        raw = DataLoader(train_ds, batch_size=batch_size, shuffle=False,
                         collate_fn=kws_ds._make_collate(), num_workers=0)
        mean, std = kws_ds._compute_stats(raw)
        _CACHE["base"] = (train_ds, val_ds, mean, std)
    train_ds, val_ds, mean, std = _CACHE["base"]

    g_train = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=kws_ds._make_collate(mean, std),
        generator=g_train, num_workers=0)
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=kws_ds._make_collate(mean, std), num_workers=0)
    return train_loader, val_loader, mean, std
