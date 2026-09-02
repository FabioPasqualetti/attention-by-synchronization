"""
KWS dataset: Google Speech Commands v0.02, 10-class subset.

Features: log-mel spectrogram, 40 bins, 25 ms window, 20 ms hop, T=49 frames.
Normalization: per-feature mean/std computed from the training split.

Download happens automatically on first use (~2.3 GB compressed).
Default cache: <OSCILLATOR_DATA_CACHE>/speech_commands
"""

import os
from training import paths

import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import DataLoader, Subset

CLASSES_10 = ["yes", "no", "up", "down", "left", "right",
              "on", "off", "stop", "go"]
LABEL_MAP  = {c: i for i, c in enumerate(CLASSES_10)}

CACHE_DIR   = os.path.join(paths.cache_root(), "speech_commands")
SAMPLE_RATE = 16_000
N_MELS      = 40
WIN_LENGTH  = 400   # 25 ms at 16 kHz
HOP_LENGTH  = 320   # 20 ms at 16 kHz → T = floor((16000-400)/320)+1 = 49
N_FRAMES    = 49

_mel = T.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=512,
    win_length=WIN_LENGTH,
    hop_length=HOP_LENGTH,
    n_mels=N_MELS,
)
_db = T.AmplitudeToDB()


def _extract(waveform: torch.Tensor) -> torch.Tensor:
    """(1, samples) → (T=49, n_mels=40) log-mel spectrogram."""
    n = waveform.shape[-1]
    if n < SAMPLE_RATE:
        waveform = F.pad(waveform, (0, SAMPLE_RATE - n))
    else:
        waveform = waveform[..., :SAMPLE_RATE]
    log_mel = _db(_mel(waveform)).squeeze(0).T   # (T_raw, 40)
    T_raw = log_mel.shape[0]
    if T_raw < N_FRAMES:
        log_mel = F.pad(log_mel, (0, 0, 0, N_FRAMES - T_raw))
    return log_mel[:N_FRAMES]                    # (49, 40)


def _collate(batch, mean=None, std=None):
    feats, labels = [], []
    for waveform, _sr, label, *_ in batch:
        if label not in LABEL_MAP:
            continue
        feat = _extract(waveform)
        if mean is not None:
            feat = (feat - mean) / std
        feats.append(feat)
        labels.append(LABEL_MAP[label])
    if not feats:
        return None, None
    return torch.stack(feats), torch.tensor(labels, dtype=torch.long)


def _make_collate(mean=None, std=None):
    def fn(batch):
        return _collate(batch, mean=mean, std=std)
    return fn


def _load_split(subset: str):
    os.makedirs(CACHE_DIR, exist_ok=True)
    ds = torchaudio.datasets.SPEECHCOMMANDS(
        root=CACHE_DIR, download=True, subset=subset)
    indices = [i for i, (_, _, label, *_) in enumerate(ds)
               if label in LABEL_MAP]
    return Subset(ds, indices)


def _compute_stats(loader) -> tuple:
    """Single pass → (mean, std) each shape (40,)."""
    chunks = []
    for feats, _ in loader:
        if feats is None:
            continue
        chunks.append(feats.reshape(-1, feats.shape[-1]))
    all_f = torch.cat(chunks, dim=0)
    return all_f.mean(0), all_f.std(0).clamp(min=1e-5)


def get_loaders(batch_size: int = 64, smoke_test: bool = False,
                seed: int = 0):
    """
    Returns (train_loader, val_loader, mean, std).

    mean, std: (40,) float32 tensors, computed from the training split.
    """
    train_ds = _load_split("training")
    val_ds   = _load_split("validation")

    if smoke_test:
        g = torch.Generator().manual_seed(seed)
        train_ds = Subset(train_ds,
                          torch.randperm(len(train_ds), generator=g)[:200].tolist())
        val_ds   = Subset(val_ds,
                          torch.randperm(len(val_ds), generator=g)[:50].tolist())

    raw = DataLoader(train_ds, batch_size=batch_size, shuffle=False,
                     collate_fn=_make_collate(), num_workers=0)
    mean, std = _compute_stats(raw)

    g_train = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=_make_collate(mean, std),
        generator=g_train, num_workers=0)
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=_make_collate(mean, std), num_workers=0)

    return train_loader, val_loader, mean, std


def get_test_loader(mean: torch.Tensor, std: torch.Tensor,
                    batch_size: int = 64):
    test_ds = _load_split("testing")
    return DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                      collate_fn=_make_collate(mean, std), num_workers=0)
