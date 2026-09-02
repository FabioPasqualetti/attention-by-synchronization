"""
SVA dataset: synthetic Linzen-style subject-verb agreement.

Sentence templates:
  Simple:      the [NOUN] [VERB] .
  With PP:     the [NOUN] [PREP] the [NOUN2] [VERB] .

Label 1 = verb agrees with subject; 0 = disagrees.
50 % label-1, 50 % label-0.
PP sentences: 50 % distractor agrees with subject (easy),
              50 % distractor disagrees (hard).

Splits (fixed seeds):
  train 40 000 (seed 0) · val 4 000 (seed 1) · test 4 000 (seed 2)

Public interface mirroring the research-code dataset_sva.py + loaders:
  VOCAB            list[str]
  WORD2IDX         dict[str, int]
  encode(tokens)   → list[int]
  make_sva_loaders(batch_size, n_train=-1, n_val=-1)
                   → (train_loader, val_loader)
"""

import json
import os
import random

import torch
from torch.utils.data import DataLoader, Dataset

# ── Wordlists ──────────────────────────────────────────────────────────────────

NOUNS_SG = [
    "key", "book", "cat", "dog", "child", "box", "plant", "lamp",
    "table", "chair", "map", "cup", "bag", "ball", "coin", "door",
    "bird", "fish", "tree", "car", "star", "ring", "shoe", "hat",
    "pen", "rock", "leaf", "wall", "flag", "seed", "duck", "fox",
    "frog", "mouse", "horse", "goat", "wolf", "bear", "deer", "owl",
    "ant", "bee", "fly", "crab", "boat", "ship", "tank", "van",
    "bus", "jet",
]
NOUNS_PL = [
    "keys", "books", "cats", "dogs", "children", "boxes", "plants", "lamps",
    "tables", "chairs", "maps", "cups", "bags", "balls", "coins", "doors",
    "birds", "fish", "trees", "cars", "stars", "rings", "shoes", "hats",
    "pens", "rocks", "leaves", "walls", "flags", "seeds", "ducks", "foxes",
    "frogs", "mice", "horses", "goats", "wolves", "bears", "deer", "owls",
    "ants", "bees", "flies", "crabs", "boats", "ships", "tanks", "vans",
    "buses", "jets",
]
VERBS_SG = [
    "is", "has", "was", "sits", "runs", "falls", "works", "stands",
    "flies", "swims", "waits", "plays", "stays", "eats", "sleeps",
    "moves", "looks", "grows", "lives", "walks", "rests", "talks",
    "stops", "drops", "rolls", "spins", "hops", "digs", "pulls", "bites",
]
VERBS_PL = [
    "are", "have", "were", "sit", "run", "fall", "work", "stand",
    "fly", "swim", "wait", "play", "stay", "eat", "sleep",
    "move", "look", "grow", "live", "walk", "rest", "talk",
    "stop", "drop", "roll", "spin", "hop", "dig", "pull", "bite",
]
PREPS = ["on", "under", "near", "beside", "above", "with", "by"]

assert len(NOUNS_SG) == 50 and len(NOUNS_PL) == 50
assert len(VERBS_SG) == 30 and len(VERBS_PL) == 30

# ── Vocabulary ─────────────────────────────────────────────────────────────────

PAD_TOKEN = "<PAD>"   # idx 0
BOS_TOKEN = "<BOS>"   # idx 1
EOS_TOKEN = "<EOS>"   # idx 2
UNK_TOKEN = "<UNK>"   # idx 3
SPECIAL   = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN]

_all_words = (["the"] + NOUNS_SG + NOUNS_PL + VERBS_SG + VERBS_PL
              + PREPS + ["."])
VOCAB    = SPECIAL + sorted(set(_all_words))
WORD2IDX = {w: i for i, w in enumerate(VOCAB)}


def encode(tokens: list) -> list:
    return [WORD2IDX.get(t, WORD2IDX[UNK_TOKEN]) for t in tokens]


# ── Sentence generator ─────────────────────────────────────────────────────────

def _make_sentence(rng: random.Random) -> dict:
    has_pp   = rng.random() < 0.60
    subj_sg  = rng.random() < 0.50
    subj_noun = rng.choice(NOUNS_SG if subj_sg else NOUNS_PL)
    correct  = rng.random() < 0.50
    label    = 1 if correct else 0

    if subj_sg:
        verb = rng.choice(VERBS_SG) if correct else rng.choice(VERBS_PL)
    else:
        verb = rng.choice(VERBS_PL) if correct else rng.choice(VERBS_SG)

    if not has_pp:
        tokens           = ["the", subj_noun, verb, "."]
        subject_idx      = 1
        verb_idx         = 2
        distractor_idx   = None
        distractor_agrees = None
    else:
        prep = rng.choice(PREPS)
        dist_agrees = rng.random() < 0.50
        dist_noun   = rng.choice(NOUNS_SG if subj_sg else NOUNS_PL) \
            if dist_agrees else rng.choice(NOUNS_PL if subj_sg else NOUNS_SG)
        tokens           = ["the", subj_noun, prep, "the", dist_noun, verb, "."]
        subject_idx      = 1
        verb_idx         = 5
        distractor_idx   = 4
        distractor_agrees = dist_agrees

    return {
        "tokens":            tokens,
        "label":             label,
        "subject_idx":       subject_idx,
        "verb_idx":          verb_idx,
        "distractor_idx":    distractor_idx,
        "has_distractor":    has_pp,
        "distractor_agrees": distractor_agrees,
    }


def _generate(n: int, seed: int) -> list:
    rng = random.Random(seed)
    return [_make_sentence(rng) for _ in range(n)]


# ── Caching (jsonl) ────────────────────────────────────────────────────────────

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "sva")


def _save_jsonl(records: list, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _load_jsonl(path: str) -> list:
    with open(path) as f:
        return [json.loads(line) for line in f]


def build_datasets(force: bool = False) -> dict:
    """Load or generate train/val/test splits (fixed seeds)."""
    paths = {
        "train": os.path.join(_DATA_DIR, "sva_train.jsonl"),
        "val":   os.path.join(_DATA_DIR, "sva_val.jsonl"),
        "test":  os.path.join(_DATA_DIR, "sva_test.jsonl"),
    }
    if not force and all(os.path.exists(p) for p in paths.values()):
        return {k: _load_jsonl(v) for k, v in paths.items()}

    train = _generate(40_000, seed=0)
    val   = _generate(4_000,  seed=1)
    test  = _generate(4_000,  seed=2)
    _save_jsonl(train, paths["train"])
    _save_jsonl(val,   paths["val"])
    _save_jsonl(test,  paths["test"])
    return {"train": train, "val": val, "test": test}


# ── PyTorch Dataset + collate ──────────────────────────────────────────────────

class SVADataset(Dataset):
    def __init__(self, records: list):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        ids      = encode([BOS_TOKEN] + r["tokens"] + [EOS_TOKEN])
        verb_pos = r["verb_idx"] + 1   # +1 for BOS prepend
        return {
            "ids":               ids,
            "label":             r["label"],
            "verb_pos":          verb_pos,
            "has_distractor":    bool(r["has_distractor"]),
            "distractor_agrees": bool(r["distractor_agrees"])
                                 if r["distractor_agrees"] is not None else False,
        }


def _collate(batch: list) -> dict:
    max_len  = max(len(b["ids"]) for b in batch)
    tokens   = torch.zeros(len(batch), max_len, dtype=torch.long)
    pad_mask = torch.ones(len(batch), max_len, dtype=torch.bool)
    labels   = torch.zeros(len(batch), dtype=torch.long)
    verb_idx = torch.zeros(len(batch), dtype=torch.long)
    has_dist, dist_agrees = [], []

    for i, b in enumerate(batch):
        L = len(b["ids"])
        tokens[i, :L]   = torch.tensor(b["ids"], dtype=torch.long)
        pad_mask[i, :L] = False
        labels[i]        = b["label"]
        verb_idx[i]      = b["verb_pos"]
        has_dist.append(b["has_distractor"])
        dist_agrees.append(b["distractor_agrees"])

    return {
        "tokens":            tokens,
        "labels":            labels,
        "pad_mask":          pad_mask,
        "verb_idx":          verb_idx,
        "has_distractor":    has_dist,
        "distractor_agrees": dist_agrees,
    }


def make_sva_loaders(batch_size: int, n_train: int = -1,
                     n_val: int = -1):
    """Returns (train_loader, val_loader)."""
    splits    = build_datasets()
    train_rec = splits["train"] if n_train < 0 else splits["train"][:n_train]
    val_rec   = splits["val"]   if n_val   < 0 else splits["val"][:n_val]
    train_loader = DataLoader(SVADataset(train_rec), batch_size=batch_size,
                              shuffle=True,  collate_fn=_collate)
    val_loader   = DataLoader(SVADataset(val_rec),   batch_size=batch_size,
                              shuffle=False, collate_fn=_collate)
    return train_loader, val_loader


if __name__ == "__main__":
    splits = build_datasets(force=True)
    for name, recs in splits.items():
        n    = len(recs)
        hard = sum(r["has_distractor"] and r["distractor_agrees"] is False
                   for r in recs)
        print(f"SVA {name}: {n} samples | "
              f"hard={hard/n*100:.1f}% | vocab={len(VOCAB)}")
