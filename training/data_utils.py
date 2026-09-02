"""
Data utilities for WikiText-2 and TinyStories language-model experiments.

WikiText-2 preprocessing:
  vocab: top-10K tokens, max_seq_len=50, chunked with BOS/EOS bookends.

TinyStories preprocessing matches dataset_ts.py:
  vocab: 8004 tokens, max_seq_len=128.

Both functions check a local cache first, then download via HuggingFace.
"""

import os
from training import paths  # noqa: E402
import pickle
import re
import string
from collections import Counter
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader

# ── Constants ──────────────────────────────────────────────────────────────────

PAD_IDX = 0
BOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3


# ── WikiText-2 ─────────────────────────────────────────────────────────────────

def _wt2_tokenize(text: str):
    text = text.lower()
    text = re.sub(r"([.,!?;:()\[\]{}\"'])", r" \1 ", text)
    return [t for t in text.split() if t]


def _wt2_build_vocab(texts, max_vocab=10_000):
    SPECIAL = ["<PAD>", "<BOS>", "<EOS>", "<UNK>"]
    counter = Counter(t for sent in texts for t in sent)
    most_common = [w for w, _ in counter.most_common(max_vocab - len(SPECIAL))]
    vocab = {w: i for i, w in enumerate(SPECIAL)}
    for w in most_common:
        vocab[w] = len(vocab)
    return vocab


def _wt2_encode(tokens, vocab):
    return [vocab.get(t, UNK_IDX) for t in tokens]


def _wt2_chunk(ids, max_len):
    chunks = []
    step = max_len - 2
    for i in range(0, len(ids), step):
        chunk = ids[i: i + step]
        if len(chunk) < 2:
            continue
        chunks.append([BOS_IDX] + chunk + [EOS_IDX])
    return chunks


def load_wikitext2(max_seq_len: int = 50, data_dir: Optional[str] = None):
    """
    Load WikiText-2 as chunked sequences.

    Returns (train_seqs, val_seqs, test_seqs, vocab) where seqs are
    lists of integer lists, each of length ≤ max_seq_len.
    """
    # Silent fast path: pre-built cache from development environment (skipped if absent)
    _dev_cache = os.path.expanduser(
        os.path.join(paths.cache_root(), "ptb", "ptb_maxlen50.pt"))
    if os.path.exists(_dev_cache):
        return torch.load(_dev_cache, weights_only=False, map_location="cpu")

    # Fall back to data_dir cache
    if data_dir is not None:
        cache = os.path.join(data_dir, f"wikitext2_maxlen{max_seq_len}.pt")
        if os.path.exists(cache):
            return torch.load(cache, weights_only=False, map_location="cpu")
    else:
        data_dir = os.path.join(paths.cache_root(), "wikitext2")

    # Build from scratch
    import warnings
    warnings.filterwarnings("ignore")
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1")

    def _texts(split):
        return [_wt2_tokenize(row["text"]) for row in ds[split] if row["text"].strip()]

    train_toks = _texts("train")
    val_toks   = _texts("validation")
    test_toks  = _texts("test")
    vocab = _wt2_build_vocab(train_toks)

    def _to_chunks(toks_list):
        all_ids = [_wt2_encode(t, vocab) for t in toks_list]
        flat = [idx for ids in all_ids for idx in ids]
        return _wt2_chunk(flat, max_seq_len)

    train_seqs = _to_chunks(train_toks)
    val_seqs   = _to_chunks(val_toks)
    test_seqs  = _to_chunks(test_toks)

    os.makedirs(data_dir, exist_ok=True)
    payload = (train_seqs, val_seqs, test_seqs, vocab)
    cache = os.path.join(data_dir, f"wikitext2_maxlen{max_seq_len}.pt")
    torch.save(payload, cache)
    return payload


# ── TinyStories ────────────────────────────────────────────────────────────────

_TS_TRANS = str.maketrans(string.punctuation, " " * len(string.punctuation))


def load_tinystories(data_dir: Optional[str] = None, max_len: int = 128,
                     rebuild: bool = False):
    """
    Load TinyStories preprocessed chunks.
    Returns (vocab, idx2word, train_chunks, val_chunks).

    Precedence, and why it is this way round:

    1. `data_dir` given -- use that directory wholesale. This is the named opt-in for
       evaluating on a split other than the published one.
    2. The shipped eval cache (data/tinystories_eval/) supplies the VOCABULARY and the
       VALIDATION split whenever it is present: it is the split the published numbers
       were computed on, and its vocabulary is what indexes the released checkpoints.
       The TRAIN split comes from the local cache if one has been fetched.
    3. Only if the shipped cache is absent does the local cache supply all three.
    4. Otherwise build from the corpus.

    The shipped cache deliberately outranks the fetched one. data/fetch_data.py rebuilds
    from the current upstream corpus and does not reproduce the published split, so if
    the fetched cache won, running the documented setup would silently move every
    evaluation onto different data -- including the results/ vs runs/ diff the README
    recommends. Fetching exists to obtain the ~1 GB train split, which is not shipped.

    `rebuild=True` skips every read path and builds from the corpus; fetch_data.py uses
    it, since otherwise the fetcher would return a cache instead of creating one.
    """
    _dev_cache_dir = os.path.expanduser(
        os.path.join(paths.cache_root(), "tinystories"))
    _ship_dir = os.path.join(paths.REPO_ROOT, "data", "tinystories_eval")

    def _read_dir(d):
        vd = torch.load(os.path.join(d, "vocab.pt"), weights_only=False,
                        map_location="cpu")
        with open(os.path.join(d, "train_chunks.pkl"), "rb") as f:
            train_chunks = pickle.load(f)
        with open(os.path.join(d, "val_chunks.pkl"), "rb") as f:
            val_chunks = pickle.load(f)
        return vd["vocab"], vd["idx2word"], train_chunks, val_chunks

    # 1. explicit directory -- the named opt-in for a non-published split.
    # If it is named but does not resolve, fail here: falling through would silently
    # relocate evaluation onto the fetched split, which is what this ordering exists to
    # prevent.
    if not rebuild and data_dir is not None:
        if not (os.path.isdir(data_dir)
                and os.path.exists(os.path.join(data_dir, "vocab.pt"))):
            raise FileNotFoundError(
                f"data_dir={data_dir!r} does not contain vocab.pt. Refusing to fall back "
                f"to another split: pass a directory holding vocab.pt / train_chunks.pkl "
                f"/ val_chunks.pkl, or omit data_dir to use the shipped evaluation cache.")
        return _read_dir(data_dir)

    # 2. shipped eval cache wins for vocabulary + validation
    if (not rebuild and data_dir is None
            and os.path.exists(os.path.join(_ship_dir, "vocab.pt"))
            and os.path.exists(os.path.join(_ship_dir, "val_chunks.pkl"))):
        vd = torch.load(os.path.join(_ship_dir, "vocab.pt"),
                        weights_only=False, map_location="cpu")
        with open(os.path.join(_ship_dir, "val_chunks.pkl"), "rb") as f:
            val_chunks = pickle.load(f)
        _tc = os.path.join(_dev_cache_dir, "train_chunks.pkl")
        if os.path.exists(_tc):
            with open(_tc, "rb") as f:
                train_chunks = pickle.load(f)
        else:
            train_chunks = []  # eval-only; run data/fetch_data.py before training
        return vd["vocab"], vd["idx2word"], train_chunks, val_chunks

    # 3. no shipped cache (older tree) -- the local cache supplies all three
    if (not rebuild and os.path.isdir(_dev_cache_dir)
            and os.path.exists(os.path.join(_dev_cache_dir, "vocab.pt"))):
        return _read_dir(_dev_cache_dir)

    # Build from scratch
    _out_dir = data_dir or os.path.join(paths.cache_root(), "tinystories")
    from datasets import load_dataset
    print("[TS] Downloading TinyStories ...", flush=True)
    ds = load_dataset("roneneldan/TinyStories")
    train_stories = [ex["text"] for ex in ds["train"]]
    val_stories   = [ex["text"] for ex in ds["validation"]]

    def _tok(text):
        return text.lower().translate(_TS_TRANS).split()

    # Prefer the shipped authoritative vocabulary so a rebuilt cache indexes the
    # released checkpoints exactly. Rebuilding via most_common() below is a
    # fallback only: tie-breaking among equal-frequency words and any drift in the
    # upstream corpus can reorder the vocabulary and silently reindex embeddings.
    _ship_vocab = os.path.join(paths.REPO_ROOT, "data", "tinystories_eval", "vocab.pt")
    if os.path.exists(_ship_vocab):
        vd = torch.load(_ship_vocab, weights_only=False, map_location="cpu")
        vocab, idx2word = vd["vocab"], vd["idx2word"]
    else:
        print("[TS] WARNING: shipped vocab.pt not found; rebuilding vocabulary "
              "from corpus. This may not match released checkpoints.", flush=True)
        total = Counter()
        for story in train_stories:
            total.update(_tok(story))
        SPECIAL = ["<pad>", "<bos>", "<eos>", "<unk>"]
        vocab = {tok: i for i, tok in enumerate(SPECIAL)}
        for w, _ in total.most_common(8000):
            vocab[w] = len(vocab)
        idx2word = {i: w for w, i in vocab.items()}

    def _enc(text):
        return [BOS_IDX] + [vocab.get(w, UNK_IDX) for w in _tok(text)] + [EOS_IDX]

    def _chunks(stories):
        result = []
        for text in stories:
            ids = _enc(text)
            for i in range(0, len(ids), max_len):
                c = ids[i: i + max_len]
                if len(c) > 4:
                    result.append(c)
        return result

    train_chunks = _chunks(train_stories)
    val_chunks   = _chunks(val_stories)

    os.makedirs(_out_dir, exist_ok=True)
    torch.save({"vocab": vocab, "idx2word": idx2word},
               os.path.join(_out_dir, "vocab.pt"))
    with open(os.path.join(_out_dir, "train_chunks.pkl"), "wb") as f:
        pickle.dump(train_chunks, f, protocol=4)
    with open(os.path.join(_out_dir, "val_chunks.pkl"), "wb") as f:
        pickle.dump(val_chunks, f, protocol=4)
    return vocab, idx2word, train_chunks, val_chunks


# ── Dataset classes ────────────────────────────────────────────────────────────

class SeqDataset(Dataset):
    def __init__(self, seqs: list, max_len: int):
        self.seqs    = seqs
        self.max_len = max_len

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        ids = self.seqs[idx]
        T   = len(ids)
        pad = self.max_len - T
        tokens   = torch.tensor(ids + [PAD_IDX] * pad, dtype=torch.long)
        pad_mask = torch.tensor([False] * T + [True] * pad, dtype=torch.bool)
        return {"tokens": tokens, "pad_mask": pad_mask}


def make_lm_loaders(seqs_train, seqs_val, max_len: int, batch_size: int = 64,
                    num_workers: int = 0):
    val_ds     = SeqDataset(seqs_val, max_len)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers)
    if not seqs_train:
        return None, val_loader
    train_ds     = SeqDataset(seqs_train, max_len)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers)
    return train_loader, val_loader


# ── Training / eval utilities ─────────────────────────────────────────────────

import math


def train_lm_epoch(model, loader, optimizer, scheduler, grad_clip, device):
    model.train()
    crit = torch.nn.CrossEntropyLoss(ignore_index=0)
    total_loss, n_batches = 0.0, 0
    for batch in loader:
        tokens   = batch["tokens"].to(device)
        pad_mask = batch["pad_mask"].to(device)
        inp, tgt = tokens[:, :-1], tokens[:, 1:]
        pm_inp   = pad_mask[:, :-1]
        optimizer.zero_grad(set_to_none=True)
        logits  = model(inp, padding_mask=pm_inp)
        B, T, V = logits.shape
        loss    = crit(logits.reshape(B * T, V), tgt.reshape(B * T))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total_loss += loss.item()
        n_batches  += 1
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def eval_ppl(model, loader, device):
    model.eval()
    crit = torch.nn.CrossEntropyLoss(ignore_index=0, reduction="sum")
    total_loss, total_toks = 0.0, 0
    for batch in loader:
        tokens   = batch["tokens"].to(device)
        pad_mask = batch["pad_mask"].to(device)
        inp, tgt = tokens[:, :-1], tokens[:, 1:]
        pm_inp   = pad_mask[:, :-1]
        logits   = model(inp, padding_mask=pm_inp)
        B, T, V  = logits.shape
        mask     = ~pad_mask[:, 1:].reshape(-1)
        total_loss += crit(logits.reshape(B * T, V)[mask], tgt.reshape(B * T)[mask]).item()
        total_toks += mask.sum().item()
    return math.exp(total_loss / max(total_toks, 1))
