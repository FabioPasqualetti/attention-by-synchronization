"""
dataset_sva.py — Subject-verb agreement dataset (synthetic).

Sentence templates:
  Simple:  the [NOUN] [VERB] .
  With PP: the [NOUN] [PREP] the [NOUN2] [VERB] .

Label 1 = verb agrees with subject, 0 = disagrees.
50% label-1, 50% label-0.
For PP sentences: 50% distractor agrees with subject (easy),
                  50% distractor disagrees (hard).
"""

import json
import os
import random

# The shipped corpus, the same one training/sva_dataset.py reads. This module used
# to keep its own copy under figures/lib/data/, which a normal figure run rebuilt --
# 8 MB of untracked files, byte-identical to what already ships.
DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "sva")

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

# ── Special tokens ─────────────────────────────────────────────────────────────
PAD_TOKEN = "<PAD>"   # idx 0
BOS_TOKEN = "<BOS>"   # idx 1
EOS_TOKEN = "<EOS>"   # idx 2
UNK_TOKEN = "<UNK>"   # idx 3

SPECIAL = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN]

# Build shared vocabulary
_all_words = (["the"] + NOUNS_SG + NOUNS_PL + VERBS_SG + VERBS_PL
              + PREPS + ["."])
VOCAB = SPECIAL + sorted(set(_all_words))
WORD2IDX = {w: i for i, w in enumerate(VOCAB)}


def _make_sentence(rng: random.Random):
    """Generate one SVA sentence dict."""
    has_pp = rng.random() < 0.60

    subj_sg = rng.random() < 0.50            # subject number
    subj_noun = rng.choice(NOUNS_SG if subj_sg else NOUNS_PL)

    correct_verb = rng.random() < 0.50       # label
    label = 1 if correct_verb else 0

    if subj_sg:
        verb = rng.choice(VERBS_SG) if correct_verb else rng.choice(VERBS_PL)
    else:
        verb = rng.choice(VERBS_PL) if correct_verb else rng.choice(VERBS_SG)

    if not has_pp:
        tokens = ["the", subj_noun, verb, "."]
        subject_idx  = 1
        verb_idx     = 2
        distractor_idx    = None
        distractor_agrees = None
    else:
        prep = rng.choice(PREPS)
        dist_agrees_with_subj = rng.random() < 0.50
        if dist_agrees_with_subj:
            dist_noun = rng.choice(NOUNS_SG if subj_sg else NOUNS_PL)
        else:
            dist_noun = rng.choice(NOUNS_PL if subj_sg else NOUNS_SG)

        tokens = ["the", subj_noun, prep, "the", dist_noun, verb, "."]
        subject_idx       = 1
        verb_idx          = 5
        distractor_idx    = 4
        distractor_agrees = dist_agrees_with_subj

    return {
        "tokens":            tokens,
        "label":             label,
        "subject_idx":       subject_idx,
        "verb_idx":          verb_idx,
        "distractor_idx":    distractor_idx,
        "has_distractor":    has_pp,
        "distractor_agrees": distractor_agrees,
    }


def generate(n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    return [_make_sentence(rng) for _ in range(n)]


def save_jsonl(records: list[dict], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def build_datasets(force: bool = False):
    paths = {
        "train": os.path.join(DATA_DIR, "sva_train.jsonl"),
        "val":   os.path.join(DATA_DIR, "sva_val.jsonl"),
        "test":  os.path.join(DATA_DIR, "sva_test.jsonl"),
    }
    if all(os.path.exists(p) for p in paths.values()):
        if force:
            raise RuntimeError(
                f"refusing to regenerate into the shipped corpus at {DATA_DIR}. "
                "The generator is deterministic and reproduces these files byte for "
                "byte, so there is nothing to gain; remove them deliberately if you "
                "really intend to rebuild.")
        return {k: load_jsonl(v) for k, v in paths.items()}

    train = generate(40_000, seed=0)
    val   = generate(4_000,  seed=1)
    test  = generate(4_000,  seed=2)
    save_jsonl(train, paths["train"])
    save_jsonl(val,   paths["val"])
    save_jsonl(test,  paths["test"])
    return {"train": train, "val": val, "test": test}


def encode(tokens: list[str]) -> list[int]:
    return [WORD2IDX.get(t, WORD2IDX[UNK_TOKEN]) for t in tokens]


def print_stats(splits: dict):
    for name, records in splits.items():
        n     = len(records)
        with_d = sum(r["has_distractor"] for r in records)
        hard  = sum(
            r["has_distractor"] and r["distractor_agrees"] is False
            for r in records
        )
        mean_len = sum(len(r["tokens"]) for r in records) / n
        print(f"  SVA {name}: {n:6d} samples | mean_len={mean_len:.1f} | "
              f"with_distractor={with_d/n*100:.1f}% | "
              f"hard={hard/n*100:.1f}%")
    print(f"  SVA vocab: {len(VOCAB)} tokens")


if __name__ == "__main__":
    splits = build_datasets()
    print_stats(splits)
