"""Populate the data cache for TinyStories and WikiText-2 via HuggingFace `datasets`.

Cache root: the OSCILLATOR_DATA_CACHE environment variable, else <repo>/data/cache.
Penn Treebank is LDC-licensed and is NOT downloaded here: place your own licensed,
tokenized copy at <cache>/ptb/ptb_maxlen50.pt (see the README "Data" section).

Not needed for evaluation: data/tinystories_eval/ ships the authoritative TinyStories
vocabulary and validation chunks, so the figures and every eval path work on a clean
clone without running this script. Run it when you want to TRAIN, which needs the
~1 GB train split that is not shipped.

Usage:
  python data/fetch_data.py                 # WikiText-2 + TinyStories
  python data/fetch_data.py --wikitext2     # one corpus
  python data/fetch_data.py --tinystories
Requires: pip install datasets  (and torch).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training import paths                                   # noqa: E402
from training.data_utils import load_wikitext2, load_tinystories  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--wikitext2", action="store_true")
    ap.add_argument("--tinystories", action="store_true")
    args = ap.parse_args()
    both = not (args.wikitext2 or args.tinystories)
    cr = paths.cache_root()
    os.makedirs(cr, exist_ok=True)
    print(f"Cache root: {cr}")

    built = []
    if both or args.wikitext2:
        print("Fetching WikiText-2 (HuggingFace: wikitext-2-raw-v1) ...", flush=True)
        load_wikitext2(max_seq_len=50)
        built.append(("WikiText-2", os.path.join(cr, "wikitext2"), None))
    if both or args.tinystories:
        print("Fetching TinyStories (HuggingFace: roneneldan/TinyStories) ...", flush=True)
        # rebuild=True: without it load_tinystories returns from the shipped-eval cache
        # with an empty train split and this script would report success having built
        # nothing. The fetcher's job is to produce the cache, so it must not read one.
        _, _, train_chunks, val_chunks = load_tinystories(max_len=128, rebuild=True)
        built.append(("TinyStories", os.path.join(cr, "tinystories"),
                      (len(train_chunks), len(val_chunks))))

    # Report what is actually on disk, not what was attempted.
    print()
    ok = True
    for name, d, counts in built:
        files = sorted(os.listdir(d)) if os.path.isdir(d) else []
        if not files:
            print(f"  {name}: NOTHING WRITTEN to {d}")
            ok = False
            continue
        extra = f"  ({counts[0]:,} train / {counts[1]:,} val chunks)" if counts else ""
        print(f"  {name}: {d}{extra}")
        for fn in files:
            print(f"      {fn}  ({os.path.getsize(os.path.join(d, fn)) / 1e6:.1f} MB)")

    ptb = os.path.join(cr, "ptb", "ptb_maxlen50.pt")
    if not os.path.exists(ptb):
        print("\nPenn Treebank: NOT downloaded (LDC-licensed, not redistributable).")
        print(f"  Place your own licensed, tokenized copy at:\n    {ptb}")
        print("  See the README 'Data' section for the expected format.")
    if not ok:
        print("\nFAILED: the cache was not produced. Check the `datasets` install and "
              "network access, then re-run.")
        sys.exit(1)
    print("\nDone.")


if __name__ == "__main__":
    main()
