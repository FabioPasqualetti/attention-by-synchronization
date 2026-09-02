"""Shared harness utilities: git hash, resumable per-run JSON, timing, device."""
import json
import os
import subprocess
import time

from . import paths

REPO_ROOT = paths.REPO_ROOT

# Two roots, deliberately separate (see training/paths.py):
#   REFERENCE_ROOT -- results/, the run behind the paper. Tracked. READ ONLY from here.
#   RUNS_ROOT      -- runs/, this machine's output. Gitignored, absent on a clean clone.
# Drivers write to RUNS_ROOT, so a re-run cannot touch the published numbers; comparing
# is then `diff results/<exp>/<key>.json runs/<exp>/<key>.json`. Readers (figures,
# analyses) resolve against REFERENCE_ROOT, which OSCILLATOR_RESULTS can repoint at runs/.
# There is deliberately no RESULTS_ROOT: every call site has to say which one it means.
REFERENCE_ROOT = paths.results_root()
RUNS_ROOT = paths.runs_root()


def git_hash():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT).decode().strip()
    except Exception:
        return "unknown"


def run_dir(exp, *parts):
    """A directory under this run's output root (created). Use for checkpoints and
    any per-experiment output directory."""
    d = os.path.join(RUNS_ROOT, exp, *parts)
    os.makedirs(d, exist_ok=True)
    return d


def result_path(exp, key):
    """WRITE / resume target, under runs/. `exists` and `load_result` follow it, so a
    driver resumes from its own prior work, never from the reference.

    Does not create the directory -- `exists()` calls this, and merely asking whether a
    result is there should not litter runs/ with empty directories. save_result creates
    it at write time."""
    return os.path.join(RUNS_ROOT, exp, f"{key}.json")


def reference_path(exp, key):
    """READ path for a published value, under results/."""
    return os.path.join(REFERENCE_ROOT, exp, f"{key}.json")


def exists(exp, key):
    return os.path.exists(result_path(exp, key))


def reference_exists(exp, key):
    return os.path.exists(reference_path(exp, key))


def load_reference(exp, key):
    with open(reference_path(exp, key)) as f:
        return json.load(f)


def _is_degenerate(obj):
    """True if a payload looks like an empty or failed run — something that must not be
    allowed to overwrite a good committed result. Covers {}/[]/None and any run whose
    only list of records is entirely error/skip stubs."""
    if not obj:
        return True
    if isinstance(obj, dict):
        for v in obj.values():
            if isinstance(v, list) and v and all(
                    isinstance(e, dict) and any(
                        k in e for k in ("skipped", "error", "VERIFY_FAIL", "SANITY_FAIL"))
                    for e in v):
                return True
    return False


def guarded_dump(path, obj, indent=2):
    """Write obj to path as JSON, but refuse to overwrite a committed non-degenerate
    result with a degenerate (empty / all-error) one. On refusal write <path>.partial and
    exit non-zero, so a failed re-run cannot silently truncate a published value.

    Two ways a degenerate result is refused:

      1. `path` already holds a good result -- the classic case, a failed re-run must not
         truncate what is there.
      2. `path` does not exist, but the REFERENCE tree holds a good result for the same
         relative key. Since drivers write to runs/, a first run into an empty runs/ hits
         no existing file, so (1) alone would let a driver that computed nothing write {}
         and report success. If we published a real result for that key, producing nothing
         is a failure, not a write.
    """
    def _good(p):
        try:
            with open(p) as f:
                return not _is_degenerate(json.load(f))
        except Exception:
            return os.path.exists(p) and os.path.getsize(p) > 2

    if _is_degenerate(obj):
        blocker = None
        if os.path.exists(path) and _good(path):
            blocker = path
        else:
            # same key under the reference tree?
            ap = os.path.abspath(path)
            if ap.startswith(os.path.abspath(RUNS_ROOT) + os.sep):
                rel = os.path.relpath(ap, os.path.abspath(RUNS_ROOT))
                ref = os.path.join(REFERENCE_ROOT, rel)
                if os.path.exists(ref) and _good(ref):
                    blocker = ref
        if blocker is not None:
            import sys
            alt = path + ".partial"
            os.makedirs(os.path.dirname(alt), exist_ok=True)
            with open(alt, "w") as f:
                json.dump(obj, f, indent=indent)
            print(f"[guard] refusing to write a degenerate (empty / all-error) result to "
                  f"{path}: {blocker} holds a real result for this key, so producing "
                  f"nothing is a failure. Wrote {alt} and exiting non-zero.", file=sys.stderr)
            sys.exit(1)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=indent)
    os.replace(tmp, path)
    return path


def save_result(exp, key, payload):
    """Write a per-run JSON with standard envelope fields (through the overwrite guard)."""
    payload = dict(payload)
    payload.setdefault("git_hash", git_hash())
    payload.setdefault("exp", exp)
    payload.setdefault("key", key)
    return guarded_dump(result_path(exp, key), payload)


def load_result(exp, key):
    with open(result_path(exp, key)) as f:
        return json.load(f)


def pick_device(pref="mps"):
    import torch
    if pref == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if pref == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def free_memory(device=None):
    """Release cached allocator memory between runs in long-lived processes.

    MPS does not always return freed memory to the system between successive
    model builds/evals in one process; over many iterations this can saturate
    RAM/swap. Call this at the end of each run in a multi-run loop.
    """
    import gc
    gc.collect()
    try:
        import torch
        dev_type = getattr(device, "type", device)
        if dev_type == "mps" and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif dev_type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass


class Timer:
    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *a):
        self.wall = time.time() - self.t0
