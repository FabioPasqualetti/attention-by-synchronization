"""kws_frame_shuffle — Frame-shuffle positional-signal probe (evaluation study).

For every kws_position_matched condition (mechanism in {softmax, osc} x PE in {none, sinusoidal,
learned_abs}) x 5 seeds, evaluate KWS validation accuracy under:
  (a) intact inputs
  (b) a random permutation of the T=49 log-mel frames (one fixed permutation per draw,
      applied to every utterance; 3 draws with fixed seeds 0,1,2; report mean)
and report the drop  Delta = acc_intact - mean(acc_shuffled).

Checkpoints: kws_position_matched persists checkpoints for all conditions (results/kws_position_matched/ckpt/). Any that are
missing (kws_position_matched runs that completed before checkpoint-saving was enabled for non-'none' PEs)
are REGENERATED here by deterministic re-training via position_matched.run_one (same seed +
cached loaders => bit-identical model). This keeps the probe self-contained; it is otherwise eval-only.

SANITY CHECK: softmax with pe='none' is permutation-invariant (per-frame proj -> position-
free softmax attention -> mean-pool), so its Delta MUST be ~0. If |Delta| exceeds
SANITY_TOL for any softmax/none seed, the permutation harness is buggy => STOP and report
(exit code 2) rather than proceeding.

Interpretation: Delta for the plain oscillator (osc/none) measures how much positional
signal the learned per-position anchors absorbed.

Device note: this driver may train (to regenerate missing checkpoints), so it should not
run concurrently with other training on the same device.
"""
import csv
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from training import harness, paths  # noqa: E402

paths.ensure_paths()
import torch  # noqa: E402

# import the kws_position_matched driver (for run_one + ARCH + MECHS + CKPT dir) via file loader
e1 = paths.load_py(os.path.join(paths.REPO_ROOT, "keyword_spotting", "position_matched.py"),
                   "kws_position_matched_mod")
from oscillator_attention.kws_pe_model import KWSTransformerPE  # noqa: E402
from training.kws_data import cached_loaders  # noqa: E402

EXP = "kws_frame_shuffle"
ARCH = e1.ARCH
MECHS = e1.MECHS            # {"softmax":"softmax", "osc":"lohe"}
PES = e1.PES               # ["none", "sinusoidal", "learned_abs"]
SEEDS = e1.SEEDS
CKPT = e1.CKPT
PERM_SEEDS = [0, 1, 2]
SANITY_TOL = 0.002          # 0.2 pp; softmax/none must be ~0


def eval_acc(model, loader, device, perm=None):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for feats, labels in loader:
            if feats is None:
                continue
            if perm is not None:
                feats = feats[:, perm, :]
            feats, labels = feats.to(device), labels.to(device)
            correct += (model(feats).argmax(1) == labels).sum().item()
            total += labels.size(0)
    return correct / max(total, 1)


VERIFY_TOL = 1e-9  # regenerated best-epoch acc must equal kws_position_matched's JSON val_acc (exact)


def ensure_ckpt(mech, pe, seed, device):
    """Return (path, regenerated, regen_best_acc).

    regen_best_acc is the best-epoch val accuracy returned by the deterministic
    re-training (same quantity kws_position_matched stores in its JSON), or None if the checkpoint
    already existed (loaded directly, no regeneration).
    """
    path = os.path.join(CKPT, f"{mech}_{pe}_s{seed}_final.pt")
    if os.path.exists(path):
        return path, False, None
    print(f"  [regen] ckpt missing -> deterministically re-training {mech}_{pe}_s{seed}",
          flush=True)
    best = e1.run_one(mech, pe, seed, device, save_ckpt=True)  # writes to CKPT
    return path, True, best


def verify_regen(mech, pe, seed, regen_best):
    """Compare a regenerated checkpoint's best-epoch acc to the kws_position_matched result JSON.

    Returns (ok, e1_ref, msg). ok=False => do NOT use this checkpoint in kws_frame_shuffle.
    """
    key = f"{mech}_{pe}_s{seed}"
    try:
        e1_ref = harness.load_reference("kws_position_matched", key)["val_acc"]
    except Exception as e:
        return False, None, f"kws_position_matched JSON unavailable ({type(e).__name__}: {e})"
    if regen_best is None:
        return False, e1_ref, "no regen accuracy captured"
    if abs(regen_best - e1_ref) <= VERIFY_TOL:
        return True, e1_ref, "exact match"
    return False, e1_ref, (f"MISMATCH regen={regen_best*100:.6f}% vs "
                           f"kws_position_matched={e1_ref*100:.6f}% (Δ={abs(regen_best-e1_ref)*100:.6f}pp)")


def main():
    device = harness.pick_device("mps")
    print(f"kws_frame_shuffle device={device}", flush=True)
    _, val, _, _ = cached_loaders(0, batch_size=64)
    perms = {ps: torch.randperm(ARCH["T"],
                                generator=torch.Generator().manual_seed(ps))
             for ps in PERM_SEEDS}

    for mech in MECHS:
        for pe in PES:
            for seed in SEEDS:
                key = f"{mech}_{pe}_s{seed}"
                if harness.exists(EXP, key):
                    continue
                ckpt, regen, regen_best = ensure_ckpt(mech, pe, seed, device)

                # Verify any REGENERATED checkpoint against the kws_position_matched result JSON.
                # Exact match required; otherwise exclude this condition from kws_frame_shuffle.
                if regen:
                    ok, e1_ref, msg = verify_regen(mech, pe, seed, regen_best)
                    if not ok:
                        harness.save_result(EXP, key, {
                            "mech": mech, "pe": pe, "seed": seed,
                            "ckpt_regenerated": True, "VERIFY_FAIL": True,
                            "regen_best_acc": regen_best, "e1_val_acc": e1_ref,
                            "verify_msg": msg})
                        print(f"VERIFY FAIL {key}: {msg} — excluded from kws_frame_shuffle table.",
                              flush=True)
                        continue
                    print(f"  [verify] {key}: regen best={regen_best*100:.4f}% "
                          f"== kws_position_matched {e1_ref*100:.4f}% ({msg})", flush=True)

                model = KWSTransformerPE(attn_type=MECHS[mech], p=1, pe=pe,
                                         **ARCH).to(device)
                model.load_state_dict(torch.load(ckpt, map_location="cpu",
                                                 weights_only=False))
                intact = eval_acc(model, val, device, perm=None)
                shuffled = [eval_acc(model, val, device, perm=perms[ps])
                            for ps in PERM_SEEDS]
                shuffled_mean = sum(shuffled) / len(shuffled)
                delta = intact - shuffled_mean

                # Sanity gate on softmax/none permutation invariance
                if mech == "softmax" and pe == "none" and abs(delta) > SANITY_TOL:
                    harness.save_result(EXP, key, {
                        "mech": mech, "pe": pe, "seed": seed,
                        "intact_acc": intact, "shuffled_accs": shuffled,
                        "shuffled_mean": shuffled_mean, "delta": delta,
                        "SANITY_FAIL": True})
                    print(f"SANITY FAIL {key}: softmax/none delta={delta*100:.3f}pp "
                          f"> {SANITY_TOL*100:.1f}pp — permutation harness buggy. STOP.",
                          flush=True)
                    sys.exit(2)

                payload = {
                    "mech": mech, "pe": pe, "seed": seed, "ckpt_regenerated": regen,
                    "intact_acc": intact, "shuffled_accs": shuffled,
                    "shuffled_mean": shuffled_mean, "delta": delta,
                    "perm_seeds": PERM_SEEDS, "verified": True}
                if regen:
                    payload["regen_best_acc"] = regen_best
                    payload["e1_val_acc"] = harness.load_reference("kws_position_matched", key)["val_acc"]
                harness.save_result(EXP, key, payload)
                print(f"DONE {key}: intact={intact*100:.2f}% "
                      f"shuffled={shuffled_mean*100:.2f}% delta={delta*100:.2f}pp",
                      flush=True)

    _write_csv()
    print("kws_frame_shuffle COMPLETE", flush=True)


def _write_csv():
    import glob
    import json
    rows = []
    for p in sorted(glob.glob(os.path.join(harness.RUNS_ROOT, EXP, "*.json"))):
        with open(p) as f:
            rows.append(json.load(f))
    if not rows:
        return
    out = os.path.join(harness.RUNS_ROOT, EXP, "frame_shuffle_all.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mech", "pe", "seed", "intact_acc", "shuffled_mean", "delta_pp",
                    "ckpt_regenerated"])
        for r in rows:
            w.writerow([r["mech"], r["pe"], r["seed"],
                        f"{r['intact_acc']*100:.3f}", f"{r['shuffled_mean']*100:.3f}",
                        f"{r['delta']*100:.3f}", r.get("ckpt_regenerated", False)])
    print(f"kws_frame_shuffle CSV -> {out}", flush=True)


if __name__ == "__main__":
    main()
