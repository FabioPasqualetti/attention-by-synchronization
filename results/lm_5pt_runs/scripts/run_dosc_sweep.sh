#!/usr/bin/env bash
# 5-point d_osc scaling-law extension sweep (Table 4 / Figure 6, d_osc in {4,16}).
# 20 runs: d_osc in {4,16} x seeds {0..4} x {WT2, TinyStories}.
#
# Order alternates datasets to spread risk (one full point per dataset early):
#   1) WT2 d4 s0-4   2) TS d4 s0-4   3) WT2 d16 s0-4   4) TS d16 s0-4
#
# Sequential only (M1 has no headroom for parallel). Each run is idempotent:
# re-running this script skips runs whose JSON already exists, so it is safely
# resumable after an interruption. Wrap with caffeinate so the Mac stays awake:
#   caffeinate -dimsu bash results/lm_5pt_runs/scripts/run_dosc_sweep.sh
set -u
cd "$(dirname "$0")/../.."                 # -> code/
V="${PYTHON:-python3}"
RUN="$V results/lm_5pt_runs/scripts/run_one_5pt.py"
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] SWEEP: $*"; }

log "START 20-run sweep (WT2~45min/run, TS longer)"
for spec in "wt2 4" "ts 4" "wt2 16" "ts 16"; do
  set -- $spec
  DS="$1"; D="$2"
  for S in 0 1 2 3 4; do
    log "begin $DS d_osc=$D seed=$S"
    $RUN --dataset "$DS" --d_osc "$D" --seed "$S"
    log "end   $DS d_osc=$D seed=$S"
  done
done
log "DONE 20-run sweep"
