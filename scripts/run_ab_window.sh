#!/bin/bash
# run_ab_window.sh - BF16-vs-FP8 KV evaluation window on OUR host. See docs/ab-protocol.md.
#
#   CONFIRM=yes PROMPTS=/data/glm_test/eval/prompts_full OUT=/data/glm_test/eval/ab_$(date +%m%d) bash scripts/run_ab_window.sh
#
# Takes the public service down three times (restarts) and keeps the public front door stopped for the whole
# window. NOT YET RUN END TO END; its parts are tested separately (docs/validation.md).
set -u
[ "${CONFIRM:-}" = "yes" ] || { echo "refusing: this stops the shared service. Re-run with CONFIRM=yes at an agreed time."; exit 2; }
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(dirname "$HERE")
PROMPTS=${PROMPTS:?set PROMPTS to a directory made by: ab_eval.py prep}
OUT=${OUT:?set OUT}
FRONT=${FRONT:-glm-front}
AB="python3 $REPO/tests/ab_eval.py"
LOG=$OUT/window.log
mkdir -p "$OUT"
say() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

[ -s "$PROMPTS/ppl.jsonl" ] && [ -s "$PROMPTS/mmlu.jsonl" ] || { echo "prompts missing in $PROMPTS"; exit 2; }
docker image inspect "${IMG:-glm53-fp8kv:v0.13.1-sm86}" >/dev/null 2>&1 || { echo "image missing - build it first (docker/Dockerfile)"; exit 2; }
$AB selftest | tail -1 | tee -a "$LOG" | grep -q "SELFTEST OK" || { echo "selftest failed - comparison logic cannot be trusted"; exit 2; }

RESTORED=0
restore() {
  [ "$RESTORED" = "1" ] && return
  if ! curl -sf -m 5 localhost:18010/health >/dev/null || [ "${FINAL_OK:-0}" != "1" ]; then
    say "restoring service with the production configuration"
    IMG=${IMG:-glm53-fp8kv:v0.13.1-sm86} KV_DTYPE=fp8_ds_mla GPU_UTIL=0.88 MAX_BT=8192 SPEC=1 bash "$HERE/launch_host_a40.sh" >>"$LOG" 2>&1 \
      || { say "image launch failed; falling back to the previous known-good launcher"; bash /data/glm_test/run_glm_fp8kv.sh >>"$LOG" 2>&1; }
  fi
  if [ "${FINAL_OK:-0}" = "1" ] && [ -f "$REPO/scripts/warmup.py" ]; then
    say "warming up the production instance before reopening the front door"
    python3 "$REPO/scripts/warmup.py" >>"$LOG" 2>&1 || say "warm-up reported an error (continuing)"
  fi
  docker start "$FRONT" >/dev/null 2>&1 && say "front door restored"
  [ -s "$OUT/crontab.bak" ] && crontab "$OUT/crontab.bak" && say "monitor cron restored"
  RESTORED=1
}
trap restore EXIT

say "stopping public front door ($FRONT) so the evaluation is the only load"
docker stop "$FRONT" >/dev/null 2>&1
# the cron canary would add requests to the evaluation batches; pause it for the window
crontab -l > "$OUT/crontab.bak" 2>/dev/null; crontab -l 2>/dev/null | grep -v glm_monitor | crontab - ; say "monitor cron paused"

run_arm() {   # $1 = label (ref|cand)  $2 = KV dtype  $3 = number of ppl/greedy runs
  local label=$1 kv=$2 nruns=${3:-2}
  say "=== arm $label (kv-cache-dtype=$kv), evaluation configuration"
  KV_DTYPE=$kv GPU_UTIL=0.80 MAX_BT=2048 SPEC=1 bash "$HERE/launch_host_a40.sh" >>"$LOG" 2>&1 || { say "arm $label failed to start"; exit 1; }
  export AB_EVAL_SAFE=1
  for r in $(seq 1 "$nruns"); do
    say "$label run $r: ppl"; $AB ppl --prompts "$PROMPTS" --out "$OUT/${label}$r" >>"$LOG" 2>&1 || { say "ppl failed"; exit 1; }
    say "$label run $r: greedy"; $AB greedy --prompts "$PROMPTS" --out "$OUT/${label}$r" >>"$LOG" 2>&1 || { say "greedy failed"; exit 1; }
  done
  say "$label: mmlu"; $AB mmlu --prompts "$PROMPTS" --out "$OUT/${label}1" >>"$LOG" 2>&1 || { say "mmlu failed"; exit 1; }
  unset AB_EVAL_SAFE
  if [ -f /data/glm_test/glm_bench.py ]; then
    say "$label: speed (idle host, evaluation configuration)"
    ( cd /data/glm_test && timeout 600 python3 glm_bench.py single; timeout 600 python3 glm_bench.py conc ) >"$OUT/speed_${label}.txt" 2>&1
  fi
  docker logs glm53flash 2>&1 | grep -E "GPU KV cache size|Maximum concurrency" | tail -2 >"$OUT/kv_${label}.txt"
}

run_arm ref auto 2          # two runs: gives the reference's own run-to-run noise floor
run_arm cand fp8_ds_mla 1   # one run is enough for the verdict (decided before any data; see docs/ab-protocol.md)

say "=== comparison"
$AB compare --ref "$OUT/ref1" --ref2 "$OUT/ref2" --cand "$OUT/cand1" --report "$OUT/report" | tee -a "$LOG" | tail -40

say "=== final: FP8 production configuration"
IMG=${IMG:-glm53-fp8kv:v0.13.1-sm86} KV_DTYPE=fp8_ds_mla GPU_UTIL=0.88 MAX_BT=8192 SPEC=1 bash "$HERE/launch_host_a40.sh" >>"$LOG" 2>&1 && FINAL_OK=1
say "window finished; report in $OUT/report.md"
