#!/usr/bin/env bash
# Run ONE MaxKernel problem in its own headless Claude Code conversation.
#
#   tools/nightly/run_one.sh 63p_GDN
#   tools/nightly/run_one.sh DSv4/csa_kernel
#
# The argument is a path relative to $MK/problems. Each invocation is a fresh
# `claude -p` process => a fresh conversation, a fresh context window, its own
# run_dir under $MK/workspace, and its own exit code.
set -uo pipefail

MK="${MK:-/home/cathygao_google_com/maxkernel}"
PROBLEM="${1:?usage: run_one.sh <problem-dir relative to problems/>}"
PROBLEM_DIR="$MK/problems/$PROBLEM"
[ -d "$PROBLEM_DIR" ] || { echo "no such problem: $PROBLEM_DIR" >&2; exit 64; }

NIGHTLY="$MK/workspace/_nightly"
LOGDIR="$NIGHTLY/logs/$(date +%Y-%m-%d)"
RESULTS="$NIGHTLY/results.csv"
LASTRUN="$NIGHTLY/last_run.tsv"
mkdir -p "$LOGDIR"

SLUG="${PROBLEM//\//_}"
MODEL="${MK_MODEL:-opus}"
# Past runs took 6-24h of wall clock for 5 iterations. Cap so one stuck problem
# cannot eat the whole window.
PER_PROBLEM_TIMEOUT="${MK_TIMEOUT:-10h}"

# --- pick the entry file the skill should treat as the source ----------------
ENTRY=""
for cand in baseline.py reference.py kernel.py source.py; do
  [ -f "$PROBLEM_DIR/$cand" ] && { ENTRY="$PROBLEM_DIR/$cand"; break; }
done
if [ -z "$ENTRY" ]; then
  ENTRY="$(find "$PROBLEM_DIR" -maxdepth 1 -name '*.py' ! -name '__init__.py' | sort | head -1)"
fi
[ -n "$ENTRY" ] || { echo "no entry .py under $PROBLEM_DIR" >&2; exit 65; }

# --- pre-create the run dir so problem -> run_dir is deterministic -----------
# (If we let the skill invent a run_id we would have to guess which of the
# concurrently-created workspace/run_* dirs belongs to this problem.)
RUN_ID="run_$(python3 -c 'import uuid;print(uuid.uuid4().hex[:8])')"
RUN_DIR="$MK/workspace/$RUN_ID"
mkdir -p "$RUN_DIR"
LOG="$LOGDIR/${SLUG}.${RUN_ID}.log"

JOB_LINE=""
if [ -f "$PROBLEM_DIR/job.json" ]; then
  JOB_LINE="Job file: $PROBLEM_DIR/job.json — it declares the input type, entry point,
inputs_fn/init_inputs_fn, the conversion method, the tolerances and
loop.max_iterations. Start from it and skip the guessing; do not override what
it declares."
fi

TASK_YAML_LINE=""
if [ -f "$PROBLEM_DIR/kernel_task.yaml" ]; then
  TASK_YAML_LINE="Task spec: $PROBLEM_DIR/kernel_task.yaml — it contains input_gen_code for
get_inputs() and the rtol/atol for this problem. Use them instead of inventing
your own inputs or tolerances."
fi

read -r -d '' PROMPT <<EOF
Optimize this code into a Pallas TPU kernel. Use the maxkernel skill.

Source file: $ENTRY
Problem directory (read it for context): $PROBLEM_DIR
$JOB_LINE
$TASK_YAML_LINE

Use run_dir = $RUN_DIR for this run. That directory already exists and is
empty: do NOT generate a new run id, and write $RUN_DIR/state.json there.

Use the TPU described in $MK/tpu_config.json.

This is an unattended batch run. There is no human watching, so never stop to
ask a question: if something is ambiguous, pick the most reasonable option,
record the choice in $RUN_DIR/maxkernel_debug_history.md, and keep going. Run
the loop until it stops on its own criteria (state.stopping.recommend_stop) or
exhausts max_iterations, then end with the normal final summary. Converging in
two iterations is a success, not a short run -- do not keep going to fill the
budget.
EOF

START=$(date +%s)
echo "[$(date -Is)] START $PROBLEM  run_id=$RUN_ID  log=$LOG"

cd "$MK" || exit 66
timeout --signal=INT --kill-after=5m "$PER_PROBLEM_TIMEOUT" \
  claude -p "$PROMPT" \
    --model "$MODEL" \
    --dangerously-skip-permissions \
    --permission-prompts none \
    --output-format stream-json --verbose \
    ${MK_BUDGET_USD:+--max-budget-usd "$MK_BUDGET_USD"} \
    </dev/null >>"$LOG" 2>&1
RC=$?

END=$(date +%s)
MINS=$(( (END - START) / 60 ))

# --- harvest the result from state.json, not from the model's prose ----------
read -r ITER SPEEDUP BASE_MS OPT_MS BEST <<<"$(python3 - "$RUN_DIR/state.json" <<'PY'
import json, sys
try:
    s = json.load(open(sys.argv[1]))
except Exception:
    print("0 NA NA NA NA"); raise SystemExit
def g(k):
    v = s.get(k)
    return "NA" if v is None else v
print(s.get("iteration", 0), g("best_speedup"), g("base_time_ms"),
      g("best_optimized_time"), g("best_code_path"))
PY
)"

[ -f "$RESULTS" ] || echo "finished_at,problem,run_id,exit_code,minutes,iterations,best_speedup,base_time_ms,best_optimized_time,best_code_path,log" >"$RESULTS"
echo "$(date -Is),$PROBLEM,$RUN_ID,$RC,$MINS,$ITER,$SPEEDUP,$BASE_MS,$OPT_MS,$BEST,$LOG" >>"$RESULTS"

# Rotation bookkeeping: remember when this problem was last attempted.
touch "$LASTRUN"
{ awk -F'\t' -v p="$PROBLEM" '$1 != p' "$LASTRUN"; printf '%s\t%s\n' "$PROBLEM" "$END"; } \
  | sort >"$LASTRUN.tmp" && mv "$LASTRUN.tmp" "$LASTRUN"

echo "[$(date -Is)] END   $PROBLEM  rc=$RC  ${MINS}min  iters=$ITER  speedup=$SPEEDUP"
exit $RC
