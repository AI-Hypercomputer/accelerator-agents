#!/usr/bin/env bash
# Run the jaxbench_level2 ops, each in its OWN headless Claude conversation, on
# the GKE v6e cluster.
#
#   tools/nightly/run_jaxbench_level2.sh                       # all 15 ops
#   tools/nightly/run_jaxbench_level2.sh 51p_DeepSeek_V4_CSA   # a subset
#   MK_JOBS=4 tools/nightly/run_jaxbench_level2.sh
#
# Deliberately separate from run_one.sh / run_batch.sh, which resolve problems
# under problems/, pick an entry file by guesswork and target whatever TPU
# tpu_config.json happens to name. This one is pinned to jaxbench_level2/:
# reference.py is the source, kernel_task.yaml supplies every input config, and
# the TPU is always the GKE cluster.
#
# Knobs (env):
#   MK_JOBS      ops running concurrently                (default 4)
#   MK_TIMEOUT   per-op wall-clock cap                   (default 10h)
#   MK_DEADLINE  cap for the whole sweep                 (default 120h)
#   MK_MODEL     model alias passed to claude            (default opus)
#   MK_STATUS_INTERVAL  status-file refresh seconds      (default 60)
#   MK_BUDGET_USD  optional --max-budget-usd per op
set -uo pipefail

MK="${MK:-/home/cathygao_google_com/maxkernel}"
export MK
export PATH="$HOME/.local/bin:$PATH"

# xargs re-invokes this script for each op; resolve it absolutely so that works
# regardless of the caller's cwd.
SELF="$(readlink -f "$0")"

VENV_PY="${MK_VENV_PYTHON:-/home/cathygao_google_com/maxkernel_venv/bin/python}"
BENCH="$MK/jaxbench_level2"
NIGHTLY="$MK/_nightly"
MARKERS="$NIGHTLY/jaxbench_level2_runs"
RESULTS="$NIGHTLY/jaxbench_level2_results.csv"
LOGROOT="$NIGHTLY/logs/jaxbench_level2"
FORWARD="$MK/tools/nightly/gke_tpu_forward.sh"
STATUS_PY="$MK/tools/nightly/jaxbench_status.py"

TPU_PORT="${MK_TPU_PORT:-8000}"
GKE_CLUSTER="${MK_GKE_CLUSTER:-maxkernel-v6e-cluster}"
GKE_ZONE="${MK_GKE_ZONE:-us-east5-b}"
GKE_PROJECT="${MK_GKE_PROJECT:-tpu-prod-env-multipod}"

mkdir -p "$MARKERS" "$LOGROOT"

# --------------------------------------------------------------------------
# single-op mode (what xargs invokes)
# --------------------------------------------------------------------------
run_one_op() {
  local OP="$1"
  local OP_DIR="$BENCH/$OP"
  local ENTRY="$OP_DIR/reference.py"
  local TASK="$OP_DIR/kernel_task.yaml"

  [ -f "$ENTRY" ] || { echo "no reference.py for $OP" >&2; return 64; }
  [ -f "$TASK" ]  || { echo "no kernel_task.yaml for $OP" >&2; return 64; }

  # The forward is this run's only route to a TPU; never start an op without it.
  bash "$FORWARD" ensure || { echo "GKE forward down for $OP" >&2; return 69; }
  # Cheap when the packages are already there; re-run per op because a pod that
  # restarted mid-sweep comes back without them.
  bash "$FORWARD" deps || echo "[warn] $OP: extra deps not fully installed" >&2

  local RUN_ID RUN_DIR LOG N_CFG ATOL RTOL START
  RUN_ID="run_$(python3 -c 'import uuid;print(uuid.uuid4().hex[:8])')"
  RUN_DIR="$MK/workspace/$RUN_ID"
  mkdir -p "$RUN_DIR"
  LOG="$LOGROOT/$(date +%Y-%m-%d)/${OP}.${RUN_ID}.log"
  mkdir -p "$(dirname "$LOG")"

  # ---- pre-seed get_inputs.py from kernel_task.yaml ------------------------
  # Writing this here (rather than letting generate-test-file author it) is what
  # guarantees "all configs from kernel_task.yaml": the snippet already returns
  # a list of (dynamic_args, static_args), one per config, which is exactly the
  # shape tools/test_harness_template.py loops over.
  # kernel_task.yaml is also the only source of tolerances on this branch --
  # there is no job file for the skill to read -- so they go into the prompt.
  read -r N_CFG ATOL RTOL <<<"$(python3 - "$TASK" "$RUN_DIR/get_inputs.py" <<'PY'
import re, sys, yaml
task = yaml.safe_load(open(sys.argv[1]))
code = task["input_gen_code"]
if "def get_inputs" not in code:
    raise SystemExit("input_gen_code has no get_inputs()")
hdr = (
    "# Generated verbatim from kernel_task.yaml input_gen_code by\n"
    "# tools/nightly/run_jaxbench_level2.sh. Every config in the task file is\n"
    "# returned, so the harness checks correctness on all of them.\n"
)
open(sys.argv[2], "w").write(hdr + code.rstrip() + "\n")
print(len(re.findall(r"'name':", code)),
      task.get("atol", 1e-2), task.get("rtol", 1e-2))
PY
)"
  [ -n "${N_CFG:-}" ] && [ -s "$RUN_DIR/get_inputs.py" ] \
    || { echo "could not seed get_inputs.py for $OP" >&2; return 65; }

  # ---- per-run tpu_config.json: the GKE endpoint, not the local TPU --------
  # get_tpu_config() prefers <run_dir>/tpu_config.json whenever --run_dir is
  # passed, so this overrides the repo-level file without editing it. The
  # cluster hands each job a single v6e chip (TPU_VISIBLE_CHIPS=2,
  # TPU_CHIPS_PER_PROCESS_BOUNDS=1,1,1), so device_count is 1 here even though
  # the local VM's config says 8.
  cat >"$RUN_DIR/tpu_config.json" <<EOF
{
  "tpus": [
    {
      "mode": "local",
      "local_port": $TPU_PORT,
      "tpu_version": "TPU v6e",
      "tpu_spec": {
        "device_kind": "TPU v6 lite",
        "device_count": 1
      }
    }
  ]
}
EOF

  START=$(date +%s)
  python3 - "$MARKERS/$OP.json" <<PY
import json, sys
json.dump({
  "op": "$OP", "run_id": "$RUN_ID", "run_dir": "$RUN_DIR", "log": "$LOG",
  "started_at": $START, "finished_at": None, "pid": None,
  "status": "running", "exit_code": None, "configs": $N_CFG,
}, open(sys.argv[1], "w"), indent=2)
PY
  python3 "$STATUS_PY" >/dev/null 2>&1

  echo "[$(date -Is)] START $OP  run_id=$RUN_ID  configs=$N_CFG  log=$LOG"

  read -r -d '' PROMPT <<EOF
Optimize this into a faster Pallas TPU kernel. Use the maxkernel skill.

Source file: $ENTRY
Problem directory: $OP_DIR
Task spec: $TASK

The source is already a JAX/Pallas module, so it IS the baseline: copy it to
$RUN_DIR/base.py unchanged. Its module-level entry point is \`computation\`
(in some ops bound as \`computation = workload\` rather than defined with
\`def\`; both are fine). There is nothing to port and no golden capture.

Use run_dir = $RUN_DIR for this run. It already exists: do NOT generate a new
run id, and write $RUN_DIR/state.json there.

CORRECTNESS ON ALL CONFIGS. $TASK declares $N_CFG input
configs. $RUN_DIR/get_inputs.py has already been written verbatim from that
file's input_gen_code and returns all $N_CFG of them as a list of
(dynamic_args, static_args) tuples. Do NOT rewrite it, do NOT reduce it to one
config, and do NOT change any shape or dtype in it. The optimized kernel must
be correct on every one of the $N_CFG configs; a kernel that only works on a
subset is a failed iteration. Skip the generate-test-file step and assemble the
harness from the get_inputs.py that is already there:
  $VENV_PY $MK/tools/assemble_test_harness.py \\
    $RUN_DIR/base.py $RUN_DIR/get_inputs.py $RUN_DIR/test_kernel.py \\
    --atol $ATOL --rtol $RTOL
Those tolerances come from $TASK; do not loosen them.

TPU: use the GKE cluster, not the local TPU VM.
  gcloud container clusters get-credentials $GKE_CLUSTER --zone=$GKE_ZONE --project=$GKE_PROJECT
A kubectl port-forward is already live and binds 127.0.0.1:$TPU_PORT to that
cluster's maxkernel-tpu-service. $RUN_DIR/tpu_config.json points at it, so pass
--run_dir $RUN_DIR on every tools/tpu_client.py call and every TPU job will run
on cluster hardware. Do NOT start a local TPU server, do NOT run
tpu_client.py --add_tpu, and do NOT fall back to the local VM's TPU. If
127.0.0.1:$TPU_PORT/health stops answering, run
  bash $FORWARD ensure
to rebuild the forward, then carry on. Each cluster job sees ONE v6e chip, so
plan and size grids for a single device.

Use $VENV_PY as the Python interpreter.

The baseline here is ALREADY an optimized Pallas kernel, not idiomatic JAX.
Beating it is genuinely hard: a final speedup of ~1.0x is an acceptable,
honest outcome. Report the measured number, whatever it is. Never edit
get_inputs.py, loosen the tolerances, or drop a config to manufacture a win.

STOPPING. Do not run to a fixed iteration count. Let the measured stopping
criteria in general_stopping.md end the run: stop when state.stopping
.recommend_stop is true for the iteration that just finished. Converging in two
iterations is a success, not a short run, and there is no budget to fill.
Equally, do not stop early on a hunch while the criteria still say continue.

DO NOT END YOUR TURN UNTIL THE LOOP IS FINISHED. This is a headless
\`claude -p\` process: the moment you stop producing output the process exits
and everything in flight dies. Nothing will resume you, and there is no one to
hand off to. Never end a turn with "iteration 1 is running", "the worker is
going now", "I'll report when it returns" or any other handoff -- that ends the
run at iteration 0 and the whole op is wasted. Dispatch the worker subagent,
WAIT for its result, act on it, dispatch the next iteration, and keep going in
one continuous turn. You may only finish after state.stopping.recommend_stop is
true for the iteration that just finished (or the loop is otherwise genuinely
complete), state.json records a real best_optimized_time, and you have written
the final summary.

This is an unattended batch run. There is no human watching, so never stop to
ask a question: if something is ambiguous, pick the most reasonable option,
record the choice in $RUN_DIR/maxkernel_debug_history.md, and keep going, then
end with the normal final summary.
EOF

  cd "$MK" || return 66
  local RC
  if [ -n "${MK_DRY_RUN:-}" ]; then
    # Exercises everything except the conversation: seeding, tpu_config,
    # markers, CSV, status. Use it to check the wiring before a multi-day sweep.
    { echo "[dry-run] would launch: claude -p --model ${MK_MODEL:-opus}"
      echo "--- prompt ---"; echo "$PROMPT"; } >>"$LOG"
    RC=0
  else
    timeout --signal=INT --kill-after=5m "${MK_TIMEOUT:-10h}" \
      claude -p "$PROMPT" \
        --model "${MK_MODEL:-opus}" \
        --dangerously-skip-permissions \
        --permission-prompts none \
        --output-format stream-json --verbose \
        ${MK_BUDGET_USD:+--max-budget-usd "$MK_BUDGET_USD"} \
        </dev/null >>"$LOG" 2>&1
    RC=$?
  fi

  local END MINS
  END=$(date +%s)
  MINS=$(( (END - START) / 60 ))

  # ---- harvest from state.json, not from the model's prose ----------------
  read -r ITER SPEEDUP BASE_MS OPT_MS NHIST BEST <<<"$(python3 - "$RUN_DIR/state.json" <<'PY'
import json, sys
try:
    s = json.load(open(sys.argv[1]))
except Exception:
    print("0 NA NA NA 0 NA"); raise SystemExit
def g(k):
    v = s.get(k)
    return "NA" if v is None else v
print(s.get("iteration", 0), g("best_speedup"), g("base_time_ms"),
      g("best_optimized_time"), len(s.get("history") or []), g("best_code_path"))
PY
)"

  local STATUS="done"
  if [ "$RC" -eq 124 ] || [ "$RC" -eq 137 ]; then
    STATUS="timeout"
  elif [ "$RC" -ne 0 ]; then
    STATUS="failed"
  elif [ "${ITER:-0}" -eq 0 ] || [ "${NHIST:-0}" -eq 0 ]; then
    # rc=0 but nothing was optimized. The orchestrator ended its turn after
    # setup -- typically narrating "iteration 1 is running" -- which exits the
    # headless process at iteration 0. Exiting cleanly is not the same as
    # finishing, so do not let it be recorded as done.
    STATUS="incomplete"
  fi

  [ -f "$RESULTS" ] || echo "finished_at,op,run_id,exit_code,minutes,iterations,configs,best_speedup,base_time_ms,best_optimized_time,best_code_path,log" >"$RESULTS"
  echo "$(date -Is),$OP,$RUN_ID,$RC,$MINS,$ITER,$N_CFG,$SPEEDUP,$BASE_MS,$OPT_MS,$BEST,$LOG" >>"$RESULTS"

  python3 - "$MARKERS/$OP.json" "$END" "$RC" "$STATUS" <<'PY'
import json, sys
p = sys.argv[1]
m = json.load(open(p))
m["finished_at"] = int(sys.argv[2])
m["exit_code"] = int(sys.argv[3])
m["status"] = sys.argv[4]
json.dump(m, open(p, "w"), indent=2)
PY
  python3 "$STATUS_PY" >/dev/null 2>&1

  echo "[$(date -Is)] END   $OP  rc=$RC  ${MINS}min  iters=$ITER  speedup=$SPEEDUP"
  return $RC
}

if [ "${1:-}" = "--one" ]; then
  shift
  run_one_op "${1:?usage: --one <op>}"
  exit $?
fi

# --------------------------------------------------------------------------
# sweep mode
# --------------------------------------------------------------------------
JOBS="${MK_JOBS:-4}"
DEADLINE="${MK_DEADLINE:-120h}"

exec 9>"$NIGHTLY/.jaxbench_level2.lock"
if ! flock -n 9; then
  echo "[$(date -Is)] a jaxbench_level2 sweep is already running; exiting"
  exit 0
fi

if [ "$#" -gt 0 ]; then
  SELECTED=("$@")
else
  mapfile -t SELECTED < <(
    find "$BENCH" -maxdepth 2 -name kernel_task.yaml -printf '%h\n' \
      | sed "s#^$BENCH/##" | sort
  )
fi
[ "${#SELECTED[@]}" -gt 0 ] || { echo "nothing to run"; exit 0; }

for OP in "${SELECTED[@]}"; do
  [ -f "$BENCH/$OP/kernel_task.yaml" ] || {
    echo "unknown op: $OP (not under $BENCH)" >&2; exit 64; }
done

# Every op must be seedable before a multi-day sweep commits to the set: a
# kernel_task.yaml whose input_gen_code has no get_inputs() would only surface
# hours in, after the ops ahead of it had finished.
python3 - "$BENCH" "${SELECTED[@]}" <<'PY' || exit 65
import pathlib, sys, yaml
bench = pathlib.Path(sys.argv[1])
bad = []
for op in sys.argv[2:]:
    try:
        task = yaml.safe_load((bench / op / "kernel_task.yaml").read_text())
        if "def get_inputs" not in (task.get("input_gen_code") or ""):
            bad.append(f"{op}: input_gen_code has no get_inputs()")
    except Exception as e:
        bad.append(f"{op}: {e}")
for b in bad:
    print("unusable:", b, file=sys.stderr)
sys.exit(1 if bad else 0)
PY

bash "$FORWARD" ensure || { echo "cannot reach the GKE TPU service" >&2; exit 69; }
bash "$FORWARD" deps  || echo "[warn] extra deps not fully installed" >&2

# Plain `&`, not setsid: these two are the sweep's own helpers and should die
# with it. Detachment is the launcher's job -- start the sweep itself under
# setsid (see the README) or it dies with the shell that started it.
bash "$FORWARD" supervise >/dev/null 2>&1 </dev/null &
FORWARD_SUP=$!

"${MK_STATUS_PYTHON:-python3}" "$STATUS_PY" \
  --watch "${MK_STATUS_INTERVAL:-60}" >/dev/null 2>&1 </dev/null &
STATUS_SUP=$!

cleanup() {
  kill "$FORWARD_SUP" "$STATUS_SUP" 2>/dev/null
  python3 "$STATUS_PY" >/dev/null 2>&1
}
trap cleanup EXIT

echo "[$(date -Is)] jaxbench_level2 sweep: ${#SELECTED[@]} op(s), ${JOBS} at a time"
printf '  %s\n' "${SELECTED[@]}"
echo "  status file: $NIGHTLY/JAXBENCH_LEVEL2_STATUS.md"

printf '%s\n' "${SELECTED[@]}" \
  | timeout --signal=INT --kill-after=10m "$DEADLINE" \
      xargs -r -P "$JOBS" -I{} "$SELF" --one {}

echo "[$(date -Is)] sweep done"
echo "results: $RESULTS"
