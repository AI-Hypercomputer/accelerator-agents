#!/usr/bin/env bash
# Entrypoint script for running Claude Code headlessly inside a pod on a single TPU device.
set -euo pipefail

MK_ROOT="${MK_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MODEL_NAME="${MODEL_NAME:-opus}"
TPU_PORT="${TPU_PORT:-5463}"
TIMEOUT="${TIMEOUT:-10h}"
OUTPUT_DIR="${OUTPUT_DIR:-${MK_ROOT}/workspace/_pod_run}"

DRY_RUN=false
PROBLEM_DIR=""

while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    --problem-dir)
      if [ $# -lt 2 ]; then
        echo "Error: --problem-dir requires a path argument" >&2
        exit 1
      fi
      PROBLEM_DIR="$2"
      shift 2
      ;;
    --problem-dir=*)
      PROBLEM_DIR="${1#*=}"
      shift
      ;;
    --output-dir)
      if [ $# -lt 2 ]; then
        echo "Error: --output-dir requires a path argument" >&2
        exit 1
      fi
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --output-dir=*)
      OUTPUT_DIR="${1#*=}"
      shift
      ;;
    --model|--model-name)
      if [ $# -lt 2 ]; then
        echo "Error: $1 requires a model name argument" >&2
        exit 1
      fi
      MODEL_NAME="$2"
      shift 2
      ;;
    --model=*|--model-name=*)
      MODEL_NAME="${1#*=}"
      shift
      ;;
    --tpu-port)
      if [ $# -lt 2 ]; then
        echo "Error: --tpu-port requires a port argument" >&2
        exit 1
      fi
      TPU_PORT="$2"
      shift 2
      ;;
    --tpu-port=*)
      TPU_PORT="${1#*=}"
      shift
      ;;
    --timeout)
      if [ $# -lt 2 ]; then
        echo "Error: --timeout requires a duration argument" >&2
        exit 1
      fi
      TIMEOUT="$2"
      shift 2
      ;;
    --timeout=*)
      TIMEOUT="${1#*=}"
      shift
      ;;
    -h|--help)
      echo "Usage: $0 [--dry-run] [--problem-dir <path>] [positional_problem_path]"
      exit 0
      ;;
    -*)
      echo "Error: Unknown option: $1" >&2
      exit 1
      ;;
    *)
      if [ -z "$PROBLEM_DIR" ]; then
        PROBLEM_DIR="$1"
        shift
      else
        echo "Error: Unexpected positional argument: $1" >&2
        exit 1
      fi
      ;;
  esac
done

if [ -z "$PROBLEM_DIR" ]; then
  echo "Error: Missing required problem directory. Provide --problem-dir <path> or positional problem path." >&2
  exit 1
fi

if [ ! -d "$PROBLEM_DIR" ]; then
  echo "Error: Problem directory not found: $PROBLEM_DIR" >&2
  exit 1
fi

# Locate the entry source file
ENTRY=""
for candidate in baseline.py reference.py kernel.py source.py; do
  if [ -f "$PROBLEM_DIR/$candidate" ]; then
    ENTRY="$PROBLEM_DIR/$candidate"
    break
  fi
done

if [ -z "$ENTRY" ]; then
  ENTRY="$(find "$PROBLEM_DIR" -maxdepth 1 -name '*.py' ! -name '__init__.py' 2>/dev/null | sort | head -1 || true)"
fi

if [ -z "$ENTRY" ]; then
  echo "Error: No Python entry file found in problem directory: $PROBLEM_DIR" >&2
  exit 1
fi

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

read -r -d '' PROMPT <<EOF || true
Optimize this code into a Pallas TPU kernel. Use the maxkernel skill.

Source file: $ENTRY
Problem directory (read it for context): $PROBLEM_DIR
$JOB_LINE
$TASK_YAML_LINE

Use run_dir = $OUTPUT_DIR for this run. That directory already exists:
do NOT generate a new run id, and write $OUTPUT_DIR/state.json there.

Use the TPU described in tpu_config.json.

This is an unattended batch run. There is no human watching, so never stop to
ask a question: if something is ambiguous, pick the most reasonable option,
record the choice in $OUTPUT_DIR/maxkernel_debug_history.md, and keep going. Run
the loop until it stops on its own criteria (state.stopping.recommend_stop) or
exhausts max_iterations, then end with the normal final summary. Converging in
two iterations is a success, not a short run -- do not keep going to fill the
budget.
EOF

TPU_SERVER_PID=""

cleanup() {
  local exit_code=$?
  if [ -n "$TPU_SERVER_PID" ] && kill -0 "$TPU_SERVER_PID" 2>/dev/null; then
    echo "Stopping background TPU server (PID: $TPU_SERVER_PID)..."
    kill "$TPU_SERVER_PID" 2>/dev/null || true
    wait "$TPU_SERVER_PID" 2>/dev/null || true
  fi
  exit "$exit_code"
}
trap cleanup EXIT INT TERM

check_tpu_health() {
  curl -s -f "http://127.0.0.1:${TPU_PORT}/health" >/dev/null 2>&1 || \
  python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:${TPU_PORT}/health', timeout=1)" >/dev/null 2>&1
}

# 1. Check node and claude CLI are in PATH
if [ "$DRY_RUN" = "true" ]; then
  echo "[dry-run] Checking node and claude CLI in PATH"
else
  command -v node >/dev/null 2>&1 || { echo "Error: node not found in PATH" >&2; exit 1; }
  command -v claude >/dev/null 2>&1 || { echo "Error: claude CLI not found in PATH" >&2; exit 1; }
fi

# 2. Detect if Vertex AI env vars are set
if [ "${CLAUDE_CODE_USE_VERTEX:-0}" = "1" ] || [ "${CLAUDE_CODE_USE_VERTEX:-}" = "true" ]; then
  echo "Detected Vertex AI configuration (CLAUDE_CODE_USE_VERTEX=${CLAUDE_CODE_USE_VERTEX})"
fi

# 3. Start local TPU daemon if not already healthy
if [ "$DRY_RUN" = "true" ]; then
  echo "[dry-run] Checking TPU server health on 127.0.0.1:${TPU_PORT}/health"
  echo "[dry-run] Starting TPU daemon if unhealthy: python3 ${MK_ROOT}/server/tpu_server.py --port=${TPU_PORT} &"
  echo "[dry-run] Polling 127.0.0.1:${TPU_PORT}/health until healthy (up to 30s)"
else
  if ! check_tpu_health; then
    echo "Starting TPU server on port ${TPU_PORT}..."
    python3 "${MK_ROOT}/server/tpu_server.py" --port="${TPU_PORT}" &
    TPU_SERVER_PID=$!

    healthy=0
    for _ in $(seq 1 30); do
      if check_tpu_health; then
        healthy=1
        echo "TPU server is healthy on port ${TPU_PORT}"
        break
      fi
      sleep 1
    done

    if [ "$healthy" -ne 1 ]; then
      echo "Error: TPU server failed to become healthy on port ${TPU_PORT} within 30s" >&2
      exit 1
    fi
  else
    echo "TPU server is already healthy on port ${TPU_PORT}"
  fi
fi

# 4. Run installer
if [ "$DRY_RUN" = "true" ]; then
  echo "[dry-run] Running installer: python3 ${MK_ROOT}/install.py --with-guard"
else
  python3 "${MK_ROOT}/install.py" --with-guard
fi

# 5. Point tpu_config.json at 127.0.0.1:${TPU_PORT}
if [ "$DRY_RUN" = "true" ]; then
  echo "[dry-run] Pointing tpu_config.json at 127.0.0.1:${TPU_PORT}"
else
  mkdir -p "${OUTPUT_DIR}"
  cat >"${OUTPUT_DIR}/tpu_config.json" <<EOF
{
  "tpus": [
    {
      "mode": "local",
      "local_port": ${TPU_PORT},
      "tpu_version": "TPU v6e",
      "tpu_spec": {
        "device_kind": "TPU v6 lite",
        "device_count": 1
      },
      "tpu_spec_verified": true
    }
  ]
}
EOF
  cp "${OUTPUT_DIR}/tpu_config.json" "${MK_ROOT}/tpu_config.json" 2>/dev/null || true
fi

# 6. Execute claude or print dry-run command sequence
if [ "$DRY_RUN" = "true" ]; then
  echo "[dry-run] Command sequence:"
  echo "[dry-run]   timeout --signal=INT --kill-after=5m ${TIMEOUT} claude -p \"$PROMPT\" --model ${MODEL_NAME} --dangerously-skip-permissions --permission-prompts none --output-format stream-json --verbose > ${OUTPUT_DIR}/claude_stream.json 2>&1"
  echo "[dry-run]   python3 ${MK_ROOT}/tools/export_run_metrics.py --run_dir ${OUTPUT_DIR} --output ${OUTPUT_DIR}/summary_metrics.json"
else
  mkdir -p "${OUTPUT_DIR}"
  echo "Starting headless Claude Code run for problem: ${PROBLEM_DIR}"
  set +e
  timeout --signal=INT --kill-after=5m "${TIMEOUT}" \
    claude -p "$PROMPT" \
      --model "$MODEL_NAME" \
      --dangerously-skip-permissions \
      --permission-prompts none \
      --output-format stream-json --verbose \
      </dev/null >"${OUTPUT_DIR}/claude_stream.json" 2>&1
  CLAUDE_RC=$?
  set -e

  echo "Claude Code execution finished with exit code ${CLAUDE_RC}"

  # 7. Export run metrics
  if [ -f "${OUTPUT_DIR}/state.json" ]; then
    python3 "${MK_ROOT}/tools/export_run_metrics.py" \
      --run_dir "${OUTPUT_DIR}" \
      --output "${OUTPUT_DIR}/summary_metrics.json"
  else
    echo "Warning: No state.json found at ${OUTPUT_DIR}/state.json; skipping metrics export." >&2
  fi

  exit "$CLAUDE_RC"
fi
