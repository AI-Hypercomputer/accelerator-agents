#!/usr/bin/env bash
# run_level0.sh — Automation script to run Level 0 (Phase 0) baseline benchmark on remote TPU VM.
set -ex

VENV_DIR="$HOME/vllm_env"
PORT=8000

# Clone benchmarking client if not present
if [ ! -d "$HOME/bench_serving" ]; then
  echo "Cloning bench_serving client..."
  git clone https://github.com/kimbochen/bench_serving.git "$HOME/bench_serving"
fi

# Activate virtual environment
source "$VENV_DIR/bin/activate"

export TPU_MULTIHOST_BACKEND=ray
export RAY_ADDRESS="auto"

# Enable persistent JAX compiler caching to skip 45-minute compile cycles on restarts
export JAX_COMPILER_CACHE_DIR="$HOME/.jax_cache"
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=-1
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0

if [ -n "${PHASED_PROFILING_DIR:-}" ]; then
  export PHASED_PROFILING_DIR
fi

echo "=== Starting Qwen2.5-Coder-3B vLLM Server in Dummy Mode ==="
# Change directory to root (/) to avoid Python importing from local vllm source directory
cd /
BLOCK_SIZE=${BLOCK_SIZE:-""}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-4}
KV_CACHE_DTYPE=${KV_CACHE_DTYPE:-"auto"}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
NUM_PROMPTS=${NUM_PROMPTS:-320}
MAX_CONCURRENCY=${MAX_CONCURRENCY:-64}

TP_SIZE=${TP_SIZE:-8}

server_args=(
    --model /home/gvanica_google_com/qwen3-8b-dummy
    --load-format dummy
    --tensor-parallel-size ${TP_SIZE}
    --port ${PORT}
    --max-model-len ${MAX_MODEL_LEN}
    --max-num-seqs ${MAX_NUM_SEQS}
    --kv-cache-dtype ${KV_CACHE_DTYPE}
    --gpu-memory-utilization 0.95
    --distributed-executor-backend ray
)

if [ -n "$BLOCK_SIZE" ]; then
    server_args+=(--block-size "$BLOCK_SIZE")
fi

python3 -m vllm.entrypoints.openai.api_server "${server_args[@]}" > "$HOME/vllm_server.log" 2>&1 &

SERVER_PID=$!

# Wait for server to start and become completely ready
echo "Waiting for the vLLM server to be fully ready on port ${PORT}..."
sleep 30
while ! curl -s --fail http://localhost:${PORT}/health > /dev/null; do
  echo "Server is still compiling graph overlays. Waiting..."
  sleep 30
done
echo "vLLM server is UP and listening!"

# Run Benchmark mix
REGIMES=(
  "Balanced 1024 1024"
)

for regime in "${REGIMES[@]}"; do
  read -r name input_len output_len <<< "$regime"
  echo "========================================================="
  echo "Running benchmark: $name (input_len=$input_len, output_len=$output_len)"
  echo "========================================================="

  args=(
      --ignore-eos
      --model=/home/gvanica_google_com/qwen3-8b-dummy
      --backend=vllm
      --port=${PORT}
      --dataset-name=random
      --random-input-len=${input_len}
      --random-output-len=${output_len}
      --random-range-ratio=0.8
      --num-prompts=${NUM_PROMPTS}
      --max-concurrency=${MAX_CONCURRENCY}
      --request-rate=inf
      --percentile-metrics='ttft,tpot,itl,e2el'
  )

  python3 "$HOME/bench_serving/benchmark_serving.py" "${args[@]}" > "$HOME/benchmark_${name}.log" 2>&1
  cat "$HOME/benchmark_${name}.log"
done

# Generate/verify numeric equivalence reference if run_numeric_equiv.py exists
if [ -f "$HOME/run_numeric_equiv.py" ]; then
  echo "=== Running Numeric Equivalence (Generate/Verify) ==="
  if [ ! -f "$HOME/reference_outputs/prompt_0.json" ]; then
    echo "No reference outputs found. Generating references..."
    python3 "$HOME/run_numeric_equiv.py" --generate --port ${PORT}
  else
    echo "Reference outputs found. Comparing..."
    python3 "$HOME/run_numeric_equiv.py" --compare --port ${PORT}
  fi
fi

echo "=== Terminating vLLM Server ==="
kill -9 $SERVER_PID || true
ray stop --force || true

echo "=== Level 0 Baseline Run Complete! ==="
