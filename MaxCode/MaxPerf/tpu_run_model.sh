# server
USE_BATCHED_RPA_KERNEL=1 MODEL_IMPL_TYPE=vllm vllm serve Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8 --max-model-len=9216 --max-num-batched-tokens=8192 --max-num-seqs=512 --kv-cache-dtype=fp8 --no-enable-prefix-caching --gpu-memory-utilization=0.8 --tensor-parallel-size=8 --download_dir=/mnt/disks/persist --async-scheduling --block-size=256 --enable-expert-parallel



# client
#!/bin/bash

DEFAULT_HOST=0.0.0.0
DEFAULT_PORT=8000

nc -zv $DEFAULT_HOST $DEFAULT_PORT
while [ $? -ne 0 ]; do
  echo "Waiting for the server to start..."
  sleep 15
  nc -zv $DEFAULT_HOST $DEFAULT_PORT
done


for config in "1024 1024" "8192 1024" "1024 8192"; do
  set -- $config
  echo "Running benchmark with input_len=$1 and output_len=$2"

  args=(
      --ignore-eos
      --model=Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8
      --backend=vllm
      --port=$DEFAULT_PORT
      --dataset-name=random
      --random-input-len=$1
      --random-output-len=$2
      --random-range-ratio=0.8
      --num-prompts=320
      --max-concurrency=64
      --request-rate=inf
      --ignore-eos
      --percentile-metrics='ttft,tpot,itl,e2el'
  )
  # git clone https://github.com/kimbochen/bench_serving.git
  python3 /home/kyuyeunk_google_com/workspace/bench_serving/benchmark_serving.py "${args[@]}"
done