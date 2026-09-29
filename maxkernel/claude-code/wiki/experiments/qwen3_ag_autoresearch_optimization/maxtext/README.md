# Qwen3 8B — maxtext lane

This lane uses the native MaxText implementation of Qwen3 8B to provide a baseline ceiling to compare against the torchax and native JAX ports. MaxText represents Google's highly optimized, scalable LLM codebase written in pure Python/JAX targeting TPUs.

## How to run (smoke test)

The maxtext code lives in `raw/code/maxtext/`.

```bash
cd raw/code/maxtext

# Set up variables
export RUN_NAME="2026-06-05-qwen3-maxtext-v001-baseline"
export BASE_OUTPUT_DIRECTORY="gs://<your-bucket>/autoresearch/qwen3_cc/"

# Launch a workload via XPK
xpk workload create \
  --workload="alekseyv-qwen3-maxtext-v001" \
  --cluster="<your-cluster>" \
  --project="<your-project>" \
  --zone="<your-region>" \
  --tpu-type="v6e-8" \
  --base-docker-image="python:3.12" \
  --command="bash src/dependencies/scripts/setup.sh && python3 src/maxtext/trainers/pre_train/train.py src/maxtext/configs/base.yml run_name=$RUN_NAME base_output_directory=$BASE_OUTPUT_DIRECTORY model_name=qwen3-8b dataset_type=synthetic per_device_batch_size=1 max_target_length=8192 enable_checkpointing=false steps=12 profiler=xplane"
```

Note: A docker container containing maxtext dependencies may need to be built and specified if the default setup.sh doesn't work, but setup.sh is generally sufficient.

## Recent issues / debugging notes
