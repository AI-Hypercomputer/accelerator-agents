# Deploying the trainer to GKE (XPK)

Generic instructions for running the Llama 3 8B torchax trainer on a multi-host
TPU GKE cluster managed by [XPK](https://github.com/AI-Hypercomputer/xpk).
Cluster details (project, zone, cluster name) are intentionally omitted —
substitute the values for your environment.

## 0. Prerequisites

- A GKE cluster with TPU node-pools (created via `xpk cluster create` or
  similar). The trainer is tested on **v6e-8 (2 hosts × 4 chips, single 2x4
  topology slice)** but the same image works on any v6e/v7x slice topology
  where `global_batch_size % fsdp_size == 0`.
- A GCS bucket reachable from the cluster, optionally mounted into the pod via
  XPK's `--storage=<gcsfuse-storage-name>` flag for the persistent compile
  cache and HF model cache.
- An Artifact Registry repo to push the trainer image into.
- Local installs: `gcloud`, `kubectl`, `docker`, `xpk` (`pip install xpk`).

## 1. Build the trainer image

The Dockerfile expects two top-level directories in the build context:
- `trainer/` — this folder (the trainer code).
- `torchax_src/` — a copy of the torchax repository at the version the wiki
  pinned (the submodule under `raw/code/torchax/` of this repo).

Stage them and build:

```bash
WIKI_ROOT="<path-to-this-wiki-checkout>"
CTX="$(mktemp -d)/llama3-trainer-build"
mkdir -p "$CTX/trainer" "$CTX/torchax_src"
rsync -a "$WIKI_ROOT/wiki/experiments/llama3_8B_autoresearch_optimization/torchax/" \
  --exclude='__pycache__' --exclude=experiments --exclude='*.md' \
  "$CTX/trainer/"
rsync -a "$WIKI_ROOT/raw/code/torchax/" \
  --exclude='__pycache__' --exclude='.git' --exclude='*.egg-info' \
  "$CTX/torchax_src/"
cp "$WIKI_ROOT/wiki/experiments/llama3_8B_autoresearch_optimization/torchax/Dockerfile" \
  "$CTX/Dockerfile"

cd "$CTX"
IMAGE="<your-region>-docker.pkg.dev/<your-project>/<your-repo>/llama3-8b-torchax:latest"
docker build -t "$IMAGE" .
docker push "$IMAGE"
```

## 2. Pre-stage the model + dataset to GCS (optional)

If you want to avoid download contention across pods (gcsfuse can race when
multiple hosts write the same `refs/main` commit-hash file in the HF cache),
download the HF cache locally once and rsync it into the bucket the cluster
mounts. The trainer will then read from cache without network access.

```bash
HF_TOKEN=<your-token> python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B')
AutoModelForCausalLM.from_pretrained('meta-llama/Meta-Llama-3-8B', torch_dtype='bfloat16')
"
gsutil -m rsync -r ~/.cache/huggingface gs://<your-bucket>/cache/huggingface
```

Same applies for the wikitext dataset (small, but each pod hits it).

## 3. Submit the workload

The example below targets **v6e-8** (single 2x4 slice). For other topologies,
adjust `--tpu-type` and `--batch_size` so `global_batch = batch_size × fsdp`
is a multiple of `fsdp_size = jax.device_count() / tp_parallelism`.

```bash
W=llama3-8b-baseline-$(date +%Y%m%d-%H%M%S)
IMAGE="<your-region>-docker.pkg.dev/<your-project>/<your-repo>/llama3-8b-torchax:latest"
STORAGE="<your-gcsfuse-storage-name>"   # gcsfuse mount, e.g. /data
HF_TOKEN="<your-hf-token>"

xpk workload create \
  --workload="$W" \
  --cluster=<your-cluster-name> \
  --zone=<your-zone> \
  --project=<your-project> \
  --tpu-type=v6e-8 --num-slices=1 \
  --docker-image="$IMAGE" \
  --storage="$STORAGE" \
  --env=WORKERS_0_HOSTNAME="$W-slice-job-0-0.$W" \
  --env=LIBTPU_INIT_ARGS=--xla_tpu_scoped_vmem_limit_kib=65536 \
  --env=HF_TOKEN="$HF_TOKEN" \
  --env=HF_HOME=/data/cache/huggingface \
  --env=HF_DATASETS_CACHE=/tmp/hf_datasets_cache \
  --env=JAX_COMPILATION_CACHE_DIR=/data/cache/xla \
  --command "cd /app/trainer && python -u train.py \
              --model_id=meta-llama/Meta-Llama-3-8B \
              --batch_size=4 --seqlen=1024 \
              --train_steps=15 --weights_dtype=bf16"
```

Notes:
- **`--storage`**: XPK mounts a GCS bucket into the pod at `/data`. The trainer
  uses `/data/cache/xla` (compile cache) and `/data/cache/huggingface`
  (HF model cache) so subsequent submissions hit a warm cache.
- **`HF_DATASETS_CACHE=/tmp/...`**: per-pod local FS — avoids gcsfuse
  filelock contention when both hosts try to materialize the dataset at once.
- **`global_batch = batch_size × fsdp`**: on v6e-8 (fsdp=8), `--batch_size=4`
  → global=32. On v6e-4 (fsdp=4), `--batch_size=4` → global=16. The trainer
  asserts divisibility — invalid combos raise at trace time.
- **`WORKERS_0_HOSTNAME`** must literally match `$W-slice-job-0-0.$W` so the
  TPU runtime's distributed coordinator finds rank 0.

## 4. Monitor

```bash
# Wait for both pods to enter Running.
kubectl get pods -l jobset.sigs.k8s.io/jobset-name="$W" -w

# Stream rank-0 (jax-tpu) logs.
POD0=$(kubectl get pods -o name | grep "$W-slice-job-0-0-" | head -1 | cut -d/ -f2)
kubectl logs -f -c jax-tpu "$POD0"
```

Healthy progression printed by the trainer:

```
[dist] global_devices=8 local_devices=4 hosts=2
[mesh] fsdp=8 tp=1 mesh=Mesh(...)
[load] model_id=meta-llama/Meta-Llama-3-8B weights_dtype=bf16 (meta init)
[load] model has 8.03 B parameters
[shard] model.embed_tokens.weight (128256, 4096) torch.bfloat16 -> ('fsdp', 'tp')
... (per-weight shard)
[data] wikitext-2-raw-v1, global_batch=32, seqlen=1024
[train] starting...
program size: 1.22 m chars      ← cold compile fingerprint
End compiling 178.4 s            ← cold compile time (cache miss)
                                 ← or End compiling ~10 s (cache hit)
[step  0/15] loss=10.83 step_time=178426ms throughput=...
[step  1/15] loss=10.31 step_time=350ms throughput=...
... (steady state)
================ summary ================
global_batch    : 32
seqlen          : 1024
steps measured  : 13
avg throughput  : <X> tok/s (<Y>/chip)
approx MFU      : <Z>% (v6e bf16 peak)
==========================================
```

## 5. Compile cache reuse

The first submission populates `gs://<your-bucket>/cache/xla/` (mounted in-pod
at `/data/cache/xla`). Subsequent submissions on the **same mesh shape and
program** skip the cold compile entirely — first step jumps from ~3 min to
~10-30 s. The cache key includes:
- HLO program bytes (so changing config or trainer code produces a fresh entry)
- Mesh device count + topology
- libtpu / JAX version

So bumping any of these invalidates entries; the cache grows additively.

## 6. Kill / cleanup

```bash
xpk workload delete --workload="$W" \
  --cluster=<your-cluster-name> --zone=<your-zone> --project=<your-project>
```

## Common failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| `Sharding ... implies that array axis 0 is partitioned N times, but the dimension size is M` at `start training` | `batch_size × fsdp` is not divisible by `fsdp_size` | Bump `--batch_size` so global batch ≥ fsdp_size and divisible |
| `OSError: [Errno 116] Stale file handle` on HF download | gcsfuse race on `refs/main` write — both hosts try simultaneously | Pre-cache HF model to GCS (step 2) and/or set `HF_HUB_OFFLINE=1` once cached |
| Silent `EXIT_CODE=1` after `Initializing weight ...` and before any step | Older JAX coordinator port mismatch (JAX 0.8.1+ `8482` vs GCP firewall `8476`) | Image uses a JAX version where this is patched; if it recurs, set `JAX_DISTRIBUTED_INITIALIZATION_TIMEOUT` higher |
| `RESOURCE_EXHAUSTED` HBM at step 1 | Activations + Adam state exceed per-chip 32 GB | Lower `--batch_size` or `--seqlen`, or add gradient checkpointing |
| `ValueError: ... could not be resolved unambiguously` on a gather op | JAX `Explicit` axis types | The trainer already uses `axis_types=Auto`; if you fork it, do the same |
