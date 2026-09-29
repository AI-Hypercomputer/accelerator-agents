# Qwen3 8B torchax trainer — TPU image.
#
# Adapted from the llama3_8B torchax lane Dockerfile. Same base + layering;
# differences: qwen3 trainer code, `google-cloud-storage` added (for the HLO
# uploader `upload_dir.py`), and TORCH_DEVICE_BACKEND_AUTOLOAD=0 baked in
# (defensive — a clean jax-ai-image has no torch_tpu, but py312 dev boxes do and
# the flag is harmless when torch_tpu is absent).
#
# Base: jax-ai-image/tpu (Cloud TPU images, JAX 0.9 + libtpu).
#
# Build context: this directory plus the torchax source tree mounted as
# `torchax_src/` (parent's `raw/code/torchax/`). See DEPLOY.md for the build cmd.

FROM us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu:jax0.9.0-rev1

# Layer 1 — extra Python deps. transformers latest carries Qwen3.
RUN pip install --no-cache-dir \
    transformers \
    accelerate \
    datasets \
    optax \
    fire \
    pyyaml \
    huggingface_hub \
    google-cloud-storage

# Layer 2 — torchax (install from local submodule source).
WORKDIR /torchax
COPY torchax_src /torchax
RUN pip install -e .

# Layer 3 — trainer code (train.py, model/, data.py, helper.py, profiling.py,
# upload_dir.py, config.yaml).
WORKDIR /app/trainer
COPY trainer /app/trainer

# Layer 4 — runtime defaults.
#  - PYTHONUNBUFFERED so streaming logs work under kubectl logs -f.
#  - PYTHONPATH so `import helper` / `import model` / `import profiling` resolve.
#  - TORCH_DEVICE_BACKEND_AUTOLOAD=0 so torchax wins the privateuse1 backend.
#  - HF_HOME / JAX_COMPILATION_CACHE_DIR default to in-pod /tmp; DEPLOY.md
#    overrides JAX_COMPILATION_CACHE_DIR to a GCS path for cross-run cache reuse.
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/trainer \
    TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
    HF_HOME=/tmp/hf_cache \
    HF_DATASETS_CACHE=/tmp/hf_datasets_cache \
    JAX_COMPILATION_CACHE_DIR=/tmp/jax_xla_cache

CMD ["python", "-u", "train.py", "--help"]
