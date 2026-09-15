# Llama 3 8B torchax trainer — TPU image.
#
# Base: jax-ai-image/tpu (Cloud TPU images, JAX 0.9 + libtpu).
# Layered on top: HF transformers + datasets, optax, torchax (from local source
# matching the wiki submodule), and the trainer code.
#
# Build context: this directory plus the torchax source tree mounted as
# `torchax_src/` (parent's `raw/code/torchax/`). See DEPLOY.md for the build
# command.

FROM us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu:jax0.9.0-rev1

# Layer 1 — extra Python deps.
RUN pip install --no-cache-dir \
    transformers \
    accelerate \
    datasets \
    optax \
    fire \
    pyyaml \
    huggingface_hub

# Layer 2 — torchax (install from local submodule source).
WORKDIR /torchax
COPY torchax_src /torchax
RUN pip install -e .

# Layer 3 — trainer code.
WORKDIR /app/trainer
COPY trainer /app/trainer

# Layer 4 — runtime defaults.
#  - PYTHONUNBUFFERED so streaming logs work under kubectl logs -f.
#  - PYTHONPATH so `import helper` and `import model` resolve.
#  - HF_HOME and JAX_COMPILATION_CACHE_DIR are expected to point at
#    persistent storage when running on a multi-pod cluster (see DEPLOY.md);
#    they default to the in-pod /tmp here so a single-host smoke run works
#    out of the box.
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/trainer \
    HF_HOME=/tmp/hf_cache \
    HF_DATASETS_CACHE=/tmp/hf_datasets_cache \
    JAX_COMPILATION_CACHE_DIR=/tmp/jax_xla_cache

CMD ["python", "-u", "train.py", "--help"]
