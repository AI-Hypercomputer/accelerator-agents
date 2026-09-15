# Llama 3 8B native-JAX (Flax NNX) trainer — TPU image.
#
# Base: jax-ai-image/tpu (Cloud TPU images, JAX 0.9 + libtpu).
# Layered on top: HF transformers + datasets, optax, fire, flax, and the
# trainer code. NO torchax or torch — this is the pure-JAX path.
#
# Build context: this directory + a `tokamax_lib/` subdir holding the
# tokamax wheel / source tree. See README.md for the build command.

FROM us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu:jax0.9.0-rev1

# Layer 1 — extra Python deps. flax is the NNX runtime.
RUN pip install --no-cache-dir \
    transformers \
    accelerate \
    datasets \
    optax \
    fire \
    pyyaml \
    huggingface_hub \
    flax \
    safetensors

# Layer 2 — uninstall any pip-installed tokamax (older versions lack
# chunked_xla / tokamax-splash). Local source from tokamax_lib/ takes over
# via PYTHONPATH below.
RUN pip uninstall -y tokamax || true

# Layer 3 — trainer code + local tokamax (matches torchax Dockerfile layout).
WORKDIR /app/trainer
COPY trainer /app/trainer
COPY tokamax_lib /app/trainer/tokamax_lib

# Layer 4 — runtime defaults. PYTHONPATH puts /app/trainer/tokamax_lib
# AHEAD of site-packages so `import tokamax` finds our newer source.
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/trainer:/app/trainer/tokamax_lib \
    HF_HOME=/tmp/hf_cache \
    HF_DATASETS_CACHE=/tmp/hf_datasets_cache \
    JAX_COMPILATION_CACHE_DIR=/tmp/jax_xla_cache

CMD ["python", "-u", "train.py", "--help"]
