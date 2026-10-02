# Qwen3 native-JAX (Flax NNX) trainer — TPU image.
#
# Reuses the torchax lane image as the base — it already carries the full stack
# this lane needs (jax + libtpu from jax-ai-image, flax, optax, transformers,
# fire, datasets, google-cloud-storage). We only swap in the jax trainer code,
# so the build reuses every heavy cached layer and pushes one tiny code layer.
#
# The jax trainer imports NO torch/torchax (pure JAX); the torch* packages from
# the base image are simply unused here.
#
# Build context: this directory's trainer files staged under `trainer/`.

# v008b layer = torchax base + dev tokamax (chunked_xla CE) — see exp v008.
FROM <your-registry>/torchtitan-images/qwen3-8b-jax:v6e8-qwen3-8b-jax-20260612-v008b-cxla

# Replace the torchax trainer code with the jax trainer code.
RUN rm -rf /app/trainer
WORKDIR /app/trainer
COPY trainer /app/trainer

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/trainer

CMD ["python", "-u", "train.py", "--help"]
