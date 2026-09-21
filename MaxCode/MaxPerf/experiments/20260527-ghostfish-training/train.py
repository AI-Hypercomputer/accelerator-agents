# pylint: skip-file
from absl import app
from absl import logging
import jax
import jax.numpy as jnp

def main(_):
  logging.info("Starting Ghostfish Training Experiment")

  # Check for TPU
  devices = jax.devices()
  logging.info(f"Available devices: {devices}")

  if any(device.platform == "tpu" for device in devices):
    logging.info("TPU detected!")
  else:
    logging.warning("No TPU detected. Running on CPU/GPU.")

  # Simple computation to verify TPU usage
  x = jnp.ones((1024, 1024))
  y = jnp.dot(x, x)
  logging.info(f"Computation result sum: {jnp.sum(y)}")

  logging.info("Ghostfish Training Experiment Complete")

if __name__ == "__main__":
  app.run(main)
