# Imports
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
alpha = 1.0
batch_size = 16
dim = 16384
key = random.PRNGKey(0)
x = random.normal(key, (batch_size, dim))


# Computation
def kernel(x_ref, out_ref):
  """Pallas kernel for the ELU activation function."""
  x = x_ref[...]
  # The ELU formula is:
  # - x if x > 0
  # - alpha * (exp(x) - 1) if x <= 0
  out_ref[...] = jnp.where(x > 0, x, alpha * (jnp.exp(x) - 1.0))


result = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct(shape=(batch_size, dim), dtype=jnp.float32),
  grid_spec=pl.GridSpec(
    grid=(2, 128),
    block_mapping=[
      pl.BlockMapping(
        block_shape=(8, 128),
        index_map=lambda i, j: (i * 8, j * 128),
        vmem_shape=pl.VmemShape(shape=(8, 128), memory_space=pl.VMEM),
      ),
      pl.BlockMapping(
        block_shape=(8, 128),
        index_map=lambda i, j: (i * 8, j * 128),
        vmem_shape=pl.VmemShape(shape=(8, 128), memory_space=pl.VMEM),
      ),
    ],
  ),
)(x).block_until_ready()
