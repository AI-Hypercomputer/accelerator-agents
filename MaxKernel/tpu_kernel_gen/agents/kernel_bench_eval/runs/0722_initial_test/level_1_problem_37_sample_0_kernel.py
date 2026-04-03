# Imports
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
batch_size = 16
features = 64
dim1 = 256
dim2 = 256
key = random.PRNGKey(0)
x = random.normal(key, (batch_size, features, dim1, dim2))
block_d1 = 128
block_d2 = 256

# Computation
norm = jnp.sqrt(jnp.sum(x**2))


def kernel(x_ref, norm_val, out_ref):
  """Divides the input block by a scalar norm.

  Args:
    x_ref: A reference to the input block.
    norm_val: The pre-computed Frobenius norm of the entire tensor. This is
      passed as a scalar argument to the kernel.
    out_ref: A reference to the output block to be written to.
  """
  out_ref[...] = x_ref[...] / norm_val


result = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
  grid=(
    x.shape[0],
    x.shape[1],
    x.shape[2] // block_d1,
    x.shape[3] // block_d2,
  ),
  in_specs=[
    pl.BlockSpec(
      block_shape=(1, 1, block_d1, block_d2),
      index_map=lambda i, j, k, l: (i, j, k * block_d1, l * block_d2),
    ),
    pl.ScalarSpec(),
  ],
  out_specs=pl.BlockSpec(
    block_shape=(1, 1, block_d1, block_d2),
    index_map=lambda i, j, k, l: (i, j, k * block_d1, l * block_d2),
  ),
)(x, norm).block_until_ready()
