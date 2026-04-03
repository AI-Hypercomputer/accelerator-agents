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


def kernel(x_ref, y_ref):
  """Pallas kernel to normalize a tensor.

  This kernel computes the L2 norm of an input block (x_ref) and normalizes
  the block by dividing it by the computed norm. The result is written to
  the output block (y_ref).

  Args:
    x_ref: A reference to the input tensor block.
    y_ref: A reference to the output tensor block for the normalized result.
  """
  # Calculate the L2 norm of the input block.
  norm = jnp.sqrt(jnp.sum(x_ref[...] * x_ref[...], axis=-1))
  # Divide the input block by its norm and write to the output block.
  y_ref[...] = x_ref[...] / norm[:, None]


# Define block sizes for tiling to fit in TPU VMEM
block_d1 = 128
block_d2 = 128

# Computation
y = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
  grid=(batch_size, features, dim1 // block_d1, dim2 // block_d2),
  in_specs=[
    pl.BlockSpec(
      (1, 1, block_d1, block_d2),
      lambda i, j, k, l: (i, j, k * block_d1, l * block_d2),
    )
  ],
  out_specs=pl.BlockSpec(
    (1, 1, block_d1, block_d2),
    lambda i, j, k, l: (i, j, k * block_d1, l * block_d2),
  ),
)(x).block_until_ready()
