# Imports
import jax
import jax.nn
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
batch_size = 16
dim = 16384
key = random.PRNGKey(0)
x = random.normal(key, (batch_size, dim))
block_size = 2048


# Computation
def kernel(x_ref, out_ref):
  # Apply the GELU activation function element-wise to the input block.
  # The result is then stored in the output block.
  out_ref[...] = jax.nn.gelu(x_ref[...])


# The block shape's first dimension must be a multiple of 8 on TPU.
block_shape = (8, block_size)
result = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
  grid=(x.shape[0] // block_shape[0], x.shape[1] // block_shape[1]),
  in_specs=[
    pl.BlockSpec(
      lambda i, j: (i, j),
      (block_shape[0], block_shape[1]),
    )
  ],
  out_specs=pl.BlockSpec(
    lambda i, j: (i, j),
    (block_shape[0], block_shape[1]),
  ),
  grid_mapping=pl.GridMapping(
    block_mapping=[
      pl.BlockMapping(
        block_shape,
        index_map=lambda i, j: (i, j),
      )
    ]
  ),
)(x).block_until_ready()
