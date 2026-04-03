# Imports
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
batch_size = 128
input_shape = (4000,)
key = random.PRNGKey(0)
key_x, key_mask = random.split(key)
x = random.normal(key_x, (batch_size, *input_shape))
mask = random.randint(key_mask, x.shape, 0, 2).astype(jnp.bool_)


# Computation
def kernel(x_ref, mask_ref, out_ref):
  # Perform the element-wise multiplication of the input by the mask.
  masked_x = x_ref[...] * mask_ref[...]

  # Use jax.lax.scan to compute the cumulative sum along the last axis.
  y = jnp.cumsum(masked_x, axis=1)

  # Write the result to the output reference.
  out_ref[...] = y


# The block size for the first dimension. Must be divisible by 8 for TPU.
block_size_m = 8

result = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
  grid=(batch_size // block_size_m,),
  in_specs=[
    pl.BlockSpec(block_shape=(block_size_m, *input_shape), index_map=lambda i: (i * block_size_m, 0)),
    pl.BlockSpec(block_shape=(block_size_m, *input_shape), index_map=lambda i: (i * block_size_m, 0)),
  ],
  out_specs=pl.BlockSpec(block_shape=(block_size_m, *input_shape), index_map=lambda i: (i * block_size_m, 0)),
)(x, mask).block_until_ready()
