# Imports
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
batch_size = 16
dim = 16384
key = random.PRNGKey(0)
x = random.normal(key, (batch_size, dim))
# Block size for the 'dim' dimension
b_dim = 128


# Computation
def kernel(x_ref, out_ref):
  """Pallas kernel for element-wise hard_sigmoid.

  This kernel computes jnp.clip(x * 0.2 + 0.5, 0., 1.) on a block of the
  input tensor.

  Args:
    x_ref: A reference to a block of the input tensor.
    out_ref: A reference to a block of the output tensor for storing the result.
  """
  # Load the input block from SRAM into registers.
  x = x_ref[...]
  # Apply the hard_sigmoid function element-wise.
  # The hard_sigmoid function is defined as clip(x * 0.2 + 0.5, 0, 1).
  result = jnp.clip(x * 0.2 + 0.5, 0.0, 1.0)
  # Write the result back to the output block in SRAM.
  out_ref[...] = result


# The hard_sigmoid operation is element-wise. We can parallelize it by
# splitting the input tensor `x` into blocks. We create a 1D grid of kernels
# along the `dim` dimension. Each kernel instance processes a vertical slice
# of the input tensor of shape (batch_size, b_dim).
result = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
  grid=(x.shape[1] // b_dim,),
  in_specs=[pl.BlockSpec(block_shape=(x.shape[0], b_dim), index_map=lambda i: (0, i * b_dim))],
  out_specs=pl.BlockSpec(block_shape=(x.shape[0], b_dim), index_map=lambda i: (0, i * b_dim)),
)(x)
