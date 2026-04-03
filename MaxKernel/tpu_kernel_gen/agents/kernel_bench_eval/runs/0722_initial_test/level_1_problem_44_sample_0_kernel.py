# Imports
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
batch_size = 16
in_channels = 32
input_length = 128
kernel_size = 4
stride = 2
# Note: PyTorch padding=1 with these parameters is equivalent to 'SAME' padding in JAX/Flax.
padding = "SAME"
key = random.PRNGKey(0)
# Note: Flax expects input shape (batch, length, features) for 1D pooling
x = random.normal(key, (batch_size, input_length, in_channels))
output_length = -(-input_length // stride)  # Equivalent to ceil(input_length / stride)


# Computation
def kernel(x_ref, out_ref):
  """Pallas kernel for 1D average pooling."""
  # x_ref corresponds to the window of input data for a single output.
  # Its shape is (1, kernel_size, in_channels).
  # We compute the mean over the window dimension (axis=1).
  # The result has shape (1, 1, in_channels), which matches out_ref.
  out_ref[...] = jnp.mean(x_ref, axis=1)


output = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct((batch_size, output_length, in_channels), x.dtype),
  grid=(batch_size, output_length),
  in_specs=[
    pl.BlockSpec(
      block_shape=(1, kernel_size, in_channels),
      index_map=lambda i, j: (i, j * stride, 0),
    )
  ],
  out_specs=pl.BlockSpec(block_shape=(1, 1, in_channels), index_map=lambda i, j: (i, j, 0)),
  grid_spec=pl.GridSpec(block_shape_in_out_dim=((1, kernel_size, in_channels), (1, 1, in_channels))),
)(x).block_until_ready()
