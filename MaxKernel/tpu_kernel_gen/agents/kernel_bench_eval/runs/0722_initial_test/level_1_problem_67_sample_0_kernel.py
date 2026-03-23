# Imports
import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
batch_size = 16
in_channels = 3
out_channels = 64
kernel_size = 3
length = 512
stride = 1
padding = "VALID"
dilation = 1
groups = 1
bias = False

key = random.PRNGKey(0)
key_x, key_params = random.split(key)

# Note: JAX convention is channels-last (N, L, C) vs PyTorch's channels-first (N, C, L)
x = random.normal(key_x, (batch_size, length, in_channels))

conv1d = nn.Conv(
  features=out_channels,
  kernel_size=(kernel_size,),
  strides=(stride,),
  padding=padding,
  kernel_dilation=(dilation,),
  feature_group_count=groups,
  use_bias=bias,
)
params = conv1d.init(key_params, x)["params"]
kernel_weights = params["kernel"]

# Calculate output shape
output_length = length - kernel_size + 1
output_shape = (batch_size, output_length, out_channels)


def kernel(x_ref, kernel_weights_ref, out_ref):
  """Pallas kernel for 1D convolution.

  This kernel computes the convolution for a single batch item. The grid is
  set up to iterate over all batch items.

  Args:
    x_ref: A reference to the input tensor for a single batch item.
      Shape: (1, length, in_channels)
    kernel_weights_ref: A reference to the entire kernel weights tensor.
      Shape: (kernel_size, in_channels, out_channels)
    out_ref: A reference to the output tensor for a single batch item, to be
      written to. Shape: (1, output_length, out_channels)
  """
  # Load the full input slice and the entire kernel into memory.
  x = x_ref[...]
  kernel_weights = kernel_weights_ref[...]

  # A convolution can be expressed as a matrix multiplication (dot product)
  # by first extracting sliding patches from the input.
  # `conv_general_dilated_patches` is a JAX primitive that does this efficiently.
  patches = jax.lax.conv_general_dilated_patches(
    lhs=x,
    kernel_size=(kernel_size,),
    window_strides=(stride,),
    padding=padding,
    dimension_numbers=("NLC", "OIC", "NLC"),
  )
  # The patches have shape (1, output_length, kernel_size * in_channels)

  # Reshape the kernel for the dot product.
  # (kernel_size, in_channels, out_channels) -> (kernel_size * in_channels, out_channels)
  kernel_reshaped = jnp.reshape(kernel_weights, (-1, out_channels))

  # Perform the dot product.
  # (1, output_length, K*C_in) @ (K*C_in, C_out) -> (1, output_length, C_out)
  out_ref[...] = jnp.dot(patches, kernel_reshaped)


# Computation
output = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct(output_shape, x.dtype),
  grid=(batch_size,),
  in_specs=[
    pl.BlockSpec(block_shape=(1, length, in_channels), index_map=lambda i: (i, 0, 0)),
    pl.BlockSpec(block_shape=(kernel_size, in_channels, out_channels), index_map=lambda i: (0, 0, 0)),
  ],
  out_specs=pl.BlockSpec(block_shape=(1, output_length, out_channels), index_map=lambda i: (i, 0, 0)),
)(x, kernel_weights).block_until_ready()
