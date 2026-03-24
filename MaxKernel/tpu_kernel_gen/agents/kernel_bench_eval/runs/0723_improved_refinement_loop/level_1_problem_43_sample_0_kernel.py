# Imports
import jax
import jax.numpy as jnp
import jax.random as random
from jax import lax
from jax.experimental import pallas as pl

# Initialization
batch_size = 16
channels = 32
dim1 = 64
dim2 = 64
dim3 = 64
kernel_size = 3
stride = 2
padding = 1
dilation = 3
# return_indices and ceil_mode are not directly applicable in JAX/Flax in the same way.
# The behavior is replicated by using explicit padding with lax.reduce_window.
key = random.PRNGKey(0)
x = random.normal(key, (batch_size, dim1, dim2, dim3, channels))  # Note: JAX uses channels-last format


# Computation
def kernel(x_ref, out_ref):
  """Pallas kernel for 3D max pooling."""
  # Hardcoded parameters from the source lax.reduce_window call
  kernel_size_d3 = 3
  stride_d3 = 2
  padding_d3 = 1
  dilation_d3 = 3

  # The output fiber shape is (1, 1, 1, out_dim3, channels)
  # We iterate over each element in the output fiber's 3rd dimension.
  for k in range(out_ref.shape[3]):
    # Initialize max value for each output element
    max_val = jnp.full(out_ref.shape[4], -jnp.inf, dtype=x_ref.dtype)

    # The receptive field for the first two spatial dimensions is fully contained
    # within the loaded x_ref block. We only need to slide the window
    # across the 3rd dimension of x_ref.
    for l in range(kernel_size_d3):
      # Calculate the index in the 3rd dimension of the input block
      idx_d3 = k * stride_d3 - padding_d3 + l * dilation_d3

      # Check if the index is within the valid bounds of the input's 3rd dim
      if 0 <= idx_d3 < x_ref.shape[3]:
        # Reduce over the first two spatial dimensions of the input block
        # for the current slice of the 3rd dimension.
        # x_ref.shape is (1, in_block_dim1, in_block_dim2, dim3, channels)
        # We take a slice at idx_d3, resulting in a shape of
        # (in_block_dim1, in_block_dim2, channels)
        # Then we reduce over axes 0 and 1.
        current_slice_max = jnp.max(x_ref[0, :, :, idx_d3, :], axis=(0, 1))
        max_val = lax.max(max_val, current_slice_max)

    # Write the final max value to the output reference
    out_ref[0, 0, 0, k, :] = max_val


# Helper function to calculate output dimensions based on JAX's convolution formula
def get_out_dim(in_dim, stride, padding, kernel_size, dilation):
  effective_kernel_size = (kernel_size - 1) * dilation + 1
  return (in_dim + 2 * padding - effective_kernel_size) // stride + 1


# Calculate output dimensions from the initialization parameters
out_dim1 = get_out_dim(dim1, 1, 0, (kernel_size - 1) * dilation + 1, 1)
out_dim2 = get_out_dim(dim2, 1, 0, (kernel_size - 1) * dilation + 1, 1)
out_dim3 = get_out_dim(dim3, stride, padding, kernel_size, dilation)

# Calculate the size of the input block required for one window pass
in_block_dim1 = (kernel_size - 1) * dilation + 1
in_block_dim2 = (kernel_size - 1) * dilation + 1

output = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct((batch_size, out_dim1, out_dim2, out_dim3, channels), x.dtype),
  grid=(batch_size, out_dim1, out_dim2),
  in_specs=[pl.BlockSpec((1, in_block_dim1, in_block_dim2, dim3, channels), lambda b, i, j: (b, i, j, 0, 0))],
  out_specs=[pl.BlockSpec((1, 1, 1, out_dim3, channels), lambda b, i, j: (b, i, j, 0, 0))],
)(x).block_until_ready()
