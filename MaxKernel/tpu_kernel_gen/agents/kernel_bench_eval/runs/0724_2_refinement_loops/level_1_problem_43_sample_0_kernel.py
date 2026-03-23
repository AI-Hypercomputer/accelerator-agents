# Imports
import jax
import jax.numpy as jnp
import jax.random as random
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

key = random.PRNGKey(0)
# JAX and Flax use a channels-last convention (N, H, W, D, C)
x = random.normal(key, (batch_size, dim1, dim2, dim3, channels))

# Calculate the output spatial dimension based on the reduce_window parameters
dilated_kernel_size = (kernel_size - 1) * dilation + 1
out_spatial_dim = (dim1 + 2 * padding - dilated_kernel_size) // stride + 1

# Pre-pad the input tensor with a value that won't affect the max operation.
pad_width = ((0, 0), (padding, padding), (padding, padding), (padding, padding), (0, 0))
x_padded = jnp.pad(x, pad_width, mode="constant", constant_values=-jnp.inf)
padded_dim1, padded_dim2, padded_dim3 = x_padded.shape[1:4]


# Computation
def kernel(x_padded_ref, output_ref):
  """
  Pallas kernel for 3D max pooling on a pre-padded input.

  This kernel processes a 1D slice of the output tensor along the x-dimension.
  The `pallas_call` maps this kernel over the batch, z, and y dimensions of the output.
  """
  # Get the z and y indices for this kernel instance from the grid.
  oz = pl.program_id(1)
  oy = pl.program_id(2)

  # Iterate over the output x-dimension for the assigned slice.
  for ox in range(out_spatial_dim):
    # For each channel, compute the max value in the pooling window.
    for c in range(channels):
      max_val = -jnp.inf
      # Iterate over the kernel window.
      for kz in range(kernel_size):
        for ky in range(kernel_size):
          for kx in range(kernel_size):
            # Calculate input coordinates relative to the loaded slice `x_padded_ref`.
            iz_slice = kz * dilation
            iy_slice = ky * dilation
            ix_slice = ox * stride + kx * dilation

            val = x_padded_ref[0, iz_slice, iy_slice, ix_slice, c]
            max_val = jax.lax.max(max_val, val)

      # Write the computed max value to the output tensor slice.
      output_ref[0, 0, 0, ox, c] = max_val


# The pallas_call replaces the original computation.
# It parallelizes the operation across the batch, z, and y dimensions.
output = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct(
    (batch_size, out_spatial_dim, out_spatial_dim, out_spatial_dim, channels),
    jnp.float32,  # Assuming float32 dtype from jax.random.normal
  ),
  grid=(batch_size, out_spatial_dim, out_spatial_dim),
  in_specs=[
    pl.BlockSpec(
      # Load a slice covering the receptive field for the output row.
      block_shape=(1, dilated_kernel_size, dilated_kernel_size, padded_dim3, channels),
      index_map=lambda b, oz, oy: (b, oz * stride, oy * stride, 0, 0),
    )
  ],
  out_specs=pl.BlockSpec(
    block_shape=(1, 1, 1, out_spatial_dim, channels),
    index_map=lambda b, oz, oy: (b, oz, oy, 0, 0),
  ),
)(x_padded).block_until_ready()
