# Imports
import jax
import jax.numpy as jnp
import jax.random as random
from flax import linen as nn
from jax import lax
from jax.experimental import pallas as pl

# Initialization
batch_size = 16
in_channels = 32
out_channels = 64
kernel_size = 3
height_in = 64
width_in = 128
stride = 5
padding = 1
dilation = 2
bias = False

key = random.PRNGKey(0)
key_x, key_init = random.split(key)

x = random.normal(key_x, (batch_size, height_in, width_in, in_channels))
conv_transpose2d = nn.ConvTranspose(
  features=out_channels,
  kernel_size=(kernel_size, kernel_size),
  strides=(stride, stride),
  padding=padding,
  kernel_dilation=(dilation, dilation),
  use_bias=bias,
)
params = conv_transpose2d.init(key_init, x)["params"]


def kernel(x_ref, kernel_ref, y_ref):
  """
  Pallas kernel for 2D transposed convolution.

  This kernel implements a "scatter-add" approach. It iterates over each
  pixel in the input block, and for each input pixel, it iterates over the
  kernel. It calculates the corresponding output pixel location and adds
  (scatters) the computed value to an accumulator for the output block. This
  is more efficient than a "gather" approach as it avoids expensive checks
  for stride validity inside the main loop.
  """
  # Hardcoded convolution parameters from the source code
  stride = 5
  padding = 1
  dilation = 2

  # Get the dimensions of the blocks and the kernel
  _, bH_out, bW_out, out_channels = y_ref.shape
  _, bH_in, bW_in, _ = x_ref.shape
  kH, kW, _, _ = kernel_ref.shape

  # Get the index of the current block being processed
  h_out_blk_idx = pl.program_id(1)
  w_out_blk_idx = pl.program_id(2)

  # Calculate the starting global coordinates of the input block (`x_ref`).
  # This logic must exactly match the `index_map` in the `pallas_call`.
  h_in_start_global = (h_out_blk_idx * bH_out + padding - dilation * (kH - 1)) // stride
  w_in_start_global = (w_out_blk_idx * bW_out + padding - dilation * (kW - 1)) // stride

  # Accumulator for the output block, initialized to zeros.
  acc = jnp.zeros((bH_out, bW_out, out_channels), dtype=y_ref.dtype)

  # Iterate over each pixel in the input block
  for h_in_local in range(bH_in):
    for w_in_local in range(bW_in):
      # Calculate global input coordinates
      h_in = h_in_start_global + h_in_local
      w_in = w_in_start_global + w_in_local

      # Read the input value patch (all input channels for one pixel).
      x_val = x_ref[0, h_in_local, w_in_local, :]

      # Iterate over the kernel's spatial dimensions
      for kh in range(kH):
        for kw in range(kW):
          # Calculate the global output coordinates this input pixel contributes to
          h_out = h_in * stride - padding + kh * dilation
          w_out = w_in * stride - padding + kw * dilation

          # Convert global output coordinates to local coordinates within the output block
          h_out_local = h_out - h_out_blk_idx * bH_out
          w_out_local = w_out - w_out_blk_idx * bW_out

          # Check if the calculated output coordinates fall within the current output block
          is_in_bounds = (h_out_local >= 0) & (h_out_local < bH_out) & (w_out_local >= 0) & (w_out_local < bW_out)

          # Use lax.cond to conditionally update the accumulator. This is a
          # JAX-traceable way to handle data-dependent control flow.
          def true_fn(acc):
            kernel_val = kernel_ref[kh, kw, :, :]
            # Use einsum for a clear and robust dot product specification.
            update = jnp.einsum("i,io->o", x_val, kernel_val)
            return acc.at[h_out_local, w_out_local, :].add(update)

          def false_fn(acc):
            return acc

          acc = lax.cond(is_in_bounds, true_fn, false_fn, acc)

  # Write the final accumulated values to the output block
  y_ref[0, :, :, :] = acc


# Computation
# Define output block dimensions that are TPU-compatible
bH_out = 32
bW_out = 128

# Define input block dimensions
# These are calculated based on the transposed convolution parameters to ensure
# they are large enough for any output block. The width is padded to be a
# multiple of 8 for TPU compatibility.
bH_in = 8
bW_in = 32

# Calculate grid dimensions based on output shape and block size
height_out = (height_in - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1
width_out = (width_in - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1
grid_h = (height_out + bH_out - 1) // bH_out
grid_w = (width_out + bW_out - 1) // bW_out

y = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct((batch_size, height_out, width_out, out_channels), x.dtype),
  grid=(batch_size, grid_h, grid_w),
  in_specs=[
    pl.BlockSpec(
      block_shape=(1, bH_in, bW_in, in_channels),
      index_map=lambda i, j, k: (
        i,
        (j * bH_out + padding - dilation * (kernel_size - 1)) // stride,
        (k * bW_out + padding - dilation * (kernel_size - 1)) // stride,
        0,
      ),
    ),
    pl.BlockSpec(block_shape=params["kernel"].shape, index_map=lambda i, j, k: (0, 0, 0, 0)),
  ],
  out_specs=pl.BlockSpec(
    block_shape=(1, bH_out, bW_out, out_channels),
    index_map=lambda i, j, k: (i, j * bH_out, k * bW_out, 0),
  ),
)(x, params["kernel"]).block_until_ready()
