# Imports
import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
batch_size = 16
in_channels = 32
out_channels = 64
kernel_size = (3, 5)
height = 128
width = 256
stride = (1, 1)
padding = (1, 2)
key = random.PRNGKey(0)
key_params, key_x = random.split(key)
x = random.normal(key_x, (batch_size, height, width, in_channels))
conv_transpose2d = nn.ConvTranspose(
  features=out_channels, kernel_size=kernel_size, strides=stride, padding=padding, use_bias=False
)
params = conv_transpose2d.init(key_params, x)["params"]
b_height = 32
b_width = 64


# Computation
def kernel(x_ref, kernel_ref, out_ref):
  """Pallas kernel for 2D transposed convolution."""
  # This kernel implements transposed convolution as a standard convolution
  # with a flipped kernel. This avoids atomics and uses a more efficient
  # "gather" pattern suitable for TPUs.
  k_h, k_w, _, _ = kernel_ref.shape
  _, h_block, w_block, _ = out_ref.shape
  in_channels = x_ref.shape[-1]

  # For a conv_transpose with stride=1, the equivalent convolution has padding
  # that makes it a "SAME" convolution.
  # For kernel (3, 5), this is ((1,1), (2,2)). We only need the low padding.
  pad_h_low = (k_h - 1) // 2
  pad_w_low = (k_w - 1) // 2

  # For a conv implementation, we iterate over the output block.
  for y_out_local in range(h_block):
    for x_out_local in range(w_block):
      # Accumulator for the output pixel vector.
      acc = jnp.zeros((out_channels,), dtype=out_ref.dtype)

      # Iterate over the kernel filter.
      for ky in range(k_h):
        for kx in range(k_w):
          # Calculate the input coordinate corresponding to this output & kernel pos.
          # This is the "gather" part of the convolution.
          y_in_local = y_out_local + ky - pad_h_low
          x_in_local = x_out_local + kx - pad_w_low

          # We manually handle padding. If the calculated coordinate is out of
          # bounds of our input block, we use a zero vector.
          in_bounds = (y_in_local >= 0) & (y_in_local < h_block) & (x_in_local >= 0) & (x_in_local < w_block)

          input_pixel_vec = jax.lax.select(
            in_bounds, x_ref[0, y_in_local, x_in_local, :], jnp.zeros(in_channels, dtype=x_ref.dtype)
          )

          # The kernel needs to be flipped spatially for the conv equivalence.
          kernel_slice = kernel_ref[k_h - 1 - ky, k_w - 1 - kx, :, :]

          acc += jnp.dot(input_pixel_vec, kernel_slice)

      # Write the accumulated result to the output.
      out_ref[0, y_out_local, x_out_local, :] = acc


result = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct((batch_size, height, width, out_channels), x.dtype),
  grid=(batch_size, height // b_height, width // b_width),
  in_specs=[
    pl.BlockSpec(
      block_shape=(1, b_height, b_width, in_channels), index_map=lambda i, j, k: (i, j * b_height, k * b_width, 0)
    ),
    pl.BlockSpec(block_shape=params["kernel"].shape, index_map=lambda i, j, k: tuple([0] * params["kernel"].ndim)),
  ],
  out_specs=pl.BlockSpec(
    block_shape=(1, b_height, b_width, out_channels), index_map=lambda i, j, k: (i, j * b_height, k * b_width, 0)
  ),
)(x, params["kernel"]).block_until_ready()
