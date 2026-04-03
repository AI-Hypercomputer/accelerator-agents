# Imports
import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
batch_size = 128
in_channels = 32
out_channels = 16
height, width = 16, 16
kernel_size = 4
stride = 2
padding = 1
# output_padding is not directly supported in flax, but the combination of other parameters achieves the desired output shape

# Calculated output shape based on ConvTranspose parameters
# out_h = (16 - 1) * 2 - 2*1 + 4 = 32
# out_w = (16 - 1) * 2 - 2*1 + 4 = 32
out_height, out_width = 32, 32

# Define block sizes for tiling the output's spatial dimensions
bH = 8
bW = 8

conv_transpose = nn.ConvTranspose(
  features=out_channels, kernel_size=(kernel_size, kernel_size), strides=(stride, stride), padding=padding
)

key = random.PRNGKey(0)
key, bias_key, x_key, params_key = random.split(key, 4)

bias = random.normal(bias_key, (1, 1, 1, out_channels))
x = random.normal(x_key, (batch_size, height, width, in_channels))
params = conv_transpose.init(params_key, x)["params"]


# Computation
def kernel(x_ref, kernel_ref, conv_bias_ref, bias_ref, out_ref):
  """Pallas kernel for a fused ConvTranspose -> subtract -> tanh operation.

  This kernel computes a tile of the output tensor. It performs a full
  ConvTranspose for a given batch element, adds the convolution bias,
  extracts the relevant output tile, subtracts a second bias, applies
  the tanh activation, and writes the result to the output buffer.

  Args:
    x_ref: A reference to the input tensor tile.
      Shape: (1, H, W, C_in)
    kernel_ref: A reference to the ConvTranspose kernel weights.
      Shape: (KH, KW, C_in, C_out)
    conv_bias_ref: A reference to the bias vector of the ConvTranspose layer.
      Shape: (C_out,)
    bias_ref: A reference to the external bias tensor to be subtracted.
      Shape: (1, 1, 1, C_out)
    out_ref: A reference to the output tensor tile to be written to.
      Shape: (1, bH, bW, C_out)
  """
  # Get the program IDs to determine the output tile index.
  # program_id(0) is the batch index, handled by the input BlockSpec.
  # program_id(1) is the output height block index.
  # program_id(2) is the output width block index.
  i, j = pl.program_id(1), pl.program_id(2)

  # Accumulator for the output tile. JAX will manage memory placement.
  acc = jnp.zeros((bH, bW, out_channels), dtype=x_ref.dtype)
  kernel_val = kernel_ref[...]

  # Manually implement the transposed convolution.
  # Iterate over each input pixel and "splat" its contribution to the output.
  for h_in in range(height):
    for w_in in range(width):
      in_pixel_vec = x_ref[0, h_in, w_in, :]
      for kh in range(kernel_size):
        for kw in range(kernel_size):
          # Calculate output coordinates, accounting for stride and padding.
          h_out = h_in * stride + kh - padding
          w_out = w_in * stride + kw - padding

          # Check if the output coordinate falls within the tile of this kernel.
          is_in_tile = (h_out >= i * bH) & (h_out < (i + 1) * bH) & (w_out >= j * bW) & (w_out < (j + 1) * bW)

          def _update_acc(current_acc):
            h_tile = h_out - i * bH
            w_tile = w_out - j * bW
            # Perform the vector-matrix product and accumulate.
            # Reshape to 2D to ensure matmul lowering, then squeeze back.
            update = jnp.dot(in_pixel_vec.reshape(1, in_channels), kernel_val[kh, kw, :, :]).squeeze(axis=0)
            return current_acc.at[h_tile, w_tile, :].add(update)

          # Conditionally update the accumulator using jax.lax.cond.
          acc = jax.lax.cond(is_in_tile, _update_acc, lambda x: x, acc)

  # Add the convolution layer's bias.
  y_tile = acc + conv_bias_ref[...]
  # Subtract the external bias. Broadcasting makes y_tile (1, bH, bW, C_out).
  y_tile = y_tile - bias_ref[...]
  # Apply the tanh activation function.
  y_tile = jnp.tanh(y_tile)

  # Write the final result to the output reference.
  out_ref[...] = y_tile


x = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct((batch_size, out_height, out_width, out_channels), x.dtype),
  grid=(batch_size, out_height // bH, out_width // bW),
  in_specs=[
    # Input image `x`: Pass the full (H, W, C_in) slice for each batch element.
    pl.BlockSpec(block_shape=(1, height, width, in_channels), index_map=lambda b, i, j: (b, 0, 0, 0)),
    # Conv kernel weights: Pass the entire kernel to every instance.
    pl.BlockSpec(
      block_shape=(kernel_size, kernel_size, in_channels, out_channels), index_map=lambda b, i, j: (0, 0, 0, 0)
    ),
    # Conv kernel bias: Pass the entire bias vector to every instance.
    pl.BlockSpec(block_shape=(out_channels,), index_map=lambda b, i, j: (0,)),
    # External bias: Pass the entire bias tensor to every instance.
    pl.BlockSpec(block_shape=(1, 1, 1, out_channels), index_map=lambda b, i, j: (0, 0, 0, 0)),
  ],
  out_specs=pl.BlockSpec(block_shape=(1, bH, bW, out_channels), index_map=lambda b, i, j: (b, i * bH, j * bW, 0)),
)(x, params["kernel"], params["bias"], bias).block_until_ready()
