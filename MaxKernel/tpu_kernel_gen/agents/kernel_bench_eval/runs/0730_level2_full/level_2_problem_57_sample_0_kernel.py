# Imports
import jax
import jax.numpy as jnp
import jax.random as random
from flax import linen as nn
from jax.experimental import pallas as pl

# Initialization
batch_size = 128
in_channels = 3
out_channels = 16
height, width = 32, 32
kernel_size = 3

key = random.PRNGKey(0)
key, x_key, params_key = random.split(key, 3)

# JAX uses channels-last convention by default
x = random.normal(x_key, (batch_size, height, width, in_channels))
conv = nn.Conv(features=out_channels, kernel_size=(kernel_size, kernel_size))
variables = conv.init(params_key, x)


def kernel(x_ref, kernel_ref, bias_ref, out_ref):
  """Pallas kernel for a fused Conv-ReLU-HardSwish operation.

  This kernel computes one full row of the output.
  """
  # Create a buffer for the output row.
  row_out = jnp.zeros((width, out_channels), dtype=x_ref.dtype)

  # Loop over the width of the output row.
  for w in range(width):
    # x_ref is a slice of the padded input. We extract a patch for each
    # output pixel using a sliding window approach.
    patch = lax.dynamic_slice(x_ref, (0, 0, w, 0), (1, kernel_size, kernel_size, in_channels))
    # Squeeze the patch to (KH, KW, C_in) for the einsum.
    patch_squeezed = jnp.squeeze(patch, axis=0)

    # Perform convolution for one pixel.
    # kernel_ref is (KH, KW, C_in, C_out)
    out_pixel = jnp.einsum("ijk,ijko->o", patch_squeezed, kernel_ref[...])
    row_out = row_out.at[w].set(out_pixel)

  # Add bias and apply activations to the whole row.
  y = row_out + bias_ref[...]
  y = jnp.maximum(y, 0)
  y = y * jnp.clip((y + 3) / 6, 0, 1)

  # Write the final result to the output buffer.
  # y has shape (width, out_channels).
  # out_ref has shape (1, 1, width, out_channels).
  out_ref[...] = y.reshape(1, 1, width, out_channels)


# Computation
# Pad the input to handle convolution boundaries, achieving 'SAME' padding.
pad_size = kernel_size // 2
x_padded = jnp.pad(x, ((0, 0), (pad_size, pad_size), (pad_size, pad_size), (0, 0)))

x = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct((batch_size, height, width, out_channels), x.dtype),
  grid=(batch_size, height),
  in_specs=[
    pl.BlockSpec(block_shape=(1, kernel_size, width + 2 * pad_size, in_channels), index_map=lambda b, h: (b, h, 0, 0)),
    pl.BlockSpec(block_shape=variables["params"]["kernel"].shape, index_map=lambda b, h: (0, 0, 0, 0)),
    pl.BlockSpec(block_shape=variables["params"]["bias"].shape, index_map=lambda b, h: (0,)),
  ],
  out_specs=pl.BlockSpec(block_shape=(1, 1, width, out_channels), index_map=lambda b, h: (b, h, 0, 0)),
)(x_padded, variables["params"]["kernel"], variables["params"]["bias"]).block_until_ready()
