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
height_in = 16
width_in = 32
stride = (1, 1)
# PyTorch padding=(0,0) and output_padding=(0,0) is equivalent to padding='VALID' in Flax.
dilation = (1, 1)
groups = 1
bias = False

conv_transpose2d = nn.ConvTranspose(
  features=out_channels,
  kernel_size=kernel_size,
  strides=stride,
  padding="VALID",
  kernel_dilation=dilation,
  use_bias=bias,
)

key = random.PRNGKey(0)
key_params, key_x = random.split(key)

# JAX uses the channel-last convention (N, H, W, C)
x = random.normal(key_x, (batch_size, height_in, width_in, in_channels))
params = conv_transpose2d.init(key_params, x)["params"]

# Calculate output shape based on 'VALID' transposed convolution rules
height_out = (height_in - 1) * stride[0] + kernel_size[0]
width_out = (width_in - 1) * stride[1] + kernel_size[1]


# Computation
def kernel(x_ref, kernel_ref, out_ref):
  """
  Pallas kernel for 2D transposed convolution.

  This kernel computes the transposed convolution for a single item in the batch.
  It uses jax.lax.conv_general_dilated to perform the core convolution
  operation on the input data slice (x_ref) and the full kernel (kernel_ref).

  Args:
    x_ref: A reference to a slice of the input tensor, corresponding to one
      batch item.
    kernel_ref: A reference to the entire convolution kernel tensor.
    out_ref: A reference to the output buffer, corresponding to a single
      item in the output batch, where the result is written in-place.
  """
  # The jax.lax.conv_general_dilated primitive is not implemented in Pallas
  # for TPU. The transposed convolution must be implemented manually.
  # This implementation iterates over each input pixel and "scatters" its
  # contribution to a patch in the output, scaled by the kernel weights.
  # This requires atomic additions to the output buffer.
  out_ref[...] = jnp.zeros(out_ref.shape, dtype=out_ref.dtype)

  def h_in_loop_body(h_in, carry):
    def w_in_loop_body(w_in, inner_carry):
      in_pixel_vector = x_ref[0, h_in, w_in, :]
      for kh in range(kernel_size[0]):
        for kw in range(kernel_size[1]):
          h_out = h_in * stride[0] + kh
          w_out = w_in * stride[1] + kw
          kernel_slice = kernel_ref[kh, kw, :, :]
          contribution = jnp.einsum("i,io->o", in_pixel_vector, kernel_slice)
          pl.atomic_add(out_ref, (0, h_out, w_out, slice(None)), contribution)
      return inner_carry

    jax.lax.fori_loop(0, width_in, w_in_loop_body, None)
    return carry

  jax.lax.fori_loop(0, height_in, h_in_loop_body, None)


output = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct((batch_size, height_out, width_out, out_channels), x.dtype),
  grid=(batch_size,),
  in_specs=[
    pl.BlockSpec(
      block_shape=(1, height_in, width_in, in_channels),
      index_map=lambda b: (b, 0, 0, 0),
    ),
    pl.BlockSpec(
      block_shape=(*kernel_size, in_channels, out_channels),
      index_map=lambda b: (0, 0, 0, 0),
    ),
  ],
  out_specs=pl.BlockSpec(
    block_shape=(1, height_out, width_out, out_channels),
    index_map=lambda b: (b, 0, 0, 0),
  ),
)(x, params["kernel"]).block_until_ready()
