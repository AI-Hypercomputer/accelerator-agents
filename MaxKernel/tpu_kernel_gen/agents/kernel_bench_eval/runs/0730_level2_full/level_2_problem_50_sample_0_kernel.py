# Imports
import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1
scale1 = 0.5
scale2 = 1.0
key = random.PRNGKey(0)
key_x, key_params, key_bias = random.split(key, 3)
x = random.normal(key_x, (batch_size, depth, height, width, in_channels))
conv_transpose = nn.ConvTranspose(
  features=out_channels,
  kernel_size=(kernel_size, kernel_size, kernel_size),
  strides=(stride, stride, stride),
  padding=padding,
  transpose_kernel=False,
)
params = conv_transpose.init(key_params, x)["params"]
# The kernel needs to be spatially flipped for the equivalent convolution operation
flipped_kernel = jnp.flip(params["kernel"], axis=(0, 1, 2))
# The bias shape must match the output of the pooling operation
pooled_out_shape = (1, 15, 31, 31, out_channels)
bias = random.normal(key_bias, pooled_out_shape)


# Computation
def kernel(x_ref, kernel_ref, bias_ref, out_ref):
  """Pallas kernel for a sequence of conv transpose, pooling, and arithmetic ops."""
  # Hardcoded constants from the original computation
  scale1 = 0.5
  scale2 = 1.0
  # A transposed convolution with stride S is equivalent to a regular convolution
  # with input (LHS) dilation S and a stride of 1.
  conv_strides = (1, 1, 1)
  lhs_dilation = (2, 2, 2)
  # The padding for the equivalent convolution is K - 1 - P_transpose
  # Here K=3, P_transpose=1, so P_conv = 3 - 1 - 1 = 1.
  padding = [(1, 1), (1, 1), (1, 1)]
  dimension_numbers = ("NDHWC", "DHWIO", "NDHWC")

  # 1. 3D Convolution (emulating transposed convolution)
  # This operation is not directly supported in Pallas on TPU. We emulate it
  # using a standard convolution with LHS dilation.
  conv_out = jax.lax.conv_general_dilated(
    x_ref[...],
    kernel_ref[...],
    window_strides=conv_strides,
    padding=padding,
    lhs_dilation=lhs_dilation,
    dimension_numbers=dimension_numbers,
  )

  # 2. First Scaling Operation
  conv_out = conv_out * scale1

  # 3. Average Pooling
  # This is implemented using jax.lax.reduce_window, which sums the elements
  # in the window. We then divide by the window size to get the average.
  window_dims = (1, 2, 2, 2, 1)
  pool_strides = (1, 2, 2, 2, 1)
  window_size = 2 * 2 * 2

  pooled_out = jax.lax.reduce_window(
    conv_out,
    init_value=0.0,
    computation=jax.lax.add,
    window_dimensions=window_dims,
    window_strides=pool_strides,
    padding="VALID",
  )
  pooled_out = pooled_out / window_size

  # 4. Add Bias
  # The bias is broadcast-added to the result of the pooling operation.
  biased_out = pooled_out + bias_ref[...]

  # 5. Second Scaling Operation
  final_out = biased_out * scale2

  # 6. Store the final result in the output buffer
  out_ref[...] = final_out


x = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct((batch_size, 15, 31, 31, out_channels), x.dtype),
  grid=(batch_size,),
  in_specs=[
    pl.BlockSpec(block_shape=(1, depth, height, width, in_channels), index_map=lambda i: (i, 0, 0, 0, 0)),
    pl.BlockSpec(block_shape=flipped_kernel.shape, index_map=lambda i: (0,) * flipped_kernel.ndim),
    pl.BlockSpec(block_shape=bias.shape, index_map=lambda i: (0,) * bias.ndim),
  ],
  out_specs=pl.BlockSpec(block_shape=(1, 15, 31, 31, out_channels), index_map=lambda i: (i, 0, 0, 0, 0)),
)(x, flipped_kernel, bias).block_until_ready()
