# Imports
from functools import partial

import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
batch_size = 128
in_channels = 3
out_channels = 16
height, width = 32, 32
kernel_size = 3
num_groups = 8
scale_shape = (1, 1, 1, out_channels)
maxpool_kernel_size = 2
clamp_min = 0.0
clamp_max = 1.0

key = random.PRNGKey(0)
key_x, key_conv, key_gnorm = random.split(key, 3)

x = random.normal(key_x, (batch_size, height, width, in_channels))

conv = nn.Conv(features=out_channels, kernel_size=(kernel_size, kernel_size), padding="VALID")
group_norm = nn.GroupNorm(num_groups=num_groups)

conv_vars = conv.init(key_conv, x)
conv_x = conv.apply(conv_vars, x)
gnorm_vars = group_norm.init(key_gnorm, conv_x)

scale = jnp.ones(scale_shape)

maxpool = partial(
  nn.max_pool,
  window_shape=(maxpool_kernel_size, maxpool_kernel_size),
  strides=(maxpool_kernel_size, maxpool_kernel_size),
)


# Computation
def kernel(
  x_ref,
  conv_kernel_ref,
  conv_bias_ref,
  gnorm_scale_ref,
  gnorm_bias_ref,
  scale_ref,
  clamp_min_ref,
  clamp_max_ref,
  out_ref,
):
  """
  Pallas kernel that applies a sequence of operations: Conv, GroupNorm,
  scaling, MaxPool, and clipping.

  Args:
    x_ref: Input tensor reference of shape (1, H, W, C_in).
    conv_kernel_ref: Convolution kernel reference.
    conv_bias_ref: Convolution bias reference.
    gnorm_scale_ref: GroupNorm scale reference.
    gnorm_bias_ref: GroupNorm bias reference.
    scale_ref: Scaling factor reference.
    clamp_min_ref: Minimum value for clipping.
    clamp_max_ref: Maximum value for clipping.
    out_ref: Output tensor reference.
  """
  # Import necessary libraries within the kernel
  import jax
  import jax.numpy as jnp
  from jax.experimental.pallas.ops.tpu import convolution

  # Define constants used in the computation
  # These are derived from the source code's initialization
  out_channels = 16
  num_groups = 8
  maxpool_kernel_size = 2
  epsilon = 1e-6  # Default epsilon for nn.GroupNorm

  # 1. Convolution
  # Use the Pallas-specific convolution operator for TPUs, which is more
  # efficient and fuses the bias addition.
  conv_out = convolution(
    x_ref[...],
    conv_kernel_ref[...],
    conv_bias_ref[...],
    window_strides=(1, 1),
    padding="VALID",
    dimension_numbers=("NHWC", "HWIO", "NHWC"),
  )

  # 2. Group Normalization
  # We replicate the logic of flax.linen.GroupNorm.
  # The convolution output already has a batch dimension of 1.
  gn_input = conv_out

  # Reshape to expose the channel groups: (N, H, W, G, C//G)
  # Here N=1.
  reshaped_shape = (
    gn_input.shape[0],
    gn_input.shape[1],
    gn_input.shape[2],
    num_groups,
    out_channels // num_groups,
  )
  x_reshaped = gn_input.reshape(reshaped_shape)

  # Calculate mean and variance over spatial and intra-group channel axes.
  # For a (N, H, W, G, C//G) tensor, these are axes (1, 2, 4).
  reduction_axes = (1, 2, 4)
  mean = jnp.mean(x_reshaped, axis=reduction_axes, keepdims=True)
  var = jnp.var(x_reshaped, axis=reduction_axes, keepdims=True)

  # Normalize the reshaped tensor
  x_normalized = (x_reshaped - mean) / jnp.sqrt(var + epsilon)

  # Reshape back to the original tensor shape (N, H', W', C_out)
  x_normalized = x_normalized.reshape(gn_input.shape)

  # Apply the learned scale and bias from GroupNorm
  gnorm_out = x_normalized * gnorm_scale_ref[...] + gnorm_bias_ref[...]

  # 3. Element-wise Scaling
  # The scale_ref will broadcast correctly.
  scaled_out = gnorm_out * scale_ref[...]

  # 4. Max Pooling
  # We use jax.lax.reduce_window to perform max pooling.
  # The input `scaled_out` already has the necessary batch dimension.
  pooled_out = jax.lax.reduce_window(
    operand=scaled_out,
    init_value=-jnp.inf,
    computation=jnp.maximum,
    window_dimensions=(1, maxpool_kernel_size, maxpool_kernel_size, 1),
    window_strides=(1, maxpool_kernel_size, maxpool_kernel_size, 1),
    padding="VALID",
  )

  # 5. Clipping
  # Clip the final result to the specified range.
  clipped_out = jnp.clip(pooled_out, clamp_min_ref[...], clamp_max_ref[...])

  # 6. Write to Output
  # The shape of `clipped_out` matches the shape of `out_ref`.
  out_ref[...] = clipped_out


x = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct(
    (
      batch_size,
      (height - kernel_size + 1) // maxpool_kernel_size,
      (width - kernel_size + 1) // maxpool_kernel_size,
      out_channels,
    ),
    x.dtype,
  ),
  grid=(batch_size,),
  in_specs=[
    pl.BlockSpec(
      block_shape=(1, height, width, in_channels),
      index_map=lambda i: (i, 0, 0, 0),
    ),
    pl.BlockSpec(block_shape=conv_vars["params"]["kernel"].shape, index_map=lambda i: (0, 0, 0, 0)),
    pl.BlockSpec(block_shape=conv_vars["params"]["bias"].shape, index_map=lambda i: (0,)),
    pl.BlockSpec(block_shape=gnorm_vars["params"]["scale"].shape, index_map=lambda i: (0,)),
    pl.BlockSpec(block_shape=gnorm_vars["params"]["bias"].shape, index_map=lambda i: (0,)),
    pl.BlockSpec(block_shape=scale.shape, index_map=lambda i: (0, 0, 0, 0)),
    pl.BlockSpec(block_shape=(1,), index_map=lambda i: (0,)),
    pl.BlockSpec(block_shape=(1,), index_map=lambda i: (0,)),
  ],
  out_specs=pl.BlockSpec(
    block_shape=(
      1,
      (height - kernel_size + 1) // maxpool_kernel_size,
      (width - kernel_size + 1) // maxpool_kernel_size,
      out_channels,
    ),
    index_map=lambda i: (i, 0, 0, 0),
  ),
)(
  x,
  conv_vars["params"]["kernel"],
  conv_vars["params"]["bias"],
  gnorm_vars["params"]["scale"],
  gnorm_vars["params"]["bias"],
  scale,
  jnp.array([clamp_min]),
  jnp.array([clamp_max]),
).block_until_ready()
