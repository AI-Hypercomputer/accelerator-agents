# Imports
import jax
import jax.numpy as jnp
import jax.random as random
from flax import linen as nn
from jax.experimental import pallas as pl
from jax.experimental.pallas.ops.tpu import convolution

# Initialization
batch_size = 128
in_channels = 3
out_channels = 16
height, width = 32, 32
kernel_size = 3
groups = 8
eps = 1e-5

key = random.PRNGKey(0)
key, x_key, conv_key, gn_key = random.split(key, 4)

# JAX uses channels-last convention (N, H, W, C)
x = random.normal(x_key, (batch_size, height, width, in_channels))

conv_layer = nn.Conv(features=out_channels, kernel_size=(kernel_size, kernel_size))
conv_params = conv_layer.init(conv_key, x)["params"]

# The output shape of the convolution is needed to initialize the next layer
# With 'VALID' padding: new_size = old_size - kernel_size + 1
conv_out_shape = (batch_size, height - kernel_size + 1, width - kernel_size + 1, out_channels)
group_norm = nn.GroupNorm(num_groups=groups, epsilon=eps)
group_norm_params = group_norm.init(gn_key, jnp.ones(conv_out_shape))["params"]


def kernel(
  x_ref,
  conv_kernel_ref,
  conv_bias_ref,
  gn_scale_ref,
  gn_bias_ref,
  out_ref,
  # The following are constants passed from the source code's initialization
  out_channels: int = 16,
  groups: int = 8,
  eps: float = 1e-5,
):
  """
  Pallas kernel that fuses Convolution, GroupNorm, and other activations.
  """
  # 1. Convolution
  # The input x_ref is a slice of the input tensor `x`. The convolution
  # is performed on this slice.
  # lhs shape: (1, kernel_size, width, in_channels)
  # rhs shape: (kernel_size, kernel_size, in_channels, out_channels)
  x_conv = convolution.conv(
    lhs=x_ref[...],
    rhs=conv_kernel_ref[...],
    window_strides=(1, 1),
    padding="VALID",
    dimension_numbers=("NHWC", "HWIO", "NHWC"),
  )
  # Add the convolution bias. It will broadcast over the N, H, W dimensions.
  x_conv = x_conv + conv_bias_ref[...]
  # Resulting x_conv shape: (1, 1, width - kernel_size + 1, out_channels)

  # 2. Group Normalization
  # Reshape the convolution output to group channels for normalization.
  # Shape changes from (N, H, W, C) to (N, H, W, G, C/G).
  x_conv_reshaped = x_conv.reshape(*x_conv.shape[:-1], groups, out_channels // groups)
  # Calculate mean and variance over the spatial and channel-group dimensions.
  # For the reshaped tensor, these are axes (1, 2, 4) -> (H, W, C/G).
  mean = jnp.mean(x_conv_reshaped, axis=(1, 2, 4), keepdims=True)
  var = jnp.var(x_conv_reshaped, axis=(1, 2, 4), keepdims=True)
  x_norm_reshaped = (x_conv_reshaped - mean) / jnp.sqrt(var + eps)
  # Reshape back to the original tensor layout.
  x_norm = x_norm_reshaped.reshape(*x_conv.shape)
  # Apply learnable scale and bias parameters.
  x_norm = x_norm * gn_scale_ref[...] + gn_bias_ref[...]

  # 3. Tanh Activation
  x_tanh = jax.nn.tanh(x_norm)

  # 4. Hard Swish Activation
  x_hard_swish = jax.nn.hard_swish(x_tanh)

  # 5. Residual Connection
  x_res = x_conv + x_hard_swish

  # 6. LogSumExp
  # Reduce the result over the channel dimension (axis=3).
  x_logsumexp = jax.nn.logsumexp(x_res, axis=3, keepdims=True)
  # Resulting shape: (1, 1, width - kernel_size + 1, 1)

  # 7. Write to Output
  # The final result has the same shape as the output block.
  out_ref[...] = x_logsumexp


# Computation
x_logsumexp = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct((batch_size, height - kernel_size + 1, width - kernel_size + 1, 1), x.dtype),
  grid=(batch_size, height - kernel_size + 1),
  in_specs=[
    # Input tensor x
    pl.BlockSpec(
      block_shape=(1, kernel_size, width, in_channels),
      index_map=lambda i, j: (i, j, 0, 0),
    ),
    # Convolution kernel weights
    pl.BlockSpec(
      block_shape=(kernel_size, kernel_size, in_channels, out_channels),
      index_map=lambda i, j: (0, 0, 0, 0),
    ),
    # Convolution bias
    pl.BlockSpec(block_shape=(out_channels,), index_map=lambda i, j: (0,)),
    # GroupNorm scale
    pl.BlockSpec(block_shape=(out_channels,), index_map=lambda i, j: (0,)),
    # GroupNorm bias
    pl.BlockSpec(block_shape=(out_channels,), index_map=lambda i, j: (0,)),
  ],
  out_specs=pl.BlockSpec(
    block_shape=(1, 1, width - kernel_size + 1, 1),
    index_map=lambda i, j: (i, j, 0, 0),
  ),
)(
  x,
  conv_params["kernel"],
  conv_params["bias"],
  group_norm_params["scale"],
  group_norm_params["bias"],
).block_until_ready()
