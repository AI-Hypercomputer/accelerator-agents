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
groups = 4
eps = 1e-5

key = random.PRNGKey(0)
key_x, key_conv, key_gn = random.split(key, 3)

# JAX uses channels-last convention: (N, D, H, W, C)
x = random.normal(key_x, (batch_size, depth, height, width, in_channels))

conv_transpose = nn.ConvTranspose(
  features=out_channels,
  kernel_size=(kernel_size, kernel_size, kernel_size),
  strides=(stride, stride, stride),
  padding=padding,
  use_bias=True,
)
params_conv = conv_transpose.init(key_conv, x)["params"]

# Determine output shape of conv_transpose to initialize group_norm
conv_output_shape_struct = jax.eval_shape(lambda: conv_transpose.apply({"params": params_conv}, x))
dummy_gn_input = jnp.zeros(conv_output_shape_struct.shape, dtype=conv_output_shape_struct.dtype)

group_norm = nn.GroupNorm(num_groups=groups, epsilon=eps)
params_gn = group_norm.init(key_gn, dummy_gn_input)["params"]


# Computation
def kernel(x_ref, kernel_ref, bias_ref, scale_ref, gn_bias_ref, out_ref):
  """
  Pallas kernel that fuses ConvTranspose, SiLU, GroupNorm, and HardSwish.

  Args:
    x_ref: Input tensor reference for a single batch item.
    kernel_ref: Convolution kernel weights reference.
    bias_ref: Convolution bias reference.
    scale_ref: Group normalization scale (gamma) reference.
    gn_bias_ref: Group normalization bias (beta) reference.
    out_ref: Output tensor reference.
  """
  # Hardcoded parameters from the original model definition.
  strides = (2, 2, 2)
  num_groups = 4
  eps = 1e-5
  # JAX standard dimension layout: (Batch, Depth, Height, Width, Channels)
  # We implement conv_transpose via conv_general_dilated, which needs a
  # transposed kernel. The layout for the transposed kernel is DHWIO.
  dimension_numbers = ("NDHWC", "DHWIO", "NDHWC")

  # Load all inputs from memory into registers.
  x = x_ref[...]
  kernel_w = kernel_ref[...]
  conv_b = bias_ref[...]
  gn_scale = scale_ref[...]
  gn_bias = gn_bias_ref[...]

  # --- 1. Transposed Convolution (emulated with forward convolution) ---
  # lax.conv_transpose is not implemented in Pallas TPU. We emulate it by
  # using lax.conv_general_dilated with a transposed kernel and input dilation.
  # Flax kernel (D,H,W,O,I) -> Transposed kernel (D,H,W,I,O)
  kernel_w_transposed = jnp.transpose(kernel_w, (0, 1, 2, 4, 3))

  # Use conv_general_dilated to perform the operation.
  # window_strides are 1, and lhs_dilation is the stride of the transpose conv.
  conv_out = jax.lax.conv_general_dilated(
    x,
    kernel_w_transposed,
    window_strides=(1, 1, 1),
    padding=((1, 1), (1, 1), (1, 1)),
    lhs_dilation=strides,
    dimension_numbers=dimension_numbers,
  )

  # Add the convolution bias.
  conv_out_biased = conv_out + conv_b

  # --- 2. SiLU Activation ---
  silu_out = nn.silu(conv_out_biased)

  # --- 3. Group Normalization ---
  # Reshape to group channels for statistics calculation.
  # Shape changes from (N, D, H, W, C) to (N, D*H*W, num_groups, C/num_groups).
  group_channels = silu_out.shape[-1] // num_groups
  x_reshaped = silu_out.reshape((1, -1, num_groups, group_channels))

  # Calculate mean and variance over the spatial and grouped channel dimensions.
  mean = jnp.mean(x_reshaped, axis=(1, 3), keepdims=True)
  var = jnp.var(x_reshaped, axis=(1, 3), keepdims=True)

  # Normalize and reshape back to the original tensor shape.
  x_norm = (x_reshaped - mean) / jnp.sqrt(var + eps)
  x_norm_reshaped = x_norm.reshape(silu_out.shape)

  # Apply scale and bias.
  gn_out = x_norm_reshaped * gn_scale + gn_bias

  # --- 4. HardSwish Activation ---
  hard_swish_out = nn.hard_swish(gn_out)

  # --- 5. Write final result to the output buffer ---
  out_ref[...] = hard_swish_out


# Determine the output shape and dtype for the pallas_call
# This is done by evaluating the shape of the original computation.
out_struct = jax.eval_shape(
  lambda x, p_conv, p_gn: nn.hard_swish(
    group_norm.apply(
      {"params": p_gn},
      nn.silu(conv_transpose.apply({"params": p_conv}, x)),
    )
  ),
  x,
  params_conv,
  params_gn,
)

# The grid is defined to parallelize the computation over the batch dimension.
# Each kernel instance will process one item from the batch.
grid = (batch_size,)

result = pl.pallas_call(
  kernel,
  out_shape=out_struct,
  grid=grid,
  in_specs=[
    # Input tensor 'x' is chunked along the batch dimension.
    pl.BlockSpec(block_shape=(1, *x.shape[1:]), index_map=lambda i: (i, 0, 0, 0, 0)),
    # Convolution kernel parameters are shared across all kernel instances.
    # The index_map returns a tuple of zeros to broadcast the entire tensor.
    pl.BlockSpec(block_shape=params_conv["kernel"].shape, index_map=lambda i: (0, 0, 0, 0, 0)),
    pl.BlockSpec(block_shape=params_conv["bias"].shape, index_map=lambda i: (0,)),
    # Group normalization parameters are also shared.
    pl.BlockSpec(block_shape=params_gn["scale"].shape, index_map=lambda i: (0,)),
    pl.BlockSpec(block_shape=params_gn["bias"].shape, index_map=lambda i: (0,)),
  ],
  # The output is chunked along the batch dimension, corresponding to the grid.
  out_specs=pl.BlockSpec(block_shape=(1, *out_struct.shape[1:]), index_map=lambda i: (i, 0, 0, 0, 0)),
)(
  x,
  params_conv["kernel"],
  params_conv["bias"],
  params_gn["scale"],
  params_gn["bias"],
).block_until_ready()
