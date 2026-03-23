# Imports
import jax
import jax.numpy as jnp
import jax.random as random
from flax import linen as nn
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

# Initialization
batch_size = 128
in_channels = 3
out_channels = 16
height, width = 32, 32
kernel_size = 3
divide_by = 2.0

key = random.PRNGKey(0)
key_x, key_conv, key_norm = random.split(key, 3)

# JAX uses channels-last convention
x = random.normal(key_x, (batch_size, height, width, in_channels))

conv = nn.Conv(features=out_channels, kernel_size=(kernel_size, kernel_size))
conv_params = conv.init(key_conv, x)["params"]

# InstanceNorm infers channel dimension from input during init
instance_norm = nn.InstanceNorm()
dummy_conv_output = jnp.ones((batch_size, height, width, out_channels))
norm_params = instance_norm.init(key_norm, dummy_conv_output)["params"]


# Computation
def kernel(x_ref, kernel_ref, conv_bias_ref, norm_scale_ref, norm_bias_ref, divide_by_ref, out_ref):
  """
  Pallas kernel that fuses Convolution, Instance Normalization, and division.

  Args:
    x_ref: Input image block.
    kernel_ref: Convolution kernel weights.
    conv_bias_ref: Convolution bias.
    norm_scale_ref: InstanceNorm scale parameter.
    norm_bias_ref: InstanceNorm bias parameter.
    divide_by_ref: Scalar division factor.
    out_ref: Output block.
  """
  # 1. Convolution Layer
  # We use the Pallas-specific TPU convolution primitive, which is optimized for the hardware.
  # It assumes 'NHWC' for input/output and 'HWIO' for the kernel.
  # Strides are (1, 1) and padding is 'SAME', matching Flax's nn.Conv defaults.
  # This primitive also handles the bias addition.
  conv_out = pltpu.convolution(
    lhs=x_ref,
    rhs=kernel_ref,
    bias=conv_bias_ref,
    window_strides=(1, 1),
    padding="SAME",
    dimension_numbers=("NHWC", "HWIO", "NHWC"),
  )

  # 2. Instance Normalization Layer
  # InstanceNorm calculates statistics over the spatial dimensions (H, W)
  # for each channel independently. For a (N, H, W, C) tensor, these are axes (1, 2).
  # The default epsilon from flax.linen.normalization.InstanceNorm is 1e-5.
  reduction_axes = (1, 2)
  epsilon = 1e-5

  # Calculate mean and variance. keepdims=True is important for broadcasting.
  mean = jnp.mean(conv_out, axis=reduction_axes, keepdims=True)
  var = jnp.var(conv_out, axis=reduction_axes, keepdims=True)

  # Normalize the output of the convolution.
  norm_out = (conv_out - mean) * jax.lax.rsqrt(var + epsilon)

  # Apply the learned scale and bias parameters.
  scaled_out = norm_out * norm_scale_ref[...] + norm_bias_ref[...]

  # 3. Division Operation
  # Perform the final element-wise division.
  final_out = scaled_out / divide_by_ref[...]

  # Write the final result to the output buffer.
  out_ref[...] = final_out


# The 'divide_by' scalar is converted to a 1-element array to be passed via in_specs.
divide_by_arr = jnp.array([divide_by], dtype=x.dtype)

# The fused computation is invoked via a pallas_call to the kernel.
x = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct((batch_size, height, width, out_channels), x.dtype),
  grid=(batch_size,),
  in_specs=[
    # Input images are sliced along the batch dimension.
    pl.BlockSpec(block_shape=(1, height, width, in_channels), index_map=lambda i: (i, 0, 0, 0)),
    # All other parameters are broadcasted to all kernel instances.
    pl.BlockSpec(block_shape=conv_params["kernel"].shape, index_map=lambda i: tuple([0] * conv_params["kernel"].ndim)),
    pl.BlockSpec(block_shape=conv_params["bias"].shape, index_map=lambda i: (0,)),
    pl.BlockSpec(block_shape=norm_params["scale"].shape, index_map=lambda i: (0,)),
    pl.BlockSpec(block_shape=norm_params["bias"].shape, index_map=lambda i: (0,)),
    pl.BlockSpec(block_shape=(1,), index_map=lambda i: (0,)),
  ],
  out_specs=pl.BlockSpec(block_shape=(1, height, width, out_channels), index_map=lambda i: (i, 0, 0, 0)),
)(
  x, conv_params["kernel"], conv_params["bias"], norm_params["scale"], norm_params["bias"], divide_by_arr
).block_until_ready()
