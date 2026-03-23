# Imports
import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.random as random
from flax.core import freeze, unfreeze
from jax import lax
from jax.experimental import pallas as pl

# Initialization
batch_size = 16
in_channels = 16
out_channels = 32
depth, height, width = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1
bias = True

key = random.PRNGKey(0)
key_init, key_x = random.split(key)

# JAX uses channels-last format
x = random.normal(key_x, (batch_size, depth, height, width, in_channels))

# In Flax, a sequence of layers is best represented by nn.Sequential
model = nn.Sequential(
  [
    nn.ConvTranspose(features=out_channels, kernel_size=kernel_size, strides=stride, padding=padding, use_bias=bias),
    # The `use_running_average` parameter must be passed to either the constructor
    # or the __call__ method. Since nn.Sequential does not support passing it
    # during the call, we set it in the constructor. For training, it should be False.
    nn.BatchNorm(use_running_average=False),
  ]
)

# Initialize the model to get parameters and batch statistics
# We run a dummy forward pass to populate the batch_stats (mean and variance)
# which are needed for the kernel.
variables = model.init(key_init, x)
_, updated_state = model.apply(variables, x, mutable=["batch_stats"])
variables = unfreeze(variables)
variables["batch_stats"] = updated_state["batch_stats"]
variables = freeze(variables)


def kernel(x_ref, conv_kernel_ref, conv_bias_ref, bn_scale_ref, bn_bias_ref, bn_mean_ref, bn_var_ref, out_ref):
  """
  Pallas kernel for a sequence of ConvTranspose, BatchNorm, and mean subtraction.

  NOTE: This kernel assumes an inference-like application of BatchNorm, using
  pre-computed running statistics for mean and variance. This is a necessary
  simplification because computing batch-wide statistics in training mode is
  not straightforward with the provided grid decomposition (`grid=(batch_size,)`),
  which processes each batch item independently.

  Args:
    x_ref: Input tensor reference for a single batch item.
    conv_kernel_ref: ConvTranspose kernel weights.
    conv_bias_ref: ConvTranspose bias.
    bn_scale_ref: BatchNorm scale parameter.
    bn_bias_ref: BatchNorm bias parameter.
    bn_mean_ref: BatchNorm running mean.
    bn_var_ref: BatchNorm running variance.
    out_ref: Output tensor reference.
  """
  # In JAX (N, D, H, W, C), spatial dimensions are 1, 2, 3
  # Kernel weights are (D, H, W, I, O)
  # We define the dimension numbers for JAX's channels-last format.
  dimension_numbers = ("NDHWC", "DHWIO", "NDHWC")
  stride_val = (stride, stride, stride)
  padding_val = ((padding, padding), (padding, padding), (padding, padding))

  # 1. ConvTranspose operation
  # The stride is applied to the spatial dimensions.
  conv_out = lax.conv_transpose(
    lhs=x_ref[...],
    rhs=conv_kernel_ref[...],
    strides=stride_val,
    padding=padding_val,
    dimension_numbers=dimension_numbers,
  )

  # 2. Add bias
  # The bias has shape (out_channels,) and is broadcast.
  conv_out_biased = conv_out + conv_bias_ref[...]

  # 3. BatchNorm operation (inference mode)
  # y = scale * (x - mean) / sqrt(variance + epsilon) + bias
  epsilon = 1e-5
  # The mean, var, scale, and bias have shape (out_channels,) and are broadcast.
  normalized = (conv_out_biased - bn_mean_ref[...]) * lax.rsqrt(bn_var_ref[...] + epsilon)
  bn_out = normalized * bn_scale_ref[...] + bn_bias_ref[...]

  # 4. Final mean subtraction over spatial dimensions
  # For a single item of shape (1, D, H, W, C), the spatial axes are 1, 2, 3.
  spatial_mean = jnp.mean(bn_out, axis=(1, 2, 3), keepdims=True)
  final_output = bn_out - spatial_mean

  # 5. Write result to the output reference
  out_ref[...] = final_output


# Calculate the output dimensions after the ConvTranspose operation
out_depth = (depth - 1) * stride - 2 * padding + kernel_size
out_height = (height - 1) * stride - 2 * padding + kernel_size
out_width = (width - 1) * stride - 2 * padding + kernel_size
output_shape = (batch_size, out_depth, out_height, out_width, out_channels)

# Computation
# Extract parameters and batch stats from the initialized model variables
# For nn.Sequential, layers are named by default: 'layers_0', 'layers_1', ...
conv_kernel = variables["params"]["layers_0"]["kernel"]
conv_bias = variables["params"]["layers_0"]["bias"]
bn_scale = variables["params"]["layers_1"]["scale"]
bn_bias = variables["params"]["layers_1"]["bias"]
bn_mean = variables["batch_stats"]["layers_1"]["mean"]
bn_var = variables["batch_stats"]["layers_1"]["var"]

# The computation is parallelized over the batch dimension. Each kernel instance
# processes one item from the batch.
x = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct(output_shape, x.dtype),
  grid=(batch_size,),
  in_specs=[
    pl.BlockSpec(block_shape=(1, depth, height, width, in_channels), index_map=lambda i: (i, 0, 0, 0, 0)),
    pl.BlockSpec(block_shape=conv_kernel.shape, index_map=lambda i: (0, 0, 0, 0, 0)),
    pl.BlockSpec(block_shape=conv_bias.shape, index_map=lambda i: (0,)),
    pl.BlockSpec(block_shape=bn_scale.shape, index_map=lambda i: (0,)),
    pl.BlockSpec(block_shape=bn_bias.shape, index_map=lambda i: (0,)),
    pl.BlockSpec(block_shape=bn_mean.shape, index_map=lambda i: (0,)),
    pl.BlockSpec(block_shape=bn_var.shape, index_map=lambda i: (0,)),
  ],
  out_specs=pl.BlockSpec(
    block_shape=(1, out_depth, out_height, out_width, out_channels), index_map=lambda i: (i, 0, 0, 0, 0)
  ),
)(x, conv_kernel, conv_bias, bn_scale, bn_bias, bn_mean, bn_var).block_until_ready()
