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
# Adjusted spatial dimensions to meet TPU block shape requirements after convolution.
# The output spatial dimensions must be divisible by 8.
# With kernel_size=3 and 'VALID' padding, out_dim = in_dim - 2.
# To make out_dim divisible by 8, in_dim must be 2 + a multiple of 8.
# e.g., height=34 -> out_height=32. width=34 -> out_width=32.
depth, height, width = 16, 34, 34
kernel_size = 3
# For JAX (..., features), multiplier should broadcast on the last axis
multiplier_shape = (1, 1, 1, 1, out_channels)
clamp_min = -1.0
clamp_max = 1.0
epsilon = 1e-5  # Default for flax.linen.InstanceNorm

key = random.PRNGKey(0)
key_x, key_conv, key_multiplier, key_norm = random.split(key, 4)

# JAX convention is (..., spatial_dims..., features)
x_shape = (batch_size, depth, height, width, in_channels)
x = random.normal(key_x, x_shape)

# Use use_bias=True to get a bias parameter, which is needed by the kernel.
conv = nn.Conv(features=out_channels, kernel_size=(kernel_size,) * 3, use_bias=True)
conv_params = conv.init(key_conv, x)["params"]

multiplier = random.normal(key_multiplier, multiplier_shape)

# Flax's InstanceNorm has learnable scale and bias parameters by default.
instance_norm = nn.InstanceNorm()
# The input shape for initialization must match the output shape of the convolution.
# Flax's 'VALID' padding default reduces spatial dims: D' = D - K + 1.
conv_output_shape = (
  batch_size,
  depth - kernel_size + 1,
  height - kernel_size + 1,
  width - kernel_size + 1,
  out_channels,
)
dummy_conv_output = jnp.ones(conv_output_shape)
# The initialized parameters will include 'scale' and 'bias'.
norm_params = instance_norm.init(key_norm, dummy_conv_output)["params"]


# Computation
def kernel(x_ref, multiplier_ref, norm_scale_ref, norm_bias_ref, out_ref):
  """
  Pallas kernel that combines instance normalization and other element-wise
  operations. The convolution is performed outside the kernel.
  """
  # The input x_ref is the result of the convolution.
  x = x_ref[...]

  # 1. First Multiplication
  x = x * multiplier_ref[...]

  # 2. Instance Normalization
  # Calculate mean and variance over spatial dimensions (D, H, W) for each channel.
  # The axes are (1, 2, 3) because the shape is (N, D, H, W, C).
  mean = jnp.mean(x, axis=(1, 2, 3), keepdims=True)
  var = jnp.var(x, axis=(1, 2, 3), keepdims=True)
  # Normalize the tensor.
  x_normalized = (x - mean) / jnp.sqrt(var + epsilon)
  # Apply learnable scale and bias parameters.
  x = norm_scale_ref[...] * x_normalized + norm_bias_ref[...]

  # 3. Clipping
  x = jnp.clip(x, a_min=clamp_min, a_max=clamp_max)

  # 4. Second Multiplication
  x = x * multiplier_ref[...]

  # 5. Max Reduction
  # Find the maximum value along the channel axis (-1).
  final_out = jnp.max(x, axis=-1)

  # 6. Store Result
  # The shape of final_out matches the out_ref block shape.
  out_ref[...] = final_out


# Perform the 3D convolution using standard Flax/JAX, as it's not supported
# inside a Pallas kernel on TPU.
conv_out = conv.apply({"params": conv_params}, x)

# The kernel combines normalization, and element-wise operations.
# We parallelize over the batch dimension. Each kernel instance handles one item.
# The grid will be of size (batch_size,).

# Calculate the spatial dimensions of the output of the convolution.
out_depth = depth - kernel_size + 1
out_height = height - kernel_size + 1
out_width = width - kernel_size + 1

# The final output shape after the max reduction along the channel axis.
final_out_shape = (batch_size, out_depth, out_height, out_width)

# The kernel takes the output of the convolution, the multiplier, and
# instance norm parameters (scale, bias).
x = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct(final_out_shape, x.dtype),
  grid=(batch_size,),
  in_specs=[
    # Input tensor (convolution output): sliced along the batch dimension.
    pl.BlockSpec(block_shape=(1, out_depth, out_height, out_width, out_channels), index_map=lambda i: (i, 0, 0, 0, 0)),
    # Multiplier: static, same for all instances.
    pl.BlockSpec(block_shape=multiplier.shape, index_map=lambda i: (0,) * multiplier.ndim),
    # InstanceNorm scale: static, same for all instances.
    pl.BlockSpec(block_shape=norm_params["scale"].shape, index_map=lambda i: (0,)),
    # InstanceNorm bias: static, same for all instances.
    pl.BlockSpec(block_shape=norm_params["bias"].shape, index_map=lambda i: (0,)),
  ],
  out_specs=pl.BlockSpec(block_shape=(1, out_depth, out_height, out_width), index_map=lambda i: (i, 0, 0, 0)),
)(conv_out, multiplier, norm_params["scale"], norm_params["bias"]).block_until_ready()
