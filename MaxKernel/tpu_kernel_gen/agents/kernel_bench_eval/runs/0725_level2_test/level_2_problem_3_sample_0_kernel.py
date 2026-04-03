# Imports
import flax.linen as nn
import jax
import jax.nn
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
batch_size = 128
in_channels = 32
out_channels = 64
depth, height, width = 16, 32, 32
kernel_size = (3, 3, 3)
stride = (2, 2, 2)
# The PyTorch padding=(1,1,1), stride=(2,2,2), kernel_size=(3,3,3) and output_padding=(1,1,1)
# configuration for ConvTranspose3d results in doubling the spatial dimensions.
# This is equivalent to padding='SAME' in Flax's ConvTranspose.
padding = "SAME"
sum_weight_value = 1.0
# The PyTorch LayerNorm(norm_shape) with norm_shape=(64,) normalizes over the last input dimension, which is width (W).
# To replicate this on a channels-last (N, D, H, W, C) tensor, we must normalize over the second to last axis (-2).
pool_kernel_size = (2, 2, 2)

key = random.PRNGKey(0)
key, x_key, params_key = random.split(key, 3)

# JAX and Flax use a channels-last convention by default (N, D, H, W, C)
x = random.normal(x_key, (batch_size, depth, height, width, in_channels))


class Model(nn.Module):
  """A Flax module encapsulating the sequence of operations."""

  @nn.compact
  def __call__(self, x):
    # In PyTorch, layers are instantiated with all parameters.
    # In Flax, layers are defined and parameters are created on the first call.
    x = nn.ConvTranspose(features=out_channels, kernel_size=kernel_size, strides=stride, padding=padding)(x)

    # nn.Parameter in PyTorch is equivalent to self.param in Flax.
    sum_weight = self.param(
      "sum_weight",
      nn.initializers.constant(sum_weight_value),
      (),  # A scalar parameter
    )
    x = x + sum_weight

    x = nn.LayerNorm(reduction_axes=-2, feature_axes=-2)(x)

    # Pooling and activation functions are often functional in Flax.
    x = nn.avg_pool(x, window_shape=pool_kernel_size, strides=pool_kernel_size)
    x = nn.gelu(x)
    return x


# Instantiate the model and initialize its parameters.
model = Model()
params = model.init(params_key, x)["params"]


# Computation
def kernel(x_ref, conv_kernel_ref, conv_bias_ref, sum_weight_ref, ln_scale_ref, ln_bias_ref, out_ref):
  """
  Pallas kernel that replicates the forward pass of the described Flax model.

  This kernel processes a single item from the input batch and applies the
  following sequence of operations:
  1. 3D Transposed Convolution
  2. Addition of a scalar weight
  3. Layer Normalization over the width dimension
  4. 3D Average Pooling
  5. GELU activation

  Args:
    x_ref: Reference to the input tensor slice for one batch item.
    conv_kernel_ref: Reference to the ConvTranspose kernel weights.
    conv_bias_ref: Reference to the ConvTranspose bias.
    sum_weight_ref: Reference to the custom scalar parameter.
    ln_scale_ref: Reference to the LayerNorm scale parameter (gamma).
    ln_bias_ref: Reference to the LayerNorm bias parameter (beta).
    out_ref: Reference to the output tensor slice for the result.
  """
  # Load the single batch item from HBM into SRAM.
  x = x_ref[...]

  # Load all model parameters. These are broadcasted to every kernel instance.
  conv_kernel = conv_kernel_ref[...]
  conv_bias = conv_bias_ref[...]
  # The scalar was reshaped to (1,) for pallas_call, so we extract the value.
  sum_weight = sum_weight_ref[0]
  ln_scale = ln_scale_ref[...]
  ln_bias = ln_bias_ref[...]

  # 1. Apply 3D Transposed Convolution.
  # The dimension numbers specify the layout for channels-last data.
  # Input/Output: (N, D, H, W, C), Kernel: (D, H, W, O, I)
  dn = ("NDHWC", "DHWOI", "NDHWC")
  x = jax.lax.conv_transpose(x, conv_kernel, strides=(2, 2, 2), padding="SAME", dimension_numbers=dn)
  x = x + conv_bias

  # 2. Add the custom scalar weight.
  x = x + sum_weight

  # 3. Apply Layer Normalization over the second-to-last axis (width).
  # Using the default epsilon from Flax's LayerNorm.
  mean = jnp.mean(x, axis=-2, keepdims=True)
  var = jnp.var(x, axis=-2, keepdims=True)
  x = (x - mean) / jnp.sqrt(var + 1e-6)
  # Apply learnable scale and bias. Their shapes must be reshaped to broadcast
  # over the correct (width) dimension.
  ln_scale_reshaped = jnp.reshape(ln_scale, (1, 1, 1, -1, 1))
  ln_bias_reshaped = jnp.reshape(ln_bias, (1, 1, 1, -1, 1))
  x = x * ln_scale_reshaped + ln_bias_reshaped

  # 4. Apply 3D Average Pooling.
  # The window and strides apply to the spatial dimensions (D, H, W).
  x = jax.lax.avg_pool(x, window_shape=(2, 2, 2), strides=(2, 2, 2), padding="VALID")

  # 5. Apply GELU activation function.
  x = jax.nn.gelu(x)

  # 6. Write the final result to the output buffer in HBM.
  out_ref[...] = x


# Create a list of parameters in the order expected by the kernel signature.
# This is more robust than relying on the alphabetical sorting of tree_flatten.
params_list = [
  params["ConvTranspose_0"]["kernel"],
  params["ConvTranspose_0"]["bias"],
  params["sum_weight"],
  params["LayerNorm_0"]["scale"],
  params["LayerNorm_0"]["bias"],
]

# Prepare the parameters for pallas_call:
# 1. Transpose the convolution kernel from (D,H,W,I,O) to (D,H,W,O,I)
#    to match the 'DHWOI' dimension spec used in the kernel.
params_list[0] = params_list[0].transpose((0, 1, 2, 4, 3))
# 2. Reshape the scalar weight to have at least one dimension.
params_list[2] = jnp.reshape(params_list[2], (1,))

params_for_pallas = params_list

# Define BlockSpec for the main input tensor `x`, chunking along the batch dimension.
in_spec_x = pl.BlockSpec(block_shape=(1, depth, height, width, in_channels), index_map=lambda i: (i, 0, 0, 0, 0))

# Define BlockSpecs for the model parameters. Each parameter is broadcasted
# in its entirety to every kernel instance.
in_specs_params = [
  pl.BlockSpec(block_shape=p.shape, index_map=lambda i, p=p: tuple([0] * p.ndim)) for p in params_for_pallas
]

# Combine all input specifications.
in_specs = [in_spec_x] + in_specs_params

# The output shape after the full model pass.
# ConvT (stride 2) doubles spatial dims, AvgPool (stride 2) halves them.
# x_in: (..., 16, 32, 32, 32) -> ConvT -> (..., 32, 64, 64, 64) -> AvgPool -> (..., 16, 32, 32, 64)
final_output_shape = (batch_size, 16, 32, 32, out_channels)

# Define the output shape structure for pallas_call.
out_shape = jax.ShapeDtypeStruct(final_output_shape, x.dtype)

# Define BlockSpec for the output tensor, chunking along the batch dimension.
out_spec = pl.BlockSpec(block_shape=(1, 16, 32, 32, out_channels), index_map=lambda i: (i, 0, 0, 0, 0))

# Invoke the Pallas kernel.
# The grid is defined by the batch size, so each kernel instance processes one batch item.
# The inputs to the kernel are the original data `x` and the ordered parameters.
x = pl.pallas_call(
  kernel,
  out_shape=out_shape,
  grid=(batch_size,),
  in_specs=in_specs,
  out_specs=out_spec,
)(x, *params_for_pallas).block_until_ready()
