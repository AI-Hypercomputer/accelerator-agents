# Imports
import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.random as random
from jax import lax
from jax.experimental import pallas as pl

# Initialization
batch_size = 128
in_channels = 32
out_channels = 64
D, H, W = 16, 16, 16
kernel_size = 3
stride = 2
padding = "SAME"  # In Flax, 'SAME' padding with stride gives output_size = input_size * stride


class ConvTransposeModel(nn.Module):
  @nn.compact
  def __call__(self, x):
    # PyTorch's (in_channels, out_channels) becomes features=out_channels in Flax
    # PyTorch's padding/output_padding logic to get output_size = input_size * stride
    # is simplified to padding='SAME' in Flax/JAX.
    return nn.ConvTranspose(
      features=out_channels,
      kernel_size=(kernel_size, kernel_size, kernel_size),
      strides=(stride, stride, stride),
      padding=padding,
    )(x)


key = random.PRNGKey(0)
key, key_init, key_x, key_add = random.split(key, 4)

# Flax models are stateless, so we initialize parameters separately.
model = ConvTransposeModel()
# JAX uses channel-last convention (N, D, H, W, C) by default.
dummy_x = jnp.empty((batch_size, D, H, W, in_channels), dtype=jnp.float32)
params = model.init(key_init, dummy_x)["params"]

x = random.normal(key_x, (batch_size, D, H, W, in_channels))
add_input = random.normal(key_add, (batch_size, D * stride, H * stride, W * stride, out_channels))


# Computation
def kernel(x_ref, kernel_ref, bias_ref, add_input_ref, out_ref):
  """Pallas kernel for fused ConvTranspose + Add + HardSwish.

  Args:
    x_ref: Input tensor block.
    kernel_ref: Convolution kernel weights.
    bias_ref: Convolution bias.
    add_input_ref: Tensor to be added element-wise.
    out_ref: Output tensor block to be written to.
  """
  # Each program in the grid computes a slice of the full output.
  # The full transposed convolution is computed first, as it's not easily divisible.

  # Define the dimension numbers for a 3D convolution with channel-last format.
  # Input: (N, D, H, W, C) -> batch=0, feature=4, spatial=(1,2,3)
  # Kernel: (D, H, W, I, O) -> out_feature=4, in_feature=3, spatial=(0,1,2)
  # Output: (N, D, H, W, C) -> batch=0, feature=4, spatial=(1,2,3)
  dn = lax.ConvDimensionNumbers(lhs_spec=(0, 4, 1, 2, 3), rhs_spec=(4, 3, 0, 1, 2), out_spec=(0, 4, 1, 2, 3))

  # Perform the 3D transposed convolution.
  # The strides and padding match the Flax nn.ConvTranspose configuration.
  # The input `x_ref` already has the correct shape (1, D, H, W, C_in)
  # which `lax.conv_transpose` treats as a batch of 1.
  conv_out = lax.conv_transpose(
    x_ref[...], kernel_ref[...], strides=(stride, stride, stride), padding=padding, dimension_numbers=dn
  )

  # Add the bias term. It will be broadcasted over the spatial dimensions.
  conv_out_bias = conv_out + bias_ref[...]

  # The grid is (batch_size, D * stride), so program_id(1) corresponds to the
  # output depth dimension. We slice the result of the full convolution
  # to get the part this program is responsible for.
  j = pl.program_id(1)
  # `lax.dynamic_slice_in_dim` extracts a slice of size 1 at index `j` along axis 1 (the depth dim).
  y = lax.dynamic_slice_in_dim(conv_out_bias, j, slice_size=1, axis=1)

  # Add the second input tensor. `add_input_ref` is already the correct slice.
  y = y + add_input_ref[...]

  # Apply the hard_swish activation function: x * relu6(x + 3) / 6
  y = y * jnp.maximum(0.0, jnp.minimum(6.0, y + 3.0)) / 6.0

  # Write the final result to the output buffer.
  out_ref[...] = y


# The kernel would perform the fused ConvTranspose + Add + HardSwish operation.
# The grid is designed to parallelize work across the batch and output depth dimensions.
# The output shape is determined by the final operation in the sequence.
final_output_shape = (batch_size, D * stride, H * stride, W * stride, out_channels)

# In Flax, the parameters for a layer are nested within a dictionary named after the layer class.
# The default name for the ConvTranspose layer is 'ConvTranspose_0'.
conv_params = params["ConvTranspose_0"]

y = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct(final_output_shape, x.dtype),
  grid=(batch_size, D * stride),
  in_specs=[
    # x: Full slice per batch item, as ConvTranspose needs a receptive field.
    pl.BlockSpec(block_shape=(1, D, H, W, in_channels), index_map=lambda i, j: (i, 0, 0, 0, 0)),
    # kernel: Not chunked, needed by all instances.
    pl.BlockSpec(
      block_shape=conv_params["kernel"].shape, index_map=lambda i, j: tuple([0] * conv_params["kernel"].ndim)
    ),
    # bias: Not chunked, needed by all instances.
    pl.BlockSpec(block_shape=conv_params["bias"].shape, index_map=lambda i, j: tuple([0] * conv_params["bias"].ndim)),
    # add_input: Chunked along the grid dimensions, matching the output.
    pl.BlockSpec(block_shape=(1, 1, H * stride, W * stride, out_channels), index_map=lambda i, j: (i, j, 0, 0, 0)),
  ],
  out_specs=pl.BlockSpec(
    block_shape=(1, 1, H * stride, W * stride, out_channels), index_map=lambda i, j: (i, j, 0, 0, 0)
  ),
)(x, conv_params["kernel"], conv_params["bias"], add_input).block_until_ready()
