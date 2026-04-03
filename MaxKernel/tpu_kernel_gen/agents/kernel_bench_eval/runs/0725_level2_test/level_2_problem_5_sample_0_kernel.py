# Imports
import flax.linen as nn
import jax
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
batch_size = 128
in_channels = 32
out_channels = 16
height, width = 16, 16
kernel_size = (4, 4)
stride = (2, 2)
# The padding argument in Flax's ConvTranspose is used to calculate the output shape.
# To match PyTorch's output shape given its padding and output_padding,
# we need to calculate the equivalent padding for Flax.
# PyTorch H_out = (H_in - 1) * stride - 2 * padding + kernel_size + output_padding
# JAX H_out = (H_in - 1) * stride + kernel_size - (pad_lo + pad_hi)
# Equating them: 2 * padding - output_padding = pad_lo + pad_hi
# With padding=1, output_padding=1, we get pad_lo + pad_hi = 2*1 - 1 = 1.
# We can choose (1, 0) for the padding.
padding = ((1, 0), (1, 0))
bias_shape = (1, 1, 1, out_channels)

key = random.PRNGKey(0)
key, params_key, bias_key, x_key = random.split(key, 4)

conv_transpose = nn.ConvTranspose(features=out_channels, kernel_size=kernel_size, strides=stride, padding=padding)

# Flax uses channels-last (NHWC) format, so the input shape is adjusted.
x = random.normal(x_key, (batch_size, height, width, in_channels))
params = conv_transpose.init(params_key, x)["params"]
bias = random.normal(bias_key, bias_shape)


# Computation
def kernel(x_ref, kernel_ref, conv_bias_ref, ext_bias_ref, out_ref):
  """
  Pallas kernel for a fused ConvTranspose -> Bias Subtraction -> Tanh operation.

  This kernel computes a single row of the output feature map for a single
  batch item.

  Args:
    x_ref: Reference to the input feature map for one batch item.
           Shape: (1, H_in, W_in, C_in)
    kernel_ref: Reference to the transposed convolution kernel weights.
                Shape: (kH, kW, C_in, C_out)
    conv_bias_ref: Reference to the bias from the ConvTranspose layer.
                   Shape: (C_out,)
    ext_bias_ref: Reference to the external bias to be subtracted.
                  Shape: (1, 1, 1, C_out)
    out_ref: Reference to the output buffer for a single row.
             Shape: (1, 1, W_out, C_out)
  """
  # These parameters are fixed by the original Flax layer definition.
  strides = (2, 2)
  padding = ((1, 0), (1, 0))

  # Get the index for the output row this kernel instance is responsible for.
  h_out_idx = pl.program_id(1)

  # The input `x_ref` has a leading dimension of 1. We need to prepare it for
  # `conv_transpose`, which expects a batch dimension.
  # `x_ref[0]` gives shape (H, W, C_in), `[None, ...]` adds the batch dim.
  lhs = x_ref[0][None, ...]

  # Perform the transposed convolution. This computes the full output feature map
  # for the given input slice.
  conv_out = jax.lax.conv_transpose(
    lhs=lhs,  # Shape: (1, H_in, W_in, C_in)
    rhs=kernel_ref[...],  # Shape: (kH, kW, C_in, C_out)
    strides=strides,
    padding=padding,
    dimension_numbers=("NHWC", "HWIO", "NHWC"),
    transpose_kernel=True,
  )

  # Fuse the subsequent operations:
  # 1. Add the convolution's own bias.
  # 2. Subtract the external bias.
  # 3. Apply the tanh activation function.
  # Broadcasting handles the shapes correctly.
  y = conv_out + conv_bias_ref[...]
  y = y - ext_bias_ref[...]
  y = jax.nn.tanh(y)

  # From the full computed output `y` (shape: 1, H_out, W_out, C_out),
  # select the single row that this kernel instance is assigned to compute.
  output_row = y[0, h_out_idx]  # Shape: (W_out, C_out)

  # Write the result to the output buffer, reshaping to match the out_spec.
  out_ref[...] = output_row.reshape(out_ref.shape)


# Calculate the output shape of the transposed convolution
H_out = (height - 1) * stride[0] + kernel_size[0] - (padding[0][0] + padding[0][1])
W_out = (width - 1) * stride[1] + kernel_size[1] - (padding[1][0] + padding[1][1])
output_shape = (batch_size, H_out, W_out, out_channels)

# The Pallas kernel will perform the fused operation:
# ConvTranspose -> Bias Subtraction -> Tanh
# The grid is defined over the batch and output height dimensions.
# Each kernel instance computes one row of the output feature map for one batch item.
x = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct(output_shape, x.dtype),
  grid=(batch_size, H_out),
  in_specs=[
    # Input feature map 'x': Pass the relevant batch item to each kernel instance.
    pl.BlockSpec(block_shape=(1, height, width, in_channels), index_map=lambda b, h: (b, 0, 0, 0)),
    # Conv kernel weights: Pass the full weight tensor to every kernel instance.
    pl.BlockSpec(block_shape=params["kernel"].shape, index_map=lambda b, h: tuple([0] * params["kernel"].ndim)),
    # Conv bias: Pass the full bias vector to every kernel instance.
    pl.BlockSpec(block_shape=params["bias"].shape, index_map=lambda b, h: (0,)),
    # External bias: Pass the full bias tensor to every kernel instance.
    pl.BlockSpec(block_shape=bias.shape, index_map=lambda b, h: tuple([0] * bias.ndim)),
  ],
  # Output spec: Each kernel instance (b, h) writes to the h-th row of the b-th output feature map.
  out_specs=pl.BlockSpec(block_shape=(1, 1, W_out, out_channels), index_map=lambda b, h: (b, h, 0, 0)),
)(x, params["kernel"], params["bias"], bias).block_until_ready()
