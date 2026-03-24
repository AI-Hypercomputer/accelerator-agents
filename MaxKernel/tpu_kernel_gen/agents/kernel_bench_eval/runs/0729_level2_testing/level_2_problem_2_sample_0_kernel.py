# Imports
import jax
import jax.numpy as jnp
import jax.random as random
from flax import linen as nn
from jax.experimental import pallas as pl

# Initialization
batch_size = 128
in_channels = 3
out_channels = 16
height, width = 32, 32
kernel_size = 3
stride = 2
# NOTE: The PyTorch combination of padding=1, stride=2, output_padding=1
# for a kernel_size=3 is a common way to achieve 2x upsampling.
# The equivalent in Flax is to use explicit padding that sums to 1
# (i.e., 2 * pytorch_padding - pytorch_output_padding).
padding = ((0, 1), (0, 1))
# Ensure scaling_factor is a JAX type, which can be more robust in JIT contexts.
scaling_factor = jnp.float32(2.0)

key = random.PRNGKey(0)
key_x, key_params, key_bias = random.split(key, 3)

# JAX uses channels-last convention (N, H, W, C)
x = random.normal(key_x, (batch_size, height, width, in_channels))

# Use use_bias=False as we are adding the bias manually in the kernel.
conv_transpose = nn.ConvTranspose(
  features=out_channels,
  kernel_size=(kernel_size, kernel_size),
  strides=(stride, stride),
  padding=padding,
  use_bias=False,
)
params = conv_transpose.init(key_params, x)["params"]
bias = random.normal(key_bias, (1, 1, 1, out_channels))

# Perform the transposed convolution using standard Flax/JAX.
y_conv = conv_transpose.apply({"params": params}, x)


# Define a Pallas kernel for the fused element-wise operations.
def fused_kernel(y_conv_ref, bias_ref, out_ref, scaling_factor):
  y = y_conv_ref[...] + bias_ref[...]
  y = jnp.clip(y, a_min=0.0, a_max=1.0)
  y = y * scaling_factor
  y = jnp.clip(y, a_min=0.0, a_max=1.0)
  y = y / scaling_factor
  out_ref[...] = y


# The output shape of the Pallas call is the same as the conv output.
output_shape = y_conv.shape

# Invoke the Pallas kernel.
result = pl.pallas_call(
  fused_kernel,
  out_shape=jax.ShapeDtypeStruct(output_shape, x.dtype),
  grid=(batch_size,),
  in_specs=[
    pl.BlockSpec(block_shape=(1, output_shape[1], output_shape[2], output_shape[3]), index_map=lambda i: (i, 0, 0, 0)),
    # Pass the entire bias tensor to each kernel instance.
    pl.BlockSpec(block_shape=bias.shape, index_map=lambda i: (0, 0, 0, 0)),
  ],
  out_specs=pl.BlockSpec(
    block_shape=(1, output_shape[1], output_shape[2], output_shape[3]), index_map=lambda i: (i, 0, 0, 0)
  ),
)(y_conv, bias, scaling_factor)

result.block_until_ready()
