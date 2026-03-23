# Imports
import flax.linen as nn
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
batch_size = 128
in_channels = 3
out_channels = 16
height, width = 32, 32
kernel_size = 3
bias_shape = (1, 1, 1, out_channels)
key = random.PRNGKey(0)
key_x, key_conv, key_bias = random.split(key, 3)
x = random.normal(key_x, (batch_size, height, width, in_channels))
conv_transpose = nn.ConvTranspose(features=out_channels, kernel_size=(kernel_size, kernel_size))
params = conv_transpose.init(key_conv, x)["params"]
bias = random.normal(key_bias, bias_shape)


# Computation
def kernel(x_ref, kernel_ref, bias_ref, out_ref):
  """
  Pallas kernel implementing a sequence of neural network operations.
  """
  # Step 1: Perform transposed convolution.
  # The dimension numbers specify the layout of the input, kernel, and output tensors.
  # 'NHWC' for input/output: Batch, Height, Width, Channels.
  # 'HWIO' for kernel: Height, Width, Input Channels, Output Channels.
  dn = lax.ConvDimensionNumbers(
    lhs_spec=(0, 3, 1, 2),  # NHWC -> NCHW
    rhs_spec=(2, 3, 0, 1),  # HWIO -> IOHW
    out_spec=(0, 3, 1, 2),  # NCHW -> NHWC
  )
  # lax.conv_transpose performs the core operation of nn.ConvTranspose.
  # Strides are (1, 1) and padding is 'SAME' to match the default Flax behavior.
  # transpose_kernel=True is crucial for using a standard forward kernel with
  # conv_transpose in a forward pass.
  conv_out = lax.conv_transpose(
    x_ref[...], kernel_ref[...], strides=(1, 1), padding="SAME", dimension_numbers=dn, transpose_kernel=True
  )

  # Step 2: Compute the mean across spatial dimensions (height and width).
  # keepdims=True maintains the rank of the tensor for broadcasting in the next step.
  mean_out = jnp.mean(conv_out, axis=(1, 2), keepdims=True)

  # Step 3: Add the bias term.
  # Broadcasting applies the bias across the batch dimension.
  biased_out = mean_out + bias_ref[...]

  # Step 4: Apply logsumexp across the channel dimension.
  # This is a common activation or normalization function.
  logsumexp_out = jax.scipy.special.logsumexp(biased_out, axis=3, keepdims=True)

  # Step 5: Sum the results across the remaining spatial and channel dimensions.
  # This reduces the tensor to a shape of (batch_size, 1).
  sum_out = jnp.sum(logsumexp_out, axis=(2, 3))

  # Step 6: Scale the final result and write to the output reference.
  out_ref[...] = sum_out * 10.0


x = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct((128, 1), x.dtype),
  grid=(16,),
  in_specs=[
    pl.BlockSpec(block_shape=(8, 32, 32, 3), index_map=lambda i: (i * 8, 0, 0, 0)),
    pl.BlockSpec(block_shape=(3, 3, 3, 16), index_map=lambda i: (0, 0, 0, 0)),
    pl.BlockSpec(block_shape=(1, 1, 1, 16), index_map=lambda i: (0, 0, 0, 0)),
  ],
  out_specs=pl.BlockSpec(block_shape=(8, 1), index_map=lambda i: (i * 8, 0)),
)(x, params["kernel"], bias).block_until_ready()
