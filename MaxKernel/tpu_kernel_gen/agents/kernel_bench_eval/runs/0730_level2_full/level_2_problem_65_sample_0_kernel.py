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
height, width = 32, 32
kernel_size = 3
pool_kernel_size = 2

key = random.PRNGKey(0)
key_params, key_x = random.split(key)

x = random.normal(key_x, (batch_size, height, width, in_channels))

conv = nn.Conv(features=out_channels, kernel_size=(kernel_size, kernel_size))
params = conv.init(key_params, x)["params"]

# Define pooled dimensions for clarity
pooled_height = height // pool_kernel_size
pooled_width = width // pool_kernel_size


# Computation
def kernel(x_ref, kernel_ref, bias_ref, out_ref):
  """
  Pallas kernel that performs convolution, average pooling, and sigmoid activation.
  The final sum reduction is performed outside the kernel.
  """
  # Constants from the original code context
  pool_kernel_size = 2
  in_channels = x_ref.shape[-1]
  out_channels = kernel_ref.shape[-1]
  height, width = x_ref.shape[1], x_ref.shape[2]
  kernel_h, kernel_w = kernel_ref.shape[0], kernel_ref.shape[1]

  # Load data for a single batch item.
  x = x_ref[0]
  conv_kernel = kernel_ref[...]
  conv_bias = bias_ref[...]

  # 'SAME' padding calculation
  pad_h = (kernel_h - 1) // 2
  pad_w = (kernel_w - 1) // 2
  x_padded = jnp.pad(x, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)))

  # Efficient convolution implementation using lax.slice and jnp.dot
  conv_out = jnp.zeros((height, width, out_channels), dtype=x.dtype)
  for kh in range(kernel_h):
    for kw in range(kernel_w):
      x_slice = jax.lax.slice(x_padded, start_indices=(kh, kw, 0), limit_indices=(kh + height, kw + width, in_channels))
      k_slice = conv_kernel[kh, kw, :, :]
      conv_out += jnp.dot(x_slice, k_slice)

  conv_out = conv_out + conv_bias

  # Manual average pooling layer
  pooled_height = height // pool_kernel_size
  pooled_width = width // pool_kernel_size
  # Reshape for non-overlapping pooling
  conv_out_reshaped = conv_out.reshape(pooled_height, pool_kernel_size, pooled_width, pool_kernel_size, out_channels)
  # Sum over the pooling window axes and average
  sum_pooled = jnp.sum(conv_out_reshaped, axis=(1, 3))
  avg_pooled = sum_pooled / (pool_kernel_size * pool_kernel_size)

  # 3. Sigmoid activation
  activated = jax.nn.sigmoid(avg_pooled)

  # 4. Write the activated result (before reduction) to the output reference
  # Add a batch dimension back to match the out_ref shape
  out_ref[...] = jnp.expand_dims(activated, axis=0)


pallas_out = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct((batch_size, pooled_height, pooled_width, out_channels), x.dtype),
  grid=(batch_size,),
  in_specs=[
    pl.BlockSpec((1, height, width, in_channels), lambda i: (i, 0, 0, 0)),
    pl.BlockSpec(block_shape=params["kernel"].shape, index_map=lambda i: (0, 0, 0, 0)),
    pl.BlockSpec(block_shape=params["bias"].shape, index_map=lambda i: (0,)),
  ],
  out_specs=pl.BlockSpec((1, pooled_height, pooled_width, out_channels), lambda i: (i, 0, 0, 0)),
)(x, params["kernel"], params["bias"])

# Perform the final sum reduction outside of the Pallas kernel
result = jnp.sum(pallas_out, axis=(1, 2, 3)).block_until_ready()
