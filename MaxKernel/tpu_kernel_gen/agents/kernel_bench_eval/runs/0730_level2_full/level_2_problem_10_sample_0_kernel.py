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
height, width = 16, 16
kernel_size = 4
stride = 2
padding = "SAME"  # Note: Flax padding behavior differs from PyTorch. 'SAME' is used to match output size.
maxpool_kernel_size = 2
maxpool_stride = 2
hardtanh_min = -1.0
hardtanh_max = 1.0
key = random.PRNGKey(0)
key, x_key, params_key = random.split(key, 3)
x = random.normal(x_key, (batch_size, height, width, in_channels))
conv_transpose = nn.ConvTranspose(
  features=out_channels, kernel_size=(kernel_size, kernel_size), strides=(stride, stride), padding=padding
)
params = conv_transpose.init(params_key, x)["params"]


def kernel(x_ref, out_ref):
  # Constants from the source code
  hardtanh_min = -1.0
  hardtanh_max = 1.0

  # Load inputs into registers
  x = x_ref[...]

  # 3. Hard Tanh
  x = jnp.clip(x, hardtanh_min, hardtanh_max)

  # 4. Mean over spatial dimensions
  x = jnp.mean(x, axis=(1, 2), keepdims=True)

  # 5. Tanh
  x = jnp.tanh(x)

  # Write the final result to the output buffer
  out_ref[...] = x


# Computation
# 1. ConvTranspose (performed outside the kernel)
conv_out = conv_transpose.apply({"params": params}, x)

# 2. MaxPool (performed outside the kernel)
maxpool_out = lax.reduce_window(
  conv_out,
  -jnp.inf,
  jnp.maximum,
  window_dimensions=(1, maxpool_kernel_size, maxpool_kernel_size, 1),
  window_strides=(1, maxpool_stride, maxpool_stride, 1),
  padding="VALID",
)


# The rest of the operations are performed inside the Pallas kernel
x = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct((batch_size, 1, 1, out_channels), x.dtype),
  grid=(batch_size,),
  in_specs=[
    pl.BlockSpec(
      block_shape=(1, maxpool_out.shape[1], maxpool_out.shape[2], out_channels), index_map=lambda i: (i, 0, 0, 0)
    ),
  ],
  out_specs=pl.BlockSpec(block_shape=(1, 1, 1, out_channels), index_map=lambda i: (i, 0, 0, 0)),
)(maxpool_out).block_until_ready()
