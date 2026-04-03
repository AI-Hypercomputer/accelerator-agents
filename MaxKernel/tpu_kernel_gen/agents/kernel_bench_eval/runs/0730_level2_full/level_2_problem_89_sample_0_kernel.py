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
depth, height, width = 16, 32, 32
kernel_size = 3
stride = 2
padding = "SAME"
pool_kernel_size = 2
pool_stride = 2
pool_padding = "VALID"  # PyTorch padding=0 is 'VALID' in JAX/Flax
key = random.PRNGKey(0)
key, x_key, params_key, subtract_key = random.split(key, 4)
x = random.normal(x_key, (batch_size, depth, height, width, in_channels))
conv_transpose = nn.ConvTranspose(
  features=out_channels,
  kernel_size=(kernel_size, kernel_size, kernel_size),
  strides=(stride, stride, stride),
  padding=padding,
)
conv_params = conv_transpose.init(params_key, x)["params"]
subtract_param = random.normal(subtract_key, (out_channels,))

# Computation
# The unsupported conv_transpose and reduce_window operations are performed outside of the Pallas kernel.
x_conv = conv_transpose.apply({"params": conv_params}, x)
x_pooled = jax.lax.reduce_window(
  x_conv,
  -jnp.inf,
  jax.lax.max,
  (1, pool_kernel_size, pool_kernel_size, pool_kernel_size, 1),
  (1, pool_stride, pool_stride, pool_stride, 1),
  "VALID",
)


def kernel(x_pooled_ref, subtract_param_ref, out_ref):
  # Load the result of the convolution and pooling
  x = x_pooled_ref[...]

  # 3. Softmax
  x = nn.softmax(x, axis=-1)

  # 4. Subtract
  x = x - subtract_param_ref[...]

  # 5. SiLU Activation
  x = nn.silu(x)

  # Store the result before the final reduction
  out_ref[...] = x


# The output shape of the kernel is the same as the input shape
kernel_out_shape = x_pooled.shape

# Define block sizes for tiling to fit into memory.
# x_pooled.shape is (128, 16, 32, 32, 16)
# We tile along the (16, 32, 32) dimensions.
d_bs, h_bs, w_bs = 4, 8, 8
grid = (
  batch_size,
  x_pooled.shape[1] // d_bs,
  x_pooled.shape[2] // h_bs,
  x_pooled.shape[3] // w_bs,
)
block_shape = (1, d_bs, h_bs, w_bs, out_channels)

x_pre_reduction = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct(kernel_out_shape, x.dtype),
  grid=grid,
  in_specs=[
    pl.BlockSpec(block_shape, lambda i, jd, jh, jw: (i, jd, jh, jw, 0)),
    pl.BlockSpec(subtract_param.shape, lambda *_: (0,)),
  ],
  out_specs=pl.BlockSpec(block_shape, lambda i, jd, jh, jw: (i, jd, jh, jw, 0)),
)(x_pooled, subtract_param)

# 6. Max Reduction (performed outside the kernel)
x = jnp.max(x_pre_reduction, axis=-1).block_until_ready()
