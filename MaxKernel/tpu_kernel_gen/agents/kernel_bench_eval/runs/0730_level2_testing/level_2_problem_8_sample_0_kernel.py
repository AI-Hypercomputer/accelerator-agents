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
depth = 16
height = 32
width = 32
kernel_size = (3, 3, 3)
divisor = 2.0
pool_size = (2, 2, 2)
sum_dim = -1
key = random.PRNGKey(0)
key_x, key_conv, key_bias = random.split(key, 3)
x = random.normal(key_x, (batch_size, depth, height, width, in_channels))
conv = nn.Conv(
  features=out_channels,
  kernel_size=kernel_size,
  kernel_init=nn.initializers.normal(),
  bias_init=nn.initializers.normal(),
)
params = conv.init(key_conv, x)["params"]
bias_shape = (1, 1, 1, 1, out_channels)
bias = random.normal(key_bias, bias_shape)

# Perform convolution and max pooling outside of Pallas kernel
dn = jax.lax.ConvDimensionNumbers(
  lhs_spec=(0, 4, 1, 2, 3),
  rhs_spec=(4, 3, 0, 1, 2),
  out_spec=(0, 4, 1, 2, 3),
)
conv_out = jax.lax.conv_general_dilated(
  lhs=x,
  rhs=params["kernel"],
  window_strides=(1, 1, 1),
  padding="SAME",
  dimension_numbers=dn,
)
conv_out = conv_out + params["bias"]
x_divided = conv_out / divisor
pool_window_shape = (1, *pool_size, 1)
pool_strides = (1, *pool_size, 1)
padding_valid = [(0, 0)] * len(pool_window_shape)
pooled_out = jax.lax.reduce_window(
  x_divided,
  init_value=-jnp.inf,
  computation=jax.lax.max,
  window_dimensions=pool_window_shape,
  window_strides=pool_strides,
  padding=padding_valid,
)


# Computation
def kernel(pooled_out_ref, bias_ref, out_ref):
  """
  Pallas kernel that implements a sequence of neural network operations.
  """
  # 1. Mean reduction
  x = jnp.mean(pooled_out_ref[...], axis=(1, 2, 3), keepdims=True)

  # 2. Add second bias
  x = x + bias_ref[...]

  # 3. Sum reduction
  out_ref[0, 0, 0, 0] = jnp.sum(x)


result = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct((batch_size, 1, 1, 1), x.dtype),
  grid=(batch_size,),
  in_specs=[
    pl.BlockSpec(
      block_shape=(1, *pooled_out.shape[1:]),
      index_map=lambda i: (i, 0, 0, 0, 0),
    ),
    pl.BlockSpec(block_shape=bias.shape, index_map=lambda i: tuple([0] * bias.ndim)),
  ],
  out_specs=pl.BlockSpec(block_shape=(1, 1, 1, 1), index_map=lambda i: (i, 0, 0, 0)),
)(pooled_out, bias).block_until_ready()
