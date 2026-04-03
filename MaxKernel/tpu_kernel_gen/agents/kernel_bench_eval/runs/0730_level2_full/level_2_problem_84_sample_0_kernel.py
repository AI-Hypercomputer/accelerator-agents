# Imports
import jax
import jax.numpy as jnp
import jax.random as random
from flax import linen as nn
from jax.experimental import pallas as pl

# Initialization
batch_size = 128
in_features = 1024
out_features = 512
bn_eps = 1e-5
bn_momentum = 0.1
scale_shape = (1,)
key = random.PRNGKey(0)
key_x, key_params = random.split(key)
x = random.normal(key_x, (batch_size, in_features))


class Model(nn.Module):
  out_features: int
  bn_eps: float
  bn_momentum: float

  @nn.compact
  def __call__(self, x, use_running_average: bool):
    x = nn.Dense(features=self.out_features)(x)
    x = nn.BatchNorm(use_running_average=use_running_average, epsilon=self.bn_eps, momentum=self.bn_momentum)(x)
    return x


model = Model(out_features=out_features, bn_eps=bn_eps, bn_momentum=bn_momentum)
variables = model.init(key_params, x, use_running_average=True)
params = variables["params"]
batch_stats = variables["batch_stats"]
scale = jnp.ones(scale_shape)


# Computation
def kernel(
  x_ref,
  dense_kernel_ref,
  dense_bias_ref,
  bn_scale_ref,
  bn_bias_ref,
  bn_mean_ref,
  bn_var_ref,
  scale_ref,
  out_ref,
):
  # Constants from the source code
  bn_eps = 1e-5

  # Load data from memory into registers
  x = x_ref[...]
  dense_kernel = dense_kernel_ref[...]
  dense_bias = dense_bias_ref[...]
  bn_scale = bn_scale_ref[...]
  bn_bias = bn_bias_ref[...]
  bn_mean = bn_mean_ref[...]
  bn_var = bn_var_ref[...]
  scale = scale_ref[0]  # scale_ref is a 1-element array

  # Dense layer computation
  y = x @ dense_kernel + dense_bias

  # BatchNorm layer computation (inference mode)
  # Equivalent to (y - mean) / sqrt(var + eps) * scale + bias
  y_norm = (y - bn_mean) * jax.lax.rsqrt(bn_var + bn_eps)
  y_bn = y_norm * bn_scale + bn_bias

  # Final scaling
  y_scaled = y_bn * scale

  # Softmax activation (stable implementation)
  max_val = jnp.max(y_scaled, axis=1, keepdims=True)
  exp_y = jnp.exp(y_scaled - max_val)
  sum_exp_y = jnp.sum(exp_y, axis=1, keepdims=True)
  softmax_y = exp_y / sum_exp_y

  # Write the final result to the output buffer
  out_ref[...] = softmax_y


batch_block_size = 8

x = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct((batch_size, out_features), x.dtype),
  grid=(batch_size // batch_block_size,),
  in_specs=[
    pl.BlockSpec(block_shape=(batch_block_size, in_features), index_map=lambda i: (i, 0)),
    pl.BlockSpec(block_shape=(in_features, out_features), index_map=lambda i: (0, 0)),
    pl.BlockSpec(block_shape=(out_features,), index_map=lambda i: (0,)),
    pl.BlockSpec(block_shape=(out_features,), index_map=lambda i: (0,)),
    pl.BlockSpec(block_shape=(out_features,), index_map=lambda i: (0,)),
    pl.BlockSpec(block_shape=(out_features,), index_map=lambda i: (0,)),
    pl.BlockSpec(block_shape=(out_features,), index_map=lambda i: (0,)),
    pl.BlockSpec(block_shape=(1,), index_map=lambda i: (0,)),
  ],
  out_specs=pl.BlockSpec(block_shape=(batch_block_size, out_features), index_map=lambda i: (i, 0)),
)(
  x,
  params["Dense_0"]["kernel"],
  params["Dense_0"]["bias"],
  params["BatchNorm_0"]["scale"],
  params["BatchNorm_0"]["bias"],
  batch_stats["BatchNorm_0"]["mean"],
  batch_stats["BatchNorm_0"]["var"],
  scale,
).block_until_ready()
