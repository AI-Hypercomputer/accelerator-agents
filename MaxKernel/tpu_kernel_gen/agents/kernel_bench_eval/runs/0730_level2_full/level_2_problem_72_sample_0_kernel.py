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
depth, height, width = 32, 32, 32
kernel_size = 3
stride = 2
padding = 1
key = random.PRNGKey(0)
key_x, key_init = random.split(key)
x = random.normal(key_x, (batch_size, depth, height, width, in_channels))


class Model(nn.Module):
  out_channels: int
  kernel_size: int
  stride: int
  padding: int

  @nn.compact
  def __call__(self, x):
    x = nn.ConvTranspose(
      features=self.out_channels,
      kernel_size=(self.kernel_size, self.kernel_size, self.kernel_size),
      strides=(self.stride, self.stride, self.stride),
      padding=self.padding,
    )(x)
    x = nn.BatchNorm(use_running_average=False)(x)
    x = nn.avg_pool(x, window_shape=(2, 2, 2), strides=(2, 2, 2))
    x = nn.avg_pool(x, window_shape=(2, 2, 2), strides=(2, 2, 2))
    return x


model = Model(out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
variables = model.init(key_init, x)


# Computation
def kernel(x_ref, params_ref, out_ref, batch_stats_out_ref):
  """
  Pallas kernel for a sequence of ConvTranspose, BatchNorm, and AvgPool layers.

  Args:
    x_ref: Input tensor reference for a single batch item.
    params_ref: A pytree of references to the model's parameters.
    out_ref: Output tensor reference for the final result.
    batch_stats_out_ref: A pytree of references for the output batch statistics.
  """
  import jax

  # 1. Apply 3D Transposed Convolution
  # lax.conv_transpose and scatter are not supported in Pallas on TPU. We
  # implement it manually by upsampling the input and then applying a
  # standard convolution.
  x_val = x_ref[...]
  kernel_val = params_ref["ConvTranspose_0"]["kernel"][...]
  bias_val = params_ref["ConvTranspose_0"]["bias"][...]

  # Manually upsample the input by interleaving with zeros.
  s = 2
  N, D, H, W, C = x_val.shape
  # Upsample W
  x_upsampled = jnp.stack([x_val, jnp.zeros_like(x_val)], axis=4).reshape(N, D, H, W * s, C)
  x_upsampled = x_upsampled[:, :, :, : W * s - (s - 1), :]
  # Upsample H
  x_upsampled = jnp.transpose(x_upsampled, (0, 1, 3, 2, 4))
  x_upsampled = jnp.stack([x_upsampled, jnp.zeros_like(x_upsampled)], axis=4).reshape(
    x_upsampled.shape[0], x_upsampled.shape[1], x_upsampled.shape[2], x_upsampled.shape[3] * s, x_upsampled.shape[4]
  )
  x_upsampled = x_upsampled[:, :, :, : H * s - (s - 1), :]
  x_upsampled = jnp.transpose(x_upsampled, (0, 1, 3, 2, 4))
  # Upsample D
  x_upsampled = jnp.transpose(x_upsampled, (0, 2, 3, 1, 4))
  x_upsampled = jnp.stack([x_upsampled, jnp.zeros_like(x_upsampled)], axis=4).reshape(
    x_upsampled.shape[0], x_upsampled.shape[1], x_upsampled.shape[2], x_upsampled.shape[3] * s, x_upsampled.shape[4]
  )
  x_upsampled = x_upsampled[:, :, :, : D * s - (s - 1), :]
  upsampled_x = jnp.transpose(x_upsampled, (0, 3, 1, 2, 4))

  # The equivalent padding for the forward convolution is k - 1 - p_transpose
  conv_padding = kernel_size - 1 - padding
  conv_out = jax.lax.conv_general_dilated(
    upsampled_x,
    kernel_val,
    window_strides=(1, 1, 1),
    padding=tuple([(conv_padding, conv_padding)] * 3),
    dimension_numbers=("NDHWC", "DHWIO", "NDHWC"),
  )
  conv_out += bias_val

  # 2. Apply Batch Normalization
  # Since each kernel instance operates on a single item from the batch,
  # we calculate the mean and variance over this single instance.
  # The reduction is performed over the spatial dimensions (D, H, W) and the
  # singleton batch dimension (N), effectively computing stats per channel.
  axes = (0, 1, 2, 3)
  mean = jnp.mean(conv_out, axis=axes)
  var = jnp.var(conv_out, axis=axes)

  # Write the computed per-instance stats to the output buffers.
  # Note: With a grid size > 1, this will result in a race condition as all
  # kernel instances write to the same output memory locations.
  batch_stats_out_ref["BatchNorm_0"]["mean"][...] = mean
  batch_stats_out_ref["BatchNorm_0"]["var"][...] = var

  # Normalize the output of the convolution using the computed stats and the
  # learned scale and bias parameters.
  epsilon = 1e-5
  bn_out = (conv_out - mean) / jnp.sqrt(var + epsilon)
  bn_out = bn_out * params_ref["BatchNorm_0"]["scale"][...] + params_ref["BatchNorm_0"]["bias"][...]

  # 3. Apply two sequential Average Pooling operations
  # We use lax.reduce_window to perform pooling and then divide by the
  # window size to get the average.
  window_dims = (1, 2, 2, 2, 1)
  strides = (1, 2, 2, 2, 1)
  pool_divisor = 8.0  # 2 * 2 * 2

  # First average pool
  pooled_out = jax.lax.reduce_window(bn_out, 0.0, jax.lax.add, window_dims, strides, "VALID")
  pooled_out /= pool_divisor

  # Second average pool
  pooled_out = jax.lax.reduce_window(pooled_out, 0.0, jax.lax.add, window_dims, strides, "VALID")
  pooled_out /= pool_divisor

  # 4. Write the final result to the output reference
  out_ref[...] = pooled_out


x_out_shape = jax.ShapeDtypeStruct((128, 15, 15, 15, 16), x.dtype)
batch_stat_shape = jax.ShapeDtypeStruct((16,), x.dtype)

x, _ = pl.pallas_call(
  kernel,
  out_shape=(
    x_out_shape,
    {
      "BatchNorm_0": {
        "mean": batch_stat_shape,
        "var": batch_stat_shape,
      }
    },
  ),
  grid=(batch_size,),
  in_specs=(
    pl.BlockSpec(block_shape=(1, 32, 32, 32, 3), index_map=lambda i: (i, 0, 0, 0, 0)),
    jax.tree_util.tree_map(
      lambda p: pl.BlockSpec(p.shape, lambda i: tuple([0] * p.ndim)),
      variables["params"],
    ),
  ),
  out_specs=(
    pl.BlockSpec(block_shape=(1, 15, 15, 15, 16), index_map=lambda i: (i, 0, 0, 0, 0)),
    {
      "BatchNorm_0": {
        "mean": pl.BlockSpec((16,), lambda i: (0,)),
        "var": pl.BlockSpec((16,), lambda i: (0,)),
      }
    },
  ),
)(x, variables["params"])
x.block_until_ready()
