# Imports
import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

# Initialization
batch_size = 16
features = 64
dim1 = 256
dim2 = 256
H_BLOCK = 16  # Tile size for the height dimension

key = random.PRNGKey(0)
key_x, key_params = random.split(key)

# Flax layers typically expect channels-last format: (N, H, W, C)
x = random.normal(key_x, (batch_size, dim1, dim2, features))
inorm = nn.InstanceNorm()
params = inorm.init(key_params, x)["params"]


# Computation
def kernel(x_ref, scale_ref, bias_ref, out_ref):
  """
  Pallas kernel for instance normalization with in-kernel tiling to manage memory.
  This kernel processes a single instance from the batch. It calculates the
  mean and variance across the spatial dimensions (height and width) for that
  instance, normalizes the input, and then applies the learned scale and
  bias parameters. It processes the data in tiles to avoid exceeding SRAM limits.
  Args:
    x_ref: A reference to the input tensor in HBM for a single instance,
      with shape (1, dim1, dim2, features).
    scale_ref: A reference to the scale parameters in SRAM, with shape (features,).
    bias_ref: A reference to the bias parameters in SRAM, with shape (features,).
    out_ref: A reference to the output tensor in HBM where the result is stored,
      with the same shape as x_ref.
  """
  epsilon = 1e-6
  mean_acc = jnp.zeros(features, dtype=jnp.float32)
  var_acc = jnp.zeros(features, dtype=jnp.float32)

  # Pass 1: Tiled calculation of sum and sum of squares
  def pass1_body(i, accs):
    mean_acc, var_acc = accs
    h_offset = i * H_BLOCK
    # Load a tile from HBM into SRAM.
    x_tile = pl.load(x_ref, (0, h_offset, 0, 0), block_shape=(1, H_BLOCK, dim2, features))
    # Calculate sum and sum of squares over the spatial axes of the tile.
    tile_sum = jnp.sum(x_tile, axis=(1, 2))
    tile_sum_sq = jnp.sum(x_tile * x_tile, axis=(1, 2))
    return mean_acc + tile_sum, var_acc + tile_sum_sq

  mean_acc, var_acc = jax.lax.fori_loop(0, dim1 // H_BLOCK, pass1_body, (mean_acc, var_acc))

  # Finalize mean and variance
  n_elements = dim1 * dim2
  mean = mean_acc / n_elements
  var = var_acc / n_elements - mean * mean

  # scale_ref and bias_ref are in SRAM, so access them directly.
  scale = scale_ref[...]
  bias = bias_ref[...]
  inv_std = jax.lax.rsqrt(var + epsilon)

  # Pass 2: Tiled normalization, scaling, and storing
  def pass2_body(i, _):
    h_offset = i * H_BLOCK
    # Load the same tile again.
    x_tile = pl.load(x_ref, (0, h_offset, 0, 0), block_shape=(1, H_BLOCK, dim2, features))
    # Apply normalization, scale, and bias.
    out_tile = (x_tile - mean.astype(x_tile.dtype)) * inv_std.astype(x_tile.dtype) * scale.astype(
      x_tile.dtype
    ) + bias.astype(x_tile.dtype)
    # Store the resulting tile back to HBM.
    pl.store(out_ref, (0, h_offset, 0, 0), out_tile)

  jax.lax.fori_loop(0, dim1 // H_BLOCK, pass2_body, None)


result = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
  grid_spec=pltpu.GridSpec(
    num_programs=batch_size,
    in_specs=[
      pl.BlockSpec(lambda i: (i, 0, 0, 0), (1, dim1, dim2, features)),
      pl.BlockSpec(lambda i: (0,), (features,)),
      pl.BlockSpec(lambda i: (0,), (features,)),
    ],
    out_specs=pl.BlockSpec(lambda i: (i, 0, 0, 0), (1, dim1, dim2, features)),
  ),
)(x, params["scale"], params["bias"]).block_until_ready()
