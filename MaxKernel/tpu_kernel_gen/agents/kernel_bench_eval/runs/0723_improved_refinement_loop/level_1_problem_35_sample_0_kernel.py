# Imports
import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
batch_size = 16
features = 64
num_groups = 8
dim1 = 256
dim2 = 256

key = random.PRNGKey(0)
key_x, key_params = random.split(key)

# JAX/Flax convention is channels-last (NHWC)
x = random.normal(key_x, (batch_size, dim1, dim2, features))
gn = nn.GroupNorm(num_groups=num_groups)
params = gn.init(key_params, x)["params"]


# Computation
def kernel(x_ref, scale_ref, bias_ref, out_ref):
  """Pallas kernel for pixel-wise Group Normalization.

  This kernel applies Group Normalization to each pixel's feature vector
  independently. Statistics (mean and variance) are computed across groups
  of channels for each spatial location.

  Args:
    x_ref: Input features reference, shaped (1, 1, 8, features).
    scale_ref: Scaling factor reference, shaped (features,).
    bias_ref: Bias term reference, shaped (features,).
    out_ref: Output reference for the normalized features, shaped
      (1, 1, 8, features).
  """
  # Hardcoded values from the source context.
  num_groups = 8
  features = 64
  group_size = features // num_groups
  epsilon = 1e-5

  # The block contains 8 pixels; we process them in a loop.
  for i in range(8):
    # Iterate over channel groups to compute statistics and normalize.
    for g in range(num_groups):
      # Define the start index for the current group.
      start_idx = g * group_size

      # Manually compute sum and sum-of-squares for the group.
      # This avoids creating a sliced array, which is problematic on TPU.
      group_sum = 0.0
      group_sum_sq = 0.0
      for j in range(group_size):
        val = x_ref[0, 0, i, start_idx + j]
        group_sum += val
        group_sum_sq += val * val

      # Calculate mean and variance for the group.
      mean = group_sum / group_size
      var = group_sum_sq / group_size - mean * mean

      # Normalize the group and write the result back.
      for j in range(group_size):
        idx = start_idx + j
        val = x_ref[0, 0, i, idx]
        normalized_val = (val - mean) / jnp.sqrt(var + epsilon)

        # Apply the learned scale and bias.
        result = normalized_val * scale_ref[idx] + bias_ref[idx]

        # Write the result back to the output reference.
        out_ref[0, 0, i, idx] = result


result = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
  grid=(batch_size, dim1, dim2 // 8),
  in_specs=[
    pl.BlockSpec(block_shape=(1, 1, 8, features), index_map=lambda b, d1, d2: (b, d1, d2 * 8, 0)),
    pl.BlockSpec(block_shape=(features,), index_map=lambda *_: (0,)),
    pl.BlockSpec(block_shape=(features,), index_map=lambda *_: (0,)),
  ],
  out_specs=pl.BlockSpec(block_shape=(1, 1, 8, features), index_map=lambda b, d1, d2: (b, d1, d2 * 8, 0)),
)(x, params["scale"], params["bias"]).block_until_ready()
