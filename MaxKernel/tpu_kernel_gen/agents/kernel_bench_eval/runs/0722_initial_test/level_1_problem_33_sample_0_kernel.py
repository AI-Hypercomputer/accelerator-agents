# Imports
import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
batch_size = 16
features = 64
dim1 = 256
dim2 = 256

key = random.PRNGKey(0)
key_x, key_init = random.split(key)

# JAX/Flax convention is channels-last: (N, H, W, C)
x = random.normal(key_x, (batch_size, dim1, dim2, features))
bn = nn.BatchNorm(use_running_average=False)  # Corresponds to training mode
variables = bn.init(key_init, x)


# Computation
def kernel(x_ref, scale_ref, bias_ref, y_ref, mean_ref, var_ref):
  """
  Pallas kernel for Batch Normalization in training mode.

  This kernel processes a single feature/channel of the input tensor.
  It computes the mean and variance across the batch and spatial dimensions,
  normalizes the input, applies scale and bias, and writes out the
  normalized feature data as well as the computed mean and variance.

  Args:
    x_ref: A reference to the input data for a single feature.
           Shape: (1, N, H, W)
    scale_ref: A reference to the scale parameter for this feature (scalar).
    bias_ref: A reference to the bias parameter for this feature (scalar).
    y_ref: A reference to the output buffer for the normalized feature data.
           Shape: (1, N, H, W)
    mean_ref: A reference to the output buffer for the computed batch mean (scalar).
    var_ref: A reference to the output buffer for the computed batch variance (scalar).
  """
  # Epsilon for numerical stability, matching the Flax default.
  epsilon = 1e-5

  # Calculate the mean and variance over the batch and spatial dimensions for the current feature.
  # x_ref holds all data for this feature, so we can operate on it directly.
  mean = jnp.mean(x_ref)
  var = jnp.var(x_ref)

  # Normalize the input feature data.
  denominator = jnp.sqrt(var + epsilon)
  normalized_x = (x_ref[...] - mean) / denominator

  # Apply the learned scale and bias parameters and write to the output reference.
  # scale_ref[...] and bias_ref[...] dereference the scalar values.
  y_ref[...] = normalized_x * scale_ref[...] + bias_ref[...]

  # Write the computed mean and variance to their respective output references.
  mean_ref[...] = mean
  var_ref[...] = var


# Transpose input from (N, H, W, C) to (C, N, H, W) for Pallas processing
x_T = jnp.transpose(x, (3, 0, 1, 2))

# Extract parameters from the initialized variables
scale = variables["params"]["scale"]
bias = variables["params"]["bias"]

# Define the output shapes. Note the transposed layout for the main output.
output_T_shape = jax.ShapeDtypeStruct(x_T.shape, x_T.dtype)
batch_stats_shape = jax.ShapeDtypeStruct(scale.shape, scale.dtype)

# Invoke the Pallas kernel
output_T, updated_mean, updated_variance = pl.pallas_call(
  kernel,
  # The primary output is the normalized x, and the other two are the computed batch statistics.
  out_shape=[output_T_shape, batch_stats_shape, batch_stats_shape],
  # Create one program instance for each feature.
  grid=(x.shape[3],),
  in_specs=[
    # For x_T, each program instance gets all data for one feature.
    pl.BlockSpec(block_shape=(1, x.shape[0], x.shape[1], x.shape[2]), index_map=lambda i: (i, 0, 0, 0)),
    # For scale and bias, each program instance gets the corresponding single value.
    pl.BlockSpec(block_shape=(1,), index_map=lambda i: (i,)),
    pl.BlockSpec(block_shape=(1,), index_map=lambda i: (i,)),
  ],
  out_specs=[
    # Each program instance writes the normalized output for its feature.
    pl.BlockSpec(block_shape=(1, x.shape[0], x.shape[1], x.shape[2]), index_map=lambda i: (i, 0, 0, 0)),
    # Each program instance writes the computed mean and variance for its feature.
    pl.BlockSpec(block_shape=(1,), index_map=lambda i: (i,)),
    pl.BlockSpec(block_shape=(1,), index_map=lambda i: (i,)),
  ],
)(x_T, scale, bias)

# Transpose the main output back to the original (N, H, W, C) layout
output = jnp.transpose(output_T, (1, 2, 3, 0))

# Reconstruct the updated state dictionary as Flax expects
updated_state = {"batch_stats": {"mean": updated_mean, "variance": updated_variance}}

# Ensure computation is complete
output.block_until_ready()
updated_state["batch_stats"]["mean"].block_until_ready()
updated_state["batch_stats"]["variance"].block_until_ready()
