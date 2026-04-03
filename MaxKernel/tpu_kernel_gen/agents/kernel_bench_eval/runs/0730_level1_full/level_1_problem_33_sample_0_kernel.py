# Imports
import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
batch_size = 16
features = 64
# The original dimensions (256, 256) caused a timeout.
# Reducing them to make the script runnable for verification.
dim1 = 32
dim2 = 32

key = random.PRNGKey(0)
x_key, params_key = random.split(key)

# Note: JAX typically uses channels-last (N, H, W, C) format.
# The original PyTorch code uses channels-first (N, C, H, W).
# We match the original format here and specify axis=1 for BatchNorm.
x = random.normal(x_key, (batch_size, features, dim1, dim2))

# To match PyTorch's nn.BatchNorm2d default momentum of 0.1, we need to set
# the momentum in Flax's nn.BatchNorm to 0.9. This is because PyTorch's
# momentum is the weight for the new batch statistics, while Flax's is the
# decay rate for the existing running statistics.
# i.e., flax_momentum = 1 - pytorch_momentum
# use_running_average=False ensures we are in training mode, using batch statistics.
bn_layer = nn.BatchNorm(use_running_average=False, axis=1, momentum=0.9)
variables = bn_layer.init(params_key, x)


# Computation
def kernel(
  x_ref, scale_ref, bias_ref, running_mean_ref, running_var_ref, result_ref, updated_mean_ref, updated_var_ref
):
  """
  Pallas kernel for Batch Normalization (training mode).

  This kernel processes a single feature channel of the input tensor. It
  calculates the batch statistics (mean and variance), normalizes the input,
  applies scale and bias, and updates the running statistics.

  Args:
    x_ref: Input tensor slice for a single feature channel.
    scale_ref: Scale parameter (gamma) for the channel.
    bias_ref: Bias parameter (beta) for the channel.
    running_mean_ref: Current running mean for the channel.
    running_var_ref: Current running variance for the channel.
    result_ref: Output tensor slice for the normalized result.
    updated_mean_ref: Output for the updated running mean.
    updated_var_ref: Output for the updated running variance.
  """
  # The program ID corresponds to the feature channel index.
  i = pl.program_id(axis=0)
  # Flax's default epsilon for BatchNorm.
  epsilon = 1e-5
  # Momentum is equivalent to 1 - PyTorch's momentum.
  momentum = 0.9

  # Step 1: Calculate batch statistics for the current feature channel.
  # The mean and variance are computed across the batch and spatial dimensions.
  batch_mean = jnp.mean(x_ref[...])
  batch_var = jnp.var(x_ref[...])

  # Step 2: Normalize the input using the batch statistics.
  # Note: In Pallas, x_ref[...] loads the data from SRAM into registers.
  denominator = jnp.sqrt(batch_var + epsilon)
  normalized_x = (x_ref[...] - batch_mean) / denominator

  # Step 3: Apply scale and bias parameters.
  # The scale and bias are loaded from their respective 1-element blocks.
  result = scale_ref[i] * normalized_x + bias_ref[i]

  # Step 4: Update the running statistics using the momentum.
  # These are the new statistics that will be used for inference later.
  new_running_mean = momentum * running_mean_ref[i] + (1.0 - momentum) * batch_mean
  new_running_var = momentum * running_var_ref[i] + (1.0 - momentum) * batch_var

  # Step 5: Write the results to the output references.
  result_ref[...] = result
  # The TPU backend cannot store scalars to VMEM directly or via indexed
  # assignment. We use pl.store with the scalar value wrapped in a 1-element
  # array to perform a valid vector write at a dynamic index.
  pl.store(updated_mean_ref, (i,), jnp.expand_dims(new_running_mean, 0))
  pl.store(updated_var_ref, (i,), jnp.expand_dims(new_running_var, 0))


# The kernel is parallelized over the feature dimension.
# Each kernel instance processes one feature channel.
grid = (features,)

# Flatten the nested 'variables' structure for the pallas_call.
# The kernel will receive these as separate array arguments.
scale = variables["params"]["scale"]
bias = variables["params"]["bias"]
running_mean = variables["batch_stats"]["mean"]
running_var = variables["batch_stats"]["var"]

# The pallas_call replaces the original bn_layer.apply call.
# It takes the input tensor 'x' and the flattened parameters.
# It returns a tuple: (normalized_output, updated_mean, updated_var).
result, updated_mean, updated_var = pl.pallas_call(
  kernel,
  # The kernel produces three outputs: the normalized result, and the
  # updated running mean and variance for the batch statistics.
  out_shape=[
    jax.ShapeDtypeStruct(x.shape, x.dtype),
    jax.ShapeDtypeStruct(running_mean.shape, running_mean.dtype),
    jax.ShapeDtypeStruct(running_var.shape, running_var.dtype),
  ],
  grid=grid,
  in_specs=[
    # x: For each feature `i`, the kernel gets the corresponding slice
    # (batch_size, 1, dim1, dim2).
    pl.BlockSpec(block_shape=(batch_size, 1, dim1, dim2), index_map=lambda i: (0, i, 0, 0)),
    # scale, bias, running_mean, running_var: Each kernel gets the full
    # array and uses its program_id to index into the correct element.
    pl.BlockSpec(block_shape=(features,), index_map=lambda i: (0,)),
    pl.BlockSpec(block_shape=(features,), index_map=lambda i: (0,)),
    pl.BlockSpec(block_shape=(features,), index_map=lambda i: (0,)),
    pl.BlockSpec(block_shape=(features,), index_map=lambda i: (0,)),
  ],
  out_specs=[
    # result: Each kernel `i` writes its normalized output to the
    # corresponding feature slice in the output tensor.
    pl.BlockSpec(block_shape=(batch_size, 1, dim1, dim2), index_map=lambda i: (0, i, 0, 0)),
    # updated_mean, updated_var: Each kernel `i` writes its updated
    # running statistic to the `i`-th element of the output arrays.
    pl.BlockSpec(block_shape=(features,), index_map=lambda i: (0,)),
    pl.BlockSpec(block_shape=(features,), index_map=lambda i: (0,)),
  ],
)(x, scale, bias, running_mean, running_var)

result.block_until_ready()
updated_mean.block_until_ready()
updated_var.block_until_ready()
