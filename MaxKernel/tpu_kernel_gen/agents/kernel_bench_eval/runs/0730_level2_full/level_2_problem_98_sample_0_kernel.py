# Imports
import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
batch_size = 128
in_features = 512
out_features = 256
pool_kernel_size = 4
scale_factor = 2.0

key = random.PRNGKey(0)
key_x, key_params = random.split(key)

x = random.normal(key_x, (batch_size, in_features))
matmul = nn.Dense(features=out_features, use_bias=True)
params = matmul.init(key_params, x)["params"]

# Block size for Pallas kernel grid
b_size = 8


# Computation
def kernel(x_ref, kernel_ref, bias_ref, out_ref):
  """
  Pallas kernel that applies a dense layer, average pooling, GELU activation,
  and scaling to a batch of inputs.
  """
  # Hardcoded constants from the source computation.
  pool_kernel_size = 4
  scale_factor = 2.0

  # Get the index for the output feature dimension
  j = pl.program_id(axis=1)

  # Step 1: Dense layer (Matmul + Bias)
  # This computes the full (b_size, out_features) intermediate result.
  x_matmul = x_ref[...] @ kernel_ref[...] + bias_ref[...]

  # Step 2: Average Pooling
  # We manually slice the result of the matmul to select the values
  # for the current pooling window, identified by `j`.
  start_index = j * pool_kernel_size
  # Use jax.lax.dynamic_slice for slicing.
  # The slice shape is (b_size, pool_kernel_size).
  x_matmul_slice = jax.lax.dynamic_slice(x_matmul, (0, start_index), (x_matmul.shape[0], pool_kernel_size))
  # Compute the mean over the pooling window (axis=1).
  x_pooled = jnp.mean(x_matmul_slice, axis=1)

  # Step 3: GELU Activation (element-wise)
  x_gelu = nn.gelu(x_pooled)

  # Step 4: Scaling (element-wise)
  x_scaled = x_gelu * scale_factor

  # Step 5: Write the result to the output buffer.
  # out_ref is (b_size, 1), x_scaled is (b_size,). Add a dimension.
  out_ref[...] = x_scaled[:, None]


# We use a 2D grid to parallelize over the batch and output feature dimensions.
grid = (batch_size // b_size, out_features // pool_kernel_size)

x_scaled = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct((batch_size, out_features // pool_kernel_size), x.dtype),
  grid=grid,
  in_specs=[
    # Each kernel instance gets a (b_size, in_features) block of x.
    # The block is selected by the batch index `i`.
    pl.BlockSpec(block_shape=(b_size, in_features), index_map=lambda i, j: (i * b_size, 0)),
    # Load the entire kernel and bias into each program.
    pl.BlockSpec(block_shape=params["kernel"].shape, index_map=lambda i, j: (0, 0)),
    pl.BlockSpec(block_shape=params["bias"].shape, index_map=lambda i, j: (0,)),
  ],
  # Each kernel instance writes a (b_size, 1) block to the output.
  # The block's position is determined by both `i` and `j`.
  out_specs=pl.BlockSpec(block_shape=(b_size, 1), index_map=lambda i, j: (i * b_size, j)),
)(x, params["kernel"], params["bias"])

# The final reduction `jnp.max(..., axis=1)` is performed as a subsequent JAX operation.
result = jnp.max(x_scaled, axis=1)
