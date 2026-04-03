# Imports
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
batch_size = 16
dim = 16384
key = random.PRNGKey(0)
x = random.normal(key, (batch_size, dim))
# Define a block size for the batch dimension that adheres to TPU constraints (divisible by 8)
b_bs = 8


# Computation
def kernel(x_ref, y_ref):
  """Pallas kernel for log_softmax.

  This kernel computes log_softmax along the second axis (axis=1).
  The computation is parallelized over the first axis (the batch dimension).
  Each program instance handles a slice of the input tensor.

  Args:
    x_ref: A reference to a block of the input tensor.
    y_ref: A reference to a block of the output tensor for storing the result.
  """
  # Load the input block from SRAM into a register.
  x = x_ref[...]

  # The log_softmax implementation is x - log(sum(exp(x))).
  # For numerical stability, we use the max trick:
  # log_softmax(x) = x - max(x) - log(sum(exp(x - max(x))))

  # 1. Find the maximum value along the reduction axis (axis=1).
  #    `keepdims=True` ensures the result is broadcastable for the subtraction.
  max_x = jnp.max(x, axis=1, keepdims=True)

  # 2. Subtract the max value for stability. This centers the values around 0.
  centered_x = x - max_x

  # 3. Calculate the log of the sum of the exponentials of the centered values.
  #    This is the log-normalizer term.
  log_normalizer = jnp.log(jnp.sum(jnp.exp(centered_x), axis=1, keepdims=True))

  # 4. Compute the final log_softmax value by subtracting the log-normalizer.
  log_softmax_val = centered_x - log_normalizer

  # 5. Write the result back to the output buffer in SRAM.
  y_ref[...] = log_softmax_val


y = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
  grid=(batch_size // b_bs,),
  in_specs=[pl.BlockSpec(block_shape=(b_bs, dim), index_map=lambda i: (i, 0))],
  out_specs=pl.BlockSpec(block_shape=(b_bs, dim), index_map=lambda i: (i, 0)),
)(x).block_until_ready()
