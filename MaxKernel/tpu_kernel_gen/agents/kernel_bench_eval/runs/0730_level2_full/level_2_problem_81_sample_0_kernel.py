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

key = random.PRNGKey(0)
key, x_key, params_key = random.split(key, 3)

x = random.normal(x_key, (batch_size, in_features))
gemm = nn.Dense(features=out_features, use_bias=True)
params = gemm.init(params_key, x)["params"]

# Block sizes for the kernel computation
bM = 128
bN = 512


# Computation
def kernel(x_ref, kernel_ref, bias_ref, out_ref):
  """
  Pallas kernel for a dense layer followed by a sequence of non-linear transformations.

  This kernel performs the following operations:
  1. Matrix multiplication (GEMM) of input `x` with `kernel` weights.
  2. Addition of the `bias` term.
  3. SiLU (Swish) activation: `x * sigmoid(x)`.
  4. Division by 2.0.
  5. Clipping the result to the range [-1.0, 1.0].
  6. Applying the tanh activation function.
  7. A final clipping operation to the range [-1.0, 1.0].

  Args:
    x_ref: A reference to the input data block.
    kernel_ref: A reference to the weight matrix block.
    bias_ref: A reference to the bias vector block.
    out_ref: A reference to the output data block where the result is stored.
  """
  # 1. Dense layer: GEMM + bias addition
  # The result of the dot product has shape (bM, bN).
  # The bias_ref has shape (bN,), which is broadcasted across the rows.
  acc = jnp.dot(x_ref[...], kernel_ref[...]) + bias_ref[...]

  # 2. SiLU activation
  acc = acc * jax.nn.sigmoid(acc)

  # 3. Division
  acc = acc / 2.0

  # 4. Clip
  acc = jnp.clip(acc, a_min=-1.0, a_max=1.0)

  # 5. Tanh
  acc = jnp.tanh(acc)

  # 6. Final Clip
  acc = jnp.clip(acc, a_min=-1.0, a_max=1.0)

  # Write the final result to the output buffer
  out_ref[...] = acc


# The pallas_call replaces the original computation block.
# The kernel is expected to perform the dense layer operation (GEMM + bias)
# and all subsequent element-wise activations and transformations.
x = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct((batch_size, out_features), x.dtype),
  grid=(batch_size // bM, out_features // bN),
  in_specs=[
    # Input x: (batch_size, in_features)
    # We parallelize over the batch dimension, so each kernel instance
    # gets a block of rows from x.
    pl.BlockSpec(block_shape=(bM, in_features), index_map=lambda i, j: (i, 0)),
    # Kernel weights: (in_features, out_features)
    # We parallelize over the output features, so each kernel instance
    # gets a block of columns from the weights.
    pl.BlockSpec(block_shape=(in_features, bN), index_map=lambda i, j: (0, j)),
    # Bias: (out_features,)
    # Each kernel instance gets the corresponding slice of the bias vector.
    pl.BlockSpec(block_shape=(bN,), index_map=lambda i, j: (j,)),
  ],
  out_specs=pl.BlockSpec(block_shape=(bM, bN), index_map=lambda i, j: (i, j)),
)(x, params["kernel"], params["bias"]).block_until_ready()
