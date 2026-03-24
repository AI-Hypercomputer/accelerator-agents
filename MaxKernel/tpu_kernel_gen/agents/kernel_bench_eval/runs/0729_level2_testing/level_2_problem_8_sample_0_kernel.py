# Imports
import os

os.environ["JAX_PLATFORMS"] = "cpu"
import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl


# Initialization
class MLP(nn.Module):
  def setup(self):
    # Flax's nn.Sequential can take functions like nn.relu directly.
    self.layers = nn.Sequential([nn.Dense(features=256), nn.relu, nn.Dense(features=10)])

  def __call__(self, x):
    # Flatten the image.
    x = x.reshape((x.shape[0], -1))
    return self.layers(x)


# In JAX, we need a PRNG key for all random operations.
key = random.PRNGKey(0)
# We split the key for different random operations to ensure reproducibility.
key, init_key, data_key = random.split(key, 3)

# Instantiate the model.
mlp = MLP()

# To initialize the model parameters, we need a dummy input.
dummy_x = jnp.ones((1, 28, 28))
# `init` returns a nested dictionary of parameters. We typically extract the 'params' part.
params = mlp.init(init_key, dummy_x)["params"]

# Generate random input data.
data = random.normal(data_key, (128, 28, 28))


def kernel(params_ref, x_ref, out_ref):
  """
  Pallas kernel for the MLP forward pass.

  Args:
    params_ref: A PyTree of references to the model parameters.
    x_ref: A reference to a chunk of the input data.
    out_ref: A reference to the output buffer for the corresponding chunk.
  """
  # Load the input data chunk and flatten it.
  # The shape changes from (b_size, 28, 28) to (b_size, 784).
  x = x_ref[...].reshape((x_ref.shape[0], -1))

  # --- First Dense Layer ---
  # Load the kernel and bias for the first dense layer.
  w1 = params_ref["layers"]["layers_0"]["kernel"][...]
  b1 = params_ref["layers"]["layers_0"]["bias"][...]
  # Apply the first dense layer: x @ w + b
  x = x @ w1 + b1

  # --- ReLU Activation ---
  x = jax.nn.relu(x)

  # --- Second Dense Layer ---
  # Load the kernel and bias for the second dense layer.
  w2 = params_ref["layers"]["layers_2"]["kernel"][...]
  b2 = params_ref["layers"]["layers_2"]["bias"][...]
  # Apply the second dense layer.
  y = x @ w2 + b2

  # Write the final result to the output reference.
  out_ref[...] = y


# Block size for chunking the batch dimension.
# Must be a multiple of 8 for TPU compatibility of the output BlockSpec.
b_size = 8

# Computation
output = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct((data.shape[0], 10), data.dtype),
  grid=(data.shape[0] // b_size,),
  in_specs=(
    # Params are not chunked; each kernel gets the full PyTree.
    jax.tree_util.tree_map(
      lambda x: pl.BlockSpec(x.shape, lambda i: tuple([0] * x.ndim)),
      params,
    ),
    # Data is chunked along the batch dimension.
    pl.BlockSpec((b_size, 28, 28), lambda i: (i * b_size, 0, 0)),
  ),
  out_specs=pl.BlockSpec((b_size, 10), lambda i: (i * b_size, 0)),
  interpret=True,
)(params, data)
