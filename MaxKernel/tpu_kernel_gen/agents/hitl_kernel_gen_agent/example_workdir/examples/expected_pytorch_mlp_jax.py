"""
Expected JAX output for PyTorch MLP example.
This shows the functional JAX equivalent without classes or GPU references.
"""

# Imports
import jax
import jax.numpy as jnp
import jax.random as random

# Initialization
batch_size = 128
input_dim = 784
hidden_dim = 256
output_dim = 10

key = random.PRNGKey(0)
key_x, key_w1, key_b1, key_w2, key_b2 = random.split(key, 5)

x = random.normal(key_x, (batch_size, input_dim))
W1 = random.normal(key_w1, (input_dim, hidden_dim)) * 0.01
b1 = jnp.zeros((hidden_dim,))
W2 = random.normal(key_w2, (hidden_dim, output_dim)) * 0.01
b2 = jnp.zeros((output_dim,))


# Computation
def computation(x: jnp.ndarray, W1: jnp.ndarray, b1: jnp.ndarray, W2: jnp.ndarray, b2: jnp.ndarray) -> jnp.ndarray:
  h = jnp.dot(x, W1) + b1
  h = jax.nn.relu(h)
  output = jnp.dot(h, W2) + b2
  return output


output = jax.block_until_ready(computation(x, W1, b1, W2, b2))
