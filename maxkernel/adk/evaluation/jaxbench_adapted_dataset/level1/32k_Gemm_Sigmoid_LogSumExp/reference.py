# Imports
import jax
import jax.numpy as jnp

# Initialization
def get_inputs(dtype=jnp.float32):
    batch_size = 16384
    input_size = 2048
    hidden_size = 4096
    output_size = 1024
    key = jax.random.key(0)
    rand_key = jax.random.key(42)
    ka, kb, kc, kd = jax.random.split(rand_key, 4)
    x = jax.random.uniform(key, (batch_size, input_size), dtype=dtype)
    w1 = jax.random.normal(ka, (hidden_size, input_size), dtype=dtype) * 0.02
    b1 = jax.random.normal(kb, hidden_size, dtype=dtype) * 0.02
    w2 = jax.random.normal(kc, (output_size, hidden_size), dtype=dtype) * 0.02
    b2 = jax.random.normal(kd, output_size, dtype=dtype) * 0.02
    dynamic_args = [x, w1, b1, w2, b2]
    static_args = []
    return dynamic_args, static_args

# Computation
def computation(x, w1, b1, w2, b2):
    x = jnp.matmul(x, w1.T) + b1
    x = jax.nn.sigmoid(x)
    x = jnp.matmul(x, w2.T) + b2
    x = jax.scipy.special.logsumexp(x, axis=1)
    return x