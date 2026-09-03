# Imports
import jax
import jax.numpy as jnp
from jax import lax

# Initialization
def get_inputs(dtype=jnp.bfloat16):
    batch = 8
    seq_len = 4096
    emb_dim = 8192
    epsilon = 1e-5

    key = jax.random.key(42)
    k1, k2 = jax.random.split(key, 2)
    x = jax.random.normal(k1, (batch, seq_len, emb_dim), dtype=dtype)
    scale = jax.random.normal(k2, (emb_dim,), dtype=dtype) * 0.1 + 1.0
    return [x, scale], [epsilon]

# Computation
def computation(x, scale, epsilon):
    x_f32 = jnp.asarray(x, jnp.float32)
    mean2 = jnp.mean(lax.square(x_f32), axis=-1, keepdims=True)
    normed = x_f32 * lax.rsqrt(mean2 + epsilon)
    normed = jnp.asarray(normed, x.dtype)
    return normed * scale