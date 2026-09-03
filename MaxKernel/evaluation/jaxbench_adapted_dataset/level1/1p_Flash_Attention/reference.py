# Imports
import jax
import jax.numpy as jnp

# Initialization
def get_inputs():
    dtype = jnp.bfloat16
    key = jax.random.key(42)
    k1, k2, k3 = jax.random.split(key, 3)
    B = 4
    S = 4096
    H = 64
    D = 128
    query = jax.random.normal(k1, (B, H, S, D), dtype=dtype)
    key_t = jax.random.normal(k2, (B, H, S, D), dtype=dtype)
    value = jax.random.normal(k3, (B, H, S, D), dtype=dtype)
    dynamic_args = [query, key_t, value]
    static_args = []
    return dynamic_args, static_args

# Computation
def computation(query, key, value):
    B, H, S, D = query.shape
    scale = D ** -0.5
    attn = jnp.einsum('bhqd,bhkd->bhqk', query, key) * scale
    mask = jnp.tril(jnp.ones((S, S)))
    attn = jnp.where(mask, attn, -1e9)
    attn = jax.nn.softmax(attn, axis=-1)
    output = jnp.einsum('bhqk,bhkd->bhqd', attn, value)
    return output