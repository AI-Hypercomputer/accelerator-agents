# Imports
import jax
import jax.numpy as jnp
from functools import partial

# Initialization
def get_inputs(dtype=jnp.bfloat16):
    CONFIG = {
        'name': 'llama3_405b_gqa',
        'model': 'Llama-3.1-405B',
        'operator': 'gqa_attention',
        'batch': 4,
        'seq_len': 4096,
        'num_query_heads': 128,
        'num_kv_heads': 8,
        'head_dim': 128,
        'emb_dim': 16384,
    }
    key = jax.random.key(42)
    k1, k2, k3 = jax.random.split(key, 3)
    B, S = CONFIG['batch'], CONFIG['seq_len']
    Hq, Hkv, D = CONFIG['num_query_heads'], CONFIG['num_kv_heads'], CONFIG['head_dim']
    query = jax.random.normal(k1, (B, S, Hq, D), dtype=dtype)
    key_t = jax.random.normal(k2, (B, S, Hkv, D), dtype=dtype)
    value = jax.random.normal(k3, (B, S, Hkv, D), dtype=dtype)
    dynamic_args = [query, key_t, value]
    static_args = []
    return dynamic_args, static_args

# Computation
def computation(query, key, value):
    B, S, Hq, D = query.shape
    Hkv = key.shape[2]
    G = Hq // Hkv
    key = jnp.repeat(key[:, :, :, None, :], G, axis=3).reshape(B, S, Hq, D)
    value = jnp.repeat(value[:, :, :, None, :], G, axis=3).reshape(B, S, Hq, D)
    q = query.transpose(0, 2, 1, 3)
    k = key.transpose(0, 2, 1, 3)
    v = value.transpose(0, 2, 1, 3)
    scale = D ** -0.5
    attn = jnp.einsum('bhqd,bhkd->bhqk', q, k) * scale
    mask = jnp.tril(jnp.ones((S, S)))
    attn = jnp.where(mask, attn, -1e9)
    attn = jax.nn.softmax(attn, axis=-1)
    out = jnp.einsum('bhqk,bhkd->bhqd', attn, v)
    return out.transpose(0, 2, 1, 3)