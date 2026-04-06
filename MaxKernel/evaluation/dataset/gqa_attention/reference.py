# Imports
import jax
import jax.numpy as jnp


# Initialization
def get_inputs():
  CONFIG = {
    "name": "llama3_405b_gqa",
    "model": "Llama-3.1-405B",
    "operator": "gqa_attention",
    "batch": 1,
    "seq_len": 2048,
    "num_query_heads": 128,
    "num_kv_heads": 8,
    "head_dim": 128,
    "emb_dim": 16384,
  }
  dtype = jnp.bfloat16
  key = jax.random.PRNGKey(42)
  k1, k2, k3 = jax.random.split(key, 3)
  B, S = CONFIG["batch"], CONFIG["seq_len"]
  Hq, Hkv, D = (
    CONFIG["num_query_heads"],
    CONFIG["num_kv_heads"],
    CONFIG["head_dim"],
  )
  query = jax.random.normal(k1, (B, Hq, S, D), dtype=dtype)
  key_t = jax.random.normal(k2, (B, Hkv, S, D), dtype=dtype)
  value = jax.random.normal(k3, (B, Hkv, S, D), dtype=dtype)
  return [query, key_t, value], []


# Computation
def computation(query, key, value):
  query = jax.lax.optimization_barrier(query)
  key = jax.lax.optimization_barrier(key)
  value = jax.lax.optimization_barrier(value)

  B, Hq, S, D = query.shape
  Hkv = key.shape[1]
  G = Hq // Hkv

  k = jnp.repeat(key[:, :, None, :, :], G, axis=2).reshape(B, Hq, S, D)
  v = jnp.repeat(value[:, :, None, :, :], G, axis=2).reshape(B, Hq, S, D)
  q = query

  scale = D**-0.5
  attn = jnp.einsum("bhqd,bhkd->bhqk", q, k) * scale
  mask = jnp.tril(jnp.ones((S, S)))
  attn = jnp.where(mask, attn, -1e9)
  attn = jax.nn.softmax(attn, axis=-1)

  out = jnp.einsum("bhqk,bhkd->bhqd", attn, v)
  return out.transpose(0, 2, 1, 3)
