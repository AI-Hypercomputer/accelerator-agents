# Imports
import jax
import jax.numpy as jnp


# Initialization
def get_inputs():
  CONFIG = {
    "name": "flash_attention_baseline",
    "model": "Baseline-MHA",
    "operator": "causal_mha",
    "batch": 1,
    "seq_len": 4096,
    "num_heads": 32,
    "head_dim": 128,
  }
  dtype = jnp.bfloat16
  key = jax.random.PRNGKey(42)
  k1, k2, k3 = jax.random.split(key, 3)
  B, S = CONFIG["batch"], CONFIG["seq_len"]
  H, D = CONFIG["num_heads"], CONFIG["head_dim"]
  query = jax.random.normal(k1, (B, H, S, D), dtype=dtype)
  key_t = jax.random.normal(k2, (B, H, S, D), dtype=dtype)
  value = jax.random.normal(k3, (B, H, S, D), dtype=dtype)
  return [query, key_t, value], []


# Computation
def computation(query, key, value):
  B, H, S, D = query.shape
  scale = D**-0.5
  attn = jnp.einsum("bhqd,bhkd->bhqk", query, key) * scale
  mask = jnp.tril(jnp.ones((S, S)))
  attn = jnp.where(mask, attn, -1e9)
  attn = jax.nn.softmax(attn, axis=-1)
  output = jnp.einsum("bhqk,bhkd->bhqd", attn, value)
  return output
