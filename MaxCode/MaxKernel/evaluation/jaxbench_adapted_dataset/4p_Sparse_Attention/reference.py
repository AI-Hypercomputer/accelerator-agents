# pylint: skip-file
# Imports
import jax
import jax.numpy as jnp


# Initialization
def get_inputs(dtype=jnp.bfloat16):
  CONFIG = {
      'name': 'llama3_70b_sparse_attention',
      'model': 'Llama-3.1-70B',
      'operator': 'sparse_attention',
      'batch': 4,
      'seq_len': 4096,
      'num_query_heads': 64,
      'num_kv_heads': 8,
      'head_dim': 128,
  }
  key = jax.random.key(42)
  k1, k2, k3 = jax.random.split(key, 3)
  S = CONFIG['seq_len']
  H_q = CONFIG['num_query_heads']
  H_kv = CONFIG['num_kv_heads']
  D = CONFIG['head_dim']
  q = jax.random.normal(k1, (H_q, S, D), dtype=dtype) * (D**-0.5)
  k = jax.random.normal(k2, (H_kv, S, D), dtype=dtype) * 0.02
  v = jax.random.normal(k3, (H_kv, S, D), dtype=dtype) * 0.02

  dynamic_args = [q, k, v]
  static_args = [S, H_q, H_kv, D]
  return dynamic_args, static_args


# Computation
def computation(q, k, v, S, H_q, H_kv, D):
  num_q_per_kv = H_q // H_kv

  k = jnp.repeat(k, num_q_per_kv, axis=0)
  v = jnp.repeat(v, num_q_per_kv, axis=0)

  attn = jnp.einsum('hqd,hkd->hqk', q, k)

  causal = jnp.tril(jnp.ones((S, S), dtype=jnp.bool_))
  attn = jnp.where(causal[None, :, :], attn, -1e30)

  attn = jax.nn.softmax(attn, axis=-1)

  out = jnp.einsum('hqk,hkd->hqd', attn, v)
  return out
