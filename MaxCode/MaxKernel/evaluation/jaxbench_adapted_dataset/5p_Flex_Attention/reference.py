# pylint: skip-file
# Imports
import jax
import jax.numpy as jnp


# Initialization
def get_inputs(dtype=jnp.bfloat16):
  CONFIG = {
      'name': 'llama3_70b_flex_attention',
      'model': 'Llama-3.1-70B',
      'operator': 'flex_attention',
      'batch': 4,
      'seq_len': 4096,
      'num_heads': 64,
      'head_dim': 128,
  }
  key = jax.random.key(42)
  k1, k2, k3, k4 = jax.random.split(key, 4)
  B = CONFIG['batch']
  S = CONFIG['seq_len']
  H = CONFIG['num_heads']
  D = CONFIG['head_dim']
  q = jax.random.normal(k1, (B, H, S, D), dtype=dtype)
  k = jax.random.normal(k2, (B, H, S, D), dtype=dtype) * 0.02
  v = jax.random.normal(k3, (B, H, S, D), dtype=dtype) * 0.02
  rel_pos_bias = jax.random.normal(k4, (H, S, S), dtype=dtype) * 0.01

  dynamic_args = [q, k, v, rel_pos_bias]
  static_args = [D, S]
  return dynamic_args, static_args


# Computation
def computation(q, k, v, rel_pos_bias, head_dim, seq_len):
  D = head_dim
  S = seq_len
  sm_scale = D**-0.5

  attn = jnp.einsum('bhqd,bhkd->bhqk', q, k) * sm_scale
  attn = attn + rel_pos_bias[None, :, :, :]
  causal = jnp.tril(jnp.ones((S, S), dtype=jnp.bool_))
  attn = jnp.where(causal[None, None, :, :], attn, -1e30)
  attn = jax.nn.softmax(attn, axis=-1)
  out = jnp.einsum('bhqk,bhkd->bhqd', attn, v)
  return out
