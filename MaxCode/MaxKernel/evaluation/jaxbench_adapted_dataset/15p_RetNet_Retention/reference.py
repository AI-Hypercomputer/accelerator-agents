# pylint: skip-file
# Imports
from functools import partial
import jax
import jax.numpy as jnp


# Initialization
def get_inputs():
  CONFIG = {
      'name': 'retnet_6_7b_retention',
      'model': 'RetNet-6.7B',
      'operator': 'multi_scale_retention',
      'batch': 4,
      'seq_len': 4096,
      'num_heads': 16,
      'head_dim': 256,
      'd_model': 4096,
  }
  dtype = jnp.bfloat16
  key = jax.random.key(42)
  keys = jax.random.split(key, 3)
  B, S = CONFIG['batch'], CONFIG['seq_len']
  H, D = CONFIG['num_heads'], CONFIG['head_dim']
  query = jax.random.normal(keys[0], (B, H, S, D), dtype=dtype)
  key_val = jax.random.normal(keys[1], (B, H, S, D), dtype=dtype)
  value = jax.random.normal(keys[2], (B, H, S, D), dtype=dtype)
  dynamic_args = [query, key_val, value]
  static_args = []
  return dynamic_args, static_args


# Computation
def computation(query, key, value):
  B, H, S, D = query.shape
  gammas = 1.0 - jnp.exp2(-5.0 - jnp.arange(H, dtype=jnp.float32))
  positions = jnp.arange(S, dtype=jnp.float32)
  distance = positions[:, None] - positions[None, :]
  causal_mask = (distance >= 0).astype(jnp.float32)
  log_gamma = jnp.log(gammas)
  decay = jnp.exp(
      log_gamma[:, None, None] * jnp.maximum(distance, 0.0)[None, :, :]
  )
  decay = decay * causal_mask[None, :, :]
  qk = jnp.einsum(
      'bhsd,bhtd->bhst', query.astype(jnp.float32), key.astype(jnp.float32)
  )
  qk = qk * decay[None, :, :, :]
  retention_sum = jnp.sum(jnp.abs(qk), axis=-1, keepdims=True)
  retention_sum = jnp.maximum(retention_sum, 1.0)
  qk = qk / retention_sum
  output = jnp.einsum('bhst,bhtd->bhsd', qk.astype(query.dtype), value)
  return output
