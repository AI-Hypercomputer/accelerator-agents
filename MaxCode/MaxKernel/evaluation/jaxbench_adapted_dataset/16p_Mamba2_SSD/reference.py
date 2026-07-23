# pylint: skip-file
# Imports
import jax
import jax.numpy as jnp


# Initialization
def get_inputs(dtype=jnp.bfloat16):
  CONFIG = {
      'name': 'mamba2_2_7b_ssd',
      'model': 'Mamba-2-2.7B',
      'operator': 'state_space_duality',
      'batch': 4,
      'seq_len': 4096,
      'num_heads': 64,
      'head_dim': 64,
      'd_state': 128,
      'd_model': 2560,
  }
  rng = jax.random.key(42)
  keys = jax.random.split(rng, 5)
  B, S = CONFIG['batch'], CONFIG['seq_len']
  H, D = CONFIG['num_heads'], CONFIG['head_dim']
  query = jax.random.normal(keys[0], (B, H, S, D), dtype=dtype)
  key = jax.random.normal(keys[1], (B, H, S, D), dtype=dtype)
  value = jax.random.normal(keys[2], (B, H, S, D), dtype=dtype)
  A_log = jax.random.normal(keys[3], (B, H, S), dtype=jnp.float32) * 0.5 - 4.0
  dynamic_args = [query, key, value, A_log]
  static_args = []
  return dynamic_args, static_args


# Computation
def computation(query, key, value, A_log):
  B, H, S, D = query.shape
  a = jax.nn.sigmoid(A_log.astype(jnp.float32))
  log_a = jnp.log(a + 1e-8)
  log_a_cumsum = jnp.cumsum(log_a, axis=-1)
  diff = log_a_cumsum[:, :, :, None] - log_a_cumsum[:, :, None, :]
  causal = jnp.tril(jnp.ones((S, S), dtype=jnp.bool_))
  L = jnp.exp(jnp.where(causal[None, None, :, :], diff, -1e30))
  scores = jnp.einsum(
      'bhsd,bhtd->bhst', query.astype(jnp.float32), key.astype(jnp.float32)
  )
  scores = scores * L
  scores_sum = jnp.sum(scores, axis=-1, keepdims=True)
  scores_sum = jnp.where(jnp.abs(scores_sum) < 1e-6, 1.0, scores_sum)
  scores = scores / jnp.maximum(jnp.abs(scores_sum), 1.0)
  output = jnp.einsum('bhst,bhtd->bhsd', scores.astype(query.dtype), value)
  return output
