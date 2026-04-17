# Imports
import jax
import jax.numpy as jnp


# Initialization
def get_inputs():
  CONFIG = {
    "name": "mamba2_2_7b_ssd",
    "model": "Mamba-2-2.7B",
    "operator": "state_space_duality",
    "batch": 1,
    "seq_len": 2048,
    "num_heads": 64,
    "head_dim": 64,
    "d_state": 128,
    "d_model": 2560,
  }
  dtype = jnp.bfloat16
  rng = jax.random.PRNGKey(42)
  keys = jax.random.split(rng, 5)
  B, S = CONFIG["batch"], CONFIG["seq_len"]
  H, D = CONFIG["num_heads"], CONFIG["head_dim"]
  query = jax.random.normal(keys[0], (B, H, S, D), dtype=dtype)
  key_t = jax.random.normal(keys[1], (B, H, S, D), dtype=dtype)
  value = jax.random.normal(keys[2], (B, H, S, D), dtype=dtype)
  A_log = jax.random.normal(keys[3], (B, H, S), dtype=jnp.float32) * 0.5 - 4.0
  dynamic_args = [query, key_t, value, A_log]
  static_args = []
  return (dynamic_args, static_args)


# Computation
def computation(query, key, value, A_log):
  B, H, S, D = query.shape

  # Compute per-position decay: a = sigmoid(A_log) to keep in (0, 1)
  a = jax.nn.sigmoid(A_log.astype(jnp.float32))  # (B, H, S)

  # Build causal mask L with cumulative decay
  # log(a) cumsum then exponentiate: L[i,j] = exp(Σ_{k=j+1}^{i} log(a_k))
  log_a = jnp.log(a + 1e-8)  # (B, H, S)
  log_a_cumsum = jnp.cumsum(log_a, axis=-1)  # (B, H, S)

  # L[i,j] = exp(cumsum[i] - cumsum[j]) for i >= j, 0 for i < j
  diff = log_a_cumsum[:, :, :, None] - log_a_cumsum[:, :, None, :]
  causal = jnp.tril(jnp.ones((S, S), dtype=jnp.bool_))
  L = jnp.exp(jnp.where(causal[None, None, :, :], diff, -1e30))

  # SSD attention: (L ⊙ CB^T) x
  # CB^T: (B, H, S, S) — "attention scores"
  scores = jnp.einsum(
    "bhsd,bhtd->bhst", query.astype(jnp.float32), key.astype(jnp.float32)
  )

  # Apply selective decay mask
  scores = scores * L

  # Normalize
  scores_sum = jnp.sum(scores, axis=-1, keepdims=True)
  scores_sum = jnp.where(jnp.abs(scores_sum) < 1e-6, 1.0, scores_sum)
  scores = scores / jnp.maximum(jnp.abs(scores_sum), 1.0)

  # Output
  output = jnp.einsum("bhst,bhtd->bhsd", scores.astype(query.dtype), value)
  return output
