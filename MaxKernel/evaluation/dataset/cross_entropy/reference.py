# Imports
import jax
import jax.numpy as jnp


# Initialization
def get_inputs():
  CONFIG = {
    "name": "llama3_8b_cross_entropy",
    "model": "Llama-3.1-8B",
    "operator": "fused_cross_entropy",
    "batch_tokens": 4096,
    "hidden_dim": 4096,
    "vocab_size": 128256,
  }
  dtype = jnp.bfloat16
  key = jax.random.PRNGKey(42)
  k1, k2, k3 = jax.random.split(key, 3)
  B, H, V = CONFIG["batch_tokens"], CONFIG["hidden_dim"], CONFIG["vocab_size"]
  hidden = jax.random.normal(k1, (B, H), dtype=dtype)
  weight = jax.random.normal(k2, (H, V), dtype=dtype) * 0.02
  labels = jax.random.randint(k3, (B,), 0, V)
  dynamic_args = [hidden, weight, labels]
  static_args = []
  return dynamic_args, static_args


# Computation
def computation(hidden, weight, labels):
  logits = jnp.dot(hidden, weight)
  log_probs = jax.nn.log_softmax(logits, axis=-1)
  one_hot = jax.nn.one_hot(labels, logits.shape[-1])
  loss = -jnp.sum(one_hot * log_probs, axis=-1)
  return jnp.mean(loss)
