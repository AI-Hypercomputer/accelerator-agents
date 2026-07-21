# Imports
import jax
import jax.numpy as jnp
from functools import partial

# Initialization
def get_inputs():
      import jax
      import jax.numpy as jnp
      from functools import partial

      configs = [
          ("decode_small", 2, 1, 4, 128, 256, 64),
          ("prefill_small", 2, 16, 4, 128, 256, 64),
          ("prefill_medium", 1, 128, 2, 256, 512, 128),
          ("odd_seq_len", 1, 9, 2, 128, 256, 64),
          ("large_topk", 1, 16, 2, 128, 1024, 512),
          ("user_decode_shape", 8, 1, 64, 512, 1024, 512),
          ("user_prefill_shape", 8, 512, 64, 512, 1024, 512),
          ("user_prefill_1024", 8, 1024, 64, 512, 1024, 512),
      ]

      outputs = []
      key = jax.random.PRNGKey(42)
      dtype = jnp.float32

      for name, batch, seq_len, num_heads, head_dim, kv_seq_len, topk in configs:
          key, k1, k2, k3, k4, k5 = jax.random.split(key, 6)

          q = jax.random.normal(k1, (batch, seq_len, num_heads, head_dim), dtype=dtype)
          kv = jax.random.normal(k2, (batch, kv_seq_len, head_dim), dtype=dtype)
          attn_sink = jax.random.normal(k3, (num_heads,), dtype=dtype)

          topk_idxs = jax.random.randint(k4, (batch, seq_len, topk), 0, kv_seq_len, dtype=jnp.int32)

          mask = jax.random.uniform(k5, (batch, seq_len, topk)) > 0.8
          topk_idxs = jnp.where(mask, -1, topk_idxs)

          softmax_scale = float(1.0 / jnp.sqrt(head_dim))
          
          q = jnp.array(q, dtype=jnp.bfloat16)
          kv = jnp.array(kv, dtype=jnp.bfloat16)   
          
          dynamic_args = [q, kv, attn_sink, topk_idxs]
          static_args = [softmax_scale]
          outputs.append((dynamic_args, static_args))

      return outputs

# Computation
@partial(jax.jit, static_argnames=['softmax_scale'])
def computation(
    q: jax.Array, 
    kv: jax.Array, 
    attn_sink: jax.Array, 
    topk_idxs: jax.Array, 
    softmax_scale: float
) -> jax.Array:
    b, m, h, d = q.shape
    _, _, topk = topk_idxs.shape
    
    valid_mask = topk_idxs != -1
    
    safe_idxs = jnp.where(valid_mask, topk_idxs, 0)
    
    batch_indices = jnp.arange(b)[:, None, None]
    kv_gathered = kv[batch_indices, safe_idxs]
    
    kv_gathered = jnp.where(jnp.expand_dims(valid_mask, -1), kv_gathered, 0.0)
    
    scores = jnp.einsum('bmhd,bmkd->bmhk', q, kv_gathered) * softmax_scale
    
    scores = jnp.where(jnp.expand_dims(valid_mask, 2), scores, -jnp.inf)
    
    scores_max = jnp.max(scores, axis=-1, keepdims=True)
    
    exp_scores = jnp.exp(scores - scores_max)
    
    sink_exp = jnp.exp(attn_sink.reshape(1, 1, h, 1) - scores_max)
    sum_exp = jnp.sum(exp_scores, axis=-1, keepdims=True) + sink_exp
    
    attn_weights = exp_scores / sum_exp
    
    out = jnp.einsum('bmhk,bmkd->bmhd', attn_weights, kv_gathered)
    
    return out