# Imports
import jax
import jax.numpy as jnp


# Initialization
def get_inputs(dtype=jnp.bfloat16):
  CONFIG = {
    "name": "llama3_70b_paged_attention",
    "model": "Llama-3.1-70B",
    "operator": "paged_attention",
    "num_seqs": 64,
    "max_seq_len": 4096,
    "num_query_heads": 64,
    "num_kv_heads": 8,
    "head_dim": 128,
    "page_size": 16,
    "pages_per_seq": 256,
  }

  key = jax.random.key(42)
  keys = jax.random.split(key, 5)
  num_seqs = CONFIG["num_seqs"]
  num_q_heads = CONFIG["num_query_heads"]
  num_kv_heads = CONFIG["num_kv_heads"]
  head_dim = CONFIG["head_dim"]
  page_size = CONFIG["page_size"]
  pages_per_seq = CONFIG["pages_per_seq"]
  total_pages = num_seqs * pages_per_seq
  max_seq_len_derived = pages_per_seq * page_size

  max_num_tokens = num_seqs
  queries = jax.random.normal(
    keys[0], (max_num_tokens, num_q_heads, head_dim), dtype=dtype
  )
  k_pages = (
    jax.random.normal(
      keys[1], (total_pages, page_size, num_kv_heads, head_dim), dtype=dtype
    )
    * 0.02
  )
  v_pages = (
    jax.random.normal(
      keys[2], (total_pages, page_size, num_kv_heads, head_dim), dtype=dtype
    )
    * 0.02
  )

  kv_lens = jnp.full((num_seqs,), max_seq_len_derived, dtype=jnp.int32)
  page_indices = jnp.arange(total_pages, dtype=jnp.int32).reshape(
    num_seqs, pages_per_seq
  )
  cu_q_lens = jnp.arange(num_seqs + 1, dtype=jnp.int32)

  dynamic_args = [queries, k_pages, v_pages, kv_lens, page_indices, cu_q_lens]
  static_args = [
    num_seqs,
    num_q_heads,
    num_kv_heads,
    head_dim,
    max_seq_len_derived,
  ]

  return dynamic_args, static_args


# Computation
def computation(
  queries,
  k_pages,
  v_pages,
  kv_lens,
  page_indices,
  cu_q_lens,
  num_seqs,
  num_q_heads,
  num_kv_heads,
  head_dim,
  max_seq_len,
):
  num_q_per_kv = num_q_heads // num_kv_heads
  sm_scale = head_dim**-0.5

  def attend_one_seq(seq_idx):
    q_start = cu_q_lens[seq_idx]
    q_end = cu_q_lens[seq_idx + 1]
    q = jax.lax.dynamic_slice(
      queries, (q_start, 0, 0), (1, num_q_heads, head_dim)
    )
    seq_pages = page_indices[seq_idx]
    k = k_pages[seq_pages].reshape(max_seq_len, num_kv_heads, head_dim)
    v = v_pages[seq_pages].reshape(max_seq_len, num_kv_heads, head_dim)
    k = jnp.repeat(k, num_q_per_kv, axis=1)
    v = jnp.repeat(v, num_q_per_kv, axis=1)
    attn = jnp.einsum("qhd,khd->hqk", q, k) * sm_scale
    kv_len = kv_lens[seq_idx]
    mask = jnp.arange(max_seq_len) < kv_len
    attn = jnp.where(mask[None, None, :], attn, -1e30)
    attn = jax.nn.softmax(attn, axis=-1)
    out = jnp.einsum("hqk,khd->qhd", attn, v)
    return out.squeeze(0)

  outputs = jax.vmap(attend_one_seq)(jnp.arange(num_seqs))
  return outputs
