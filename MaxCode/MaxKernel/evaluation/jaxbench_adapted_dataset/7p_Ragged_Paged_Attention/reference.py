# pylint: skip-file
# Imports
import math

import jax
import jax.numpy as jnp


# Initialization
def get_inputs():
  CONFIG = {
      'name': 'ragged_paged_attention_llama70b',
      'model': 'Llama-3.1-70B',
      'operator': 'ragged_paged_attention',
      'max_num_batched_tokens': 4096,
      'max_num_seqs': 64,
      'num_q_heads': 64,
      'num_kv_heads': 8,
      'head_dim': 128,
      'page_size': 16,
      'pages_per_seq': 256,
  }

  dtype = jnp.bfloat16
  key = jax.random.key(42)
  k1, k2 = jax.random.split(key, 2)
  max_tokens = CONFIG['max_num_batched_tokens']
  max_seqs = CONFIG['max_num_seqs']
  H_q = CONFIG['num_q_heads']
  H_kv = CONFIG['num_kv_heads']
  D = CONFIG['head_dim']
  page_size = CONFIG['page_size']
  pages_per_seq = CONFIG['pages_per_seq']
  head_dim = CONFIG['head_dim']
  max_num_seqs = CONFIG['max_num_seqs']
  max_num_batched_tokens = CONFIG['max_num_batched_tokens']
  num_combined_kv_heads = 2 * H_kv
  total_num_pages = max_seqs * pages_per_seq
  q = jax.random.normal(k1, (max_tokens, H_q, D), dtype=dtype)
  kv_pages = jax.random.normal(
      k2, (total_num_pages, page_size, num_combined_kv_heads, D), dtype=dtype
  )
  tokens_per_seq = max_tokens // max_seqs
  kv_len_per_seq = pages_per_seq * page_size
  kv_lens = jnp.full((max_seqs,), kv_len_per_seq, dtype=jnp.int32)
  page_indices = jnp.arange(total_num_pages, dtype=jnp.int32).reshape(
      max_seqs, pages_per_seq
  )
  cu_q_lens = jnp.arange(max_seqs + 1, dtype=jnp.int32) * tokens_per_seq
  num_seqs = jnp.array([max_seqs], dtype=jnp.int32)

  dynamic_args = [q, kv_pages, kv_lens, page_indices, cu_q_lens, num_seqs]
  static_args = [head_dim, max_num_seqs, max_num_batched_tokens]
  return dynamic_args, static_args


# Computation
def computation(
    queries,
    kv_pages,
    kv_lens,
    page_indices,
    cu_q_lens,
    num_seqs,
    head_dim,
    max_num_seqs,
    max_num_batched_tokens,
):
  DEFAULT_MASK_VALUE = -0.7 * float(jnp.finfo(jnp.dtype('float32')).max)

  sm_scale = 1.0 / math.sqrt(head_dim)
  mask_value = DEFAULT_MASK_VALUE
  _, _, num_combined_kv_heads, head_dim = kv_pages.shape
  num_kv_heads = num_combined_kv_heads // 2
  num_q_heads = queries.shape[1]
  num_query_per_kv = num_q_heads // num_kv_heads

  max_seqs = max_num_seqs
  tokens_per_seq = max_num_batched_tokens // max_seqs

  outputs = []
  for i in range(max_seqs):
    q_start = cu_q_lens[i]
    kv_len = kv_lens[i]
    indices = page_indices[i]

    q = jax.lax.dynamic_slice(
        queries, (q_start, 0, 0), (tokens_per_seq, num_q_heads, head_dim)
    )

    k = kv_pages[indices, :, 0::2, :].reshape(-1, num_kv_heads, head_dim)
    v = kv_pages[indices, :, 1::2, :].reshape(-1, num_kv_heads, head_dim)

    k = jnp.repeat(k, num_query_per_kv, axis=1)
    v = jnp.repeat(v, num_query_per_kv, axis=1)

    attn = jnp.einsum('qhd,khd->hqk', q, k, preferred_element_type=jnp.float32)
    attn *= sm_scale

    q_span = (kv_len - tokens_per_seq) + jax.lax.broadcasted_iota(
        jnp.int32, attn.shape, 1
    )
    kv_span = jax.lax.broadcasted_iota(jnp.int32, attn.shape, 2)

    mask = (q_span < kv_span) | (kv_span >= kv_len)
    attn = jnp.where(mask, mask_value, attn)

    attn = jax.nn.softmax(attn, axis=-1).astype(v.dtype)
    out = jnp.einsum('hqk,khd->qhd', attn, v).astype(queries.dtype)

    is_valid = i < num_seqs[0]
    out = jnp.where(is_valid, out, 0.0)

    outputs.append(out)

  return jnp.concatenate(outputs, axis=0)
