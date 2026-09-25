# Imports
from collections.abc import Sequence
import functools
from typing import Literal

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas.ops.tpu.paged_attention import quantization_utils
import jax.numpy as jnp
import numpy as np

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
class MultiPageAsyncCopyDescriptor:
  def __init__(
      self,
      pages_hbm_ref,
      scales_pages_hbm_ref,
      vmem_buffer,
      scales_vmem_buffer,
      sem,
      page_indices,
      page_indices_start_offset,
      num_pages_to_load,
      head_index,
  ):
    self._vmem_buffer = vmem_buffer
    self._scales_vmem_buffer = scales_vmem_buffer
    self._num_pages_to_load = num_pages_to_load
    if head_index is not None:
      self._pages_hbm_ref = pages_hbm_ref.at[head_index]
      if scales_pages_hbm_ref is not None:
        self._scales_pages_hbm_ref = scales_pages_hbm_ref.at[head_index]
      else:
        self._scales_pages_hbm_ref = None
    else:
      self._pages_hbm_ref = pages_hbm_ref
      self._scales_pages_hbm_ref = scales_pages_hbm_ref
    self._sem = sem
    self._page_indices = page_indices
    self._page_indices_start_offset = page_indices_start_offset
    self._async_copies = [
        self._make_async_copy(i) for i in range(self._num_pages_to_load)
    ]
    if (
        self._scales_pages_hbm_ref is not None
        and self._scales_vmem_buffer is not None
    ):
      self._async_copies += [
          self._make_scales_async_copy(i)
          for i in range(self._num_pages_to_load)
      ]

  def _make_async_copy(self, i):
    page_index = self._page_indices[self._page_indices_start_offset + i]
    return pltpu.make_async_copy(
        self._pages_hbm_ref.at[page_index], self._vmem_buffer.at[i], self._sem
    )

  def _make_scales_async_copy(self, i):
    page_index = self._page_indices[self._page_indices_start_offset + i]
    return pltpu.make_async_copy(
        self._scales_pages_hbm_ref.at[page_index],
        self._scales_vmem_buffer.at[i],
        self._sem,
    )

  def start(self):
    for async_copy in self._async_copies:
      async_copy.start()

  def _maybe_dequantize(self, x, x_scale, dtype=jnp.bfloat16):
    if x_scale is None:
      return x.astype(dtype)
    return quantization_utils.from_int8(x, x_scale, dtype=dtype)

  def wait_and_get_loaded(self) -> jax.Array:
    for async_copy in self._async_copies:
      async_copy.wait()
    head_dim = self._vmem_buffer.shape[-1]
    jax_array = self._vmem_buffer[...].astype(jnp.float32)
    if self._scales_vmem_buffer is not None:
      scales_jax_array = self._scales_vmem_buffer[...].astype(jnp.float32)
    else:
      scales_jax_array = None
    jax_array = self._maybe_dequantize(jax_array, scales_jax_array)
    return jax_array.reshape(-1, head_dim)


def paged_flash_attention_kernel(
    lengths_ref,
    page_indices_ref,
    buffer_index_ref,
    init_flag_ref,
    q_ref,
    k_pages_hbm_ref,
    k_scales_pages_hbm_ref,
    v_pages_hbm_ref,
    v_scales_pages_hbm_ref,
    o_ref,
    m_ref,
    l_ref,
    k_vmem_buffer,
    k_scales_vmem_buffer,
    v_vmem_buffer,
    v_scales_vmem_buffer,
    k_sems,
    v_sems,
    *,
    batch_size: int,
    pages_per_compute_block: int,
    pages_per_sequence: int,
    mask_value: float,
    attn_logits_soft_cap: float | None,
    megacore_mode: str | None,
    program_ids=(),
):
  if program_ids:
    core_index, b, h, i = program_ids
  else:
    core_index, b, h, i = (
        pl.program_id(0),
        pl.program_id(1),
        pl.program_id(2),
        pl.program_id(3),
    )
  num_kv_heads, _, page_size, _ = k_pages_hbm_ref.shape
  bk = page_size * pages_per_compute_block
  num_cores = pl.num_programs(0)

  b_step = num_cores if megacore_mode == "batch" else 1
  b_start = core_index if megacore_mode == "batch" else 0
  h_step = num_cores if megacore_mode == "kv_head" else 1
  h_start = core_index if megacore_mode == "kv_head" else 0

  h = h * h_step + h_start
  b = b * b_step + b_start
  length = lengths_ref[b]

  def compute_block_indices(b, h, i):

    def advance_b():
      next_b = b + b_step

      def advance_to_next_non_zero_length():
        next_next_b = next_b + b_step
        return lax.fori_loop(
            lax.div(next_next_b, b_step),
            lax.div(batch_size, b_step),
            lambda _, b: jnp.where(lengths_ref[b] == 0, b + b_step, b),
            next_next_b,
        )

      return (
          lax.cond(
              jnp.logical_and(
                  next_b < batch_size,
                  lengths_ref[lax.clamp(0, next_b, batch_size - 1)] == 0),
              advance_to_next_non_zero_length,
              lambda: next_b,
          ),
          h_start,
          0,
      )

    def advance_h():
      next_h = h + h_step
      return lax.cond(next_h < num_kv_heads, lambda: (b, next_h, 0), advance_b)

    return lax.cond(i * bk < lengths_ref[b], lambda: (b, h, i), advance_h)

  def create_kv_async_copy_descriptors(b, h, i, buffer_index):
    page_offset = b * pages_per_sequence + i * pages_per_compute_block
    pages_to_load = pages_per_compute_block
    async_copy_k = MultiPageAsyncCopyDescriptor(
        k_pages_hbm_ref,
        k_scales_pages_hbm_ref,
        k_vmem_buffer.at[buffer_index],
        k_scales_vmem_buffer.at[buffer_index]
        if k_scales_vmem_buffer is not None
        else None,
        k_sems.at[buffer_index],
        page_indices_ref,
        page_offset,
        pages_to_load,
        h,
    )
    async_copy_v = MultiPageAsyncCopyDescriptor(
        v_pages_hbm_ref,
        v_scales_pages_hbm_ref,
        v_vmem_buffer.at[buffer_index],
        v_scales_vmem_buffer.at[buffer_index]
        if v_scales_vmem_buffer is not None
        else None,
        v_sems.at[buffer_index],
        page_indices_ref,
        page_offset,
        pages_to_load,
        h,
    )
    return async_copy_k, async_copy_v

  @pl.when(i * bk < length)
  def flash_attention():
    init_flag = init_flag_ref[0]
    init_flag_ref[0] = 0
    buffer_index = buffer_index_ref[0]
    next_b, next_h, next_i = compute_block_indices(b, h, i + 1)

    @pl.when(init_flag)
    def prefetch_first_block():
      async_copy_k, async_copy_v = create_kv_async_copy_descriptors(
          b, h, i, buffer_index
      )
      async_copy_k.start()
      async_copy_v.start()

    @pl.when(i == 0)
    def init():
      m_ref[...] = jnp.full_like(m_ref, -jnp.inf)
      l_ref[...] = jnp.zeros_like(l_ref)
      o_ref[...] = jnp.zeros_like(o_ref)

    @pl.when(next_b < batch_size)
    def prefetch_next_block():
      next_buffer_index = jnp.where(buffer_index == 0, 1, 0)
      async_copy_next_k, async_copy_next_v = create_kv_async_copy_descriptors(
          next_b, next_h, next_i, next_buffer_index
      )
      async_copy_next_k.start()
      async_copy_next_v.start()
      buffer_index_ref[0] = next_buffer_index

    async_copy_k, async_copy_v = create_kv_async_copy_descriptors(
        b, h, i, buffer_index
    )
    q = q_ref[...].astype(jnp.float32)
    k = async_copy_k.wait_and_get_loaded()
    qk = jnp.einsum("gd,td->gt", q, k, preferred_element_type=jnp.float32)
    if attn_logits_soft_cap is not None:
      capped_qk = jnp.tanh(qk / attn_logits_soft_cap)
      qk = capped_qk * attn_logits_soft_cap

    mask = i * bk + jax.lax.broadcasted_iota(jnp.int32, qk.shape, 1) < length
    qk = qk + jnp.where(mask, 0.0, mask_value)
    m_curr = qk.max(axis=-1)

    s_curr = jnp.exp(qk - m_curr[..., None])
    m_prev, l_prev = m_ref[...], l_ref[...]
    l_curr = jax.lax.broadcast_in_dim(s_curr.sum(axis=-1), l_prev.shape, (0,))
    m_curr = jax.lax.broadcast_in_dim(m_curr, m_prev.shape, (0,))
    m_next = jnp.maximum(m_prev, m_curr)
    alpha = jnp.exp(m_prev - m_next)
    beta = jnp.exp(m_curr - m_next)
    l_next = alpha * l_prev + beta * l_curr
    m_ref[...], l_ref[...] = m_next, l_next

    v = async_copy_v.wait_and_get_loaded()
    o_curr = jnp.einsum("gt,td->gd", s_curr, v)

    o_ref[...] = (
        (l_prev * alpha * o_ref[...] + beta * o_curr) / l_next
    ).astype(o_ref.dtype)


def paged_flash_attention_kernel_inline_seq_dim(
    lengths_ref,
    page_indices_ref,
    buffer_index_ref,
    init_flag_ref,
    q_ref,
    k_pages_hbm_ref,
    k_scales_pages_hbm_ref,
    v_pages_hbm_ref,
    v_scales_pages_hbm_ref,
    o_ref,
    m_ref,
    l_ref,
    k_vmem_buffer,
    k_scales_vmem_buffer,
    v_vmem_buffer,
    v_scales_vmem_buffer,
    k_sems,
    v_sems,
    *,
    batch_size: int,
    pages_per_compute_block: int,
    pages_per_sequence: int,
    mask_value: float,
    attn_logits_soft_cap: float | None,
    megacore_mode: str | None,
):
  core_index, b, h = pl.program_id(0), pl.program_id(1), pl.program_id(2)

  m_ref[...] = jnp.full_like(m_ref, -jnp.inf)
  l_ref[...] = jnp.zeros_like(l_ref)
  o_ref[...] = jnp.zeros_like(o_ref)

  def body(i, _):
    paged_flash_attention_kernel(
        lengths_ref,
        page_indices_ref,
        buffer_index_ref,
        init_flag_ref,
        q_ref,
        k_pages_hbm_ref,
        k_scales_pages_hbm_ref,
        v_pages_hbm_ref,
        v_scales_pages_hbm_ref,
        o_ref,
        m_ref,
        l_ref,
        k_vmem_buffer,
        k_scales_vmem_buffer,
        v_vmem_buffer,
        v_scales_vmem_buffer,
        k_sems,
        v_sems,
        batch_size=batch_size,
        pages_per_compute_block=pages_per_compute_block,
        pages_per_sequence=pages_per_sequence,
        mask_value=mask_value,
        attn_logits_soft_cap=attn_logits_soft_cap,
        megacore_mode=megacore_mode,
        program_ids=(core_index, b, h, i),
    )
    return ()

  bk = pages_per_compute_block * k_pages_hbm_ref.shape[-2]

  if megacore_mode == "batch":
    num_cores = pl.num_programs(0)
    length = lengths_ref[b * num_cores + core_index]
  else:
    length = lengths_ref[b]

  lax.fori_loop(0, lax.div(length + bk - 1, bk), body, ())


@functools.partial(
    jax.jit,
    static_argnames=[
        "pages_per_compute_block",
        "attn_logits_soft_cap",
        "mask_value",
        "megacore_mode",
        "inline_seq_dim",
    ],
)
def paged_attention(
    q: jax.Array,
    k_pages: jax.Array | quantization_utils.QuantizedTensor,
    v_pages: jax.Array | quantization_utils.QuantizedTensor,
    lengths: jax.Array,
    page_indices: jax.Array,
    *,
    mask_value: float,
    attn_logits_soft_cap: float | None = None,
    pages_per_compute_block: int,
    megacore_mode: str | None = None,
    inline_seq_dim: bool = True,
) -> jax.Array:
  if isinstance(k_pages, quantization_utils.QuantizedTensor):
    k_pages, k_scales_pages = k_pages.weight, k_pages.scales
    assert isinstance(k_scales_pages, jax.Array)
    k_scales_pages = jnp.broadcast_to(
        k_scales_pages, (*k_scales_pages.shape[:-1], k_pages.shape[-1])
    )
  else:
    k_scales_pages = None
  if isinstance(v_pages, quantization_utils.QuantizedTensor):
    v_pages, v_scales_pages = v_pages.weight, v_pages.scales
    assert isinstance(v_scales_pages, jax.Array)
    v_scales_pages = jnp.broadcast_to(
        v_scales_pages, (*v_scales_pages.shape[:-1], v_pages.shape[-1])
    )
  else:
    v_scales_pages = None

  batch_size, num_q_heads, head_dim = q.shape
  num_kv_heads, _, page_size, head_dim_k = k_pages.shape
  batch_size_paged_indices, pages_per_sequence = page_indices.shape

  if k_pages.shape != v_pages.shape:
    raise ValueError(
        f"k_pages and v_pages must have the same shape. Got {k_pages.shape} and"
        f" {v_pages.shape}"
    )
  if num_q_heads % num_kv_heads != 0:
    raise ValueError(
        "Number of Q heads must be divisible by number of KV heads. Got"
        f" {num_q_heads} and {num_kv_heads}."
    )
  if head_dim_k != head_dim:
    raise ValueError(
        "head_dim of Q must be the same as that of K/V. Got"
        f" {head_dim} and {head_dim_k}."
    )
  if pages_per_sequence % pages_per_compute_block != 0:
    raise ValueError(
        "pages_per_compute_block must be divisible by pages per sequence. Got"
        f" {pages_per_compute_block} and {pages_per_sequence}."
    )
  if lengths.shape != (batch_size,):
    raise ValueError("`lengths` and `q` must have the same batch size")
  if batch_size_paged_indices != batch_size:
    raise ValueError("`page_indices` and `q` must have the same batch size")
  if lengths.dtype != jnp.int32:
    raise ValueError(
        f"The dtype of `lengths` must be int32. Got {lengths.dtype}"
    )

  if megacore_mode == "kv_head":
    if num_kv_heads % 2 != 0:
      raise ValueError(
          "number of KV heads must be even when megacore_mode is 'kv_head'"
      )
    num_cores = 2
  elif megacore_mode == "batch":
    if batch_size % 2 != 0:
      raise ValueError("batch size must be even when megacore_mode is 'batch'")
    num_cores = 2
  elif megacore_mode is None:
    num_cores = 1
  else:
    raise ValueError("megacore_mode must be one of ['kv_head', 'batch', None]")

  num_groups = num_q_heads // num_kv_heads
  if (num_groups) % 8 != 0:
    q = q.reshape(batch_size, num_q_heads, 1, head_dim)
    if megacore_mode == "kv_head":
      q_block_spec = pl.BlockSpec(
          (None, num_groups, None, head_dim),
          lambda core_index, b, h, *_: (b, h * num_cores + core_index, 0, 0),
      )
    elif megacore_mode == "batch":
      q_block_spec = pl.BlockSpec(
          (None, num_groups, None, head_dim),
          lambda core_index, b, h, *_: (b * num_cores + core_index, h, 0, 0),
      )
    else:
      q_block_spec = pl.BlockSpec(
          (None, num_groups, None, head_dim),
          lambda core_index, b, h, *_: (b, h, 0, 0),
      )
    q_dtype_for_kernel_launch = jnp.float32
  else:
    if megacore_mode == "kv_head":
      q_block_spec = pl.BlockSpec(
          (None, num_groups, head_dim),
          lambda core_index, b, h, *_: (b, h * num_cores + core_index, 0),
      )
    elif megacore_mode == "batch":
      q_block_spec = pl.BlockSpec(
          (None, num_groups, head_dim),
          lambda core_index, b, h, *_: (b * num_cores + core_index, h, 0),
      )
    else:
      q_block_spec = pl.BlockSpec(
          (None, num_groups, head_dim),
          lambda core_index, b, h, *_: (b, h, 0),
      )
    q_dtype_for_kernel_launch = q.dtype

  dimension_semantics: Sequence[Literal["parallel", "arbitrary"]]
  if inline_seq_dim:
    kernel = paged_flash_attention_kernel_inline_seq_dim
    grid = (
        num_cores,
        batch_size // num_cores if megacore_mode == "batch" else batch_size,
        num_kv_heads // num_cores
        if megacore_mode == "kv_head"
        else num_kv_heads,
    )
    dimension_semantics = ("parallel", "arbitrary", "arbitrary")
  else:
    kernel = paged_flash_attention_kernel
    grid = (
        num_cores,
        batch_size // num_cores if megacore_mode == "batch" else batch_size,
        num_kv_heads // num_cores
        if megacore_mode == "kv_head"
        else num_kv_heads,
        pages_per_sequence // pages_per_compute_block,
    )
    dimension_semantics = ("parallel", "arbitrary", "arbitrary", "arbitrary")

  if k_scales_pages is not None and v_scales_pages is not None:
    in_specs = [
        q_block_spec,
        pl.BlockSpec(memory_space=pl.ANY),
        pl.BlockSpec(memory_space=pl.ANY),
        pl.BlockSpec(memory_space=pl.ANY),
        pl.BlockSpec(memory_space=pl.ANY),
    ]
    scratch_shapes = (
        pltpu.VMEM(
            (
                2,
                pages_per_compute_block,
                page_size,
                head_dim,
            ),
            k_pages.dtype,
        ),
        pltpu.VMEM(
            (
                2,
                pages_per_compute_block,
                page_size,
                head_dim,
            ),
            k_scales_pages.dtype,
        ),
        pltpu.VMEM(
            (
                2,
                pages_per_compute_block,
                page_size,
                head_dim,
            ),
            v_pages.dtype,
        ),
        pltpu.VMEM(
            (
                2,
                pages_per_compute_block,
                page_size,
                head_dim,
            ),
            v_scales_pages.dtype,
        ),
        pltpu.SemaphoreType.DMA((2,)),
        pltpu.SemaphoreType.DMA((2,)),
    )
  else:
    in_specs = [
        q_block_spec,
        pl.BlockSpec(memory_space=pl.ANY),
        None,
        pl.BlockSpec(memory_space=pl.ANY),
        None,
    ]
    scratch_shapes = (
        pltpu.VMEM(
            (
                2,
                pages_per_compute_block,
                page_size,
                head_dim,
            ),
            k_pages.dtype,
        ),
        None,
        pltpu.VMEM(
            (
                2,
                pages_per_compute_block,
                page_size,
                head_dim,
            ),
            v_pages.dtype,
        ),
        None,
        pltpu.SemaphoreType.DMA((2,)),
        pltpu.SemaphoreType.DMA((2,)),
    )

  out, _, _ = pl.pallas_call(
      functools.partial(
          kernel,
          pages_per_sequence=pages_per_sequence,
          batch_size=batch_size,
          pages_per_compute_block=pages_per_compute_block,
          mask_value=mask_value,
          attn_logits_soft_cap=attn_logits_soft_cap,
          megacore_mode=megacore_mode,
      ),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=4,
          in_specs=in_specs,
          out_specs=[
              q_block_spec,
              q_block_spec,
              q_block_spec,
          ],
          grid=grid,
          scratch_shapes=scratch_shapes,
      ),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=dimension_semantics
      ),
      out_shape=[
          jax.ShapeDtypeStruct(q.shape, q_dtype_for_kernel_launch),
          jax.ShapeDtypeStruct((*q.shape[:-1], 1), jnp.float32),
          jax.ShapeDtypeStruct((*q.shape[:-1], 1), jnp.float32),
      ],
  )(
      lengths,
      page_indices.reshape(-1),
      jnp.zeros((1,), jnp.int32),
      jnp.ones((1,), jnp.int32),
      q.astype(q_dtype_for_kernel_launch),
      k_pages,
      k_scales_pages,
      v_pages,
      v_scales_pages,
  )
  return out.reshape(batch_size, num_q_heads, head_dim).astype(q.dtype)


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
    DEFAULT_MASK_VALUE = -0.7 * float(np.finfo(np.dtype("float32")).max)
    pages_per_compute_block = 128
    k_pages = k_pages.transpose(2, 0, 1, 3)
    v_pages = v_pages.transpose(2, 0, 1, 3)
    return paged_attention(
        queries, k_pages, v_pages, kv_lens, page_indices,
        mask_value=DEFAULT_MASK_VALUE,
        pages_per_compute_block=pages_per_compute_block,
    )