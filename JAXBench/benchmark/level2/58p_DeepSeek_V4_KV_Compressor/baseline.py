"""DeepSeek-V4 KV-Cache Compressor — Pallas TPU Kernel.

Self-contained implementation.
"""
import sys
import types
import jax
import jax.numpy as jnp
import numpy as np

# ==============================================================================
# Inlined kv_compressor modules
# ==============================================================================
"""CSA Configs.

Modes:
  HCA:
    - State Bytes: 1024 (dim) x 4 (bytes per fp32) = 4096 bytes
    - Head Dim:    512 (bf16, no quantization)
    - Cache:       [num_pages, _, 4, 128] uint8 (where _ = state_block_size * 8
    or kv_block_size * 2)

  CSA:
    - State Bytes: 2048 (dim) x 4 (bytes per fp32) = 8192 bytes
    - Head Dim:    448 fp8 + 7 scale = 462 bytes -> 512 bytes
    - Cache:       [num_pages, _, 4, 128] uint8 (where _ = state_block_size * 16
    or kv_block_size)

  CSA-Indexer:
    - State Bytes: 512 (dim) x 4 (bytes per fp32) = 2048 bytes
    - Head Dim:    128 fp8 + 1 scale = 129 bytes -> 256 bytes
    - Cache:       [num_pages, _, 4, 256] uint8 (where _ = state_block_size * 2
    or kv_block_size // 8)
"""

import dataclasses
import enum

import jax
from jax.experimental import pallas as pl
import jax.numpy as jnp

# --- physical layout constants ------------------------------------------------
LANE = 128  # bytes per sub-slot / TPU lane width
SLOT_PACK = 4  # sub-slots packed into one physical HBM slot row
N_FIELDS = 2  # values stored per token: kv + score
FP32_BYTES = 4
CSA_COMPRESS_RATIO = 4  # the compress ratio that selects Mode.CSA


class Mode(enum.Enum):
  HCA = "hca"
  CSA = "csa"
  CSA_INDEXER = "csa_indexer"


_MODE_DEFAULTS = {
    Mode.HCA: dict(
        head_dim=512,
        rope_head_dim=64,
        compress_ratio=128,
        quant_block=0,
        overlap=False,
        has_rope_cache=False,
    ),
    Mode.CSA: dict(
        head_dim=512,
        rope_head_dim=64,
        compress_ratio=4,
        quant_block=64,
        overlap=True,
        has_rope_cache=True,
    ),
    Mode.CSA_INDEXER: dict(
        head_dim=128,
        rope_head_dim=64,
        compress_ratio=4,
        quant_block=128,
        overlap=True,
        has_rope_cache=False,
    ),
}


def select_mode(head_dim: int, overlap: bool) -> Mode:
  """Picks the storage mode from the compressor's head_dim / overlap flag."""
  if head_dim == LANE:
    return Mode.CSA_INDEXER
  return Mode.CSA if overlap else Mode.HCA


def physical_page_size(
    mode: Mode, kv_cache_block_size: int, compress_ratio: int
) -> int:
  """Rows per page of the uint8 cache array allocated for `mode`.

  This must mirror the shapes ``KVCacheManager._create_dsv4_kv_caches``
  allocates, because ``compress_norm_rope_store`` re-derives the whole page
  geometry from ``cache.shape[1]``. Anything that computes slot indices for
  that array (the compressor wrapper's state-cache ``block_size``, and
  ``derive_metadata``) has to agree with it or writes land on the wrong page.
  """
  # Compressed KV tokens per page (vLLM's ``spec.storage_block_size``).
  storage_block_size = kv_cache_block_size // compress_ratio
  if mode is Mode.CSA_INDEXER:
    # (N, T // 4, 4, 256) u8
    return storage_block_size // SLOT_PACK
  if mode is Mode.CSA:
    # (N, T, 4, 128) u8
    return storage_block_size
  # HCA (N, T * 2, 4, 128) u8
  return storage_block_size * 2


def state_host_page_size(
    mode: Mode, kv_cache_block_size: int, compress_ratio: int
) -> int:
  """Rows per page of the array that *hosts* ``mode``'s f32 compressor state."""
  if mode is Mode.HCA:
    return physical_page_size(Mode.CSA, kv_cache_block_size, CSA_COMPRESS_RATIO)
  return physical_page_size(mode, kv_cache_block_size, compress_ratio)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class TileSizes:
  """Tile sizes for the kernel."""
  tile_n: int


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class Dimensions:
  """True inputs only. Anything derivable is a property below."""

  mode: Mode = dataclasses.field(metadata=dict(static=True))
  size_n: int  # num_tokens
  head_dim: int
  rope_head_dim: int
  compress_ratio: int
  physical_page_size: int
  state_physical_page_size: int
  quant_block: int
  overlap: bool
  has_rope_cache: bool
  rms_eps: float = 1e-6

  cos_sin_dtype: jax.typing.DTypeLike = jnp.float32
  rope_width: int = 128

  @property
  def is_quantized(self) -> bool:
    return self.quant_block > 0

  @property
  def has_rope(self) -> bool:
    return self.rope_head_dim > 0

  @property
  def nope_dtype(self) -> jax.typing.DTypeLike:
    return jnp.uint8 if self.is_quantized else jnp.bfloat16

  @property
  def rope_dtype(self) -> jax.typing.DTypeLike:
    return self.nope_dtype


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class Configs:
  """Configuration for the kernel."""
  tile_sizes: TileSizes
  dims: Dimensions

  # --- factory ---------------------------------------------------------------
  @classmethod
  def make(
      cls,
      mode: Mode,
      *,
      size_n,
      physical_page_size,
      state_physical_page_size=None,
      rms_eps=1e-6,
      tile_n=4,
      **overrides,
  ) -> "Configs":
    """Build a config for `mode`; `overrides` replace per-mode defaults.

    `state_physical_page_size` defaults to `physical_page_size`, i.e. the
    state and the compressed KV share one buffer.
    """
    actual_overrides = {**_MODE_DEFAULTS[mode], **overrides}
    if mode == Mode.HCA:
      actual_overrides["quant_block"] = 0

    dims = Dimensions(
        mode=mode,
        size_n=size_n,
        physical_page_size=physical_page_size,
        state_physical_page_size=(
            physical_page_size
            if state_physical_page_size is None
            else state_physical_page_size
        ),
        rms_eps=rms_eps,
        **actual_overrides,
    )
    return cls(tile_sizes=TileSizes(tile_n=tile_n), dims=dims)

  # --- compute tiling (logical, in LANE-wide tiles) --------------------------
  @property
  def overlap_factor(self) -> int:
    """State copies kept per step: 1, or 2 when windows overlap (prev+curr)."""
    return 1 + int(self.dims.overlap)

  @property
  def state_width(self) -> int:
    """Width of one stored field (head_dim, doubled when overlapping)."""
    return self.overlap_factor * self.dims.head_dim

  @property
  def head_tiles(self) -> int:
    """head_dim split into LANE-wide compute tiles (old: slots_per_part)."""
    return self.dims.head_dim // LANE

  @property
  def window(self) -> int:
    """Timesteps compressed together (x overlap_factor when overlapping)."""
    return self.dims.compress_ratio * self.overlap_factor

  # --- rope sub-layout -------------------------------------------------------
  @property
  def nope_dim(self) -> int:
    """Width of the non-rope (nope) part of a head."""
    return self.dims.head_dim - self.dims.rope_head_dim

  @property
  def nope_store_dim(self) -> int:
    """Dimension of the nope storage (contains rope if no separate rope cache)."""
    return (
        self.dims.head_dim - self.dims.rope_head_dim
        if self.dims.has_rope_cache
        else self.dims.head_dim
    )

  @property
  def half_rope(self) -> int:
    """cos/sin split point (half the rope dim)."""
    return self.dims.rope_head_dim // 2

  @property
  def rope_slot(self) -> int:
    """Head-tile holding the rope channels (the last one)."""
    return self.head_tiles - 1

  # --- output record size ----------------------------------------------------
  @property
  def record_bytes(self) -> int:
    """Bytes in one packed output record (old: total_bytes_out)."""
    if not self.dims.is_quantized:
      return self.dims.head_dim * 2  # bf16
    # fp8 payload + scale + padding, capped to a fixed record width.
    return 256 if self.dims.head_dim == LANE else 512

  @property
  def record_rows(self) -> int:
    """Packed physical rows per output record."""
    return pl.cdiv(self.record_bytes, self.row_size_bytes)

  # --- HBM storage packing ---------------------------------------------------
  @property
  def row_size_bytes(self) -> int:
    """Size of one physical HBM row in bytes."""
    return self.hbm_pack * self.last_dim_size

  @property
  def hbm_pack(self) -> int:
    """Sub-slots physically packed per slot row, <= SLOT_PACK."""
    return SLOT_PACK

  @property
  def last_dim_size(self) -> int:
    """Size of the last HBM cache dimension in bytes."""
    if self.dims.mode == Mode.CSA_INDEXER:
      return 2 * LANE
    return LANE

  @property
  def cache_last_dims(self) -> tuple[int, ...]:
    return (self.hbm_pack, self.last_dim_size)

  # --- slot translation helpers -----------------------------------------
  @property
  def tokens_in_second_minor(self) -> int:
    """Divisor to map wrapper row offset to new physical row offset."""
    tokens_per_row = self.row_size_bytes // self.record_bytes
    return max(1, tokens_per_row)

  @property
  def kv_stride(self) -> int:
    """Token stride in logical slot units."""
    if self.record_bytes <= self.row_size_bytes:
      return 1
    return self.record_rows

  # --- block sizes and physical page size ------------------------------------
  @property
  def physical_page_size(self) -> int:
    """Number of physical HBM rows per page of the compressed-KV array."""
    return self.dims.physical_page_size

  @property
  def state_physical_page_size(self) -> int:
    """Number of physical HBM rows per page of the *state* array."""
    return self.dims.state_physical_page_size

  @property
  def state_rows_per_token(self) -> int:
    """Number of physical HBM rows occupied by one token's full state (kv + score)."""
    state_bytes = N_FIELDS * self.state_width * FP32_BYTES
    return state_bytes // self.row_size_bytes

  @property
  def field_rows(self) -> int:
    """Number of physical HBM rows spanned by one field during gather."""
    field_bytes = self.dims.head_dim * FP32_BYTES
    return pl.cdiv(field_bytes, self.row_size_bytes)

  @property
  def state_block_size(self) -> int:
    """Number of state tokens per page of the state array."""
    return self.state_physical_page_size // self.state_rows_per_token

  @property
  def kv_block_size(self) -> int:
    """Number of compressed KV tokens per page."""
    page_bytes = self.physical_page_size * self.row_size_bytes
    return page_bytes // self.record_bytes

  @property
  def rope_page_size(self) -> int:
    """Number of physical HBM rows per page in RoPE cache."""
    return self.physical_page_size // self.hbm_pack

  @property
  def pages_to_buffer_per_token(self) -> int:
    """Pages that must be resident to cover one token's window (+1 guard)."""
    return pl.cdiv(self.window, self.state_block_size) + 1

  # --- shapes (single source of truth for every reshape / BlockSpec) ---------
  @property
  def _tile_n(self) -> int:
    return self.tile_sizes.tile_n

  def window_shape(self) -> tuple[int, ...]:
    """f32 window scratch: (fields, tile, window, head_tiles, lane)."""
    return (N_FIELDS, self._tile_n, self.window, self.head_tiles, LANE)

  def window_bytes_shape(self) -> tuple[int, ...]:
    """uint8 view of the window scratch; each f32 lane -> FP32_BYTES rows.

    Note: that trailing FP32_BYTES (4) is bytes-per-f32, NOT the HBM SLOT_PACK
    (also 4) -- they're numerically equal but mean different things.

    Returns:
      The shape of the window bytes scratch.
    """
    return (
        N_FIELDS,
        self._tile_n,
        self.window,
        self.head_tiles,
        FP32_BYTES,
        LANE,
    )

  def output_shape(self) -> tuple[int, ...]:
    """Packed output tile / VMEM block: (tile, record_rows) + cache_last_dims."""
    return (self._tile_n, self.record_rows) + self.cache_last_dims

  def page_buffer_shape(self) -> tuple[int, ...]:
    """Page-buffer VMEM block: (tile, pages, state page rows) + cache_last_dims.

    Pages are DMA'd whole out of the *state* array, so this is sized by
    `state_physical_page_size`, not by the compressed-KV page.
    """
    return (
        self._tile_n,
        self.pages_to_buffer_per_token,
        self.state_physical_page_size,
    ) + self.cache_last_dims

  def rope_output_shape(self) -> tuple[int, ...]:
    """RoPE output VMEM block: (tile, 4, lane)."""
    assert self.dims.has_rope_cache
    return (self._tile_n, 4, LANE)

  def cos_sin_shape(self) -> tuple[int, ...]:
    """cos/sin VMEM block: (tile, rope_head_dim)."""
    return (self._tile_n, self.dims.rope_head_dim)

  def cache_shape(self, num_pages: int) -> tuple[int, ...]:
    """Shape of the global HBM KV cache."""
    return (num_pages, self.physical_page_size) + self.cache_last_dims

  def state_cache_shape(self, num_pages: int) -> tuple[int, ...]:
    """Shape of the global HBM array hosting the f32 compressor state."""
    return (num_pages, self.state_physical_page_size) + self.cache_last_dims

  def rope_cache_shape(self, num_pages: int) -> tuple[int, ...]:
    """Shape of the global HBM RoPE cache."""
    assert self.dims.has_rope_cache
    return (num_pages, self.rope_page_size, 4, 128)

"""Utility functions for compressor benchmarking and input generation."""

from typing import Any
import jax
import jax.numpy as jnp
import numpy as np


def generate_kv_slot_mapping(
    num_boundary_tokens: int,
    num_pages: int,
    page_size: int,
    slots_per_part_out: int,
) -> jax.Array:
  """Generates kv_slot_mapping for boundary tokens."""
  kv_slot_mapping_np = np.full((num_boundary_tokens,), -1, dtype=np.int32)
  boundary_count = 0
  total_slots_needed = num_boundary_tokens * slots_per_part_out
  pages_needed = (total_slots_needed + page_size - 1) // page_size
  start_page = max(0, num_pages - pages_needed)

  for t in range(num_boundary_tokens):
    kv_slot_mapping_np[t] = start_page * page_size + boundary_count
    boundary_count += slots_per_part_out
  return jnp.array(kv_slot_mapping_np, dtype=jnp.int32)


def generate_caches(
    cfgs: Any,
    num_pages: int,
) -> tuple[jax.Array, jax.Array | None]:
  """Generates empty cache and rope_cache."""
  cache = jnp.zeros(cfgs.cache_shape(num_pages), dtype=jnp.uint8)
  if cfgs.dims.has_rope_cache:
    rope_cache = jnp.zeros(cfgs.rope_cache_shape(num_pages), dtype=jnp.uint8)
  else:
    rope_cache = None
  return cache, rope_cache

import dataclasses
from typing import Any

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp

config = types.SimpleNamespace(
    Configs=Configs, Dimensions=Dimensions, Mode=Mode, TileSizes=TileSizes, select_mode=select_mode
)

@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, kw_only=True)
class _BufferedRef(pltpu.BufferedRef):
  cfgs: config.Configs = dataclasses.field(metadata=dict(static=True))

  @classmethod
  def create(
      cls,
      spec: pl.BlockSpec,
      dtype_or_type: jax.Array,
      buffer_type: pltpu.BufferType,
      buffer_count: int,
      use_lookahead: bool,
      cfgs: config.Configs,
  ):
    standard_ref = pltpu.BufferedRef.create(
        spec=spec,
        dtype_or_type=dtype_or_type,
        buffer_type=buffer_type,
        buffer_count=buffer_count,
        grid_rank=1,
        use_lookahead=use_lookahead,
    )
    return cls(
        cfgs=cfgs,
        **{
            f.name: getattr(standard_ref, f.name)
            for f in dataclasses.fields(pltpu.BufferedRef)
        },
    )


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, kw_only=True)
class CosSinRef(_BufferedRef):

  def copy_in(self, src_ref, grid_indices):
    cos_sin_cache_ref, positions_ref = src_ref
    slot = self.current_copy_in_slot
    sem = self.sem_recvs.at[slot]
    pid = grid_indices[0]

    dest_ref = self.window_ref.at[slot]  # (tile_n, rope_head_dim)

    tile_n = self.cfgs.tile_sizes.tile_n
    compress_ratio = self.cfgs.dims.compress_ratio

    @pl.loop(0, tile_n, unroll=True)
    def loop_body(i):
      global_idx = pid * tile_n + i
      position = positions_ref[global_idx]
      compressed_pos = (position // compress_ratio) * compress_ratio

      src = cos_sin_cache_ref.at[
          compressed_pos, pl.ds(0, self.window_ref.shape[-1])
      ]
      dest = dest_ref.at[i, :]
      pltpu.make_async_copy(src, dest, sem).start()

  def wait_in(self, src_ref, grid_indices):
    slot = self.current_wait_in_slot
    sem = self.sem_recvs.at[slot]
    vmem_ref = self.window_ref.at[slot]
    pltpu.make_async_copy(vmem_ref, vmem_ref, sem).wait()


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, kw_only=True)
class PageBufferRef(_BufferedRef):
  # VMEM block: cfgs.page_buffer_shape()
  #   (tile_n, pages_to_buffer_per_token, page_size, hbm_pack, LANE) uint8

  def copy_in(self, src_ref, grid_indices):
    slot = self.current_copy_in_slot
    sem = self.sem_recvs.at[slot]
    pid = grid_indices[0]

    (
        state_cache_ref,
        positions_ref,
        block_table_ref,
        block_table_stride,
        token_to_req_indices_ref,
    ) = src_ref

    tile_n = self.cfgs.tile_sizes.tile_n
    pages_to_buffer_per_token = self.cfgs.pages_to_buffer_per_token
    block_size = self.cfgs.state_block_size

    page_buffer_ref = self.window_ref.at[slot]
    global_idx = pid * tile_n

    @pl.loop(0, tile_n * pages_to_buffer_per_token, unroll=True)
    def loop_body(idx):
      n = idx // pages_to_buffer_per_token
      p = idx % pages_to_buffer_per_token
      idx_n = global_idx + n

      position = positions_ref[idx_n]
      req_idx = token_to_req_indices_ref[idx_n]

      block_idx = position // block_size - (pages_to_buffer_per_token - 1) + p
      safe_block_idx = jnp.maximum(block_idx, 0)
      page = block_table_ref[req_idx * block_table_stride + safe_block_idx]

      src = state_cache_ref.at[page, :, :, :]
      dest = page_buffer_ref.at[n, p, :, :, :]
      pltpu.make_async_copy(src, dest, sem).start()

  def wait_in(self, src_ref, grid_indices):
    slot = self.current_wait_in_slot
    sem = self.sem_recvs.at[slot]
    vmem_ref = self.window_ref.at[slot]
    pltpu.make_async_copy(vmem_ref, vmem_ref, sem).wait()


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, kw_only=True)
class OutputRef(_BufferedRef):
  # VMEM block: cfgs.output_shape() -> (tile_n, record_rows, hbm_pack, last_dim_size)

  def copy_in(self, src_ref, grid_indices):
    # Only called in CSA_INDEXER mode (INPUT_OUTPUT)
    cache_ref, kv_slot_mapping_ref, _ = src_ref
    slot = self.current_copy_in_slot
    sem = self.sem_recvs.at[slot]
    dest_ref = self.window_ref.at[slot]
    pid = grid_indices[0]

    tile_n = self.cfgs.tile_sizes.tile_n
    global_idx = pid * tile_n

    # Cache is laid out like this:
    # [num_page, kv_block_size / 4, 4, 256]
    page_size = self.cfgs.kv_block_size

    @pl.loop(0, tile_n)
    def loop_body(n):
      idx_n = global_idx + n
      kv_slot = kv_slot_mapping_ref[idx_n]
      valid = kv_slot >= 0

      @pl.when(valid)
      def _read_nope():
        p = kv_slot // page_size
        # we fetch the entire tile (4, 128), with 4 pages per tile
        s_row = (kv_slot % page_size) // (self.cfgs.tokens_in_second_minor)
        src = cache_ref.at[p, s_row, :, :]
        dest = dest_ref.at[n, 0, :, :]
        pltpu.make_async_copy(src, dest, sem).start()

  def wait_in(self, src_ref, grid_indices):
    slot = self.current_wait_in_slot
    sem = self.sem_recvs.at[slot]
    vmem_ref = self.window_ref.at[slot]
    pid = grid_indices[0]

    tile_n = self.cfgs.tile_sizes.tile_n
    global_idx = pid * tile_n
    _, kv_slot_mapping_ref, _ = src_ref

    @pl.loop(0, tile_n)
    def loop_body(n):
      idx_n = global_idx + n
      kv_slot = kv_slot_mapping_ref[idx_n]
      valid = kv_slot >= 0

      @pl.when(valid)
      def _wait_nope():
        pltpu.make_async_copy(
            vmem_ref.at[n, 0, :, :],
            vmem_ref.at[n, 0, :, :],
            sem,
        ).wait()

  def copy_out(self, dest_ref, grid_indices):
    slot = self.current_copy_out_slot
    sem = self.sem_sends.at[slot]
    src_ref = self.window_ref.at[slot]
    pid = grid_indices[0]
    tile_n = self.cfgs.tile_sizes.tile_n
    global_idx = pid * tile_n

    if self.cfgs.dims.mode == config.Mode.CSA_INDEXER:
      cache_ref, kv_slot_mapping_ref, is_first_mask_ref = dest_ref

      @pl.loop(0, tile_n)
      def loop_body(n):
        idx_n = global_idx + n
        kv_slot = kv_slot_mapping_ref[idx_n]
        is_first = is_first_mask_ref[idx_n]
        valid = (kv_slot >= 0) & is_first

        @pl.when(valid)
        def _write_nope():
          p = kv_slot // self.cfgs.kv_block_size
          s_row = (kv_slot % self.cfgs.kv_block_size) // (
              self.cfgs.tokens_in_second_minor
          )
          src = src_ref.at[n, 0, :, :]
          dest = cache_ref.at[p, s_row, :, :]
          pltpu.make_async_copy(src, dest, sem).start()

    else:
      cache_ref, kv_slot_mapping_ref = dest_ref

      # HCA: [num_pages, kv_block_size * 2, 4, 128] uint8
      # CSA: [num_pages, kv_block_size, 4, 128] uint8
      @pl.loop(0, tile_n)
      def loop_body(n):
        idx_n = global_idx + n
        kv_slot = kv_slot_mapping_ref[idx_n]

        @pl.when(kv_slot >= 0)
        def _write_nope():
          p = kv_slot // self.cfgs.physical_page_size
          s = kv_slot % self.cfgs.physical_page_size

          @pl.loop(0, self.cfgs.record_rows, unroll=True)
          def write_loop(o):
            src_nope = src_ref.at[n, o, ...]
            dest_nope = cache_ref.at[p, s + o, ...]
            pltpu.make_async_copy(src_nope, dest_nope, sem).start()

  def wait_out(self, dest_ref, grid_indices):
    slot = self.current_wait_out_slot
    sem = self.sem_sends.at[slot]
    vmem_ref = self.window_ref.at[slot]
    pid = grid_indices[0]
    tile_n = self.cfgs.tile_sizes.tile_n
    global_idx = pid * tile_n

    is_indexer = self.cfgs.dims.mode == config.Mode.CSA_INDEXER

    if is_indexer:
      _, kv_slot_mapping_ref, is_first_mask_ref = dest_ref

      @pl.loop(0, tile_n)
      def loop_body(n):
        idx_n = global_idx + n
        kv_slot = kv_slot_mapping_ref[idx_n]
        is_first = is_first_mask_ref[idx_n]
        valid = (kv_slot >= 0) & is_first

        @pl.when(valid)
        def _wait_nope():
          pltpu.make_async_copy(
              vmem_ref.at[n, 0, :, :],
              vmem_ref.at[n, 0, :, :],
              sem,
          ).wait()

    else:
      _, kv_slot_mapping_ref = dest_ref

      @pl.loop(0, tile_n)
      def loop_body(n):
        idx_n = global_idx + n
        kv_slot = kv_slot_mapping_ref[idx_n]

        @pl.when(kv_slot >= 0)
        def _wait_nope():

          @pl.loop(0, self.cfgs.record_rows, unroll=True)
          def wait_loop(o):
            pltpu.make_async_copy(
                vmem_ref.at[n, o, :, :],
                vmem_ref.at[n, o, :, :],
                sem,
            ).wait()


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, kw_only=True)
class RoPEOutputRef(_BufferedRef):
  # VMEM block: cfgs.rope_output_shape() -> (tile_n, 1, LANE) uint8
  # Only used for CSA (has_rope_cache=True).

  def copy_in(self, src_ref, grid_indices):
    rope_cache_ref, kv_slot_mapping_ref, is_first_mask_ref = src_ref
    slot = self.current_copy_in_slot
    sem = self.sem_recvs.at[slot]
    dest_ref = self.window_ref.at[slot]
    pid = grid_indices[0]

    tile_n = self.cfgs.tile_sizes.tile_n
    global_idx = pid * tile_n

    @pl.loop(0, tile_n)
    def loop_body(n):
      idx_n = global_idx + n
      kv_slot = kv_slot_mapping_ref[idx_n]
      is_first = is_first_mask_ref[idx_n]
      valid = kv_slot >= 0

      @pl.when(valid & is_first)
      def _read_rope():
        # divide by 4 because we pack 4 pages into 1 tile
        # layout: [num_page, kv_block_size / 4, 4, 256]
        rope_slot = kv_slot // 4
        rope_p = rope_slot // self.cfgs.rope_page_size
        s_row = rope_slot % self.cfgs.rope_page_size
        src = rope_cache_ref.at[rope_p, s_row, :, :]
        dest = dest_ref.at[n, :, :]
        pltpu.make_async_copy(src, dest, sem).start()

  def wait_in(self, src_ref, grid_indices):
    slot = self.current_wait_in_slot
    sem = self.sem_recvs.at[slot]
    vmem_ref = self.window_ref.at[slot]
    pid = grid_indices[0]

    tile_n = self.cfgs.tile_sizes.tile_n
    global_idx = pid * tile_n
    _, kv_slot_mapping_ref, is_first_mask_ref = src_ref

    @pl.loop(0, tile_n)
    def loop_body(n):
      idx_n = global_idx + n
      kv_slot = kv_slot_mapping_ref[idx_n]
      is_first = is_first_mask_ref[idx_n]
      valid = kv_slot >= 0

      @pl.when(valid & is_first)
      def _wait_rope():
        pltpu.make_async_copy(
            vmem_ref.at[n, :, :],
            vmem_ref.at[n, :, :],
            sem,
        ).wait()

  def copy_out(self, dest_ref, grid_indices):
    rope_cache_ref, kv_slot_mapping_ref, is_first_mask_ref = dest_ref
    slot = self.current_copy_out_slot
    sem = self.sem_sends.at[slot]

    src_ref = self.window_ref.at[slot]
    pid = grid_indices[0]

    tile_n = self.cfgs.tile_sizes.tile_n
    global_idx = pid * tile_n

    @pl.loop(0, tile_n)
    def loop_body(n):
      idx_n = global_idx + n
      kv_slot = kv_slot_mapping_ref[idx_n]
      is_first = is_first_mask_ref[idx_n]
      valid = kv_slot >= 0

      @pl.when(valid & is_first)
      def _write_rope():
        rope_slot = kv_slot // 4
        rope_p = rope_slot // self.cfgs.rope_page_size
        s_row = rope_slot % self.cfgs.rope_page_size
        src_rope = src_ref.at[n, :, :]
        dest_rope = rope_cache_ref.at[rope_p, s_row, :, :]
        pltpu.make_async_copy(src_rope, dest_rope, sem).start()

  def wait_out(self, dest_ref, grid_indices):
    slot = self.current_wait_out_slot
    sem = self.sem_sends.at[slot]
    vmem_ref = self.window_ref.at[slot]
    pid = grid_indices[0]

    tile_n = self.cfgs.tile_sizes.tile_n
    global_idx = pid * tile_n

    _, kv_slot_mapping_ref, is_first_mask_ref = dest_ref

    @pl.loop(0, tile_n)
    def loop_body(n):
      idx_n = global_idx + n
      kv_slot = kv_slot_mapping_ref[idx_n]
      is_first = is_first_mask_ref[idx_n]
      valid = kv_slot >= 0

      @pl.when(valid & is_first)
      def _wait_rope():
        pltpu.make_async_copy(
            vmem_ref.at[n, :, :],
            vmem_ref.at[n, :, :],
            sem,
        ).wait()


def create_allocs_and_specs(
    cfgs: config.Configs,
    *,
    cache_ref,
    rope_cache_ref,
    state_cache_ref,
    cos_sin_cache_ref,
    positions_ref,
    block_table_ref,
    block_table_stride,
    token_to_req_indices_ref,
    kv_slot_mapping_ref,
    is_first_mask_ref,
    is_first_mask_rope_ref,
) -> tuple[
    tuple[CosSinRef | None, OutputRef, PageBufferRef, RoPEOutputRef | None],
    tuple[pl.BlockSpec | None, pl.BlockSpec, pl.BlockSpec, pl.BlockSpec | None],
    tuple[
        tuple[Any, ...] | None,
        tuple[Any, ...],
        tuple[Any, ...],
        tuple[Any, ...] | None,
    ],
]:
  # cos_sin
  if cfgs.dims.has_rope:
    tile_n = cfgs.tile_sizes.tile_n
    cos_sin_spec = pl.BlockSpec(
        block_shape=cfgs.cos_sin_shape(),
        memory_space=pltpu.VMEM,
        index_map=lambda i: (i * tile_n, 0),
    )
    cos_sin_alloc = CosSinRef.create(
        spec=cos_sin_spec,
        dtype_or_type=cfgs.dims.cos_sin_dtype,
        buffer_type=pltpu.BufferType.INPUT,
        buffer_count=2,
        use_lookahead=False,
        cfgs=cfgs,
    )
    cos_sin_args = (cos_sin_cache_ref, positions_ref)
  else:
    cos_sin_alloc = cos_sin_spec = cos_sin_args = None

  # output
  output_spec = pl.BlockSpec(
      block_shape=cfgs.output_shape(),
      memory_space=pltpu.VMEM,
      index_map=lambda i: (i, 0, 0, 0),
  )
  is_indexer = cfgs.dims.mode == config.Mode.CSA_INDEXER
  buffer_type = (
      pltpu.BufferType.INPUT_OUTPUT if is_indexer else pltpu.BufferType.OUTPUT
  )
  output_alloc = OutputRef.create(
      spec=output_spec,
      dtype_or_type=jnp.uint8,
      buffer_type=buffer_type,
      buffer_count=2,
      use_lookahead=False,
      cfgs=cfgs,
  )
  if is_indexer:
    output_args = (cache_ref, kv_slot_mapping_ref, is_first_mask_ref)
  else:
    output_args = (cache_ref, kv_slot_mapping_ref)

  # page_buffer
  page_buffer_spec = pl.BlockSpec(
      block_shape=cfgs.page_buffer_shape(),
      memory_space=pltpu.VMEM,
      index_map=lambda i: (i, 0, 0, 0),
  )
  page_buffer_alloc = PageBufferRef.create(
      spec=page_buffer_spec,
      dtype_or_type=jnp.uint8,
      buffer_type=pltpu.BufferType.INPUT,
      buffer_count=2,
      use_lookahead=False,
      cfgs=cfgs,
  )
  page_buffer_args = (
      cache_ref if state_cache_ref is None else state_cache_ref,
      positions_ref,
      block_table_ref,
      block_table_stride,
      token_to_req_indices_ref,
  )

  # rope
  if cfgs.dims.has_rope_cache:
    rope_spec = pl.BlockSpec(
        block_shape=cfgs.rope_output_shape(),
        memory_space=pltpu.VMEM,
        index_map=lambda i: (i, 0, 0),
    )
    rope_alloc = RoPEOutputRef.create(
        spec=rope_spec,
        dtype_or_type=jnp.uint8,
        buffer_type=pltpu.BufferType.INPUT_OUTPUT,
        buffer_count=2,
        use_lookahead=False,
        cfgs=cfgs,
    )
    rope_args = (rope_cache_ref, kv_slot_mapping_ref, is_first_mask_rope_ref)
  else:
    rope_alloc = rope_spec = rope_args = None

  return (
      (cos_sin_alloc, output_alloc, page_buffer_alloc, rope_alloc),
      (cos_sin_spec, output_spec, page_buffer_spec, rope_spec),
      (cos_sin_args, output_args, page_buffer_args, rope_args),
  )

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp


def interleaved_rope_vector(x, cos_val_32, sin_val_32):
  """Applies interleaved Rotary Position Embedding (RoPE) to a vector."""
  # x: (tile_n, 1, 128)
  # cos_val_32: (tile_n, 1, 32)
  # sin_val_32: (tile_n, 1, 32)
  tile_n = x.shape[0]

  # We work with 2D tensors for gather to keep dimensions simple.
  x_2d = jnp.squeeze(x, axis=1)  # (tile_n, 128)

  # swap adjacent pairs: [1, 0, 3, 2, ...]
  iota = jnp.arange(128)
  swap_indices = jnp.bitwise_xor(iota, 1)  # (128,)
  swap_coords = jnp.broadcast_to(swap_indices, (tile_n, 128))[
      :, :, None
  ]  # (tile_n, 128, 1)

  gather_dn = jax.lax.GatherDimensionNumbers(
      offset_dims=(),
      collapsed_slice_dims=(1,),
      start_index_map=(1,),
      operand_batching_dims=(0,),
      start_indices_batching_dims=(0,),
  )

  x_swapped_2d = jax.lax.gather(
      x_2d,
      swap_coords,
      dimension_numbers=gather_dn,
      slice_sizes=(1, 1),
      unique_indices=True,
      mode=jax.lax.GatherScatterMode.PROMISE_IN_BOUNDS,
  )

  # cos_val_32/sin_val_32 are (tile_n, 1, 32). Squeeze to (tile_n, 32)
  cos_32 = jnp.squeeze(cos_val_32, axis=1).astype(x.dtype)
  sin_32 = jnp.squeeze(sin_val_32, axis=1).astype(x.dtype)

  ones_32 = jnp.ones((tile_n, 32), dtype=x.dtype)
  zeros_32 = jnp.zeros((tile_n, 32), dtype=x.dtype)

  # pad to pairs: (tile_n, 64)
  cos_pairs = jnp.concatenate([ones_32, cos_32], axis=-1)
  sin_pairs = jnp.concatenate([zeros_32, sin_32], axis=-1)

  # duplicate indices: [0, 0, 1, 1, 2, 2, ..., 63, 63]
  dup_indices = iota // 2
  dup_coords = jnp.broadcast_to(dup_indices, (tile_n, 128))[:, :, None]

  cos_dup_2d = jax.lax.gather(
      cos_pairs,
      dup_coords,
      dimension_numbers=gather_dn,
      slice_sizes=(1, 1),
      unique_indices=False,
      mode=jax.lax.GatherScatterMode.PROMISE_IN_BOUNDS,
  )

  sin_dup_2d = jax.lax.gather(
      sin_pairs,
      dup_coords,
      dimension_numbers=gather_dn,
      slice_sizes=(1, 1),
      unique_indices=False,
      mode=jax.lax.GatherScatterMode.PROMISE_IN_BOUNDS,
  )

  # alternate sin: [-1, 1, -1, 1, ...]
  alt_mask = ((iota % 2) * 2 - 1).astype(x.dtype)[None, :]  # (1, 128)
  sin_alt_2d = sin_dup_2d * alt_mask

  out_2d = x_2d * cos_dup_2d + x_swapped_2d * sin_alt_2d

  return out_2d[:, None, :]


def quantize_fp8_tiled(x, block_size):
  # x: (tile_n, S, 128) f32
  fp8_max = float(jnp.finfo(jnp.float8_e4m3fn).max)
  tile_n, slots, width = x.shape  # (tile_n, S, 128) f32
  num_blocks = width // block_size

  qs = []
  scales = []
  for b in range(num_blocks):
    start = b * block_size
    end = start + block_size
    x_block = x[:, :, start:end]  # (tile_n, S, block_size) f32

    amax = jnp.clip(
        jnp.max(jnp.abs(x_block), axis=-1, keepdims=True), 1e-4, None
    )  # (tile_n, S, 1) f32

    log2_val = jnp.log2(amax / fp8_max)  # f32
    scale = jnp.exp2(jnp.ceil(log2_val))  # (tile_n, S, 1) f32

    q_block = (x_block * (1.0 / scale)).astype(
        jnp.float8_e4m3fn
    )  # (tile_n, S, block_size) fp8

    qs.append(q_block)
    scales.append(scale)

  q = jnp.concatenate(qs, axis=-1)  # (tile_n, S, 128) fp8
  scale = jnp.concatenate(scales, axis=-1)  # (tile_n, S, num_blocks) f32

  # Bitcast workaround directly on f32 to extract exponent
  # f32 exponent is at bits 23-30. Shift right by 23 to align it.
  scale_u32 = pltpu.bitcast(scale, jnp.uint32)
  scale_exp = scale_u32 >> 23
  scale_u8 = scale_exp.astype(jnp.uint8)
  return q, scale_u8


def pack_nope_tiled(
    q, scale, nope_dim, block_size, nope_width_bytes=512, last_dim_size=128
):
  # q: (tile_n, S, 128) fp8
  # scale: (tile_n, S, num_blocks) e8m0 (uint8)
  tile_n, _, _ = q.shape

  # Bitcast to uint8
  q_bytes = pltpu.bitcast(q, jnp.uint8)
  scale_bytes = (
      scale if scale.dtype == jnp.uint8 else pltpu.bitcast(scale, jnp.uint8)
  )

  # Flat representations
  q_flat = q_bytes.reshape(tile_n, -1)  # (tile_n, S * 128)
  scale_flat = scale_bytes.reshape(tile_n, -1)  # (tile_n, S * num_blocks)

  # Select NOPE parts
  if nope_dim < q_flat.shape[1]:
    q_nope = q_flat[:, :nope_dim]
  else:
    q_nope = q_flat

  nope_blocks = (nope_dim + block_size - 1) // block_size
  if nope_blocks < scale_flat.shape[1]:
    scale_nope = scale_flat[:, :nope_blocks]
  else:
    scale_nope = scale_flat

  # Pad with zeros
  pad_size = nope_width_bytes - (nope_dim + nope_blocks)
  zeros = jnp.zeros((tile_n, pad_size), dtype=jnp.uint8)

  nope_record_padded = jnp.concatenate(
      [q_nope, scale_nope, zeros], axis=1
  )  # (tile_n, 512)
  return nope_record_padded.reshape(tile_n, -1, last_dim_size)


def pack_rope_tiled(rope_slot_ropped, rope_head_dim_actual, rope_width=128):
  # rope_slot_ropped: (tile_n, 1, 128)
  # TODO: make this a config value rather than inferred
  start = 128 - rope_head_dim_actual
  rope_val = rope_slot_ropped[:, :, start:]  # (tile_n, 1, rope_head_dim)

  rope_bf16 = rope_val.astype(jnp.bfloat16)
  rope_width_bf16 = rope_width // 2

  if rope_head_dim_actual < rope_width_bf16:
    rope_padded_bf16 = jnp.pad(
        rope_bf16,
        ((0, 0), (0, 0), (0, rope_width_bf16 - rope_head_dim_actual)),
    )
  else:
    rope_padded_bf16 = rope_bf16

  rope_u16 = pltpu.bitcast(rope_padded_bf16, jnp.uint16)
  rope_i32 = rope_u16.astype(jnp.int32)
  rope_hi = ((rope_i32 >> 8) & 0xFF).astype(jnp.uint8)
  rope_lo = (rope_i32 & 0xFF).astype(jnp.uint8)
  rope_uint8_2d = jnp.concatenate([rope_hi, rope_lo], axis=-1)

  return rope_uint8_2d[:, None, :]


def merge_slot_updates(
    slots_val,  # (pack_factor, physical_slot_size) uint8 (the row)
    kv_slots,  # (tile_n,) int (all slots in tile)
    val_padded,  # (tile_n, record_subslots, 128) uint8 (all values in tile)
    n,  # int (current token index)
):
  """multiple KV slots are packed into a single physical 512-byte row in HBM.

  Thus, different slots may map to the same tile.

  We only send the DMA from the first tile that updates a physical row,
  defined by `is_first_mask`.

  In this function, we:
  1. Read the current value of `slot_val`
  2. Scan all other tokens in the current tile
  3. Consolidate all other updates into the single tile.
  """
  tile_n = kv_slots.shape[0]
  curr_slot = kv_slots[n]
  pack_factor = slots_val.shape[0]

  curr_row = curr_slot // pack_factor

  slots_list = [slots_val[i] for i in range(pack_factor)]

  for i in range(tile_n):
    slot_i = kv_slots[i]
    valid_i = slot_i >= 0
    row_i = slot_i // pack_factor
    sub_idx = slot_i % pack_factor

    update_cond = valid_i & (row_i == curr_row) & (curr_slot >= 0)

    val = val_padded[i].flatten()

    for target_s in range(pack_factor):
      slots_list[target_s] = jax.lax.select(
          update_cond & (sub_idx == target_s),
          val,
          slots_list[target_s],
      )

  return jnp.stack(slots_list)


def gather_from_page_buffer(
    page_buffer,
    positions_ref,
    kv_window_u8,
    score_window_u8,
    *,
    global_idx: int,
    num_tokens: int,
    window: int,
    block_size: int,
    pages_to_buffer_per_token: int,
    field_rows: int,
    state_rows_per_token: int,
    overlap: bool,
    is_indexer: bool = False,
):
  """Extracts kv_window, score_window from page buffer."""
  tile_n = page_buffer.shape[0]
  if is_indexer:
    # CSA_INDEXER (layout 32x4x256)
    # page_buffer shape: (tile_n, pages_to_buffer, 32, 4, 256)
    for n in range(tile_n):
      safe_idx = jnp.minimum(global_idx + n, num_tokens - 1)
      pos = positions_ref[safe_idx]

      block_idx_curr = pos // block_size
      pos_start = pos - window + 1

      for w in range(window):
        pos_w = pos_start + w
        block_idx_w = pos_w // block_size
        p = block_idx_w - block_idx_curr + pages_to_buffer_per_token - 1

        offset_in_block = pos_w % block_size
        token_row_start = offset_in_block * 2

        kv_row = token_row_start + 0
        score_row = token_row_start + 1

        valid_w = pos_w >= 0

        @pl.when(valid_w)
        def _():
          row_kv = page_buffer[n, p, kv_row][...]  # shape (4, 256)
          row_score = page_buffer[n, p, score_row][...]  # shape (4, 256)

          # `proj_and_save_state` stores the f32 state byte-transposed
          # (u8[s, l] == byte s of f32 l), so one (4, 256) row holds
          # 256 floats across its LANES. The prev/curr halves of the
          # overlapping state are therefore lanes 0:128 and 128:256.
          if overlap:
            is_prev = w < (window // 2)
            val_kv = jax.lax.select(
                is_prev,
                row_kv[:, 0:128],
                row_kv[:, 128:256],
            )
            val_score = jax.lax.select(
                is_prev,
                row_score[:, 0:128],
                row_score[:, 128:256],
            )
          else:
            val_kv = row_kv[:, 0:128]
            val_score = row_score[:, 0:128]

          kv_window_u8[n, w, 0, :, :] = val_kv
          score_window_u8[n, w, 0, :, :] = val_score

  else:
    # Standard HCA/CSA (last dim 128)
    slots_per_part_head = kv_window_u8.shape[2]

    for n in range(tile_n):
      safe_idx = jnp.minimum(global_idx + n, num_tokens - 1)
      pos = positions_ref[safe_idx]

      block_idx_curr = pos // block_size
      pos_start = pos - window + 1

      @pl.loop(0, window)
      def body_w(w):
        pos_w = pos_start + w
        block_idx_w = pos_w // block_size
        p = block_idx_w - block_idx_curr + pages_to_buffer_per_token - 1

        slots_per_part_row = field_rows
        slots_per_token_row = state_rows_per_token
        if overlap:
          is_prev = w < (window // 2)
          kv_slot_start_row = jax.lax.select(is_prev, 0, slots_per_part_row)
          score_slot_start_row = jax.lax.select(
              is_prev, 2 * slots_per_part_row, 3 * slots_per_part_row
          )
        else:
          kv_slot_start_row = 0
          score_slot_start_row = slots_per_part_row

        offset_in_block = pos_w % block_size

        @pl.loop(0, slots_per_part_head, unroll=True)
        def gather_loop(d_idx):
          kv_src_row = (
              offset_in_block * slots_per_token_row + kv_slot_start_row + d_idx
          )
          score_src_row = (
              offset_in_block * slots_per_token_row
              + score_slot_start_row
              + d_idx
          )

          kv_window_u8[n, w, d_idx, :, :] = page_buffer[n, p, kv_src_row, :, :]
          score_window_u8[n, w, d_idx, :, :] = page_buffer[
              n, p, score_src_row, :, :
          ]

import functools

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
buffered_ref = types.SimpleNamespace(
    create_allocs_and_specs=create_allocs_and_specs,
)
compute = types.SimpleNamespace(
    gather_from_page_buffer=gather_from_page_buffer,
    interleaved_rope_vector=interleaved_rope_vector,
    quantize_fp8_tiled=quantize_fp8_tiled,
    pack_nope_tiled=pack_nope_tiled,
    pack_rope_tiled=pack_rope_tiled,
    merge_slot_updates=merge_slot_updates,
)


def inner_kernel(
    cos_sin_vmem,
    out_vmem,
    page_buffer_vmem,
    rope_out_vmem,
    positions_ref,
    rms_weight_vmem_ref,
    window_vmem,
    kv_slot_mapping_ref,
    is_first_mask_ref,
    is_first_mask_rope_ref,
    *,
    cfgs: config.Configs,
):
  tile_n = cfgs.tile_sizes.tile_n
  window = cfgs.window
  head_tiles = cfgs.head_tiles

  pid = pl.program_id(0)
  global_idx = pid * tile_n

  # Per-token window start = position - window + 1
  start_list = []
  for i in range(tile_n):
    idx = global_idx + i
    safe_idx = jax.lax.select(idx < cfgs.dims.size_n, idx, 0)
    start_list.append(positions_ref[safe_idx] - window + 1)
  start = jnp.stack(start_list)  # (tile_n,)

  window_u8 = window_vmem.bitcast(jnp.uint8).reshape(
      2, tile_n, window, head_tiles, 4, 128
  )
  kv_window_u8 = window_u8.at[0]
  score_window_u8 = window_u8.at[1]

  compute.gather_from_page_buffer(
      page_buffer=page_buffer_vmem,
      positions_ref=positions_ref,
      kv_window_u8=kv_window_u8,
      score_window_u8=score_window_u8,
      global_idx=global_idx,
      num_tokens=cfgs.dims.size_n,
      window=window,
      block_size=cfgs.state_block_size,
      pages_to_buffer_per_token=cfgs.pages_to_buffer_per_token,
      field_rows=cfgs.field_rows,
      state_rows_per_token=cfgs.state_rows_per_token,
      overlap=cfgs.dims.overlap,
      is_indexer=cfgs.dims.mode == config.Mode.CSA_INDEXER,
  )

  kv_val = window_vmem.at[0][...]  # (tile_n, window, head_tiles, 128)
  scores_val = window_vmem.at[1][...]  # (tile_n, window, head_tiles, 128)
  rms_weight_tiled = rms_weight_vmem_ref[...].astype(
      jnp.float32
  )  # (head_tiles, 128)

  # --- windowed softmax ---
  curr_pos = start[:, None] + jnp.arange(window)[None, :]  # (tile_n, window)
  mask = curr_pos >= 0  # (tile_n, window)
  mask_float = mask.astype(scores_val.dtype)
  mask_float_reshaped = mask_float[:, :, None, None]  # (tile_n, window, 1, 1)
  neg_inf = jnp.array(-jnp.inf, dtype=scores_val.dtype)
  masked_scores = jnp.where(mask_float_reshaped > 0.5, scores_val, neg_inf)
  weights = jax.nn.softmax(masked_scores, axis=1)
  kv_val = jnp.where(mask_float_reshaped > 0.5, kv_val, 0.0)
  compressed = jnp.sum(weights * kv_val, axis=1)  # (tile_n, head_tiles, 128)

  # --- rms norm ---
  variance = jnp.mean(jnp.square(compressed), axis=(1, 2), keepdims=True)
  normed = (
      compressed
      * jax.lax.rsqrt(variance + cfgs.dims.rms_eps)
      * rms_weight_tiled[None, :, :]
  )  # (tile_n, head_tiles, 128)

  # --- rope ---
  rope_ropped = None
  if cfgs.dims.has_rope:
    rope_slot = cfgs.rope_slot
    rope_val = normed[:, rope_slot : rope_slot + 1]

    cos_sin = cos_sin_vmem[...][:, None, :]
    cos_val = cos_sin[:, :, : cfgs.half_rope]
    sin_val = cos_sin[:, :, cfgs.half_rope :]

    rope_ropped = compute.interleaved_rope_vector(rope_val, cos_val, sin_val)
    if head_tiles > 1:
      normed = jnp.concatenate([normed[:, :rope_slot], rope_ropped], axis=1)
    else:
      normed = rope_ropped

  # --- pack + store ---
  if cfgs.dims.is_quantized:
    q, scale = compute.quantize_fp8_tiled(normed, cfgs.dims.quant_block)
    nope_val_padded = compute.pack_nope_tiled(
        q,
        scale,
        cfgs.nope_store_dim,
        cfgs.dims.quant_block,
        nope_width_bytes=cfgs.record_bytes,
        last_dim_size=cfgs.last_dim_size,
    )
    if cfgs.dims.mode == config.Mode.CSA_INDEXER:
      kv_slots = []
      for i in range(tile_n):
        kv_slots.append(kv_slot_mapping_ref[global_idx + i])
      kv_slots = jnp.stack(kv_slots)

      for n in range(tile_n):
        is_first = is_first_mask_ref[global_idx + n]

        @pl.when(is_first)
        def _merge_nope():
          # only applicable to csa-indexer
          # out_vmem shape: (tile_n, 1, 4, 256)
          slots_val = out_vmem[n, 0]
          out_vmem[n, 0] = compute.merge_slot_updates(
              slots_val,
              kv_slots,
              nope_val_padded,
              n,
          )

    else:
      out_vmem[:, 0] = nope_val_padded
    if cfgs.dims.has_rope_cache:

      rope_val_padded = compute.pack_rope_tiled(
          rope_ropped, cfgs.dims.rope_head_dim, cfgs.dims.rope_width
      )  # (tile_n, 1, 128)

      rope_slots_val = rope_out_vmem[...]  # (tile_n, 4, 128)
      kv_slots = []
      for i in range(tile_n):
        kv_slots.append(kv_slot_mapping_ref[global_idx + i])
      kv_slots = jnp.stack(kv_slots)
      for n in range(tile_n):
        is_first = is_first_mask_rope_ref[global_idx + n]

        @pl.when(is_first)
        def _merge_rope():
          rope_out_vmem[n] = compute.merge_slot_updates(
              rope_slots_val[n],
              kv_slots,
              rope_val_padded,
              n,
          )

  else:
    # hca: bitcast bf16 -> uint8 and match the output block shape.
    out_vmem[...] = pltpu.bitcast(
        normed.astype(cfgs.dims.nope_dtype), jnp.uint8
    ).reshape(tile_n, cfgs.record_rows, cfgs.hbm_pack, 128)


def kernel_fn(
    # prefetched inputs (scalar)
    block_table_ref,
    positions_ref,
    token_to_req_indices_ref,
    kv_slot_mapping_ref,
    is_first_mask_ref,
    is_first_mask_rope_ref,
    grid_size_ref,
    rms_weight_ref,
    # HBM inputs (dynamic access)
    cos_sin_cache_ref,
    cache_ref,
    rope_cache_ref,
    state_cache_ref,
    # outputs (aliased)
    _out_cache_ref,
    _out_rope_cache_ref,
    window_scratch_ref,
    *,
    cfgs: config.Configs,
    block_table_stride: int,
):
  """Pallas kernel entry point."""
  grid_size = grid_size_ref[...]
  allocs, in_specs, pipeline_args = buffered_ref.create_allocs_and_specs(
      cfgs=cfgs,
      cache_ref=cache_ref,
      rope_cache_ref=rope_cache_ref,
      state_cache_ref=state_cache_ref,
      cos_sin_cache_ref=cos_sin_cache_ref,
      positions_ref=positions_ref,
      block_table_ref=block_table_ref,
      block_table_stride=block_table_stride,
      token_to_req_indices_ref=token_to_req_indices_ref,
      kv_slot_mapping_ref=kv_slot_mapping_ref,
      is_first_mask_ref=is_first_mask_ref,
      is_first_mask_rope_ref=is_first_mask_rope_ref,
  )

  pipeline_func = pltpu.emit_pipeline(
      body=functools.partial(inner_kernel, cfgs=cfgs),
      grid=(grid_size,),
      in_specs=in_specs,
      out_specs=[],
  )

  @pl.with_scoped(allocations=tuple(allocs))
  def _run(allocations):
    pipeline_func(
        *pipeline_args,
        scratches=(
            positions_ref,
            rms_weight_ref,
            window_scratch_ref,
            kv_slot_mapping_ref,
            is_first_mask_ref,
            is_first_mask_rope_ref,
        ),
        allocations=allocations,
    )

  _run()


def _select_mode(head_dim: int, overlap: bool) -> config.Mode:
  return config.select_mode(head_dim, overlap)


def derive_aliases(
    has_rope: bool, has_rope_cache: bool, num_scalar_prefetch: int
) -> dict[int, int]:
  cache_index = num_scalar_prefetch + 1 + int(has_rope)
  aliases = {cache_index: 0}
  if has_rope_cache:
    aliases[cache_index + 1] = 1
  return aliases


def compute_is_first_mask(kv_slot_mapping, tile_n, pack_factor=4):
  """Determines, for every token in a sequence, whether it is the first token within its execution tile to map to a particular physical HBM row.

  If multiple tokens in the same tile map to the same row, only the first one is
  responsible for writing the merged VMEM row buffer back to HBM. The
  subsequent tokens in the same tile that conflict will skip the HBM write to
  prevent overwriting each other's data and reduce memory traffic.
  """
  num_tokens = kv_slot_mapping.shape[0]
  pad_len = (tile_n - (num_tokens % tile_n)) % tile_n
  if pad_len > 0:
    kv_slots_padded = jnp.pad(kv_slot_mapping, (0, pad_len), constant_values=-1)
  else:
    kv_slots_padded = kv_slot_mapping

  total_tokens = kv_slots_padded.shape[0]
  num_tiles = total_tokens // tile_n
  kv_slots_tiled = kv_slots_padded.reshape(num_tiles, tile_n)

  row_idxs = kv_slots_tiled // pack_factor
  valid = kv_slots_tiled >= 0

  eq = row_idxs[:, None, :] == row_idxs[:, :, None]
  tril = jnp.tril(jnp.ones((tile_n, tile_n), dtype=bool), k=-1)
  conflict = eq & tril[None, :, :] & valid[:, None, :]
  has_conflict = jnp.any(conflict, axis=-1)
  is_first = valid & ~has_conflict

  return is_first.flatten()[:num_tokens]


@functools.partial(
    jax.jit,
    static_argnames=(
        "block_table_stride",
        "compress_ratio",
        "overlap",
        "quant_block",
        "rms_eps",
        "interpret",
        "name",
    ),
    donate_argnames=("cache", "rope_cache"),
)
def compress_norm_rope_store(
    cache: jax.Array,
    positions: jax.Array,
    block_table: jax.Array,  # [num_reqs * block_table_stride] int
    token_to_req_indices: jax.Array,
    kv_slot_mapping: jax.Array,
    rms_weight: jax.Array,
    *,
    block_table_stride: int,
    state_cache: jax.Array | None = None,
    rope_cache: jax.Array | None = None,
    cos_sin_cache: jax.Array | None = None,
    compress_ratio: int,
    overlap: bool,
    quant_block: int = 64,
    rms_eps: float = 1e-6,
    interpret: bool = False,
    name: str = "compress_norm_rope_store",
) -> tuple[jax.Array, jax.Array | None]:
  """Compresses, normalizes, applies RoPE and stores to cache."""
  assert block_table.ndim == 1
  assert block_table.shape[0] % block_table_stride == 0

  num_tokens = positions.shape[0]
  head_dim = rms_weight.shape[0]
  rope_head_dim = cos_sin_cache.shape[1] if cos_sin_cache is not None else 0

  state_operand = (
      None if state_cache is None or state_cache is cache else state_cache
  )
  state_source = cache if state_operand is None else state_operand

  cfgs = config.Configs.make(
      _select_mode(head_dim, overlap),
      size_n=num_tokens,
      physical_page_size=cache.shape[1],
      state_physical_page_size=state_source.shape[1],
      rms_eps=rms_eps,
      tile_n=4,
      head_dim=head_dim,
      rope_head_dim=rope_head_dim,
      compress_ratio=compress_ratio,
      quant_block=quant_block,
  )

  if cfgs.dims.mode in (config.Mode.CSA, config.Mode.CSA_INDEXER):
    assert cfgs.tile_sizes.tile_n % 4 == 0, (
        f"tile_n must be a multiple of 4 for {cfgs.dims.mode.value}, "
        f"got {cfgs.tile_sizes.tile_n}"
    )

  if cfgs.dims.has_rope_cache and rope_cache is None:
    raise ValueError("rope_cache must be provided when has_rope_cache is True")

  rms_weight_reshaped = rms_weight.reshape(cfgs.head_tiles, 128)

  # Compute grid size dynamically based on the maximum index in kv_slot_mapping.
  valid = kv_slot_mapping >= 0
  indices = jnp.arange(kv_slot_mapping.shape[0])
  max_idx = jnp.max(jnp.where(valid, indices, -1))
  grid_size = jnp.where(
      max_idx >= 0, pl.cdiv(max_idx + 1, cfgs.tile_sizes.tile_n), 0
  )

  is_first_mask = compute_is_first_mask(
      kv_slot_mapping,
      cfgs.tile_sizes.tile_n,
      pack_factor=cfgs.tokens_in_second_minor,
  )
  is_first_mask_rope = compute_is_first_mask(
      kv_slot_mapping,
      cfgs.tile_sizes.tile_n,
      # TODO: this is confusing, should probably find a better name for it
      pack_factor=cfgs.hbm_pack,
  )
  # Outer pallas_call operands, in call order. Optional operands are passed as
  # None to keep kernel_fn's argument positions fixed; pallas drops the Nones
  # when indexing, so alias indices count only the operands actually present.
  scalar_prefetch = (
      block_table,
      positions,
      token_to_req_indices,
      kv_slot_mapping,
      is_first_mask,
      is_first_mask_rope,
      grid_size,
  )
  cos_sin_operand = cos_sin_cache if cfgs.dims.has_rope else None
  rope_operand = rope_cache if cfgs.dims.has_rope_cache else None

  in_specs = (
      pl.BlockSpec(memory_space=pltpu.VMEM),  # rms_weight
      (
          pl.BlockSpec(memory_space=pltpu.HBM) if cfgs.dims.has_rope else None
      ),  # cos_sin
      pl.BlockSpec(memory_space=pltpu.HBM),  # cache
      (
          pl.BlockSpec(memory_space=pltpu.HBM)
          if cfgs.dims.has_rope_cache
          else None
      ),  # rope_cache
      (
          pl.BlockSpec(memory_space=pltpu.HBM)
          if state_operand is not None
          else None
      ),  # state_cache
  )
  out_specs = (
      pl.BlockSpec(memory_space=pltpu.HBM),  # cache
      (
          pl.BlockSpec(memory_space=pltpu.HBM)
          if cfgs.dims.has_rope_cache
          else None
      ),  # rope_cache
  )
  out_shapes = (
      jax.ShapeDtypeStruct(cache.shape, cache.dtype),
      jax.ShapeDtypeStruct(rope_cache.shape, rope_cache.dtype)
      if cfgs.dims.has_rope_cache
      else None,
  )

  aliases = derive_aliases(
      cfgs.dims.has_rope, cfgs.dims.has_rope_cache, len(scalar_prefetch)
  )

  grid_spec = pltpu.PrefetchScalarGridSpec(
      num_scalar_prefetch=len(scalar_prefetch),
      in_specs=in_specs,
      out_specs=out_specs,
      scratch_shapes=[pltpu.VMEM(cfgs.window_shape(), jnp.float32)],
  )

  out_cache, out_rope_cache = pl.pallas_call(
      functools.partial(
          kernel_fn, cfgs=cfgs, block_table_stride=block_table_stride
      ),
      out_shape=out_shapes,
      grid_spec=grid_spec,
      input_output_aliases=aliases,
      compiler_params=pltpu.CompilerParams(disable_bounds_checks=True),
      interpret=interpret,
      name=name,
  )(
      *scalar_prefetch,
      rms_weight_reshaped,
      cos_sin_operand,
      cache,
      rope_operand,
      state_operand,
  )

  return out_cache, out_rope_cache



CONFIGS = {
    "csa_small": {
        "name": "csa_small",
        "model": "DeepSeek-V4",
        "operator": "compress_norm_rope_store",
        "case_type": "csa",
        "batch_size": 16,
        "hidden_size": 7168,
        "head_dim": 512,
        "rope_head_dim": 64,
        "compress_ratio": 4,
        "physical_page_size": 256,
        "quant_block": 128,
        "overlap": True,
        "mode": "prefill",
    },
    "csa_decode_small": {
        "name": "csa_decode_small",
        "model": "DeepSeek-V4",
        "operator": "compress_norm_rope_store",
        "case_type": "csa",
        "batch_size": 16,
        "hidden_size": 7168,
        "head_dim": 512,
        "rope_head_dim": 64,
        "compress_ratio": 4,
        "physical_page_size": 256,
        "quant_block": 128,
        "overlap": True,
        "mode": "decode",
        "history_len": 64,
    },
    "csa_prefill_bs512": {
        "name": "csa_prefill_bs512",
        "model": "DeepSeek-V4",
        "operator": "compress_norm_rope_store",
        "case_type": "csa",
        "batch_size": 512,
        "hidden_size": 7168,
        "head_dim": 512,
        "rope_head_dim": 64,
        "compress_ratio": 4,
        "physical_page_size": 256,
        "quant_block": 128,
        "overlap": True,
        "mode": "prefill",
    },
    "csa_decode_bs512": {
        "name": "csa_decode_bs512",
        "model": "DeepSeek-V4",
        "operator": "compress_norm_rope_store",
        "case_type": "csa",
        "batch_size": 512,
        "hidden_size": 7168,
        "head_dim": 512,
        "rope_head_dim": 64,
        "compress_ratio": 4,
        "physical_page_size": 256,
        "quant_block": 128,
        "overlap": True,
        "mode": "decode",
        "history_len": 1024,
    },
    "hca_prefill_bs512": {
        "name": "hca_prefill_bs512",
        "model": "DeepSeek-V4",
        "operator": "compress_norm_rope_store",
        "case_type": "hca",
        "batch_size": 512,
        "hidden_size": 7168,
        "head_dim": 512,
        "rope_head_dim": 64,
        "compress_ratio": 128,
        "physical_page_size": 128,
        "quant_block": 0,
        "overlap": False,
        "mode": "prefill",
    },
    "hca_decode_bs512": {
        "name": "hca_decode_bs512",
        "model": "DeepSeek-V4",
        "operator": "compress_norm_rope_store",
        "case_type": "hca",
        "batch_size": 512,
        "hidden_size": 7168,
        "head_dim": 512,
        "rope_head_dim": 64,
        "compress_ratio": 128,
        "physical_page_size": 128,
        "quant_block": 0,
        "overlap": False,
        "mode": "decode",
        "history_len": 1024,
    },
    "csa_indexer_prefill_bs512": {
        "name": "csa_indexer_prefill_bs512",
        "model": "DeepSeek-V4",
        "operator": "compress_norm_rope_store",
        "case_type": "csa_indexer",
        "batch_size": 512,
        "hidden_size": 7168,
        "head_dim": 128,
        "rope_head_dim": 64,
        "compress_ratio": 4,
        "physical_page_size": 32,
        "quant_block": 128,
        "overlap": True,
        "mode": "prefill",
    },
    "csa_indexer_decode_bs512": {
        "name": "csa_indexer_decode_bs512",
        "model": "DeepSeek-V4",
        "operator": "compress_norm_rope_store",
        "case_type": "csa_indexer",
        "batch_size": 512,
        "hidden_size": 7168,
        "head_dim": 128,
        "rope_head_dim": 64,
        "compress_ratio": 4,
        "physical_page_size": 32,
        "quant_block": 128,
        "overlap": True,
        "mode": "decode",
        "history_len": 1024,
    },
}
CONFIG = CONFIGS["csa_small"]


def create_inputs(dtype=jnp.bfloat16, config=None):
  """Returns (cache, positions, block_table, token_to_req_indices, kv_slot_mapping, rms_weight, cos_sin_cache, rope_cache)."""
  cfg = (
      CONFIG
      if config is None
      else (CONFIGS[config] if isinstance(config, str) else config)
  )

  case_type = cfg.get("case_type", "csa")
  batch_size = cfg.get("batch_size", 512)
  head_dim = cfg.get("head_dim", 512)
  compress_ratio = cfg.get("compress_ratio", 4)
  physical_page_size = cfg.get("physical_page_size", 256)
  rope_head_dim = cfg.get("rope_head_dim", 64)
  quant_block = cfg.get("quant_block", 128)
  mode = cfg.get("mode", "prefill")
  history_len = cfg.get("history_len", 1024)
  rms_eps = cfg.get("rms_eps", 1e-6)
  tile_n = 4

  if case_type == "hca":
    config_mode = Mode.HCA
  elif case_type == "csa":
    config_mode = Mode.CSA
  elif case_type == "csa_indexer":
    config_mode = Mode.CSA_INDEXER
  else:
    raise ValueError(f"Unknown case_type: {case_type}")

  cfgs = Configs.make(
      config_mode,
      size_n=batch_size,
      physical_page_size=physical_page_size,
      rms_eps=rms_eps,
      tile_n=tile_n,
      head_dim=head_dim,
      rope_head_dim=rope_head_dim,
      compress_ratio=compress_ratio,
      quant_block=quant_block,
  )

  tokens_per_page = cfgs.state_block_size
  rng = np.random.default_rng(42)

  if mode == "prefill":
    num_boundary_tokens = batch_size // compress_ratio
    assert (
        num_boundary_tokens % 4 == 0
    ), f"num_boundary_tokens ({num_boundary_tokens}) must be divisible by 4"
    max_pos = num_boundary_tokens * compress_ratio
    pages_for_state = max(1, (max_pos + tokens_per_page - 1) // tokens_per_page)

    total_rows_needed = num_boundary_tokens * cfgs.record_rows
    pages_for_kv_cache = max(
        1,
        (total_rows_needed + cfgs.physical_page_size - 1)
        // cfgs.physical_page_size,
    )
    num_pages = pages_for_state + pages_for_kv_cache

    block_table = jnp.arange(pages_for_state, dtype=jnp.int32).reshape(
        1, pages_for_state
    )

    boundary_positions = (
        np.arange(num_boundary_tokens) + 1
    ) * compress_ratio - 1
    positions_np = np.pad(
        boundary_positions,
        (0, batch_size - num_boundary_tokens),
        constant_values=0,
    )
    positions = jnp.array(positions_np, dtype=jnp.int32)
    token_to_req_indices = jnp.zeros((batch_size,), dtype=jnp.int32)

    kv_slot_mapping_filtered = generate_kv_slot_mapping(
        num_boundary_tokens,
        num_pages,
        cfgs.kv_block_size,
        cfgs.kv_stride,
    )
    kv_slot_mapping = jnp.pad(
        kv_slot_mapping_filtered,
        (0, batch_size - num_boundary_tokens),
        constant_values=-1,
    )

  elif mode == "decode":
    boundary_ratio = cfg.get("boundary_ratio", 1.0 / compress_ratio)
    num_boundary_tokens = int(batch_size * boundary_ratio)
    if boundary_ratio > 0 and num_boundary_tokens == 0:
      num_boundary_tokens = 1
    num_boundary_tokens = max(
        4, ((num_boundary_tokens + tile_n - 1) // tile_n) * tile_n
    )
    num_boundary_tokens = min(num_boundary_tokens, batch_size)

    max_pos = history_len
    pages_for_state = max(1, (max_pos + tokens_per_page - 1) // tokens_per_page)

    total_rows_needed = num_boundary_tokens * cfgs.record_rows
    pages_for_kv_cache = max(
        1,
        (total_rows_needed + cfgs.physical_page_size - 1)
        // cfgs.physical_page_size,
    )
    num_pages = batch_size * pages_for_state + pages_for_kv_cache

    block_table = jnp.arange(
        batch_size * pages_for_state, dtype=jnp.int32
    ).reshape(batch_size, pages_for_state)

    max_boundary_idx = max(1, history_len // compress_ratio)
    boundary_reqs = rng.choice(
        batch_size,
        size=num_boundary_tokens,
        replace=(batch_size < num_boundary_tokens),
    )

    kv_slot_mapping_filtered = generate_kv_slot_mapping(
        num_boundary_tokens,
        num_pages,
        cfgs.kv_block_size,
        cfgs.kv_stride,
    )

    positions_active = np.zeros((num_boundary_tokens,), dtype=np.int32)
    for idx in range(num_boundary_tokens):
      q = rng.integers(0, max_boundary_idx)
      positions_active[idx] = (q + 1) * compress_ratio - 1

    positions_np = np.pad(
        positions_active,
        (0, batch_size - num_boundary_tokens),
        constant_values=0,
    )
    token_to_req_indices_np = np.pad(
        boundary_reqs,
        (0, batch_size - num_boundary_tokens),
        constant_values=0,
    )
    kv_slot_mapping_np = np.pad(
        kv_slot_mapping_filtered,
        (0, batch_size - num_boundary_tokens),
        constant_values=-1,
    )

    positions = jnp.array(positions_np, dtype=jnp.int32)
    token_to_req_indices = jnp.array(token_to_req_indices_np, dtype=jnp.int32)
    kv_slot_mapping = jnp.array(kv_slot_mapping_np, dtype=jnp.int32)
  else:
    raise ValueError(f"Unknown mode: {mode}")

  key = jax.random.key(42)
  k1, k2, k3, k4 = jax.random.split(key, 4)
  rms_weight = jax.random.normal(k1, (head_dim,), dtype=jnp.float32)
  max_p = int(np.max(positions_np)) + 1 if len(positions_np) > 0 else batch_size
  cos_sin_len = max(max_p, batch_size)
  cos_sin_cache = jax.random.normal(
      k2, (cos_sin_len, rope_head_dim), dtype=jnp.float32
  )
  cache = jax.random.randint(k3, cfgs.cache_shape(num_pages), 1, 255, dtype=jnp.uint8)
  if cfgs.dims.has_rope_cache:
    rope_cache = jax.random.randint(k4, cfgs.rope_cache_shape(num_pages), 1, 255, dtype=jnp.uint8)
  else:
    rope_cache = None

  return (
      cache,
      positions,
      block_table,
      token_to_req_indices,
      kv_slot_mapping,
      rms_weight,
      cos_sin_cache,
      rope_cache,
  )


def get_inputs(dtype=jnp.bfloat16):
  """Returns list of inputs across all defined configurations."""
  return [
      (list(create_inputs(dtype=dtype, config=cfg)), [])
      for cfg in CONFIGS.values()
  ]


def workload(
    cache,
    positions,
    block_table,
    token_to_req_indices,
    kv_slot_mapping,
    rms_weight,
    cos_sin_cache,
    rope_cache=None,
):
  """Pallas TPU KV compressor execution."""
  head_dim = rms_weight.shape[0]
  rope_head_dim = cos_sin_cache.shape[1] if cos_sin_cache is not None else 0

  if head_dim == 128:
    case_type = "csa_indexer"
    compress_ratio = 4
    overlap = True
    quant_block = 128
    has_rope_cache = False
  elif rope_cache is None:
    case_type = "hca"
    compress_ratio = 128
    overlap = False
    quant_block = 0
    has_rope_cache = False
  else:
    case_type = "csa"
    compress_ratio = 4
    overlap = True
    quant_block = 128
    has_rope_cache = True

  if block_table.ndim == 2:
    block_table_stride = block_table.shape[1]
    flat_block_table = block_table.reshape(-1)
  else:
    block_table_stride = block_table.shape[0]
    flat_block_table = block_table

  out_cache, out_rope = compress_norm_rope_store(
      cache,
      positions,
      flat_block_table,
      token_to_req_indices,
      kv_slot_mapping,
      rms_weight,
      block_table_stride=block_table_stride,
      rope_cache=rope_cache if has_rope_cache else None,
      cos_sin_cache=cos_sin_cache,
      compress_ratio=compress_ratio,
      overlap=overlap,
      quant_block=quant_block,
      rms_eps=1e-6,
      name=f"compress_store_{case_type}",
  )
  return (out_cache, out_rope) if has_rope_cache else out_cache


def get_flops(config=None):
  """Estimate total FLOPs for the KV compressor."""
  cfg = (
      CONFIG
      if config is None
      else (CONFIGS[config] if isinstance(config, str) else config)
  )
  batch_size = cfg.get("batch_size", 512)
  compress_ratio = cfg.get("compress_ratio", 4)
  head_dim = cfg.get("head_dim", 512)
  rope_head_dim = cfg.get("rope_head_dim", 64)
  overlap = cfg.get("overlap", True)
  quant_block = cfg.get("quant_block", 128)

  num_boundary_tokens = max(1, batch_size // compress_ratio)
  window_size = (1 + int(overlap)) * compress_ratio

  flops_per_token = (
      2 * window_size * head_dim
      + 3 * head_dim
      + 6 * rope_head_dim
      + (3 * head_dim if quant_block > 0 else 0)
  )
  return int(num_boundary_tokens * flops_per_token)


def benchmark():
  """Local verification and benchmarking."""
  import time

  inputs = create_inputs()
  (
      cache,
      positions,
      block_table,
      token_to_req_indices,
      kv_slot_mapping,
      rms_weight,
      cos_sin_cache,
      rope_cache,
  ) = inputs

  print(f"Testing {CONFIG['name']} on {jax.devices()[0]}...")
  out = workload(
      cache,
      positions,
      block_table,
      token_to_req_indices,
      kv_slot_mapping,
      rms_weight,
      cos_sin_cache,
      rope_cache,
  )

  leaves = jax.tree.leaves(out)
  for i, leaf in enumerate(leaves):
    print(f"Output leaf {i}: shape={leaf.shape}, dtype={leaf.dtype}")

  # Warmup and time
  jit_fn = jax.jit(workload)
  jit_out = jit_fn(*inputs)
  jax.block_until_ready(jit_out)

  t0 = time.perf_counter()
  iters = 10
  for _ in range(iters):
    res = jit_fn(*inputs)
    jax.block_until_ready(res)
  t1 = time.perf_counter()
  latency_us = (t1 - t0) / iters * 1e6
  flops = get_flops()
  gflops = (flops / (latency_us * 1e-6)) / 1e9
  print(f"Latency: {latency_us:.1f} us, FLOPs: {flops}, GFLOPS: {gflops:.2f}")


if __name__ == "__main__":
  benchmark()

