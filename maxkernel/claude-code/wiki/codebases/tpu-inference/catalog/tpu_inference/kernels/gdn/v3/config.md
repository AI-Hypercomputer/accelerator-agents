---
title: 'Module: tpu_inference/kernels/gdn/v3/config.py'
type: catalog
provenance: extracted
module: tpu_inference/kernels/gdn/v3/config.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.kernels.gdn.v3.config`/
symbols:
  GDNMode: GDNMode#
  GDNMode.BATCHED: GDNMode#BATCHED.
  GDNMode.PER_SEQ: GDNMode#PER_SEQ.
  GDNMode.get_seq_tile_size: GDNMode#get_seq_tile_size().
  GDNMode.get_chunk_size: GDNMode#get_chunk_size().
  Dtypes: Dtypes#
  Dtypes.act_in: Dtypes#act_in.
  Dtypes.act_out: Dtypes#act_out.
  Dtypes.compute: Dtypes#compute.
  Dtypes.recurrent_state: Dtypes#recurrent_state.
  Dtypes.conv_state: Dtypes#conv_state.
  GDNConfig: GDNConfig#
  GDNConfig.mode: GDNConfig#mode.
  GDNConfig.dtypes: GDNConfig#dtypes.
  GDNConfig.batch_size: GDNConfig#batch_size.
  GDNConfig.dim_size: GDNConfig#dim_size.
  GDNConfig.kernel_size: GDNConfig#kernel_size.
  GDNConfig.tile_size: GDNConfig#tile_size.
  GDNConfig.num_kq_heads: GDNConfig#num_kq_heads.
  GDNConfig.num_v_heads: GDNConfig#num_v_heads.
  GDNConfig.kq_head_dim: GDNConfig#kq_head_dim.
  GDNConfig.v_head_dim: GDNConfig#v_head_dim.
  GDNConfig.num_buffers: GDNConfig#num_buffers.
  GDNConfig.chunk_size: GDNConfig#chunk_size().
  GDNConfig.seq_tile_size: GDNConfig#seq_tile_size().
  GDNConfig.prev_kernel_size: GDNConfig#prev_kernel_size().
  GDNConfig.v_dim_size: GDNConfig#v_dim_size().
  GDNConfig.kq_dim_size: GDNConfig#kq_dim_size().
  GDNConfig.v_per_kq_head: GDNConfig#v_per_kq_head().
  GDNConfig.aligned_num_v_heads: GDNConfig#aligned_num_v_heads().
  GDNConfig.get_kernel_name: GDNConfig#get_kernel_name().
  GDNConfig.get_metadata: GDNConfig#get_metadata().
  GDNConfig.get_out_shape: GDNConfig#get_out_shape().
  GDNConfig.get_vmem_limit_bytes: GDNConfig#get_vmem_limit_bytes().
  GDNConfig.get_scratch_shape_dict: GDNConfig#get_scratch_shape_dict().
---
# Module: [`tpu_inference/kernels/gdn/v3/config.py`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v3/config.py)

## Classes
### `Dtypes`
- def: [`tpu_inference/kernels/gdn/v3/config.py:42`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v3/config.py#L42)
- signature: `class Dtypes:`
- members:
  - `act_in` — [`L43`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v3/config.py#L43)
  - `act_out` — [`L44`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v3/config.py#L44)
  - `compute` — [`L45`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v3/config.py#L45)
  - `conv_state` — [`L47`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v3/config.py#L47)
  - `recurrent_state` — [`L46`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v3/config.py#L46)

### `GDNConfig`
- def: [`tpu_inference/kernels/gdn/v3/config.py:52`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v3/config.py#L52)
- signature: `class GDNConfig:`
- members:
  - `aligned_num_v_heads(self)` — [`L90`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v3/config.py#L90)
  - `chunk_size(self)` — [`L66`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v3/config.py#L66)
  - `get_kernel_name(self)` — [`L95`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v3/config.py#L95)
  - `get_metadata(self)` — [`L99`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v3/config.py#L99)
  - `get_out_shape(self)` — [`L109`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v3/config.py#L109)
  - `get_scratch_shape_dict(self)` — [`L119`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v3/config.py#L119)
  - `get_vmem_limit_bytes(self)` — [`L115`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v3/config.py#L115)
  - `kq_dim_size(self)` — [`L82`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v3/config.py#L82)
  - `prev_kernel_size(self)` — [`L74`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v3/config.py#L74)
  - `seq_tile_size(self)` — [`L70`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v3/config.py#L70)
  - `v_dim_size(self)` — [`L78`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v3/config.py#L78)
  - `v_per_kq_head(self)` — [`L86`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v3/config.py#L86)
  - `batch_size` — [`L55`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v3/config.py#L55)
  - `dim_size` — [`L56`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v3/config.py#L56)
  - `dtypes` — [`L54`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v3/config.py#L54)
  - `kernel_size` — [`L57`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v3/config.py#L57)
  - `kq_head_dim` — [`L61`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v3/config.py#L61)
  - `mode` — [`L53`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v3/config.py#L53)
  - `num_buffers` — [`L63`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v3/config.py#L63)
  - `num_kq_heads` — [`L59`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v3/config.py#L59)
  - `num_v_heads` — [`L60`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v3/config.py#L60)
  - `tile_size` — [`L58`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v3/config.py#L58)
  - `v_head_dim` — [`L62`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v3/config.py#L62)

### `GDNMode`
- def: [`tpu_inference/kernels/gdn/v3/config.py:25`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v3/config.py#L25)
- signature: `class GDNMode(enum.StrEnum):`
- members:
  - `get_chunk_size(self, tile_size: int)` — [`L34`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v3/config.py#L34)
  - `get_seq_tile_size(self, tile_size: int)` — [`L29`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v3/config.py#L29)
  - `BATCHED` — [`L26`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v3/config.py#L26)
  - `PER_SEQ` — [`L27`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v3/config.py#L27)

