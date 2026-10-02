---
title: 'Module: tokamax/_src/ops/experimental/mla/pallas_mosaic_tpu.py'
type: catalog
provenance: extracted
module: tokamax/_src/ops/experimental/mla/pallas_mosaic_tpu.py
status: fresh
symbol_base: scip-python python tokamax 0.0.0 `tokamax._src.ops.experimental.mla.pallas_mosaic_tpu`/
symbols:
  PallasTpuMultiHeadLatentAttention._fwd: PallasTpuMultiHeadLatentAttention#_fwd().
  PallasTpuMultiHeadLatentAttention._get_heuristics_config: PallasTpuMultiHeadLatentAttention#_get_heuristics_config().
  PallasTpuMultiHeadLatentAttention._get_autotuning_configs: PallasTpuMultiHeadLatentAttention#_get_autotuning_configs().
  Config: Config#
  PallasTpuMultiHeadLatentAttention.config_cls: PallasTpuMultiHeadLatentAttention#config_cls.
  PallasTpuMultiHeadLatentAttention: PallasTpuMultiHeadLatentAttention#
  Config.chunk_prefill_size: Config#chunk_prefill_size.
  Config.num_kv_pages_per_block: Config#num_kv_pages_per_block.
  Config.num_queries_per_block: Config#num_queries_per_block.
  Config.vmem_limit_bytes: Config#vmem_limit_bytes.
  Config.decode_batch_size: Config#decode_batch_size.
  PallasTpuMultiHeadLatentAttention.supported_on: PallasTpuMultiHeadLatentAttention#supported_on().
---
# Module: [`tokamax/_src/ops/experimental/mla/pallas_mosaic_tpu.py`](../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/mla/pallas_mosaic_tpu.py)

## Classes
### `Config`
- def: [`tokamax/_src/ops/experimental/mla/pallas_mosaic_tpu.py:33`](../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/mla/pallas_mosaic_tpu.py#L33)
- signature: `class Config:`
- members:
  - `chunk_prefill_size` — [`L37`](../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/mla/pallas_mosaic_tpu.py#L37)
  - `decode_batch_size` — [`L38`](../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/mla/pallas_mosaic_tpu.py#L38)
  - `num_kv_pages_per_block` — [`L34`](../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/mla/pallas_mosaic_tpu.py#L34)
  - `num_queries_per_block` — [`L35`](../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/mla/pallas_mosaic_tpu.py#L35)
  - `vmem_limit_bytes` — [`L36`](../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/mla/pallas_mosaic_tpu.py#L36)
- used by: [`_fwd`](pallas_mosaic_tpu.md#PallasTpuMultiHeadLatentAttention._fwd), [`_get_autotuning_configs`](pallas_mosaic_tpu.md#PallasTpuMultiHeadLatentAttention._get_autotuning_configs), [`_get_heuristics_config`](pallas_mosaic_tpu.md#PallasTpuMultiHeadLatentAttention._get_heuristics_config), [`config_cls`](pallas_mosaic_tpu.md#PallasTpuMultiHeadLatentAttention.config_cls)

### `PallasTpuMultiHeadLatentAttention`  ·  implements/extends MultiHeadLatentAttention
- def: [`tokamax/_src/ops/experimental/mla/pallas_mosaic_tpu.py:41`](../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/mla/pallas_mosaic_tpu.py#L41)
- doc: Tokamax operator that invokes the Pallas kernel for Multi-Head Latent Attention.
- signature: `class PallasTpuMultiHeadLatentAttention(base.MultiHeadLatentAttention):`
- members:
  - `supported_on(self, device)` — [`L143`](../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/mla/pallas_mosaic_tpu.py#L143)
  - `config_cls` — [`L44`](../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/mla/pallas_mosaic_tpu.py#L44)
- protocol/private: `_fwd`[`L48`](../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/mla/pallas_mosaic_tpu.py#L48), `_get_autotuning_configs`[`L123`](../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/mla/pallas_mosaic_tpu.py#L123), `_get_heuristics_config`[`L113`](../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/mla/pallas_mosaic_tpu.py#L113)
- uses (calls/refs, reference-scoped): [`jaxtyped`](../../../jaxtyping.md#jaxtyped), [`mla_ragged_paged_attention`](pallas_mosaic_tpu_kernel.md#mla_ragged_paged_attention), [`MultiHeadLatentAttention`](base.md#MultiHeadLatentAttention), [`Config`](pallas_mosaic_tpu.md#Config), [`chunk_prefill_size`](pallas_mosaic_tpu.md#Config.chunk_prefill_size), [`decode_batch_size`](pallas_mosaic_tpu.md#Config.decode_batch_size), [`num_kv_pages_per_block`](pallas_mosaic_tpu.md#Config.num_kv_pages_per_block), [`num_queries_per_block`](pallas_mosaic_tpu.md#Config.num_queries_per_block), [`vmem_limit_bytes`](pallas_mosaic_tpu.md#Config.vmem_limit_bytes)
- used by: [`_get_heuristics_config`](../../op.md#Op._get_heuristics_config), [`_get_autotuning_configs`](../../op.md#Op._get_autotuning_configs), [`supported_on`](../../op.md#Op.supported_on), [`MultiHeadLatentAttention`](base.md#MultiHeadLatentAttention), [`_fwd`](base.md#MultiHeadLatentAttention._fwd), [`IMPLEMENTATIONS`](api.md#IMPLEMENTATIONS.IMPLEMENTATIONS), [`PallasTpuMhla`](pallas_mosaic_tpu_test.md#PallasTpuMhla)

