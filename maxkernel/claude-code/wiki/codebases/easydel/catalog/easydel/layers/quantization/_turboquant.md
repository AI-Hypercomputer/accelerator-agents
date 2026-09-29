---
title: 'Module: easydel/layers/quantization/_turboquant.py'
type: catalog
provenance: extracted
module: easydel/layers/quantization/_turboquant.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.layers.quantization._turboquant`/
symbols:
  TurboQuantConstants.generate: TurboQuantConstants#generate().
  TurboQuantConstants.replicate: TurboQuantConstants#replicate().
  TurboQuantConfig: TurboQuantConfig#
  TurboQuantConstants: TurboQuantConstants#
  TurboQuantConfig.qjl_dim: TurboQuantConfig#qjl_dim.
  TurboQuantConfig.bits: TurboQuantConfig#bits.
  TurboQuantConstants.key_codebook: TurboQuantConstants#key_codebook.
  TurboQuantConstants.value_codebook: TurboQuantConstants#value_codebook.
  TurboQuantConstants.rotation_matrix: TurboQuantConstants#rotation_matrix.
  TurboQuantConstants.qjl_projection: TurboQuantConstants#qjl_projection.
  TurboQuantConstants.qjl_dim: TurboQuantConstants#qjl_dim.
  TurboQuantConfig.key_codebook_bits: TurboQuantConfig#key_codebook_bits().
  TurboQuantConfig.value_codebook_bits: TurboQuantConfig#value_codebook_bits().
  TurboQuantConstants.bits: TurboQuantConstants#bits.
  TurboQuantConfig.__post_init__: TurboQuantConfig#__post_init__().
  TurboQuantConfig.key_codebook_size: TurboQuantConfig#key_codebook_size().
  TurboQuantConfig.value_codebook_size: TurboQuantConfig#value_codebook_size().
  _solve_lloyd_max_cached: _solve_lloyd_max_cached().
  TurboQuantConfig.seed: TurboQuantConfig#seed.
---
# Module: [`easydel/layers/quantization/_turboquant.py`](../../../../../../../raw/code/EasyDeL/easydel/layers/quantization/_turboquant.py)

## Classes
### `TurboQuantConfig`
- def: [`easydel/layers/quantization/_turboquant.py:48`](../../../../../../../raw/code/EasyDeL/easydel/layers/quantization/_turboquant.py#L48)
- doc: Configuration for TurboQuant KV cache quantization.
- signature: `class TurboQuantConfig:`
- members:
  - `key_codebook_bits(self)` — [`L81`](../../../../../../../raw/code/EasyDeL/easydel/layers/quantization/_turboquant.py#L81) — Bits used for key codebook indices (bits - 1 for QJL sign).
  - `key_codebook_size(self)` — [`L91`](../../../../../../../raw/code/EasyDeL/easydel/layers/quantization/_turboquant.py#L91) — Number of Lloyd-Max centroids for keys.
  - `value_codebook_bits(self)` — [`L86`](../../../../../../../raw/code/EasyDeL/easydel/layers/quantization/_turboquant.py#L86) — Bits used for value codebook indices (all bits, no QJL).
  - `value_codebook_size(self)` — [`L96`](../../../../../../../raw/code/EasyDeL/easydel/layers/quantization/_turboquant.py#L96) — Number of Lloyd-Max centroids for values.
  - `bits` — [`L72`](../../../../../../../raw/code/EasyDeL/easydel/layers/quantization/_turboquant.py#L72)
  - `qjl_dim` — [`L73`](../../../../../../../raw/code/EasyDeL/easydel/layers/quantization/_turboquant.py#L73)
  - `seed` — [`L74`](../../../../../../../raw/code/EasyDeL/easydel/layers/quantization/_turboquant.py#L74)
- protocol/private: `__post_init__`[`L76`](../../../../../../../raw/code/EasyDeL/easydel/layers/quantization/_turboquant.py#L76)
- used by: [`_import_structure`](../../__init__.md#_import_structure), [`init_operations_cache_config`](../../infra/mixins/generation.md#EasyGenerationMixin.init_operations_cache_config), [`init_ragged_pages`](../../infra/mixins/generation.md#EasyGenerationMixin.init_ragged_pages), [`generate`](_turboquant.md#TurboQuantConstants.generate), [`init`](../../caching/turboquant_ragged_page/cache.md#TurboQuantRaggedPagesCacheView.init), [`init_all_layers`](../../caching/turboquant_ragged_page/cache.md#TurboQuantRaggedPagesCacheView.init_all_layers), [`initialize_kv_cache`](../../inference/esurge/runners/model_runner.md#eSurgeRunner.initialize_kv_cache), [`create`](../../caching/turboquant_ragged_page/cache.md#TurboQuantRaggedPagesCacheConfig.create), [`quantizer`](../attention/_flexible.md#AttentionModule.quantizer), [`_use_request_distribution`](../../inference/esurge/runners/execution_manager.md#ExecutionManager._use_request_distribution), [`turboquant_config`](../../caching/turboquant_ragged_page/cache.md#TurboQuantRaggedPagesCacheConfig.turboquant_config)

### `TurboQuantConstants`
- def: [`easydel/layers/quantization/_turboquant.py:102`](../../../../../../../raw/code/EasyDeL/easydel/layers/quantization/_turboquant.py#L102)
- doc: Precomputed constants for TurboQuant, generated once at cache init.
- signature: `class TurboQuantConstants:`
- members:
  - `generate(config: TurboQuantConfig, head_dim: int, layer_index: int = 0, mesh: jax.sharding.Mesh | None = None)` — [`L143`](../../../../../../../raw/code/EasyDeL/easydel/layers/quantization/_turboquant.py#L143) — Generate all precomputed constants from config.
  - `replicate(self, mesh: jax.sharding.Mesh)` — [`L125`](../../../../../../../raw/code/EasyDeL/easydel/layers/quantization/_turboquant.py#L125) — Return a copy with all arrays replicated across the mesh.
  - `bits` — [`L123`](../../../../../../../raw/code/EasyDeL/easydel/layers/quantization/_turboquant.py#L123)
  - `key_codebook` — [`L118`](../../../../../../../raw/code/EasyDeL/easydel/layers/quantization/_turboquant.py#L118)
  - `qjl_dim` — [`L122`](../../../../../../../raw/code/EasyDeL/easydel/layers/quantization/_turboquant.py#L122)
  - `qjl_projection` — [`L121`](../../../../../../../raw/code/EasyDeL/easydel/layers/quantization/_turboquant.py#L121)
  - `rotation_matrix` — [`L120`](../../../../../../../raw/code/EasyDeL/easydel/layers/quantization/_turboquant.py#L120)
  - `value_codebook` — [`L119`](../../../../../../../raw/code/EasyDeL/easydel/layers/quantization/_turboquant.py#L119)
- uses (calls/refs, reference-scoped): [`TurboQuantConfig`](_turboquant.md#TurboQuantConfig), [`qjl_dim`](_turboquant.md#TurboQuantConfig.qjl_dim), [`bits`](_turboquant.md#TurboQuantConfig.bits), [`key_codebook_bits`](_turboquant.md#TurboQuantConfig.key_codebook_bits), [`value_codebook_bits`](_turboquant.md#TurboQuantConfig.value_codebook_bits), [`_solve_lloyd_max_cached`](_turboquant.md#_solve_lloyd_max_cached), [`seed`](_turboquant.md#TurboQuantConfig.seed)
- used by: [`_forward_v2_turboquant`](../../operations/kernels/ragged_page_attention.md#_RaggedPageAttn._forward_v2_turboquant), [`_forward_v3_turboquant`](../../operations/kernels/ragged_page_attention.md#_RaggedPageAttn._forward_v3_turboquant), [`init`](../../caching/turboquant_ragged_page/cache.md#TurboQuantRaggedPagesCacheView.init), [`init_all_layers`](../../caching/turboquant_ragged_page/cache.md#TurboQuantRaggedPagesCacheView.init_all_layers), [`constants`](../../caching/turboquant_ragged_page/cache.md#TurboQuantRaggedPagesCacheView.constants)

## Functions
- `_solve_lloyd_max_cached(bits: int, dim: int)` — [`L40`](../../../../../../../raw/code/EasyDeL/easydel/layers/quantization/_turboquant.py#L40) — Cached Lloyd-Max solver — codebooks depend only on (bits, dim).

