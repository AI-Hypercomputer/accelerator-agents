---
title: 'Module: python/sgl_jax/srt/lora/lora_memory_pool.py'
type: catalog
provenance: extracted
module: python/sgl_jax/srt/lora/lora_memory_pool.py
status: fresh
symbol_base: scip-python python sglang-jax 0.0.0 `python.sgl_jax.srt.lora.lora_memory_pool`/
symbols:
  LoRAMemoryPool.tree_flatten: LoRAMemoryPool#tree_flatten().
  LoRAMemoryPool.tree_unflatten: LoRAMemoryPool#tree_unflatten().
  LoRAMemoryPool.load_lora_weight_to_buffer: LoRAMemoryPool#load_lora_weight_to_buffer().
  LoRAMemoryPool.init_buffers: LoRAMemoryPool#init_buffers().
  LoRAMemoryPool._get_lora_b_shape: LoRAMemoryPool#_get_lora_b_shape().
  LoRAMemoryPool._replicate_kv_lora_weight: LoRAMemoryPool#_replicate_kv_lora_weight().
  logger: logger.
  LoRAMemoryPool._get_lora_a_shape: LoRAMemoryPool#_get_lora_a_shape().
  LoRAMemoryPool.max_lora_rank: LoRAMemoryPool#max_lora_rank.
  LoRAMemoryPool.prepare_lora_batch: LoRAMemoryPool#prepare_lora_batch().
  LoRAMemoryPool.A_buffer: LoRAMemoryPool#A_buffer.
  LoRAMemoryPool.B_buffer: LoRAMemoryPool#B_buffer.
  LoRAMemoryPool.can_support: LoRAMemoryPool#can_support().
  LoRAMemoryPool.num_kv_replicas: LoRAMemoryPool#num_kv_replicas.
  LoRAMemoryPool.num_layers: LoRAMemoryPool#num_layers.
  LoRAMemoryPool.target_modules: LoRAMemoryPool#target_modules.
  LoRAMemoryPool.dtype: LoRAMemoryPool#dtype.
  LoRAMemoryPool.head_dim: LoRAMemoryPool#head_dim.
  LoRAMemoryPool.original_num_kv_heads: LoRAMemoryPool#original_num_kv_heads.
  LoRAMemoryPool.hidden_size: LoRAMemoryPool#hidden_size.
  LoRAMemoryPool.buffer_id_to_uid: LoRAMemoryPool#buffer_id_to_uid.
  LoRAMemoryPool.get_available_buffer_slot: LoRAMemoryPool#get_available_buffer_slot().
  LoRAMemoryPool._extract_module_weights: LoRAMemoryPool#_extract_module_weights().
  LoRAMemoryPool._extract_and_concat_qkv: LoRAMemoryPool#_extract_and_concat_qkv().
  LoRAMemoryPool.get_ab_zero_matrix_shape: LoRAMemoryPool#get_ab_zero_matrix_shape().
  LoRAMemoryPool.max_loras_per_batch: LoRAMemoryPool#max_loras_per_batch.
  LoRAMemoryPool._handle_rank_mismatch: LoRAMemoryPool#_handle_rank_mismatch().
  LoRAMemoryPool.get_array: LoRAMemoryPool#get_array().
  LoRAMemoryPool.mesh: LoRAMemoryPool#mesh.
  LoRAMemoryPool.num_attention_heads: LoRAMemoryPool#num_attention_heads.
  LoRAMemoryPool.uid_to_buffer_id: LoRAMemoryPool#uid_to_buffer_id.
  LoRAMemoryPool._get_lora_a_sharding: LoRAMemoryPool#_get_lora_a_sharding().
  LoRAMemoryPool._get_lora_b_sharding: LoRAMemoryPool#_get_lora_b_sharding().
  LoRAMemoryPool._extract_single_module: LoRAMemoryPool#_extract_single_module().
  LoRAMemoryPool._extract_and_concat_gate_up: LoRAMemoryPool#_extract_and_concat_gate_up().
  LoRAMemoryPool.intermediate_size: LoRAMemoryPool#intermediate_size.
  EMPTY_SLOT: EMPTY_SLOT.
  LoRAMemoryPool.tp_size: LoRAMemoryPool#tp_size.
  LoRAMemoryPool.get_buffer_id: LoRAMemoryPool#get_buffer_id().
  LoRAMemoryPool.num_kv_heads: LoRAMemoryPool#num_kv_heads.
  LoRAMemoryPool.needs_kv_replication: LoRAMemoryPool#needs_kv_replication.
  EmptySlot.__new__: EmptySlot#__new__().
  EmptySlot: EmptySlot#
  LoRAMemoryPool: LoRAMemoryPool#
  EmptySlot._instance: EmptySlot#_instance.
  EmptySlot.__slots__: EmptySlot#__slots__.
  EmptySlot.__repr__: EmptySlot#__repr__().
  LoRAMemoryPool.__init__: LoRAMemoryPool#__init__().
---
# Module: [`python/sgl_jax/srt/lora/lora_memory_pool.py`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/lora_memory_pool.py)

## Classes
### `EmptySlot`
- def: [`python/sgl_jax/srt/lora/lora_memory_pool.py:36`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/lora_memory_pool.py#L36)
- doc: Singleton class to represent an empty slot in the memory pool.
- signature: `class EmptySlot:`
- protocol/private: `__new__`[`L47`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/lora_memory_pool.py#L47), `__repr__`[`L44`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/lora_memory_pool.py#L44), `__slots__`[`L42`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/lora_memory_pool.py#L42), `_instance`[`L49`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/lora_memory_pool.py#L49)
- used by: [`buffer_id_to_uid`](lora_memory_pool.md#LoRAMemoryPool.buffer_id_to_uid), [`EMPTY_SLOT`](lora_memory_pool.md#EMPTY_SLOT)

### `LoRAMemoryPool`
- def: [`python/sgl_jax/srt/lora/lora_memory_pool.py:57`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/lora_memory_pool.py#L57)
- doc: JAX-based memory pool for LoRA adapters.
- signature: `class LoRAMemoryPool:`
- members:
  - `__init__(self, max_loras_per_batch: int, max_lora_rank: int, num_layers: int, target_modules: set[str], mesh: Mesh, dtype: jnp.dtype = jnp.float16, hidden_size: int = 4096, intermediate_size: int = 11008, num_attention_heads: int = 32, num_kv_heads: int = 32, head_dim: int | None = None, original_num_kv_heads: int | None = None, tp_size: int = 1)` — [`L78`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/lora_memory_pool.py#L78) — Initialize LoRA memory pool.
  - `_extract_and_concat_gate_up(self, layer_weights: dict[str, jax.Array], layer_id: int)` — [`L662`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/lora_memory_pool.py#L662) — Extract and concatenate gate_proj, up_proj weights.
  - `_extract_and_concat_qkv(self, layer_weights: dict[str, jax.Array], layer_id: int)` — [`L609`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/lora_memory_pool.py#L609) — Extract and concatenate q_proj, k_proj, v_proj weights.
  - `_extract_module_weights(self, layer_weights: dict[str, jax.Array], layer_id: int, module_name: str)` — [`L550`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/lora_memory_pool.py#L550) — Extract LoRA A and B weights for a specific module from layer weights.
  - `_extract_single_module(self, layer_weights: dict[str, jax.Array], layer_id: int, module_name: str)` — [`L588`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/lora_memory_pool.py#L588) — Extract weights for a single module (o_proj, down_proj).
  - `_get_lora_a_shape(self, module_name: str)` — [`L232`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/lora_memory_pool.py#L232) — Get shape for LoRA A matrix.
  - `_get_lora_b_shape(self, module_name: str)` — [`L259`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/lora_memory_pool.py#L259) — Get shape for LoRA B matrix.
  - `_handle_rank_mismatch(self, weight: jax.Array, is_lora_a: bool)` — [`L700`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/lora_memory_pool.py#L700) — Handle rank mismatch between adapter and buffer.
  - `_replicate_kv_lora_weight(self, weight: jax.Array, is_lora_a: bool, module_name: str)` — [`L764`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/lora_memory_pool.py#L764) — Replicate LoRA weights for k_proj/v_proj when KV head replication is needed.
  - `can_support(self, config: LoRAConfig)` — [`L224`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/lora_memory_pool.py#L224) — Check if the memory pool can support the given LoRA config.
  - `get_ab_zero_matrix_shape(module_name: str)` — [`L424`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/lora_memory_pool.py#L424)
  - `get_array(self, module_name: str, layer_id: int, is_lora_a: bool)` — [`L857`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/lora_memory_pool.py#L857) — Get LoRA array for a specific module and layer.
  - `get_available_buffer_slot(uid: str)` — [`L377`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/lora_memory_pool.py#L377) — Find next available buffer slot (simple incremental allocation).
  - `get_buffer_id(self, lora_uid: str | None)` — [`L853`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/lora_memory_pool.py#L853) — Get buffer slot ID for a given LoRA adapter ID.
  - `init_buffers(self)` — [`L302`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/lora_memory_pool.py#L302) — Initialize GPU buffers for LoRA weights.
  - `load_lora_weight_to_buffer(self, uid: str | None, buffer_id: int, lora_adapter: LoRAAdapter | None)` — [`L409`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/lora_memory_pool.py#L409) — Load LoRA weights into buffer slot.
  - `prepare_lora_batch(self, cur_uids: set[str | None], lora_adapters: dict[str | None, LoRAAdapter])` — [`L356`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/lora_memory_pool.py#L356) — Prepare LoRA batch by loading adapters into buffer slots.
  - `tree_flatten(self)` — [`L149`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/lora_memory_pool.py#L149) — Flatten for pytree registration - only JAX arrays are children.
  - `tree_unflatten(cls, aux_data, children)` — [`L184`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/lora_memory_pool.py#L184) — Unflatten from pytree.
  - `A_buffer` — [`L146`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/lora_memory_pool.py#L146)
  - `B_buffer` — [`L147`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/lora_memory_pool.py#L147)
  - `buffer_id_to_uid` — [`L143`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/lora_memory_pool.py#L143)
  - `dtype` — [`L117`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/lora_memory_pool.py#L117)
  - `head_dim` — [`L122`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/lora_memory_pool.py#L122)
  - `hidden_size` — [`L118`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/lora_memory_pool.py#L118)
  - `intermediate_size` — [`L119`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/lora_memory_pool.py#L119)
  - `max_lora_rank` — [`L113`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/lora_memory_pool.py#L113)
  - `max_loras_per_batch` — [`L112`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/lora_memory_pool.py#L112)
  - `mesh` — [`L116`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/lora_memory_pool.py#L116)
  - `needs_kv_replication` — [`L132`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/lora_memory_pool.py#L132)
  - `num_attention_heads` — [`L120`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/lora_memory_pool.py#L120)
  - `num_kv_heads` — [`L121`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/lora_memory_pool.py#L121)
  - `num_kv_replicas` — [`L133`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/lora_memory_pool.py#L133)
  - `num_layers` — [`L114`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/lora_memory_pool.py#L114)
  - `original_num_kv_heads` — [`L125`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/lora_memory_pool.py#L125)
  - `target_modules` — [`L115`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/lora_memory_pool.py#L115)
  - `tp_size` — [`L128`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/lora_memory_pool.py#L128)
  - `uid_to_buffer_id` — [`L142`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/lora_memory_pool.py#L142)
- protocol/private: `_get_lora_a_sharding`[`L296`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/lora_memory_pool.py#L296), `_get_lora_b_sharding`[`L299`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/lora_memory_pool.py#L299)
- uses (calls/refs, reference-scoped): [`logger`](lora_memory_pool.md#logger), [`LoRAConfig`](lora_config.md#LoRAConfig), [`target_modules`](lora_config.md#LoRAConfig.target_modules), [`layers`](lora.md#LoRAAdapter.layers), [`r`](lora_config.md#LoRAConfig.r), [`LoRAAdapter`](lora.md#LoRAAdapter), [`EMPTY_SLOT`](lora_memory_pool.md#EMPTY_SLOT), [`weights`](lora.md#LoRALayer.weights), [`EmptySlot`](lora_memory_pool.md#EmptySlot), [`get_lora_a_sharding`](utils.md#get_lora_a_sharding), [`get_lora_b_sharding`](utils.md#get_lora_b_sharding)
- used by: [`memory_pool`](lora_manager.md#LoRAManager.memory_pool), [`prepare_dynamic_lora_batch`](lora_manager.md#LoRAManager.prepare_dynamic_lora_batch), [`update_lora_info`](lora_manager.md#LoRAManager.update_lora_info), [`can_support`](lora_manager.md#LoRAManager.can_support), [`get_buffer_id`](lora_manager.md#LoRAManager.get_buffer_id)

## Module values
- `EMPTY_SLOT` — [`L53`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/lora_memory_pool.py#L53)
- `logger` — [`L33`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/lora_memory_pool.py#L33)

