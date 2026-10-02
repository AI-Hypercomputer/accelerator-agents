---
title: 'Module: jax/_src/tpu_custom_call.py'
type: catalog
provenance: extracted
module: jax/_src/tpu_custom_call.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax._src.tpu_custom_call`/
symbols:
  CustomCallBackendConfig.to_json: CustomCallBackendConfig#to_json().
  _tpu_custom_call_lowering: _tpu_custom_call_lowering().
  MemorySpace: MemorySpace#
  _lowered_to_custom_call_config: _lowered_to_custom_call_config().
  _lower_to_custom_call_config: _lower_to_custom_call_config().
  lower_module_to_custom_call: lower_module_to_custom_call().
  lowered_as_tpu_kernel: lowered_as_tpu_kernel().
  MemorySpace.color: MemorySpace#color().
  as_tpu_kernel: as_tpu_kernel().
  TpuSideEffectType: TpuSideEffectType#
  _as_jax_callable: _as_jax_callable().
  tpu_custom_call_p: tpu_custom_call_p.
  Tiling: Tiling#
  CustomCallBackendConfig.downgrade_lowered_module_asm: CustomCallBackendConfig#downgrade_lowered_module_asm().
  OptLevel: OptLevel#
  CustomCallBackendConfig.__post_init__: CustomCallBackendConfig#__post_init__().
  get_ir_version: get_ir_version().
  CostEstimate: CostEstimate#
  CustomCallBackendConfig.cost_estimate: CustomCallBackendConfig#cost_estimate.
  tpu_custom_call_batcher: tpu_custom_call_batcher().
  _as_jax_callable.apply_kernel: _as_jax_callable().apply_kernel().
  CustomCallBackendConfig.output_memory_spaces: CustomCallBackendConfig#output_memory_spaces.
  CustomCallBackendConfig.input_memory_spaces: CustomCallBackendConfig#input_memory_spaces.
  CustomCallBackendConfig.tiling: CustomCallBackendConfig#tiling.
  CustomCallBackendConfig.opt_level: CustomCallBackendConfig#opt_level.
  MemorySpace.VMEM: MemorySpace#VMEM.
  MemorySpace.SMEM: MemorySpace#SMEM.
  TpuSideEffectType.PURE: TpuSideEffectType#PURE.
  CustomCallBackendConfig: CustomCallBackendConfig#
  CustomCallBackendConfig.allow_input_fusion: CustomCallBackendConfig#allow_input_fusion.
  _tpu_custom_call_abstract_eval: _tpu_custom_call_abstract_eval().
  _get_device_type: _get_device_type().
  MemorySpace.HBM: MemorySpace#HBM.
  CustomCallBackendConfig.device_type: CustomCallBackendConfig#device_type.
  FLAGS: FLAGS.
  register_extra_dialect: register_extra_dialect().
  _get_active_core_count: _get_active_core_count().
  MemorySpace.SEMAPHORE_MEM: MemorySpace#SEMAPHORE_MEM.
  MemorySpace.SC_SCALAR_SEMAPHORE_MEM: MemorySpace#SC_SCALAR_SEMAPHORE_MEM.
  TpuSideEffectType.DATAFLOW_SIDE_EFFECTING: TpuSideEffectType#DATAFLOW_SIDE_EFFECTING.
  CustomCallBackendConfig.lowered_module_asm_version: CustomCallBackendConfig#lowered_module_asm_version.
  CustomCallBackendConfig.has_communication: CustomCallBackendConfig#has_communication.
  CustomCallBackendConfig.flags: CustomCallBackendConfig#flags.
  CustomCallBackendConfig.skip_device_barrier: CustomCallBackendConfig#skip_device_barrier.
  CustomCallBackendConfig.shape_invariant_numerics: CustomCallBackendConfig#shape_invariant_numerics.
  _compact_json_object: _compact_json_object().
  MemorySpace.HOST: MemorySpace#HOST.
  MemorySpace.SC_VECTOR_SEMAPHORE_MEM: MemorySpace#SC_VECTOR_SEMAPHORE_MEM.
  TpuSideEffectType.SIDE_EFFECTING: TpuSideEffectType#SIDE_EFFECTING.
  Tiling.COMPACT: Tiling#COMPACT.
  CustomCallBackendConfig.lowered_module_asm: CustomCallBackendConfig#lowered_module_asm.
  CustomCallBackendConfig.collective_id: CustomCallBackendConfig#collective_id.
  CustomCallBackendConfig.needs_hlo_passes: CustomCallBackendConfig#needs_hlo_passes.
  CustomCallBackendConfig.needs_layout_passes: CustomCallBackendConfig#needs_layout_passes.
  CustomCallBackendConfig.vmem_limit_bytes: CustomCallBackendConfig#vmem_limit_bytes.
  CustomCallBackendConfig.serialization_format: CustomCallBackendConfig#serialization_format.
  CustomCallBackendConfig.internal_scratch_in_bytes: CustomCallBackendConfig#internal_scratch_in_bytes.
  CustomCallBackendConfig.disable_bounds_checks: CustomCallBackendConfig#disable_bounds_checks.
  CustomCallBackendConfig.disable_semaphore_checks: CustomCallBackendConfig#disable_semaphore_checks.
  CustomCallBackendConfig.active_core_count: CustomCallBackendConfig#active_core_count.
  _avals_to_layouts: _avals_to_layouts().
  _extra_dialect_loaders._extra_dialect_loaders: _extra_dialect_loaders._extra_dialect_loaders.
  _FWD_COMPAT_VERSION: _FWD_COMPAT_VERSION.
  Tiling.SPARSE_CORE: Tiling#SPARSE_CORE.
  _lower_mosaic_module_to_asm: _lower_mosaic_module_to_asm().
  _get_device_type.assign_device_type_based_on_core_type: _get_device_type().assign_device_type_based_on_core_type().
  _get_active_core_count.get_core_parallel_dim_size: _get_active_core_count().get_core_parallel_dim_size().
  CostEstimate.flops: CostEstimate#flops.
  CostEstimate.transcendentals: CostEstimate#transcendentals.
  CostEstimate.bytes_accessed: CostEstimate#bytes_accessed.
  CostEstimate.remote_bytes_transferred: CostEstimate#remote_bytes_transferred.
  CostEstimate.to_json: CostEstimate#to_json().
  OptLevel.O0: OptLevel#O0.
  OptLevel.O1: OptLevel#O1.
  OptLevel.O2: OptLevel#O2.
  OptLevel.O3: OptLevel#O3.
  CustomCallBackendConfig.__repr__: CustomCallBackendConfig#__repr__().
---
# Module: [`jax/_src/tpu_custom_call.py`](../../../../../../raw/code/jax/jax/_src/tpu_custom_call.py)

## Classes
### `CostEstimate`  ·  implements/extends _TypedDict
- def: [`jax/_src/tpu_custom_call.py:127`](../../../../../../raw/code/jax/jax/_src/tpu_custom_call.py#L127)
- signature: `class CostEstimate(TypedDict):`
- members:
  - `to_json(self)` — [`L133`](../../../../../../raw/code/jax/jax/_src/tpu_custom_call.py#L133)
  - `bytes_accessed` — [`L130`](../../../../../../raw/code/jax/jax/_src/tpu_custom_call.py#L130)
  - `flops` — [`L128`](../../../../../../raw/code/jax/jax/_src/tpu_custom_call.py#L128)
  - `remote_bytes_transferred` — [`L131`](../../../../../../raw/code/jax/jax/_src/tpu_custom_call.py#L131)
  - `transcendentals` — [`L129`](../../../../../../raw/code/jax/jax/_src/tpu_custom_call.py#L129)
- used by: [`_lower_to_custom_call`](pallas/mosaic/pallas_call_registration.md#_lower_to_custom_call), [`_lowered_to_custom_call_config`](tpu_custom_call.md#_lowered_to_custom_call_config), [`_lower_to_custom_call_config`](tpu_custom_call.md#_lower_to_custom_call_config), [`lower_module_to_custom_call`](tpu_custom_call.md#lower_module_to_custom_call), [`lowered_as_tpu_kernel`](tpu_custom_call.md#lowered_as_tpu_kernel), [`as_tpu_kernel`](tpu_custom_call.md#as_tpu_kernel), [`cost_estimate`](tpu_custom_call.md#CustomCallBackendConfig.cost_estimate)

### `CustomCallBackendConfig`
- def: [`jax/_src/tpu_custom_call.py:164`](../../../../../../raw/code/jax/jax/_src/tpu_custom_call.py#L164)
- doc: Represents an unserialized backend config for custom calls.
- signature: `class CustomCallBackendConfig:`
- members:
  - `downgrade_lowered_module_asm(self, version: int)` — [`L202`](../../../../../../raw/code/jax/jax/_src/tpu_custom_call.py#L202) — Downgrades the lowered module asm to the given version.
  - `to_json(self)` — [`L231`](../../../../../../raw/code/jax/jax/_src/tpu_custom_call.py#L231) — Serializes the backend config into JSON.
  - `active_core_count` — [`L182`](../../../../../../raw/code/jax/jax/_src/tpu_custom_call.py#L182)
  - `allow_input_fusion` — [`L176`](../../../../../../raw/code/jax/jax/_src/tpu_custom_call.py#L176)
  - `collective_id` — [`L169`](../../../../../../raw/code/jax/jax/_src/tpu_custom_call.py#L169)
  - `cost_estimate` — [`L171`](../../../../../../raw/code/jax/jax/_src/tpu_custom_call.py#L171)
  - `device_type` — [`L170`](../../../../../../raw/code/jax/jax/_src/tpu_custom_call.py#L170)
  - `disable_bounds_checks` — [`L180`](../../../../../../raw/code/jax/jax/_src/tpu_custom_call.py#L180)
  - `disable_semaphore_checks` — [`L181`](../../../../../../raw/code/jax/jax/_src/tpu_custom_call.py#L181)
  - `flags` — [`L175`](../../../../../../raw/code/jax/jax/_src/tpu_custom_call.py#L175)
  - `has_communication` — [`L168`](../../../../../../raw/code/jax/jax/_src/tpu_custom_call.py#L168)
  - `input_memory_spaces` — [`L183`](../../../../../../raw/code/jax/jax/_src/tpu_custom_call.py#L183)
  - `internal_scratch_in_bytes` — [`L178`](../../../../../../raw/code/jax/jax/_src/tpu_custom_call.py#L178)
  - `lowered_module_asm` — [`L166`](../../../../../../raw/code/jax/jax/_src/tpu_custom_call.py#L166)
  - `lowered_module_asm_version` — [`L167`](../../../../../../raw/code/jax/jax/_src/tpu_custom_call.py#L167)
  - `needs_hlo_passes` — [`L172`](../../../../../../raw/code/jax/jax/_src/tpu_custom_call.py#L172)
  - `needs_layout_passes` — [`L173`](../../../../../../raw/code/jax/jax/_src/tpu_custom_call.py#L173)
  - `opt_level` — [`L187`](../../../../../../raw/code/jax/jax/_src/tpu_custom_call.py#L187)
  - `output_memory_spaces` — [`L179`](../../../../../../raw/code/jax/jax/_src/tpu_custom_call.py#L179)
  - `serialization_format` — [`L177`](../../../../../../raw/code/jax/jax/_src/tpu_custom_call.py#L177)
  - `shape_invariant_numerics` — [`L185`](../../../../../../raw/code/jax/jax/_src/tpu_custom_call.py#L185)
  - `skip_device_barrier` — [`L184`](../../../../../../raw/code/jax/jax/_src/tpu_custom_call.py#L184)
  - `tiling` — [`L186`](../../../../../../raw/code/jax/jax/_src/tpu_custom_call.py#L186)
  - `vmem_limit_bytes` — [`L174`](../../../../../../raw/code/jax/jax/_src/tpu_custom_call.py#L174)
- protocol/private: `__post_init__`[`L189`](../../../../../../raw/code/jax/jax/_src/tpu_custom_call.py#L189), `__repr__`[`L199`](../../../../../../raw/code/jax/jax/_src/tpu_custom_call.py#L199)
- uses (calls/refs, reference-scoped): [`MemorySpace`](tpu_custom_call.md#MemorySpace), [`FrozenDict`](frozen_dict.md#FrozenDict), [`color`](tpu_custom_call.md#MemorySpace.color), [`Tiling`](tpu_custom_call.md#Tiling), [`OptLevel`](tpu_custom_call.md#OptLevel), [`CostEstimate`](tpu_custom_call.md#CostEstimate), [`SMEM`](tpu_custom_call.md#MemorySpace.SMEM), [`VMEM`](tpu_custom_call.md#MemorySpace.VMEM), [`HBM`](tpu_custom_call.md#MemorySpace.HBM), [`_compact_json_object`](tpu_custom_call.md#_compact_json_object), [`COMPACT`](tpu_custom_call.md#Tiling.COMPACT), [`_extra_dialect_loaders`](tpu_custom_call.md#_extra_dialect_loaders._extra_dialect_loaders)
- used by: [`_tpu_custom_call_lowering`](tpu_custom_call.md#_tpu_custom_call_lowering), [`_lowered_to_custom_call_config`](tpu_custom_call.md#_lowered_to_custom_call_config), [`_lower_to_custom_call_config`](tpu_custom_call.md#_lower_to_custom_call_config), [`_as_jax_callable`](tpu_custom_call.md#_as_jax_callable)

### `MemorySpace`  ·  implements/extends Enum
- def: [`jax/_src/tpu_custom_call.py:98`](../../../../../../raw/code/jax/jax/_src/tpu_custom_call.py#L98)
- signature: `class MemorySpace(enum.Enum):`
- members:
  - `color(self)` — [`L108`](../../../../../../raw/code/jax/jax/_src/tpu_custom_call.py#L108)
  - `HBM` — [`L99`](../../../../../../raw/code/jax/jax/_src/tpu_custom_call.py#L99)
  - `HOST` — [`L103`](../../../../../../raw/code/jax/jax/_src/tpu_custom_call.py#L103)
  - `SC_SCALAR_SEMAPHORE_MEM` — [`L104`](../../../../../../raw/code/jax/jax/_src/tpu_custom_call.py#L104)
  - `SC_VECTOR_SEMAPHORE_MEM` — [`L105`](../../../../../../raw/code/jax/jax/_src/tpu_custom_call.py#L105)
  - `SEMAPHORE_MEM` — [`L101`](../../../../../../raw/code/jax/jax/_src/tpu_custom_call.py#L101)
  - `SMEM` — [`L102`](../../../../../../raw/code/jax/jax/_src/tpu_custom_call.py#L102)
  - `VMEM` — [`L100`](../../../../../../raw/code/jax/jax/_src/tpu_custom_call.py#L100)
- used by: [`to_json`](tpu_custom_call.md#CustomCallBackendConfig.to_json), [`_get_memory_space_from_aval`](pallas/mosaic/pallas_call_registration.md#_get_memory_space_from_aval), [`_lowered_to_custom_call_config`](tpu_custom_call.md#_lowered_to_custom_call_config), [`_lower_to_custom_call_config`](tpu_custom_call.md#_lower_to_custom_call_config), [`lower_module_to_custom_call`](tpu_custom_call.md#lower_module_to_custom_call), [`_resolve_memory_spaces`](pallas/mosaic/pallas_call_registration.md#_resolve_memory_spaces), [`_get_memory_spaces_from_avals`](pallas/mosaic/pallas_call_registration.md#_get_memory_spaces_from_avals), [`as_tpu_kernel`](tpu_custom_call.md#as_tpu_kernel), [`input_memory_spaces`](tpu_custom_call.md#CustomCallBackendConfig.input_memory_spaces), [`output_memory_spaces`](tpu_custom_call.md#CustomCallBackendConfig.output_memory_spaces)

### `OptLevel`  ·  implements/extends Enum
- def: [`jax/_src/tpu_custom_call.py:156`](../../../../../../raw/code/jax/jax/_src/tpu_custom_call.py#L156)
- signature: `class OptLevel(enum.Enum):`
- members:
  - `O0` — [`L157`](../../../../../../raw/code/jax/jax/_src/tpu_custom_call.py#L157)
  - `O1` — [`L158`](../../../../../../raw/code/jax/jax/_src/tpu_custom_call.py#L158)
  - `O2` — [`L159`](../../../../../../raw/code/jax/jax/_src/tpu_custom_call.py#L159)
  - `O3` — [`L160`](../../../../../../raw/code/jax/jax/_src/tpu_custom_call.py#L160)
- used by: [`_lowered_to_custom_call_config`](tpu_custom_call.md#_lowered_to_custom_call_config), [`_lower_to_custom_call_config`](tpu_custom_call.md#_lower_to_custom_call_config), [`lower_module_to_custom_call`](tpu_custom_call.md#lower_module_to_custom_call), [`lowered_as_tpu_kernel`](tpu_custom_call.md#lowered_as_tpu_kernel), [`as_tpu_kernel`](tpu_custom_call.md#as_tpu_kernel), [`__init__`](pallas/mosaic/core.md#CompilerParams.__init__), [`opt_level`](tpu_custom_call.md#CustomCallBackendConfig.opt_level), [`opt_level`](pallas/mosaic/core.md#CompilerParams.opt_level)

### `Tiling`  ·  implements/extends Enum
- def: [`jax/_src/tpu_custom_call.py:151`](../../../../../../raw/code/jax/jax/_src/tpu_custom_call.py#L151)
- signature: `class Tiling(enum.Enum):`
- members:
  - `COMPACT` — [`L152`](../../../../../../raw/code/jax/jax/_src/tpu_custom_call.py#L152)
  - `SPARSE_CORE` — [`L153`](../../../../../../raw/code/jax/jax/_src/tpu_custom_call.py#L153)
- used by: [`to_json`](tpu_custom_call.md#CustomCallBackendConfig.to_json), [`_lowered_to_custom_call_config`](tpu_custom_call.md#_lowered_to_custom_call_config), [`_lower_to_custom_call_config`](tpu_custom_call.md#_lower_to_custom_call_config), [`lower_module_to_custom_call`](tpu_custom_call.md#lower_module_to_custom_call), [`lowered_as_tpu_kernel`](tpu_custom_call.md#lowered_as_tpu_kernel), [`as_tpu_kernel`](tpu_custom_call.md#as_tpu_kernel), [`_resolve_tiling`](pallas/mosaic/pallas_call_registration.md#_resolve_tiling), [`tiling`](tpu_custom_call.md#CustomCallBackendConfig.tiling)

### `TpuSideEffectType`  ·  implements/extends Enum
- def: [`jax/_src/tpu_custom_call.py:142`](../../../../../../raw/code/jax/jax/_src/tpu_custom_call.py#L142)
- signature: `class TpuSideEffectType(enum.Enum):`
- members:
  - `DATAFLOW_SIDE_EFFECTING` — [`L146`](../../../../../../raw/code/jax/jax/_src/tpu_custom_call.py#L146)
  - `PURE` — [`L144`](../../../../../../raw/code/jax/jax/_src/tpu_custom_call.py#L144)
  - `SIDE_EFFECTING` — [`L148`](../../../../../../raw/code/jax/jax/_src/tpu_custom_call.py#L148)
- used by: [`_tpu_custom_call_lowering`](tpu_custom_call.md#_tpu_custom_call_lowering), [`lower_module_to_custom_call`](tpu_custom_call.md#lower_module_to_custom_call), [`lowered_as_tpu_kernel`](tpu_custom_call.md#lowered_as_tpu_kernel), [`_resolve_side_effect_type`](pallas/mosaic/pallas_call_registration.md#_resolve_side_effect_type), [`as_tpu_kernel`](tpu_custom_call.md#as_tpu_kernel), [`_as_jax_callable`](tpu_custom_call.md#_as_jax_callable)

## Functions
- `_as_jax_callable(config: CustomCallBackendConfig, has_side_effects: TpuSideEffectType, out_type: Any, *, kernel_name: str | None, input_output_aliases: tuple[tuple[int, int], ...], metadata: Any | None)` — [`L945`](../../../../../../raw/code/jax/jax/_src/tpu_custom_call.py#L945)
- `_avals_to_layouts(avals)` — [`L388`](../../../../../../raw/code/jax/jax/_src/tpu_custom_call.py#L388)
- `_compact_json_object(**kwargs: Any)` — [`L377`](../../../../../../raw/code/jax/jax/_src/tpu_custom_call.py#L377)
- `_get_active_core_count(module: ir.Module)` — [`L544`](../../../../../../raw/code/jax/jax/_src/tpu_custom_call.py#L544)
- `_get_device_type(module: ir.Module)` — [`L506`](../../../../../../raw/code/jax/jax/_src/tpu_custom_call.py#L506) — Determines the device type based on the core_type annotations.
- `_lower_mosaic_module_to_asm(module: ir.Module, *, ir_version: int | None = None)` — [`L473`](../../../../../../raw/code/jax/jax/_src/tpu_custom_call.py#L473)
- `_lower_to_custom_call_config(module: ir.Module, *, vmem_limit_bytes: int | None, cost_estimate: CostEstimate | None, flags: dict[str, bool | int | float] | None, allow_input_fusion: Sequence[bool] | None, internal_scratch_in_bytes: int | None, collective_id: int | None, serialization_format: int | None, output_memory_spaces: tuple[MemorySpace | None, ...] | None = None, ir_version: int | None = None, disable_bounds_checks: bool = False, disable_semaphore_checks: bool = False, input_memory_spaces: tuple[MemorySpace | None, ...] | None = None, skip_device_barrier: bool = False, allow_collective_id_without_custom_barrier: bool = False, shape_invariant_numerics: bool = False, needs_layout_passes: bool | None = None, tiling: Tiling | None = None, opt_level: OptLevel | None = None)` — [`L619`](../../../../../../raw/code/jax/jax/_src/tpu_custom_call.py#L619)
- `_lowered_to_custom_call_config(lowered_module_asm: bytes, *, lowered_module_asm_version: int | None, vmem_limit_bytes: int | None, cost_estimate: CostEstimate | None, flags: dict[str, bool | int | float] | None, allow_input_fusion: Sequence[bool] | None, internal_scratch_in_bytes: int | None, collective_id: int | None, serialization_format: int | None, has_custom_barrier: bool, has_communication: bool, needs_hlo_passes: bool, needs_layout_passes: bool, device_type: str | None, output_memory_spaces: tuple[MemorySpace | None, ...] | None = None, disable_bounds_checks: bool = False, disable_semaphore_checks: bool = False, active_core_count: int | None = None, input_memory_spaces: tuple[MemorySpace | None, ...] | None = None, skip_device_barrier: bool = False, allow_collective_id_without_custom_barrier: bool = False, shape_invariant_numerics: bool = False, tiling: Tiling | None = None, opt_level: OptLevel | None = None)` — [`L681`](../../../../../../raw/code/jax/jax/_src/tpu_custom_call.py#L681)
- `_tpu_custom_call_abstract_eval(*_, out_avals, **__)` — [`L384`](../../../../../../raw/code/jax/jax/_src/tpu_custom_call.py#L384)
- `_tpu_custom_call_lowering(ctx: mlir.LoweringRuleContext, *in_nodes, config: CustomCallBackendConfig, has_side_effects: TpuSideEffectType, kernel_name: str | None, out_avals: Any, input_output_aliases: tuple[tuple[int, int], ...], metadata: Any | None)` — [`L392`](../../../../../../raw/code/jax/jax/_src/tpu_custom_call.py#L392)
- `apply_kernel(*args)` — [`L962`](../../../../../../raw/code/jax/jax/_src/tpu_custom_call.py#L962)
- `as_tpu_kernel(module: ir.Module, out_type: Any, *, cost_estimate: CostEstimate | None = None, kernel_name: str | None = None, vmem_limit_bytes: int | None = None, flags: dict[str, bool | int | float] | None = None, allow_input_fusion: Sequence[bool] | None = None, input_output_aliases: tuple[tuple[int, int], ...] = (), internal_scratch_in_bytes: int | None = None, collective_id: int | None = None, has_side_effects: TpuSideEffectType = TpuSideEffectType.PURE, serialization_format: int | None = 1, output_memory_spaces: tuple[MemorySpace | None, ...] | None = None, disable_bounds_checks: bool = False, disable_semaphore_checks: bool = False, input_memory_spaces: tuple[MemorySpace | None, ...] | None = None, shape_invariant_numerics: bool = False, needs_layout_passes: bool | None = None, metadata: Any | None = None, tiling: Tiling | None = None, _ir_version: int | None = None, opt_level: OptLevel | None = None)` — [`L823`](../../../../../../raw/code/jax/jax/_src/tpu_custom_call.py#L823) — Turns an MLIR Mosaic kernel into a JAX-compatible function.
- `assign_device_type_based_on_core_type(op: ir.Operation)` — [`L511`](../../../../../../raw/code/jax/jax/_src/tpu_custom_call.py#L511)
- `get_core_parallel_dim_size(dim_semantics: ir.ArrayAttr, iter_bounds: ir.DenseI64ArrayAttr, other_subkernel_core_dim_size: int | None = None)` — [`L546`](../../../../../../raw/code/jax/jax/_src/tpu_custom_call.py#L546)
- `get_ir_version(ctx: mlir.LoweringRuleContext)` — [`L66`](../../../../../../raw/code/jax/jax/_src/tpu_custom_call.py#L66)
- `lower_module_to_custom_call(ctx: mlir.LoweringRuleContext, *in_nodes: ir.Value, module: ir.Module, out_type: Any, kernel_name: str, cost_estimate: CostEstimate | None, vmem_limit_bytes: int | None, flags: dict[str, bool | int | float] | None, allow_input_fusion: Sequence[bool] | None, input_output_aliases: tuple[tuple[int, int], ...], internal_scratch_in_bytes: int | None, collective_id: int | None, has_side_effects: bool | TpuSideEffectType, serialization_format: int | None, output_memory_spaces: tuple[MemorySpace | None, ...] | None, disable_bounds_checks: bool = False, disable_semaphore_checks: bool = False, input_memory_spaces: tuple[MemorySpace | None, ...] | None, metadata: Any | None = None, skip_device_barrier: bool = False, allow_collective_id_without_custom_barrier: bool = False, shape_invariant_numerics: bool = False, needs_layout_passes: bool | None = None, tiling: Tiling | None = None, opt_level: OptLevel | None = None)` — [`L757`](../../../../../../raw/code/jax/jax/_src/tpu_custom_call.py#L757)
- `lowered_as_tpu_kernel(lowered_module: ir.Module, out_type: Any, *, collective_id: int | None = None, cost_estimate: CostEstimate | None = None, needs_hlo_passes: bool = False, needs_layout_passes: bool = False, has_communication: bool = False, has_side_effects: bool | TpuSideEffectType = False, has_custom_barrier: bool = False, kernel_name: str | None = None, vmem_limit_bytes: int | None = None, flags: dict[str, bool | int | float] | None = None, allow_input_fusion: Sequence[bool] | None = None, input_output_aliases: tuple[tuple[int, int], ...] = (), serialization_format: int | None = None, internal_scratch_in_bytes: int | None = None, disable_bounds_checks: bool = False, disable_semaphore_checks: bool = False, metadata: Any | None = None, allow_collective_id_without_custom_barrier: bool = False, tiling: Tiling | None = None, opt_level: OptLevel | None = None)` — [`L878`](../../../../../../raw/code/jax/jax/_src/tpu_custom_call.py#L878)
- `register_extra_dialect(loader: Callable[[ir.Context], None])` — [`L50`](../../../../../../raw/code/jax/jax/_src/tpu_custom_call.py#L50)
- `tpu_custom_call_batcher(axis_data, args, dims, **kwargs)` — [`L83`](../../../../../../raw/code/jax/jax/_src/tpu_custom_call.py#L83)

## Module values
- `FLAGS` — [`L43`](../../../../../../raw/code/jax/jax/_src/tpu_custom_call.py#L43)
- `_FWD_COMPAT_VERSION` — [`L65`](../../../../../../raw/code/jax/jax/_src/tpu_custom_call.py#L65)
- `_extra_dialect_loaders` — [`L47`](../../../../../../raw/code/jax/jax/_src/tpu_custom_call.py#L47)
- `tpu_custom_call_p` — [`L78`](../../../../../../raw/code/jax/jax/_src/tpu_custom_call.py#L78)

