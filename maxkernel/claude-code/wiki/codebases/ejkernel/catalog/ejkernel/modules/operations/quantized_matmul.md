---
title: 'Module: ejkernel/modules/operations/quantized_matmul.py'
type: catalog
provenance: extracted
module: ejkernel/modules/operations/quantized_matmul.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.modules.operations.quantized_matmul`/
symbols:
  QuantizedMatmul._candidate_cfgs_gpu_for_platform: QuantizedMatmul#_candidate_cfgs_gpu_for_platform().
  _pallas_tpu_candidate_cfgs: _pallas_tpu_candidate_cfgs().
  _quantized_matmul_impl: _quantized_matmul_impl().
  quantized_matmul: quantized_matmul().
  _xla_candidate_cfgs: _xla_candidate_cfgs().
  QuantizedMatmul.run: QuantizedMatmul#run().
  _pallas_tpu_heuristic_cfg: _pallas_tpu_heuristic_cfg().
  _triton_heuristic_cfg: _triton_heuristic_cfg().
  _cuda_heuristic_cfg: _cuda_heuristic_cfg().
  _normalize_pallas_tpu_packed_cfg_forward: _normalize_pallas_tpu_packed_cfg_forward().
  _tilelang_heuristic_cfg: _tilelang_heuristic_cfg().
  _cute_heuristic_cfg: _cute_heuristic_cfg().
  QuantizedMatmul.candidate_cfgs_gpu: QuantizedMatmul#candidate_cfgs_gpu().
  _quantized_matmul_executor._quantized_matmul_executor: _quantized_matmul_executor._quantized_matmul_executor.
  QuantizedMatmul.heuristic_cfg_gpu: QuantizedMatmul#heuristic_cfg_gpu().
  _infer_mkn: _infer_mkn().
  QuantizedMatmul.candidate_cfgs_tpu: QuantizedMatmul#candidate_cfgs_tpu().
  QuantizedMatmul._resolve_inv_platform: QuantizedMatmul#_resolve_inv_platform().
  _inv_arg: _inv_arg().
  QuantizedMatmul.get_impl: QuantizedMatmul#get_impl().
  QuantizedMatmul.heuristic_cfg_tpu: QuantizedMatmul#heuristic_cfg_tpu().
  _xla_heuristic_cfg: _xla_heuristic_cfg().
  _ceil_div: _ceil_div().
  _lcm: _lcm().
  _resolve_qparams: _resolve_qparams().
  _lookup_best_qmm_policy: _lookup_best_qmm_policy().
  _pallas_tpu_candidate_cfgs._score: _pallas_tpu_candidate_cfgs()._score().
  _xla_candidate_cfgs._score: _xla_candidate_cfgs()._score().
  quantized_matmul._inner: quantized_matmul()._inner().
  QuantizedMatmul: QuantizedMatmul#
  _maybe_tpu_predecode_once_matmul: _maybe_tpu_predecode_once_matmul().
  _packed_legal_block_n: _packed_legal_block_n().
  QuantizedMatmul.heuristic_cfg: QuantizedMatmul#heuristic_cfg().
  QuantizedMatmul.candidate_cfgs: QuantizedMatmul#candidate_cfgs().
  QuantizedMatmul.heuristic_cfg_cpu: QuantizedMatmul#heuristic_cfg_cpu().
  QuantizedMatmul.candidate_cfgs_cpu: QuantizedMatmul#candidate_cfgs_cpu().
  _static_bool: _static_bool().
  _prefer_bf16: _prefer_bf16().
  _nearest_choices: _nearest_choices().
  _is_tracer_value: _is_tracer_value().
  _should_try_tpu_predecode_once_default: _should_try_tpu_predecode_once_default().
  quantized_matmul._inner_bwd: quantized_matmul()._inner_bwd().
  _bit_aligned_values: _bit_aligned_values().
  quantized_matmul._inner_vjp: quantized_matmul()._inner_vjp().
  _parse_positive_int_env: _parse_positive_int_env().
  _tilelang_runtime_available: _tilelang_runtime_available().
  _qmm_bestcfg_mode_key: _qmm_bestcfg_mode_key().
  _normalize_pallas_tpu_packed_cfg_forward._legal: _normalize_pallas_tpu_packed_cfg_forward()._legal().
  quantized_matmul._inner_fwd: quantized_matmul()._inner_fwd().
  _expand_choices: _expand_choices().
  QuantizedMatmul.__init__: QuantizedMatmul#__init__().
  QuantizedMatmul.candidate_cfgs_shard_map_gpu: QuantizedMatmul#candidate_cfgs_shard_map_gpu.
  _static_int: _static_int().
  _ensure_aligned: _ensure_aligned().
  _pick_split_k: _pick_split_k().
  _parse_nonnegative_int_env: _parse_nonnegative_int_env().
  _QMM_TPU_BESTCFG_POLICY._QMM_TPU_BESTCFG_POLICY: _QMM_TPU_BESTCFG_POLICY._QMM_TPU_BESTCFG_POLICY.
  _xla_choices: _xla_choices().
  _tpu_tile_working_set_bytes: _tpu_tile_working_set_bytes().
  _env_tpu_default_strategy: _env_tpu_default_strategy().
  _QMM_BESTCFG_NON_AFFINE_MODES: _QMM_BESTCFG_NON_AFFINE_MODES.
  QuantizedMatmul._tilelang_num_stages: QuantizedMatmul#_tilelang_num_stages().
  QuantizedMatmul.version: QuantizedMatmul#version.
---
# Module: [`ejkernel/modules/operations/quantized_matmul.py`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/quantized_matmul.py)

## Classes
### `QuantizedMatmul`  ·  implements/extends Kernel
- def: [`ejkernel/modules/operations/quantized_matmul.py:1216`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/quantized_matmul.py#L1216)
- doc: Quantized matrix multiplication kernel with configurable tiling and backend selection.
- signature: `class QuantizedMatmul(Kernel[QuantizedMatmulConfig, Array]):`
- members:
  - `__init__(self)` — [`L1238`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/quantized_matmul.py#L1238) — Initialize the quantized matmul kernel.
  - `_candidate_cfgs_gpu_for_platform(self, inv: Invocation[QuantizedMatmulConfig, Array], resolved: Platform)` — [`L1506`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/quantized_matmul.py#L1506) — Return candidate configurations for a concrete GPU platform. — documented in [ejkernel-modules-operations-quantized_matmul](../../../../concepts/ejkernel-modules-operations-quantized_matmul.md)
  - `_resolve_inv_platform(self, inv: Invocation[QuantizedMatmulConfig, Array])` — [`L1242`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/quantized_matmul.py#L1242) — Resolve the target platform from invocation parameters.
  - `candidate_cfgs(self, inv: Invocation[QuantizedMatmulConfig, Array])` — [`L1430`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/quantized_matmul.py#L1430) — Return candidate configurations for autotuning.
  - `candidate_cfgs_cpu(self, inv: Invocation[QuantizedMatmulConfig, Array])` — [`L1691`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/quantized_matmul.py#L1691) — Return CPU-specific candidate configurations for autotuning.
  - `candidate_cfgs_gpu(self, inv: Invocation[QuantizedMatmulConfig, Array])` — [`L1486`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/quantized_matmul.py#L1486) — Return GPU-specific candidate configurations for autotuning. — documented in [ejkernel-modules-operations-quantized_matmul](../../../../concepts/ejkernel-modules-operations-quantized_matmul.md)
  - `candidate_cfgs_tpu(self, inv: Invocation[QuantizedMatmulConfig, Array])` — [`L1702`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/quantized_matmul.py#L1702) — Return TPU-specific candidate configurations for autotuning. — documented in [ejkernel-modules-operations-quantized_matmul](../../../../concepts/ejkernel-modules-operations-quantized_matmul.md)
  - `get_impl(self, cfg: QuantizedMatmulConfig)` — [`L1256`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/quantized_matmul.py#L1256) — Get the kernel implementation for the given configuration.
  - `heuristic_cfg(self, inv: Invocation[QuantizedMatmulConfig, Array])` — [`L1419`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/quantized_matmul.py#L1419) — Return default heuristic configuration for any platform.
  - `heuristic_cfg_cpu(self, inv: Invocation[QuantizedMatmulConfig, Array])` — [`L1461`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/quantized_matmul.py#L1461) — Return heuristic configuration optimized for CPU.
  - `heuristic_cfg_gpu(self, inv: Invocation[QuantizedMatmulConfig, Array])` — [`L1441`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/quantized_matmul.py#L1441) — Return heuristic configuration optimized for GPU.
  - `heuristic_cfg_tpu(self, inv: Invocation[QuantizedMatmulConfig, Array])` — [`L1472`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/quantized_matmul.py#L1472) — Return heuristic configuration optimized for TPU.
  - `run(self, x: Float[Array, "m k"], w: Array, scales: Array, zeros: Array | None = None, transpose: bool = False, group_size: int | None = None, bits: int | None = None, mode: QuantizationMode = "affine", axis: QuantizationAxis | None = None, gemv_mode: GemvMode = "auto", revsplit_k: RevSplitKMode = "auto", revsplit_k_parts: int | None = None, allow_dense_fallback: bool = True, _resolved_platform: str | None = None, platform: Literal["triton", "pallas", "cuda", "cute", "tilelang", "xla", "auto"] | None = None, *, cfg: QuantizedMatmulConfig)` — [`L1277`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/quantized_matmul.py#L1277) — Execute quantized matmul with the selected backend.
  - `candidate_cfgs_shard_map_gpu` — [`L1721`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/quantized_matmul.py#L1721)
  - `version` — [`L1236`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/quantized_matmul.py#L1236)
- protocol/private: `_tilelang_num_stages`[`L1670`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/quantized_matmul.py#L1670)
- uses (calls/refs, reference-scoped): [`backend`](configs.md#BaseOperationConfig.backend), [`platform`](configs.md#BaseOperationConfig.platform), [`Invocation`](../../ops/core/kernel.md#Invocation), [`Kernel`](../../ops/core/kernel.md#Kernel), [`kwargs`](../../ops/core/kernel.md#Invocation.kwargs), [`kernel_registry`](../../kernels/_registry.md#kernel_registry), [`Platform`](../../kernels/_registry.md#Platform), [`detect_platform`](../base.md#detect_platform), [`get`](../../kernels/_registry.md#KernelRegistry.get), [`QuantizedMatmulConfig`](configs.md#QuantizedMatmulConfig), [`_pallas_tpu_candidate_cfgs`](quantized_matmul.md#_pallas_tpu_candidate_cfgs), [`_xla_candidate_cfgs`](quantized_matmul.md#_xla_candidate_cfgs), [`_pallas_tpu_heuristic_cfg`](quantized_matmul.md#_pallas_tpu_heuristic_cfg), [`__init__`](../../ops/core/kernel.md#Kernel.__init__), [`_triton_heuristic_cfg`](quantized_matmul.md#_triton_heuristic_cfg), [`_cuda_heuristic_cfg`](quantized_matmul.md#_cuda_heuristic_cfg), [`_normalize_pallas_tpu_packed_cfg_forward`](quantized_matmul.md#_normalize_pallas_tpu_packed_cfg_forward), [`_tilelang_heuristic_cfg`](quantized_matmul.md#_tilelang_heuristic_cfg), [`PALLAS`](../../kernels/_registry.md#Platform.PALLAS), [`QuantizationAxis`](../../quantization/_utils/qparams.md#QuantizationAxis), [`_cute_heuristic_cfg`](quantized_matmul.md#_cute_heuristic_cfg), [`QuantizationMode`](../../quantization/_utils/qparams.md#QuantizationMode), [`_infer_mkn`](quantized_matmul.md#_infer_mkn), [`block_k`](configs.md#QuantizedMatmulConfig.block_k), [`block_m`](configs.md#QuantizedMatmulConfig.block_m), [`block_n`](configs.md#QuantizedMatmulConfig.block_n), [`op_id`](../../ops/core/kernel.md#Kernel.op_id), [`_inv_arg`](quantized_matmul.md#_inv_arg), [`use_bf16`](configs.md#QuantizedMatmulConfig.use_bf16), [`num_stages`](configs.md#QuantizedMatmulConfig.num_stages), [`num_warps`](configs.md#QuantizedMatmulConfig.num_warps), [`split_k`](configs.md#QuantizedMatmulConfig.split_k), [`_xla_heuristic_cfg`](quantized_matmul.md#_xla_heuristic_cfg), [`GemvMode`](../../quantization/_utils/qparams.md#GemvMode), [`RevSplitKMode`](../../quantization/_utils/qparams.md#RevSplitKMode), [`override_cfg`](../../ops/core/kernel.md#Invocation.override_cfg), [`TRITON`](../../kernels/_registry.md#Platform.TRITON), [`_resolve_qparams`](quantized_matmul.md#_resolve_qparams), [`tpu_path`](configs.md#QuantizedMatmulConfig.tpu_path), [`TILELANG`](../../kernels/_registry.md#Platform.TILELANG)  (+5 more)
- used by: [`Kernel`](../../ops/core/kernel.md#Kernel), [`candidate_cfgs`](../../ops/core/kernel.md#Kernel.candidate_cfgs), [`heuristic_cfg`](../../ops/core/kernel.md#Kernel.heuristic_cfg), [`run`](../../ops/core/kernel.md#Kernel.run), [`_quantized_matmul_impl`](quantized_matmul.md#_quantized_matmul_impl)

## Functions
- `_bit_aligned_values(bits: int)` — [`L156`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/quantized_matmul.py#L156) — Return the value count that starts and ends on a 32-bit word boundary.
- `_ceil_div(a: int, b: int)` — [`L161`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/quantized_matmul.py#L161) — Compute ceiling division of a by b.
- `_cuda_heuristic_cfg(inv: Invocation[QuantizedMatmulConfig, Array])` — [`L1108`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/quantized_matmul.py#L1108) — Generate a heuristic configuration for CUDA custom-call quantized matmul. — documented in [ejkernel-modules-operations-quantized_matmul](../../../../concepts/ejkernel-modules-operations-quantized_matmul.md)
- `_cute_heuristic_cfg(inv: Invocation[QuantizedMatmulConfig, Array])` — [`L1191`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/quantized_matmul.py#L1191) — Generate a heuristic configuration for CuTe DSL quantized matmul. — documented in [ejkernel-modules-operations-quantized_matmul](../../../../concepts/ejkernel-modules-operations-quantized_matmul.md)
- `_ensure_aligned(choices: list[int], align: int, max_choice: int)` — [`L215`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/quantized_matmul.py#L215) — Filter or round choices to ensure alignment.
- `_env_tpu_default_strategy()` — [`L446`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/quantized_matmul.py#L446) — Read the TPU default QMM strategy from the environment.
- `_expand_choices(value: int, choices: tuple[int, ...])` — [`L189`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/quantized_matmul.py#L189) — Expand a value into a neighborhood of choices.
- `_infer_mkn(inv: Invocation[QuantizedMatmulConfig, Array], group_size: int)` — [`L257`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/quantized_matmul.py#L257) — Infer the M, K, N dimensions and transpose flag from an invocation. — documented in [ejkernel-modules-operations-quantized_matmul](../../../../concepts/ejkernel-modules-operations-quantized_matmul.md)
- `_inner(xi, wi, si, zi)` — [`L2118`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/quantized_matmul.py#L2118) — Dispatch to _quantized_matmul_impl with captured quantization parameters.
- `_inner_bwd(res, g)` — [`L2176`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/quantized_matmul.py#L2176) — Custom VJP backward: dequantize weights and compute grad_x (grad_w/scales/zeros are zero).
- `_inner_fwd(xi, wi, si, zi)` — [`L2171`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/quantized_matmul.py#L2171) — Custom VJP forward: compute output and save residuals (w, scales, zeros).
- `_inner_vjp(xi, wi, si, zi)` — [`L2167`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/quantized_matmul.py#L2167) — Forward pass wrapper decorated with custom_vjp for backward compatibility.
- `_inv_arg(inv: Invocation[QuantizedMatmulConfig, Array], name: str, index: int)` — [`L241`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/quantized_matmul.py#L241) — Resolve a positional-or-keyword argument from an Invocation.
- `_is_tracer_value(x: object)` — [`L430`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/quantized_matmul.py#L430) — Return ``True`` when ``x`` is an abstract JAX tracer.
- `_lcm(a: int, b: int)` — [`L139`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/quantized_matmul.py#L139) — Compute the least common multiple of two integers.
- `_legal(block_n: int)` — [`L928`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/quantized_matmul.py#L928)
- `_lookup_best_qmm_policy(*, backend_name: str, mode: str, m_tokens: int, runtime_axis: QuantizationAxis, runtime_transpose: bool, weights_concrete: bool)` — [`L525`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/quantized_matmul.py#L525) — Return best-known policy controls for QMM runtime selection.
- `_maybe_tpu_predecode_once_matmul(x: Array, w: Array, scales: Array, zeros: Array | None, *, mode: QuantizationMode, group_size: int, bits: int)` — [`L603`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/quantized_matmul.py#L603) — Try the TPU predecode-once + dense matmul path.
- `_nearest_choices(value: int, choices: tuple[int, ...], count: int = 2)` — [`L174`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/quantized_matmul.py#L174) — Select the nearest choices to a target value from a set of options.
- `_normalize_pallas_tpu_packed_cfg_forward(*, x: Array, w: Array, scales: Array, group_size: int, bits: int, cfg: QuantizedMatmulConfig)` — [`L897`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/quantized_matmul.py#L897) — Coerce TPU Pallas forward tiles to a legal packed BlockSpec.
- `_packed_legal_block_n(n: int, *, group_size: int, bits: int, align_n: int, vmem_cap: int = 4096)` — [`L674`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/quantized_matmul.py#L674) — Return the smallest packed-legal block_n for the given N dimension.
- `_pallas_tpu_candidate_cfgs(inv: Invocation[QuantizedMatmulConfig, Array])` — [`L752`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/quantized_matmul.py#L752) — Generate candidate configurations for autotuning TPU Pallas quantized matmul. — documented in [ejkernel-modules-operations-quantized_matmul](../../../../concepts/ejkernel-modules-operations-quantized_matmul.md)
- `_pallas_tpu_heuristic_cfg(inv: Invocation[QuantizedMatmulConfig, Array])` — [`L712`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/quantized_matmul.py#L712) — Generate a heuristic configuration for TPU Pallas quantized matmul. — documented in [ejkernel-modules-operations-quantized_matmul](../../../../concepts/ejkernel-modules-operations-quantized_matmul.md)
- `_parse_nonnegative_int_env(name: str, default: int)` — [`L393`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/quantized_matmul.py#L393) — Parse a non-negative integer environment variable with a safe fallback.
- `_parse_positive_int_env(name: str, default: int)` — [`L373`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/quantized_matmul.py#L373) — Parse a positive integer environment variable with a safe fallback.
- `_pick_split_k(m: int, k: int, block_k: int)` — [`L327`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/quantized_matmul.py#L327) — Select the split-K factor for improved parallelism on small M.
- `_prefer_bf16(x: Array)` — [`L294`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/quantized_matmul.py#L294) — Determine whether to prefer bfloat16 accumulation for the given input.
- `_qmm_bestcfg_mode_key(mode: str)` — [`L508`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/quantized_matmul.py#L508) — Map a quantization mode string to a best-config policy table key.
- `_quantized_matmul_impl(x: Float[Array, "m k"], w: Array, scales: Array, zeros: Array | None = None,, *, transpose: bool = False, group_size: int | None = None, bits: int | None = None, mode: QuantizationMode = "affine", axis: QuantizationAxis | None = None, gemv_mode: GemvMode = "auto", revsplit_k: RevSplitKMode = "auto", revsplit_k_parts: int | None = None, allow_dense_fallback: bool = True, platform: Literal["triton", "pallas", "cuda", "cute", "tilelang", "xla", "auto"] | None = None, cfg: QuantizedMatmulConfig | None = None)` — [`L1741`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/quantized_matmul.py#L1741) — Execute quantized matrix multiplication with normalized qparams. — documented in [ejkernel-modules-operations-quantized_matmul](../../../../concepts/ejkernel-modules-operations-quantized_matmul.md)
- `_resolve_qparams(mode: str, group_size: int | None, bits: int | None)` — [`L98`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/quantized_matmul.py#L98) — Resolve quantization parameters from mode, group_size, and bits.
- `_score(cfg: QuantizedMatmulConfig)` — [`L869`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/quantized_matmul.py#L869)
- `_score(cfg: QuantizedMatmulConfig)` — [`L1027`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/quantized_matmul.py#L1027) — Score a config by Manhattan distance from actual matrix dimensions.
- `_should_try_tpu_predecode_once_default(*, fuse: bool, backend_name: str, runtime_axis: QuantizationAxis, runtime_transpose: bool, platform: Literal["triton", "pallas", "cuda", "cute", "tilelang", "xla", "auto"] | None, strategy_override: Literal["predecode_once", "packed"] | None = None)` — [`L555`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/quantized_matmul.py#L555) — Determine whether to attempt the TPU predecode-once dense matmul path.
- `_static_bool(value, name: str)` — [`L113`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/quantized_matmul.py#L113) — Extract a concrete boolean value, raising if it is a JAX tracer.
- `_static_int(value, name: str)` — [`L126`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/quantized_matmul.py#L126) — Extract a concrete integer value, raising if it is a JAX tracer.
- `_tilelang_heuristic_cfg(inv: Invocation[QuantizedMatmulConfig, Array])` — [`L1138`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/quantized_matmul.py#L1138) — Generate a launch-safe heuristic configuration for TileLang QMM.
- `_tilelang_runtime_available()` — [`L317`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/quantized_matmul.py#L317) — Return whether TileLang FFI can actually run in this environment.
- `_tpu_tile_working_set_bytes(*, block_m: int, block_n: int, block_k: int)` — [`L413`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/quantized_matmul.py#L413) — Estimate the per-tile bytes touched by the fused TPU QMM hot loop.
- `_triton_heuristic_cfg(inv: Invocation[QuantizedMatmulConfig, Array])` — [`L1052`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/quantized_matmul.py#L1052) — Generate a heuristic configuration for Triton GPU quantized matmul.
- `_xla_candidate_cfgs(inv: Invocation[QuantizedMatmulConfig, Array], hardware: str)` — [`L973`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/quantized_matmul.py#L973) — Generate candidate configurations for autotuning XLA quantized matmul.
- `_xla_choices(hardware: str)` — [`L356`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/quantized_matmul.py#L356) — Return block size choice tuples for XLA backend on the given hardware.
- `_xla_heuristic_cfg(inv: Invocation[QuantizedMatmulConfig, Array], hardware: str)` — [`L1035`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/quantized_matmul.py#L1035) — Generate a heuristic configuration for XLA quantized matmul.
- `quantized_matmul(x: Float[Array, "m k"], w: Array, scales: Array, zeros: Array | None = None,, *, transpose: bool = False, group_size: int | None = None, bits: int | None = None, mode: QuantizationMode = "affine", axis: QuantizationAxis | None = None, gemv_mode: GemvMode = "auto", revsplit_k: RevSplitKMode = "auto", revsplit_k_parts: int | None = None, fuse: bool = True, strict_fuse: bool | None = None, tpu_path: Literal["packed", "hybrid", "predecode"] | None = None, allow_dense_fallback: bool | None = None, use_best_config: bool = False, platform: Literal["triton", "pallas", "cuda", "cute", "tilelang", "xla", "auto"] | None = None, cfg: QuantizedMatmulConfig | None = None)` — [`L1924`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/quantized_matmul.py#L1924) — Quantized matrix multiplication with fused dequantization and custom VJP. — documented in [ejkernel-modules-operations-quantized_matmul](../../../../concepts/ejkernel-modules-operations-quantized_matmul.md)

## Module values
- `_QMM_BESTCFG_NON_AFFINE_MODES` — [`L466`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/quantized_matmul.py#L466)
- `_QMM_TPU_BESTCFG_POLICY` — [`L467`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/quantized_matmul.py#L467)
- `_quantized_matmul_executor` — [`L1724`](../../../../../../../raw/code/ejkernel/ejkernel/modules/operations/quantized_matmul.py#L1724) — documented in [ejkernel-modules-operations-quantized_matmul](../../../../concepts/ejkernel-modules-operations-quantized_matmul.md)

