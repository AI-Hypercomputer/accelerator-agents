---
title: 'Module: ejkernel/callib/_triton_call.py'
type: catalog
provenance: extracted
module: ejkernel/callib/_triton_call.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.callib._triton_call`/
symbols:
  get_or_create_triton_kernel: get_or_create_triton_kernel().
  triton_kernel_call_lowering: triton_kernel_call_lowering().
  triton_call: triton_call().
  _load_triton_kernel_cache: _load_triton_kernel_cache().
  triton_kernel_call_lowering.unwrap_triton_kernel: triton_kernel_call_lowering().unwrap_triton_kernel().
  compile_ttir_to_ptx_inplace: compile_ttir_to_ptx_inplace().
  compile_ttir_to_hsaco_inplace: compile_ttir_to_hsaco_inplace().
  _save_triton_kernel_cache: _save_triton_kernel_cache().
  triton_kernel_call_p: triton_kernel_call_p.
  get_triton_type: get_triton_type().
  CompilationResult.binary: CompilationResult#binary.
  _log_triton_cache: _log_triton_cache().
  CompilationResult.shared_mem_bytes: CompilationResult#shared_mem_bytes.
  CompilationResult: CompilationResult#
  CompilationResult.name: CompilationResult#name.
  CompilationResult.cluster_dims: CompilationResult#cluster_dims.
  compile_ttir_inplace: compile_ttir_inplace().
  _make_triton_cache_key: _make_triton_cache_key().
  _invalidate_triton_cache_entry: _invalidate_triton_cache_entry().
  _get_triton_cache_dir: _get_triton_cache_dir().
  CAN_USE_TRITON: CAN_USE_TRITON.
  GridOrLambda: GridOrLambda.
  TRITON_CACHE_ENABLED: TRITON_CACHE_ENABLED.
  TRITON_CACHE_DIR: TRITON_CACHE_DIR.
  _lru_set: _lru_set().
  safe_zip: safe_zip.
  _load_triton_kernel_call_lib: _load_triton_kernel_call_lib().
  TRITON_CACHE_VERBOSE: TRITON_CACHE_VERBOSE.
  TRITON_CACHE_MAX_ITEMS: TRITON_CACHE_MAX_ITEMS.
  normalize_grid: normalize_grid().
  _array_device_set: _array_device_set().
  _assert_single_device_args: _assert_single_device_args().
  _get_cuda_compute_capability: _get_cuda_compute_capability().
  _triton_kernel_source_hash: _triton_kernel_source_hash().
  CompilationResult.ttgir: CompilationResult#ttgir.
  CompilationResult.llir: CompilationResult#llir.
  _COMPILED_KERNEL_CACHE._COMPILED_KERNEL_CACHE: _COMPILED_KERNEL_CACHE._COMPILED_KERNEL_CACHE.
  _has_triton_kernel_call_runtime: _has_triton_kernel_call_runtime().
  Grid: Grid.
  _parse_int_env: _parse_int_env().
  _device_set_from_sharding: _device_set_from_sharding().
  aval_size_bytes: aval_size_bytes().
  _normalize_cuda_compute_capability: _normalize_cuda_compute_capability().
  safe_map: safe_map.
  _JAX_TO_TRITON_TYPE_MAP: _JAX_TO_TRITON_TYPE_MAP.
  _has_multi_accelerators: _has_multi_accelerators().
  _in_shard_map_context: _in_shard_map_context().
  get_cuda_backend: get_cuda_backend().
  get_hip_backend: get_hip_backend().
  _lru_get: _lru_get().
  triton_kernel_call_lowering.unwrap_triton_kernel.prune_configs: triton_kernel_call_lowering().unwrap_triton_kernel().prune_configs().
  triton_kernel_call_raise_on_jvp: triton_kernel_call_raise_on_jvp().
  triton_kernel_call_raise_on_vmap: triton_kernel_call_raise_on_vmap().
  unsafe_zip: unsafe_zip.
  avals_to_layouts: avals_to_layouts().
  triton_kernel_call_abstract_eval: triton_kernel_call_abstract_eval().
  unsafe_map: unsafe_map.
---
# Module: [`ejkernel/callib/_triton_call.py`](../../../../../../raw/code/ejkernel/ejkernel/callib/_triton_call.py)

## Classes
### `CompilationResult`
- def: [`ejkernel/callib/_triton_call.py:590`](../../../../../../raw/code/ejkernel/ejkernel/callib/_triton_call.py#L590) — documented in [ejkernel-callib-_triton_call](../../../concepts/ejkernel-callib-_triton_call.md)
- doc: Result of Triton kernel compilation containing binary and metadata.
- signature: `class CompilationResult:`
- members:
  - `binary` — [`L602`](../../../../../../raw/code/ejkernel/ejkernel/callib/_triton_call.py#L602) — documented in [ejkernel-callib-_triton_call](../../../concepts/ejkernel-callib-_triton_call.md)
  - `cluster_dims` — [`L605`](../../../../../../raw/code/ejkernel/ejkernel/callib/_triton_call.py#L605) — documented in [ejkernel-callib-_triton_call](../../../concepts/ejkernel-callib-_triton_call.md)
  - `llir` — [`L607`](../../../../../../raw/code/ejkernel/ejkernel/callib/_triton_call.py#L607) — documented in [ejkernel-callib-_triton_call](../../../concepts/ejkernel-callib-_triton_call.md)
  - `name` — [`L603`](../../../../../../raw/code/ejkernel/ejkernel/callib/_triton_call.py#L603) — documented in [ejkernel-callib-_triton_call](../../../concepts/ejkernel-callib-_triton_call.md)
  - `shared_mem_bytes` — [`L604`](../../../../../../raw/code/ejkernel/ejkernel/callib/_triton_call.py#L604) — documented in [ejkernel-callib-_triton_call](../../../concepts/ejkernel-callib-_triton_call.md)
  - `ttgir` — [`L606`](../../../../../../raw/code/ejkernel/ejkernel/callib/_triton_call.py#L606) — documented in [ejkernel-callib-_triton_call](../../../concepts/ejkernel-callib-_triton_call.md)
- used by: [`get_or_create_triton_kernel`](_triton_call.md#get_or_create_triton_kernel), [`_load_triton_kernel_cache`](_triton_call.md#_load_triton_kernel_cache), [`compile_ttir_to_hsaco_inplace`](_triton_call.md#compile_ttir_to_hsaco_inplace), [`compile_ttir_to_ptx_inplace`](_triton_call.md#compile_ttir_to_ptx_inplace), [`_save_triton_kernel_cache`](_triton_call.md#_save_triton_kernel_cache)

## Functions
- `_array_device_set(arg: Any)` — [`L290`](../../../../../../raw/code/ejkernel/ejkernel/callib/_triton_call.py#L290) — Return the device set for a JAX array or tracer.
- `_assert_single_device_args(array_args: Sequence[Any], device_index: int | None, *, allow_sharded_tracers: bool)` — [`L336`](../../../../../../raw/code/ejkernel/ejkernel/callib/_triton_call.py#L336) — Validate that all array arguments are on one logical device.
- `_device_set_from_sharding(sharding)` — [`L265`](../../../../../../raw/code/ejkernel/ejkernel/callib/_triton_call.py#L265) — Extract the participating devices from a sharding object.
- `_get_cuda_compute_capability(device: int)` — [`L534`](../../../../../../raw/code/ejkernel/ejkernel/callib/_triton_call.py#L534) — documented in [ejkernel-callib-_triton_call](../../../concepts/ejkernel-callib-_triton_call.md)
- `_get_triton_cache_dir()` — [`L788`](../../../../../../raw/code/ejkernel/ejkernel/callib/_triton_call.py#L788) — Return the Triton kernel cache directory, creating it if needed.
- `_has_multi_accelerators()` — [`L400`](../../../../../../raw/code/ejkernel/ejkernel/callib/_triton_call.py#L400) — Check whether more than one non-CPU accelerator is available.
- `_has_triton_kernel_call_runtime(module: Any)` — [`L109`](../../../../../../raw/code/ejkernel/ejkernel/callib/_triton_call.py#L109)
- `_in_shard_map_context()` — [`L415`](../../../../../../raw/code/ejkernel/ejkernel/callib/_triton_call.py#L415) — Check whether execution is currently inside a shard_map context.
- `_invalidate_triton_cache_entry(cache_key: str, reason: str)` — [`L872`](../../../../../../raw/code/ejkernel/ejkernel/callib/_triton_call.py#L872) — Remove a Triton kernel cache entry from disk.
- `_load_triton_kernel_cache(cache_key: str, *, platform: str, compute_capability: int)` — [`L885`](../../../../../../raw/code/ejkernel/ejkernel/callib/_triton_call.py#L885) — Load a compiled Triton kernel from the on-disk cache. — documented in [ejkernel-callib-_triton_call](../../../concepts/ejkernel-callib-_triton_call.md)
- `_load_triton_kernel_call_lib(module: Any)` — [`L123`](../../../../../../raw/code/ejkernel/ejkernel/callib/_triton_call.py#L123)
- `_log_triton_cache(msg: str)` — [`L778`](../../../../../../raw/code/ejkernel/ejkernel/callib/_triton_call.py#L778) — Print a cache diagnostic message when verbose logging is enabled.
- `_lru_get(cache: OrderedDict, key: str)` — [`L985`](../../../../../../raw/code/ejkernel/ejkernel/callib/_triton_call.py#L985) — Retrieve a value from an LRU OrderedDict cache, promoting it on hit.
- `_lru_set(cache: OrderedDict, key: str, value)` — [`L1001`](../../../../../../raw/code/ejkernel/ejkernel/callib/_triton_call.py#L1001) — Insert or update a value in an LRU OrderedDict cache.
- `_make_triton_cache_key(*, fn: triton.JITFunction, platform: str, compute_capability: int, signature: tuple[tuple[str, str], ...], specialization: tuple[str, ...], constants: tuple[tuple[str, Any], ...], num_warps: int, num_stages: int, num_ctas: int, enable_fp_fusion: bool)` — [`L818`](../../../../../../raw/code/ejkernel/ejkernel/callib/_triton_call.py#L818) — Build a deterministic SHA-256 cache key for a Triton kernel compilation.
- `_normalize_cuda_compute_capability(value: Any)` — [`L516`](../../../../../../raw/code/ejkernel/ejkernel/callib/_triton_call.py#L516)
- `_parse_int_env(name: str, default: int)` — [`L193`](../../../../../../raw/code/ejkernel/ejkernel/callib/_triton_call.py#L193) — Parse an integer environment variable with a fallback default.
- `_save_triton_kernel_cache(cache_key: str, compilation_result: CompilationResult, *, platform: str, compute_capability: int, ttir: str | None)` — [`L936`](../../../../../../raw/code/ejkernel/ejkernel/callib/_triton_call.py#L936) — Persist a compiled Triton kernel to the on-disk cache.
- `_triton_kernel_source_hash(fn: triton.JITFunction)` — [`L798`](../../../../../../raw/code/ejkernel/ejkernel/callib/_triton_call.py#L798) — Compute a SHA-256 hash of a Triton JIT function's source code.
- `aval_size_bytes(aval)` — [`L504`](../../../../../../raw/code/ejkernel/ejkernel/callib/_triton_call.py#L504) — Calculate size in bytes for an abstract value.
- `avals_to_layouts(avals)` — [`L253`](../../../../../../raw/code/ejkernel/ejkernel/callib/_triton_call.py#L253) — Convert abstract values to layout specifications.
- `compile_ttir_inplace(ttir, backend: [cb.CUDABackend | hb.HIPBackend], options: [cb.CUDAOptions | hb.HIPOptions], compute_capability, platform)` — [`L610`](../../../../../../raw/code/ejkernel/ejkernel/callib/_triton_call.py#L610) — Compile Triton IR to platform-specific binary in-place.
- `compile_ttir_to_hsaco_inplace(ttir, hip_backend: hb.HIPBackend, hip_options: hb.HIPOptions, compute_capability)` — [`L720`](../../../../../../raw/code/ejkernel/ejkernel/callib/_triton_call.py#L720) — Compile Triton IR to HSACO binary for AMD ROCm devices.
- `compile_ttir_to_ptx_inplace(ttir, cuda_backend: cb.CUDABackend, cuda_options: cb.CUDAOptions, compute_capability)` — [`L651`](../../../../../../raw/code/ejkernel/ejkernel/callib/_triton_call.py#L651) — Compile Triton IR to PTX binary for CUDA devices.
- `get_cuda_backend(device, compute_capability)` — [`L557`](../../../../../../raw/code/ejkernel/ejkernel/callib/_triton_call.py#L557) — Create CUDA backend for Triton compilation.
- `get_hip_backend(device, compute_capability)` — [`L572`](../../../../../../raw/code/ejkernel/ejkernel/callib/_triton_call.py#L572) — Create HIP backend for Triton compilation on AMD GPUs.
- `get_or_create_triton_kernel(backend_init_func, platform, fn, arg_dtypes, scalar_args, device, *, num_warps, num_stages, num_ctas, compute_capability, enable_fp_fusion, metaparams, dump: bool)` — [`L1018`](../../../../../../raw/code/ejkernel/ejkernel/callib/_triton_call.py#L1018) — Get or create a compiled Triton kernel with caching. — documented in [ejkernel-callib-_triton_call](../../../concepts/ejkernel-callib-_triton_call.md)
- `get_triton_type(obj: Any)` — [`L447`](../../../../../../raw/code/ejkernel/ejkernel/callib/_triton_call.py#L447) — Get Triton type string representation for a given object.
- `normalize_grid(grid: GridOrLambda, metaparams)` — [`L231`](../../../../../../raw/code/ejkernel/ejkernel/callib/_triton_call.py#L231) — Normalize grid specification to a 3D tuple. — documented in [ejkernel-callib-_triton_call](../../../concepts/ejkernel-callib-_triton_call.md)
- `prune_configs(configs, named_args, **kwargs)` — [`L1337`](../../../../../../raw/code/ejkernel/ejkernel/callib/_triton_call.py#L1337) — Filter autotuner configs to those compatible with the current metaparams.
- `triton_call(*args: jax.Array | bool | int | float | np.float32, kernel: triton.JITFunction | triton.runtime.Heuristics | triton.runtime.Autotuner, out_shape: ShapeDtype | Sequence[ShapeDtype], grid: GridOrLambda, name: str = "", custom_call_target_name: str = "triton_kernel_call", num_warps: int | None = None, num_stages: int | None = None, num_ctas: int = 1, device: int = 0, compute_capability: int | None = None, enable_fp_fusion: bool = True, input_output_aliases: dict[int, int] | None = None, zeroed_outputs: Sequence[int] | Callable[[dict[str, Any]], Sequence[int]] = (), debug: bool = False, serialized_metadata: bytes = b"", **metaparams: Any)` — [`L1538`](../../../../../../raw/code/ejkernel/ejkernel/callib/_triton_call.py#L1538) — Call a Triton kernel from JAX with specified parameters. — documented in [ejkernel-callib-_triton_call](../../../concepts/ejkernel-callib-_triton_call.md)
- `triton_kernel_call_abstract_eval(*_, out_shapes, **__)` — [`L490`](../../../../../../raw/code/ejkernel/ejkernel/callib/_triton_call.py#L490) — Abstract evaluation function for triton kernel call primitive.
- `triton_kernel_call_lowering(backend_init_func, ctx, *array_args, fn, scalar_args, name, custom_call_target_name, out_shapes, grid, num_warps, num_stages, num_ctas, device, compute_capability, enable_fp_fusion, input_output_aliases, zeroed_outputs, debug, serialized_metadata, **metaparams)` — [`L1242`](../../../../../../raw/code/ejkernel/ejkernel/callib/_triton_call.py#L1242) — Lower Triton kernel call to platform-specific implementation. — documented in [ejkernel-callib-_triton_call](../../../concepts/ejkernel-callib-_triton_call.md)
- `triton_kernel_call_raise_on_jvp(*args, **kwargs)` — [`L1496`](../../../../../../raw/code/ejkernel/ejkernel/callib/_triton_call.py#L1496) — Raise error for automatic differentiation on Triton kernels.
- `triton_kernel_call_raise_on_vmap(*args, **kwargs)` — [`L1517`](../../../../../../raw/code/ejkernel/ejkernel/callib/_triton_call.py#L1517) — Raise error for batching with vmap on Triton kernels.
- `unwrap_triton_kernel(kernel, configs)` — [`L1311`](../../../../../../raw/code/ejkernel/ejkernel/callib/_triton_call.py#L1311) — Recursively unwrap Triton kernel wrappers to the underlying JIT function.

## Module values
- `CAN_USE_TRITON` — [`L144`](../../../../../../raw/code/ejkernel/ejkernel/callib/_triton_call.py#L144) — documented in [ejkernel-callib-_triton_call](../../../concepts/ejkernel-callib-_triton_call.md)
- `Grid` — [`L189`](../../../../../../raw/code/ejkernel/ejkernel/callib/_triton_call.py#L189) — documented in [ejkernel-callib-_triton_call](../../../concepts/ejkernel-callib-_triton_call.md)
- `GridOrLambda` — [`L190`](../../../../../../raw/code/ejkernel/ejkernel/callib/_triton_call.py#L190) — documented in [ejkernel-callib-_triton_call](../../../concepts/ejkernel-callib-_triton_call.md)
- `TRITON_CACHE_DIR` — [`L223`](../../../../../../raw/code/ejkernel/ejkernel/callib/_triton_call.py#L223)
- `TRITON_CACHE_ENABLED` — [`L211`](../../../../../../raw/code/ejkernel/ejkernel/callib/_triton_call.py#L211)
- `TRITON_CACHE_MAX_ITEMS` — [`L219`](../../../../../../raw/code/ejkernel/ejkernel/callib/_triton_call.py#L219)
- `TRITON_CACHE_VERBOSE` — [`L215`](../../../../../../raw/code/ejkernel/ejkernel/callib/_triton_call.py#L215)
- `_COMPILED_KERNEL_CACHE` — [`L982`](../../../../../../raw/code/ejkernel/ejkernel/callib/_triton_call.py#L982)
- `_JAX_TO_TRITON_TYPE_MAP` — [`L169`](../../../../../../raw/code/ejkernel/ejkernel/callib/_triton_call.py#L169)
- `safe_map` — [`L165`](../../../../../../raw/code/ejkernel/ejkernel/callib/_triton_call.py#L165)
- `safe_zip` — [`L166`](../../../../../../raw/code/ejkernel/ejkernel/callib/_triton_call.py#L166)
- `triton_kernel_call_p` — [`L484`](../../../../../../raw/code/ejkernel/ejkernel/callib/_triton_call.py#L484)
- `unsafe_map` — [`L165`](../../../../../../raw/code/ejkernel/ejkernel/callib/_triton_call.py#L165)
- `unsafe_zip` — [`L166`](../../../../../../raw/code/ejkernel/ejkernel/callib/_triton_call.py#L166)

