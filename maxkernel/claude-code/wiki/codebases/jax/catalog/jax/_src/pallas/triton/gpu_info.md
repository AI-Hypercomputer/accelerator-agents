---
title: 'Module: jax/_src/pallas/triton/gpu_info.py'
type: catalog
provenance: extracted
module: jax/_src/pallas/triton/gpu_info.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax._src.pallas.triton.gpu_info`/
symbols:
  _get_gpu_info_impl: _get_gpu_info_impl().
  GpuVersion: GpuVersion#
  get_gpu_info: get_gpu_info().
  GpuInfo: GpuInfo#
  GpuInfo.gpu_version: GpuInfo#gpu_version.
  get_gpu_info_from_version: get_gpu_info_from_version().
  GpuInfo.arch_name: GpuInfo#arch_name.
  GpuInfo.compute_capability: GpuInfo#compute_capability.
  gpu_version_from_device_kind: gpu_version_from_device_kind().
  get_device_kind: get_device_kind().
  is_gpu_device: is_gpu_device().
  registry.registry: registry.registry.
  _GPU_VERSION_RE: _GPU_VERSION_RE.
  GpuVersion.A10: GpuVersion#A10.
  GpuVersion.A30: GpuVersion#A30.
  GpuVersion.A100: GpuVersion#A100.
  GpuVersion.H100: GpuVersion#H100.
  GpuVersion.H200: GpuVersion#H200.
  GpuVersion.GH200: GpuVersion#GH200.
  GpuVersion.B200: GpuVersion#B200.
  GpuVersion.GB200: GpuVersion#GB200.
  GpuVersion.B300: GpuVersion#B300.
  GpuVersion.GB300: GpuVersion#GB300.
  GpuVersion.GB10: GpuVersion#GB10.
  GpuVersion.L4: GpuVersion#L4.
  GpuVersion.L40: GpuVersion#L40.
  GpuVersion.T4: GpuVersion#T4.
  GpuVersion.RTX_4090: GpuVersion#RTX_4090.
  GpuVersion.RTX_PRO_4500: GpuVersion#RTX_PRO_4500.
  GpuVersion.RTX_PRO_5000: GpuVersion#RTX_PRO_5000.
  GpuVersion.RTX_PRO_6000: GpuVersion#RTX_PRO_6000.
  GpuVersion.__str__: GpuVersion#__str__().
---
# Module: [`jax/_src/pallas/triton/gpu_info.py`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/gpu_info.py)

## Classes
### `GpuInfo`
- def: [`jax/_src/pallas/triton/gpu_info.py:63`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/gpu_info.py#L63)
- doc: GPU hardware information
- signature: `class GpuInfo:`
- members:
  - `arch_name` — [`L66`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/gpu_info.py#L66)
  - `compute_capability` — [`L67`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/gpu_info.py#L67)
  - `gpu_version` — [`L65`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/gpu_info.py#L65)
- uses (calls/refs, reference-scoped): [`GpuVersion`](gpu_info.md#GpuVersion)
- used by: [`pallas_call_lowering`](pallas_call_registration.md#pallas_call_lowering), [`_get_gpu_info_impl`](gpu_info.md#_get_gpu_info_impl), [`get_gpu_info`](gpu_info.md#get_gpu_info), [`get_gpu_info_from_version`](gpu_info.md#get_gpu_info_from_version), [`registry`](gpu_info.md#registry.registry)

### `GpuVersion`  ·  implements/extends Enum
- def: [`jax/_src/pallas/triton/gpu_info.py:27`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/gpu_info.py#L27)
- doc: NVidia GPU version
- signature: `class GpuVersion(enum.Enum):`
- members:
  - `A10` — [`L30`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/gpu_info.py#L30)
  - `A100` — [`L32`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/gpu_info.py#L32)
  - `A30` — [`L31`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/gpu_info.py#L31)
  - `B200` — [`L36`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/gpu_info.py#L36)
  - `B300` — [`L38`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/gpu_info.py#L38)
  - `GB10` — [`L40`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/gpu_info.py#L40)
  - `GB200` — [`L37`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/gpu_info.py#L37)
  - `GB300` — [`L39`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/gpu_info.py#L39)
  - `GH200` — [`L35`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/gpu_info.py#L35)
  - `H100` — [`L33`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/gpu_info.py#L33)
  - `H200` — [`L34`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/gpu_info.py#L34)
  - `L4` — [`L41`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/gpu_info.py#L41)
  - `L40` — [`L42`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/gpu_info.py#L42)
  - `RTX_4090` — [`L44`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/gpu_info.py#L44)
  - `RTX_PRO_4500` — [`L45`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/gpu_info.py#L45)
  - `RTX_PRO_5000` — [`L46`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/gpu_info.py#L46)
  - `RTX_PRO_6000` — [`L47`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/gpu_info.py#L47)
  - `T4` — [`L43`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/gpu_info.py#L43)
- protocol/private: `__str__`[`L49`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/gpu_info.py#L49)
- used by: [`_get_gpu_info_impl`](gpu_info.md#_get_gpu_info_impl), [`gpu_version`](gpu_info.md#GpuInfo.gpu_version), [`get_gpu_info_from_version`](gpu_info.md#get_gpu_info_from_version), [`gpu_version_from_device_kind`](gpu_info.md#gpu_version_from_device_kind), [`_GPU_VERSION_RE`](gpu_info.md#_GPU_VERSION_RE)

## Functions
- `_get_gpu_info_impl(gpu_version: GpuVersion)` — [`L77`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/gpu_info.py#L77) — Returns the GPU hardware info for the given its version.
- `get_device_kind()` — [`L167`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/gpu_info.py#L167)
- `get_gpu_info()` — [`L144`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/gpu_info.py#L144) — Returns the GPU hardware info for the current device.
- `get_gpu_info_from_version(gpu_version: GpuVersion)` — [`L156`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/gpu_info.py#L156) — Returns the GPU hardware info for the given GPU version.
- `gpu_version_from_device_kind(device_kind: str)` — [`L56`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/gpu_info.py#L56)
- `is_gpu_device()` — [`L70`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/gpu_info.py#L70)

## Module values
- `_GPU_VERSION_RE` — [`L53`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/gpu_info.py#L53)
- `registry` — [`L74`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/gpu_info.py#L74)

