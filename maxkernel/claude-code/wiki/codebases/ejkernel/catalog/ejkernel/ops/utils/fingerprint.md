---
title: 'Module: ejkernel/ops/utils/fingerprint.py'
type: catalog
provenance: extracted
module: ejkernel/ops/utils/fingerprint.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.ops.utils.fingerprint`/
symbols:
  abstractify: abstractify().
  device_fingerprint: device_fingerprint().
  get_device_platform: get_device_platform().
  default_key_builder_with_sharding: default_key_builder_with_sharding().
  short_hash: short_hash().
  stable_json: stable_json().
  stable_json.default: stable_json().default().
  sharding_fingerprint: sharding_fingerprint().
  device_kind: device_kind().
  abstractify.leaf: abstractify().leaf().
---
# Module: [`ejkernel/ops/utils/fingerprint.py`](../../../../../../../raw/code/ejkernel/ejkernel/ops/utils/fingerprint.py)

## Functions
- `abstractify(pytree: Any)` — [`L311`](../../../../../../../raw/code/ejkernel/ejkernel/ops/utils/fingerprint.py#L311) — Replace all arrays in a PyTree with abstract shape/dtype specifications.
- `default(o)` — [`L245`](../../../../../../../raw/code/ejkernel/ejkernel/ops/utils/fingerprint.py#L245) — JSON serialization fallback for non-standard types including JAX arrays and callables.
- `default_key_builder_with_sharding(inv)` — [`L93`](../../../../../../../raw/code/ejkernel/ejkernel/ops/utils/fingerprint.py#L93) — Generate cache key that includes sharding information for device-aware caching.
- `device_fingerprint(dev: jax.Device | None = None)` — [`L124`](../../../../../../../raw/code/ejkernel/ejkernel/ops/utils/fingerprint.py#L124) — Generate a stable identifier for a JAX device including platform version.
- `device_kind()` — [`L194`](../../../../../../../raw/code/ejkernel/ejkernel/ops/utils/fingerprint.py#L194) — Get the device kind (gpu, cpu, tpu) for the default device.
- `get_device_platform(dev: jax.Device | None = None)` — [`L155`](../../../../../../../raw/code/ejkernel/ejkernel/ops/utils/fingerprint.py#L155) — Extract the platform identifier (gpu/tpu/cpu) from a JAX device.
- `leaf(x)` — [`L347`](../../../../../../../raw/code/ejkernel/ejkernel/ops/utils/fingerprint.py#L347) — Convert a single array to ShapeDtypeStruct or pass through non-array values.
- `sharding_fingerprint(x: Any)` — [`L66`](../../../../../../../raw/code/ejkernel/ejkernel/ops/utils/fingerprint.py#L66) — Extract sharding information from a JAX array for fingerprinting.
- `short_hash(obj: Any)` — [`L286`](../../../../../../../raw/code/ejkernel/ejkernel/ops/utils/fingerprint.py#L286) — Generate a short (16-character) hash from an object using stable JSON serialization.
- `stable_json(obj: Any)` — [`L215`](../../../../../../../raw/code/ejkernel/ejkernel/ops/utils/fingerprint.py#L215) — Deterministic JSON serialization that handles JAX/NumPy types and dataclasses.

