---
title: 'Module: jax/_src/pallas/triton/primitives.py'
type: catalog
provenance: extracted
module: jax/_src/pallas/triton/primitives.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax._src.pallas.triton.primitives`/
symbols:
  _atomic_lowering_rule: _atomic_lowering_rule().
  load: load().
  _atomic_rmw_discharge_rule: _atomic_rmw_discharge_rule().
  _approx_tanh_rocm_lowering: _approx_tanh_rocm_lowering().
  _elementwise_inline_asm_lowering: _elementwise_inline_asm_lowering().
  store: store().
  AtomicOpType: AtomicOpType#
  approx_tanh: approx_tanh().
  _atomic_rmw: _atomic_rmw().
  _expand_atomic_fp_min_max: _expand_atomic_fp_min_max().
  _atomic_abstract_eval: _atomic_abstract_eval().
  _max_contiguous_rule: _max_contiguous_rule().
  elementwise_inline_asm_p: elementwise_inline_asm_p.
  _atomic_cas_lowering_rule: _atomic_cas_lowering_rule().
  max_contiguous_p: max_contiguous_p.
  elementwise_inline_asm: elementwise_inline_asm().
  debug_barrier_p: debug_barrier_p.
  atomic_cas_p: atomic_cas_p.
  atomic_xchg: atomic_xchg().
  atomic_add: atomic_add().
  atomic_max: atomic_max().
  atomic_min: atomic_min().
  atomic_and: atomic_and().
  atomic_or: atomic_or().
  atomic_xor: atomic_xor().
  Slice: Slice.
  _debug_barrier_abstract_eval: _debug_barrier_abstract_eval().
  _debug_barrier_lowering: _debug_barrier_lowering().
  _atomic_cas_discharge_rule: _atomic_cas_discharge_rule().
  Ref.Ref: Ref.Ref.
  atomic_rmw_p: atomic_rmw_p.
  debug_barrier: debug_barrier().
  barrier_effect: barrier_effect.
  atomic_cas: atomic_cas().
  max_contiguous: max_contiguous().
  BarrierEffect: BarrierEffect#
  _elementwise_inline_asm_abstract_eval: _elementwise_inline_asm_abstract_eval().
  _atomic_cas_abstract_eval: _atomic_cas_abstract_eval().
  _max_contiguous_abstract_eval: _max_contiguous_abstract_eval().
  AtomicOpType.ADD: AtomicOpType#ADD.
  AtomicOpType.MAX: AtomicOpType#MAX.
  AtomicOpType.MIN: AtomicOpType#MIN.
  AtomicOpType.XCHG: AtomicOpType#XCHG.
  AtomicOpType.AND: AtomicOpType#AND.
  AtomicOpType.OR: AtomicOpType#OR.
  AtomicOpType.XOR: AtomicOpType#XOR.
---
# Module: [`jax/_src/pallas/triton/primitives.py`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/primitives.py)

## Classes
### `AtomicOpType`  ·  implements/extends Enum
- def: [`jax/_src/pallas/triton/primitives.py:299`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/primitives.py#L299)
- signature: `class AtomicOpType(enum.Enum):`
- members:
  - `ADD` — [`L301`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/primitives.py#L301)
  - `AND` — [`L304`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/primitives.py#L304)
  - `MAX` — [`L302`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/primitives.py#L302)
  - `MIN` — [`L303`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/primitives.py#L303)
  - `OR` — [`L305`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/primitives.py#L305)
  - `XCHG` — [`L300`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/primitives.py#L300)
  - `XOR` — [`L306`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/primitives.py#L306)
- used by: [`_atomic_lowering_rule`](primitives.md#_atomic_lowering_rule), [`_atomic_rmw_discharge_rule`](primitives.md#_atomic_rmw_discharge_rule), [`_atomic_rmw`](primitives.md#_atomic_rmw), [`_expand_atomic_fp_min_max`](primitives.md#_expand_atomic_fp_min_max), [`_atomic_abstract_eval`](primitives.md#_atomic_abstract_eval), [`atomic_add`](primitives.md#atomic_add), [`atomic_and`](primitives.md#atomic_and), [`atomic_max`](primitives.md#atomic_max), [`atomic_min`](primitives.md#atomic_min), [`atomic_or`](primitives.md#atomic_or), [`atomic_xchg`](primitives.md#atomic_xchg), [`atomic_xor`](primitives.md#atomic_xor)

### `BarrierEffect`  ·  implements/extends Effect
- def: [`jax/_src/pallas/triton/primitives.py:219`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/primitives.py#L219)
- signature: `class BarrierEffect(jax_core.Effect):`
- uses (calls/refs, reference-scoped): [`Effect`](../../core.md#Effect)
- used by: [`barrier_effect`](primitives.md#barrier_effect)

## Functions
- `_approx_tanh_rocm_lowering(ctx: lowering.LoweringRuleContext, *args)` — [`L162`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/primitives.py#L162) — Lower approx_tanh for ROCm.
- `_atomic_abstract_eval(*avals_flat, args_tree, atomic_type: AtomicOpType)` — [`L360`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/primitives.py#L360)
- `_atomic_cas_abstract_eval(ref_aval, cmp_aval, val_aval)` — [`L620`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/primitives.py#L620)
- `_atomic_cas_discharge_rule(in_avals, out_avals, ref, cmp, val)` — [`L651`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/primitives.py#L651)
- `_atomic_cas_lowering_rule(ctx: lowering.LoweringRuleContext, ptr, cmp, val)` — [`L658`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/primitives.py#L658)
- `_atomic_lowering_rule(ctx: lowering.LoweringRuleContext, *args_flat, args_tree, atomic_type: AtomicOpType)` — [`L446`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/primitives.py#L446)
- `_atomic_rmw(x_ref_or_view, idx, val, *, mask: Any | None = None, atomic_type: AtomicOpType)` — [`L376`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/primitives.py#L376)
- `_atomic_rmw_discharge_rule(in_avals, out_avals, *args_flat, args_tree, atomic_type: AtomicOpType)` — [`L312`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/primitives.py#L312)
- `_debug_barrier_abstract_eval()` — [`L233`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/primitives.py#L233)
- `_debug_barrier_lowering(ctx: lowering.LoweringRuleContext)` — [`L238`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/primitives.py#L238)
- `_elementwise_inline_asm_abstract_eval(*avals: jax_core.ShapedArray, result_shape_dtypes, **kwargs)` — [`L119`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/primitives.py#L119)
- `_elementwise_inline_asm_lowering(ctx: lowering.LoweringRuleContext, *args, asm, constraints, pack, result_shape_dtypes)` — [`L131`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/primitives.py#L131)
- `_expand_atomic_fp_min_max(atomic_type: AtomicOpType, ptr: ir.Value, val: ir.Value, mask: ir.Value | None = None, semantic: tt_dialect.MemSemantic = tt_dialect.MemSemantic.ACQUIRE_RELEASE, sync_scope: tt_dialect.MemSyncScope = tt_dialect.MemSyncScope.GPU)` — [`L392`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/primitives.py#L392) — Expands floating point min/max via sequence of integer min/max. Does not handle NaNs.
- `_max_contiguous_abstract_eval(aval, **_)` — [`L692`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/primitives.py#L692)
- `_max_contiguous_rule(ctx: lowering.LoweringRuleContext, x, values: Sequence[int])` — [`L697`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/primitives.py#L697)
- `approx_tanh(x: jax.Array)` — [`L49`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/primitives.py#L49) — Elementwise approximate hyperbolic tangent: :math:`\mathrm{tanh}(x)`.
- `atomic_add(x_ref_or_view, idx, val, *, mask: Any | None = None)` — [`L520`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/primitives.py#L520) — Atomically computes ``x_ref_or_view[idx] += val``.
- `atomic_and(x_ref_or_view, idx, val, *, mask: Any | None = None)` — [`L568`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/primitives.py#L568) — Atomically computes ``x_ref_or_view[idx] &= val``.
- `atomic_cas(ref, cmp, val)` — [`L634`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/primitives.py#L634) — Performs an atomic compare-and-swap of the value in the ref with the
- `atomic_max(x_ref_or_view, idx, val, *, mask: Any | None = None)` — [`L536`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/primitives.py#L536) — Atomically computes ``x_ref_or_view[idx] = max(x_ref_or_view[idx], val)``.
- `atomic_min(x_ref_or_view, idx, val, *, mask: Any | None = None)` — [`L552`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/primitives.py#L552) — Atomically computes ``x_ref_or_view[idx] = min(x_ref_or_view[idx], val)``.
- `atomic_or(x_ref_or_view, idx, val, *, mask: Any | None = None)` — [`L584`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/primitives.py#L584) — Atomically computes ``x_ref_or_view[idx] |= val``.
- `atomic_xchg(x_ref_or_view, idx, val, *, mask: Any | None = None)` — [`L504`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/primitives.py#L504) — Atomically exchanges the given value with the value at the given index.
- `atomic_xor(x_ref_or_view, idx, val, *, mask: Any | None = None)` — [`L600`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/primitives.py#L600) — Atomically computes ``x_ref_or_view[idx] ^= val``.
- `debug_barrier()` — [`L214`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/primitives.py#L214) — Synchronizes all kernel executions in the grid.
- `elementwise_inline_asm(asm: str, *, args: Sequence[jax.Array], constraints: str, pack: int, result_shape_dtypes: Sequence[jax.ShapeDtypeStruct])` — [`L82`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/primitives.py#L82) — Inline assembly applying an elementwise operation.
- `load(ref: Ref, *, mask: jax.Array | None = None, other: jax.typing.ArrayLike | None = None, cache_modifier: str | None = None, eviction_policy: str | None = None, volatile: bool = False)` — [`L244`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/primitives.py#L244) — Loads an array from the given ref.
- `max_contiguous(x, values)` — [`L684`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/primitives.py#L684) — A compiler hint that asserts the ``values`` first values of ``x`` are contiguous.
- `store(ref: Ref, val: jax.Array, *, mask: jax.Array | None = None, eviction_policy: str | None = None)` — [`L279`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/primitives.py#L279) — Stores a value to the given ref.

## Module values
- `Ref` — [`L44`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/primitives.py#L44)
- `Slice` — [`L46`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/primitives.py#L46)
- `atomic_cas_p` — [`L616`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/primitives.py#L616)
- `atomic_rmw_p` — [`L309`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/primitives.py#L309)
- `barrier_effect` — [`L222`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/primitives.py#L222)
- `debug_barrier_p` — [`L228`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/primitives.py#L228)
- `elementwise_inline_asm_p` — [`L114`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/primitives.py#L114)
- `max_contiguous_p` — [`L678`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/primitives.py#L678)

