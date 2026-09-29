---
title: 'Module: jax/experimental/key_reuse/_core.py'
type: catalog
provenance: extracted
module: jax/experimental/key_reuse/_core.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax.experimental.key_reuse._core`/
symbols:
  key_reuse_signatures.key_reuse_signatures: key_reuse_signatures.key_reuse_signatures.
  jaxpr_type_signature: jaxpr_type_signature().
  KeyReuseSignature: KeyReuseSignature#
  _SourceSinkBase.idx: _SourceSinkBase#idx.
  _cond_key_type_signature: _cond_key_type_signature().
  Forward: Forward#
  _SourceSinkBase.mask: _SourceSinkBase#mask.
  _slice_signature: _slice_signature().
  call_impl_with_key_reuse_checks: call_impl_with_key_reuse_checks().
  KeyReuseSignature.update_consumption: KeyReuseSignature#update_consumption().
  assert_consumed_value_p: assert_consumed_value_p.
  key_reuse_signature_from_eqn: key_reuse_signature_from_eqn().
  key_reuse_signature_from_primitive: key_reuse_signature_from_primitive().
  consume_p: consume_p.
  _scan_key_type_signature: _scan_key_type_signature().
  _while_key_type_signature: _while_key_type_signature().
  Sink: Sink#
  dynamic_key_reuse_signature: dynamic_key_reuse_signature().
  function_type_signature: function_type_signature().
  Source: Source#
  KeyReuseSignature._args: KeyReuseSignature#_args.
  KeyReuseSignature.sinks: KeyReuseSignature#sinks().
  jaxpr_type_signature.is_key: jaxpr_type_signature().is_key().
  KeyReuseSignature.check_signature: KeyReuseSignature#check_signature().
  _concatenate_signature: _concatenate_signature().
  key_reuse_error_with_source_traceback: key_reuse_error_with_source_traceback().
  unknown_signature.is_key: unknown_signature().is_key().
  _unstack_signature: _unstack_signature().
  KeyReuseSignature.sources: KeyReuseSignature#sources().
  DynamicKeyReuseSignature: DynamicKeyReuseSignature#
  jaxpr_type_signature.source: jaxpr_type_signature().source().
  check_key_reuse_jaxpr: check_key_reuse_jaxpr().
  unknown_signature: unknown_signature().
  jaxpr_type_signature.resolve_forwards: jaxpr_type_signature().resolve_forwards().
  jaxpr_type_signature.sink: jaxpr_type_signature().sink().
  _remat_key_type_signature: _remat_key_type_signature().
  _device_put_signature: _device_put_signature().
  _SourceSinkBase: _SourceSinkBase#
  KeyReuseSignature.forwards: KeyReuseSignature#forwards().
  _SourceSinkBase.__lt__: _SourceSinkBase#__lt__().
  assert_unconsumed: assert_unconsumed().
  assert_consumed: assert_consumed().
  jaxpr_type_signature.is_consumed: jaxpr_type_signature().is_consumed().
  _pjit_key_type_signature: _pjit_key_type_signature().
  _shard_map_type_signature: _shard_map_type_signature().
  _SourceSinkBase.__eq__: _SourceSinkBase#__eq__().
  _SourceSinkBase.__hash__: _SourceSinkBase#__hash__().
  _SourceSinkBase.__repr__: _SourceSinkBase#__repr__().
  Forward.__repr__: Forward#__repr__().
  KeyReuseSignature.__eq__: KeyReuseSignature#__eq__().
  consume: consume().
  KeyReuseSignature.__init__: KeyReuseSignature#__init__().
  KeyReuseSignature.__repr__: KeyReuseSignature#__repr__().
  KeyReuseSignature.__hash__: KeyReuseSignature#__hash__().
  check_key_reuse: check_key_reuse().
  Forward.in_idx: Forward#in_idx.
  Forward.out_idx: Forward#out_idx.
  _source_context_message: _source_context_message.
  DynamicKeyReuseSignature.signature: DynamicKeyReuseSignature#signature.
  consume_effect: consume_effect.
  assert_effect: assert_effect.
  _check_consumed_value: _check_consumed_value().
  _SourceSinkBase.__init__: _SourceSinkBase#__init__().
  _SourceSinkBase.__setattr__: _SourceSinkBase#__setattr__().
---
# Module: [`jax/experimental/key_reuse/_core.py`](../../../../../../../raw/code/jax/jax/experimental/key_reuse/_core.py)

## Classes
### `DynamicKeyReuseSignature`  ·  implements/extends NamedTuple
- def: [`jax/experimental/key_reuse/_core.py:204`](../../../../../../../raw/code/jax/jax/experimental/key_reuse/_core.py#L204)
- signature: `class DynamicKeyReuseSignature(NamedTuple):`
- members:
  - `signature` — [`L205`](../../../../../../../raw/code/jax/jax/experimental/key_reuse/_core.py#L205)
- uses (calls/refs, reference-scoped): [`JaxprEqn`](../../_src/core.md#JaxprEqn), [`KeyReuseSignature`](_core.md#KeyReuseSignature)
- used by: [`key_reuse_signatures`](_core.md#key_reuse_signatures.key_reuse_signatures), [`key_reuse_signature_from_eqn`](_core.md#key_reuse_signature_from_eqn), [`key_reuse_signature_from_primitive`](_core.md#key_reuse_signature_from_primitive), [`dynamic_key_reuse_signature`](_core.md#dynamic_key_reuse_signature)

### `Forward`
- def: [`jax/experimental/key_reuse/_core.py:135`](../../../../../../../raw/code/jax/jax/experimental/key_reuse/_core.py#L135)
- members:
  - `in_idx` — [`L136`](../../../../../../../raw/code/jax/jax/experimental/key_reuse/_core.py#L136)
  - `out_idx` — [`L137`](../../../../../../../raw/code/jax/jax/experimental/key_reuse/_core.py#L137)
- protocol/private: `__repr__`[`L139`](../../../../../../../raw/code/jax/jax/experimental/key_reuse/_core.py#L139)
- used by: [`key_reuse_signatures`](_core.md#key_reuse_signatures.key_reuse_signatures), [`jaxpr_type_signature`](_core.md#jaxpr_type_signature), [`_cond_key_type_signature`](_core.md#_cond_key_type_signature), [`_slice_signature`](_core.md#_slice_signature), [`update_consumption`](_core.md#KeyReuseSignature.update_consumption), [`_args`](_core.md#KeyReuseSignature._args), [`_concatenate_signature`](_core.md#_concatenate_signature), [`_device_put_signature`](_core.md#_device_put_signature), [`forwards`](_core.md#KeyReuseSignature.forwards), [`__lt__`](_core.md#_SourceSinkBase.__lt__)

### `KeyReuseSignature`
- def: [`jax/experimental/key_reuse/_core.py:145`](../../../../../../../raw/code/jax/jax/experimental/key_reuse/_core.py#L145)
- signature: `class KeyReuseSignature:`
- members:
  - `check_signature(self, *args, funcname="function", context=None)` — [`L172`](../../../../../../../raw/code/jax/jax/experimental/key_reuse/_core.py#L172)
  - `forwards(self)` — [`L169`](../../../../../../../raw/code/jax/jax/experimental/key_reuse/_core.py#L169)
  - `sinks(self)` — [`L161`](../../../../../../../raw/code/jax/jax/experimental/key_reuse/_core.py#L161)
  - `sources(self)` — [`L165`](../../../../../../../raw/code/jax/jax/experimental/key_reuse/_core.py#L165)
  - `update_consumption(self, args_in, args_out)` — [`L184`](../../../../../../../raw/code/jax/jax/experimental/key_reuse/_core.py#L184)
- protocol/private: `__eq__`[`L154`](../../../../../../../raw/code/jax/jax/experimental/key_reuse/_core.py#L154), `__hash__`[`L157`](../../../../../../../raw/code/jax/jax/experimental/key_reuse/_core.py#L157), `__init__`[`L148`](../../../../../../../raw/code/jax/jax/experimental/key_reuse/_core.py#L148), `__repr__`[`L151`](../../../../../../../raw/code/jax/jax/experimental/key_reuse/_core.py#L151), `_args`[`L146`](../../../../../../../raw/code/jax/jax/experimental/key_reuse/_core.py#L146)
- uses (calls/refs, reference-scoped): [`idx`](_core.md#_SourceSinkBase.idx), [`current`](../../_src/source_info_util.md#current), [`Forward`](_core.md#Forward), [`traceback`](../../_src/source_info_util.md#SourceInfo.traceback), [`mask`](_core.md#_SourceSinkBase.mask), [`Sink`](_core.md#Sink), [`Source`](_core.md#Source), [`key_reuse_error_with_source_traceback`](_core.md#key_reuse_error_with_source_traceback), [`in_idx`](_core.md#Forward.in_idx), [`out_idx`](_core.md#Forward.out_idx)
- used by: [`key_reuse_signatures`](_core.md#key_reuse_signatures.key_reuse_signatures), [`jaxpr_type_signature`](_core.md#jaxpr_type_signature), [`_cond_key_type_signature`](_core.md#_cond_key_type_signature), [`_slice_signature`](_core.md#_slice_signature), [`call_impl_with_key_reuse_checks`](_core.md#call_impl_with_key_reuse_checks), [`_scan_key_type_signature`](_core.md#_scan_key_type_signature), [`_while_key_type_signature`](_core.md#_while_key_type_signature), [`key_reuse_signature_from_eqn`](_core.md#key_reuse_signature_from_eqn), [`key_reuse_signature_from_primitive`](_core.md#key_reuse_signature_from_primitive), [`dynamic_key_reuse_signature`](_core.md#dynamic_key_reuse_signature), [`function_type_signature`](_core.md#function_type_signature), [`_concatenate_signature`](_core.md#_concatenate_signature), [`_unstack_signature`](_core.md#_unstack_signature), [`DynamicKeyReuseSignature`](_core.md#DynamicKeyReuseSignature), [`_device_put_signature`](_core.md#_device_put_signature), [`_remat_key_type_signature`](_core.md#_remat_key_type_signature), [`unknown_signature`](_core.md#unknown_signature)

### `Sink`  ·  implements/extends _SourceSinkBase
- def: [`jax/experimental/key_reuse/_core.py:127`](../../../../../../../raw/code/jax/jax/experimental/key_reuse/_core.py#L127)
- signature: `class Sink(_SourceSinkBase):`
- uses (calls/refs, reference-scoped): [`_SourceSinkBase`](_core.md#_SourceSinkBase)
- used by: [`key_reuse_signatures`](_core.md#key_reuse_signatures.key_reuse_signatures), [`jaxpr_type_signature`](_core.md#jaxpr_type_signature), [`_cond_key_type_signature`](_core.md#_cond_key_type_signature), [`_slice_signature`](_core.md#_slice_signature), [`_args`](_core.md#KeyReuseSignature._args), [`sinks`](_core.md#KeyReuseSignature.sinks), [`_concatenate_signature`](_core.md#_concatenate_signature), [`_unstack_signature`](_core.md#_unstack_signature), [`unknown_signature`](_core.md#unknown_signature), [`_SourceSinkBase`](_core.md#_SourceSinkBase)

### `Source`  ·  implements/extends _SourceSinkBase
- def: [`jax/experimental/key_reuse/_core.py:131`](../../../../../../../raw/code/jax/jax/experimental/key_reuse/_core.py#L131)
- signature: `class Source(_SourceSinkBase):`
- uses (calls/refs, reference-scoped): [`_SourceSinkBase`](_core.md#_SourceSinkBase)
- used by: [`key_reuse_signatures`](_core.md#key_reuse_signatures.key_reuse_signatures), [`jaxpr_type_signature`](_core.md#jaxpr_type_signature), [`_cond_key_type_signature`](_core.md#_cond_key_type_signature), [`_slice_signature`](_core.md#_slice_signature), [`_args`](_core.md#KeyReuseSignature._args), [`_concatenate_signature`](_core.md#_concatenate_signature), [`_unstack_signature`](_core.md#_unstack_signature), [`sources`](_core.md#KeyReuseSignature.sources), [`_SourceSinkBase`](_core.md#_SourceSinkBase)

### `_SourceSinkBase`
- def: [`jax/experimental/key_reuse/_core.py:71`](../../../../../../../raw/code/jax/jax/experimental/key_reuse/_core.py#L71)
- signature: `class _SourceSinkBase:`
- members:
  - `idx` — [`L72`](../../../../../../../raw/code/jax/jax/experimental/key_reuse/_core.py#L72)
  - `mask` — [`L73`](../../../../../../../raw/code/jax/jax/experimental/key_reuse/_core.py#L73)
- protocol/private: `__eq__`[`L98`](../../../../../../../raw/code/jax/jax/experimental/key_reuse/_core.py#L98), `__hash__`[`L113`](../../../../../../../raw/code/jax/jax/experimental/key_reuse/_core.py#L113), `__init__`[`L75`](../../../../../../../raw/code/jax/jax/experimental/key_reuse/_core.py#L75), `__lt__`[`L104`](../../../../../../../raw/code/jax/jax/experimental/key_reuse/_core.py#L104), `__repr__`[`L121`](../../../../../../../raw/code/jax/jax/experimental/key_reuse/_core.py#L121), `__setattr__`[`L95`](../../../../../../../raw/code/jax/jax/experimental/key_reuse/_core.py#L95)
- uses (calls/refs, reference-scoped): [`Forward`](_core.md#Forward), [`Sink`](_core.md#Sink), [`Source`](_core.md#Source)
- used by: [`jaxpr_type_signature`](_core.md#jaxpr_type_signature), [`_cond_key_type_signature`](_core.md#_cond_key_type_signature), [`update_consumption`](_core.md#KeyReuseSignature.update_consumption), [`_scan_key_type_signature`](_core.md#_scan_key_type_signature), [`_while_key_type_signature`](_core.md#_while_key_type_signature), [`Sink`](_core.md#Sink), [`Source`](_core.md#Source), [`check_signature`](_core.md#KeyReuseSignature.check_signature)

## Functions
- `_check_consumed_value(eqn, consumed)` — [`L273`](../../../../../../../raw/code/jax/jax/experimental/key_reuse/_core.py#L273) — Extra check for use with assert_consumed_value_p
- `_concatenate_signature(eqn)` — [`L458`](../../../../../../../raw/code/jax/jax/experimental/key_reuse/_core.py#L458)
- `_cond_key_type_signature(eqn)` — [`L481`](../../../../../../../raw/code/jax/jax/experimental/key_reuse/_core.py#L481)
- `_device_put_signature(eqn)` — [`L586`](../../../../../../../raw/code/jax/jax/experimental/key_reuse/_core.py#L586)
- `_pjit_key_type_signature(eqn)` — [`L469`](../../../../../../../raw/code/jax/jax/experimental/key_reuse/_core.py#L469)
- `_remat_key_type_signature(eqn)` — [`L572`](../../../../../../../raw/code/jax/jax/experimental/key_reuse/_core.py#L572)
- `_scan_key_type_signature(eqn)` — [`L500`](../../../../../../../raw/code/jax/jax/experimental/key_reuse/_core.py#L500)
- `_shard_map_type_signature(eqn)` — [`L475`](../../../../../../../raw/code/jax/jax/experimental/key_reuse/_core.py#L475)
- `_slice_signature(eqn)` — [`L432`](../../../../../../../raw/code/jax/jax/experimental/key_reuse/_core.py#L432)
- `_unstack_signature(eqn)` — [`L451`](../../../../../../../raw/code/jax/jax/experimental/key_reuse/_core.py#L451)
- `_while_key_type_signature(eqn)` — [`L530`](../../../../../../../raw/code/jax/jax/experimental/key_reuse/_core.py#L530)
- `assert_consumed(key, value=True)` — [`L268`](../../../../../../../raw/code/jax/jax/experimental/key_reuse/_core.py#L268) — Assert that a key is consumed
- `assert_unconsumed(key)` — [`L264`](../../../../../../../raw/code/jax/jax/experimental/key_reuse/_core.py#L264) — Assert that a key is unconsumed
- `call_impl_with_key_reuse_checks(prim: core.Primitive, raw_impl: Callable[..., Any], *args, **kwargs)` — [`L593`](../../../../../../../raw/code/jax/jax/experimental/key_reuse/_core.py#L593)
- `check_key_reuse(fun: Callable[..., Any],, *args: Any)` — [`L423`](../../../../../../../raw/code/jax/jax/experimental/key_reuse/_core.py#L423) — Function to statically check key reuse.
- `check_key_reuse_jaxpr(jaxpr: core.Jaxpr)` — [`L418`](../../../../../../../raw/code/jax/jax/experimental/key_reuse/_core.py#L418) — Check the jaxpr for key reuse.
- `consume(key)` — [`L250`](../../../../../../../raw/code/jax/jax/experimental/key_reuse/_core.py#L250) — Consume the key and return a consumed copy.
- `dynamic_key_reuse_signature(f: Callable[[core.JaxprEqn], KeyReuseSignature])` — [`L207`](../../../../../../../raw/code/jax/jax/experimental/key_reuse/_core.py#L207)
- `function_type_signature(fun: Callable[..., Any], *args: Any)` — [`L407`](../../../../../../../raw/code/jax/jax/experimental/key_reuse/_core.py#L407)
- `is_consumed(var: core.Atom)` — [`L364`](../../../../../../../raw/code/jax/jax/experimental/key_reuse/_core.py#L364)
- `is_key(var: core.Atom)` — [`L321`](../../../../../../../raw/code/jax/jax/experimental/key_reuse/_core.py#L321)
- `is_key(var: core.Atom)` — [`L345`](../../../../../../../raw/code/jax/jax/experimental/key_reuse/_core.py#L345)
- `jaxpr_type_signature(jaxpr: core.Jaxpr)` — [`L328`](../../../../../../../raw/code/jax/jax/experimental/key_reuse/_core.py#L328) — Parse the jaxpr to determine key reuse signature
- `key_reuse_error_with_source_traceback(message: str, traceback: source_info_util.Traceback | None)` — [`L53`](../../../../../../../raw/code/jax/jax/experimental/key_reuse/_core.py#L53)
- `key_reuse_signature_from_eqn(eqn: core.JaxprEqn)` — [`L210`](../../../../../../../raw/code/jax/jax/experimental/key_reuse/_core.py#L210)
- `key_reuse_signature_from_primitive(prim, *args, **params)` — [`L224`](../../../../../../../raw/code/jax/jax/experimental/key_reuse/_core.py#L224)
- `resolve_forwards(var: core.Atom)` — [`L333`](../../../../../../../raw/code/jax/jax/experimental/key_reuse/_core.py#L333)
- `sink(var: core.Atom, mask=True)` — [`L348`](../../../../../../../raw/code/jax/jax/experimental/key_reuse/_core.py#L348)
- `source(var: core.Atom, mask=False)` — [`L357`](../../../../../../../raw/code/jax/jax/experimental/key_reuse/_core.py#L357)
- `unknown_signature(eqn)` — [`L320`](../../../../../../../raw/code/jax/jax/experimental/key_reuse/_core.py#L320)

## Module values
- `_source_context_message` — [`L49`](../../../../../../../raw/code/jax/jax/experimental/key_reuse/_core.py#L49)
- `assert_consumed_value_p` — [`L256`](../../../../../../../raw/code/jax/jax/experimental/key_reuse/_core.py#L256)
- `assert_effect` — [`L254`](../../../../../../../raw/code/jax/jax/experimental/key_reuse/_core.py#L254)
- `consume_effect` — [`L241`](../../../../../../../raw/code/jax/jax/experimental/key_reuse/_core.py#L241)
- `consume_p` — [`L242`](../../../../../../../raw/code/jax/jax/experimental/key_reuse/_core.py#L242)
- `key_reuse_signatures` — [`L286`](../../../../../../../raw/code/jax/jax/experimental/key_reuse/_core.py#L286)

