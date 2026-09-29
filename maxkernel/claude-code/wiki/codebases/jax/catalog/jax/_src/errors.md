---
title: 'Module: jax/_src/errors.py'
type: catalog
provenance: extracted
module: jax/_src/errors.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax._src.errors`/
symbols:
  ConcretizationTypeError: ConcretizationTypeError#
  JAXTypeError: JAXTypeError#
  KeyReuseError: KeyReuseError#
  UnexpectedTracerError: UnexpectedTracerError#
  TracerBoolConversionError.__init__: TracerBoolConversionError#__init__().
  export: export.
  _JAXErrorMixin.__init__: _JAXErrorMixin#__init__().
  ConcretizationTypeError.__init__: ConcretizationTypeError#__init__().
  TracerArrayConversionError.__init__: TracerArrayConversionError#__init__().
  TracerIntegerConversionError.__init__: TracerIntegerConversionError#__init__().
  JAXIndexError: JAXIndexError#
  TracerArrayConversionError: TracerArrayConversionError#
  TracerIntegerConversionError: TracerIntegerConversionError#
  TracerBoolConversionError: TracerBoolConversionError#
  _JAXErrorMixin: _JAXErrorMixin#
  NonConcreteBooleanIndexError: NonConcreteBooleanIndexError#
  NonConcreteBooleanIndexError.__init__: NonConcreteBooleanIndexError#__init__().
  UnexpectedTracerError.__init__: UnexpectedTracerError#__init__().
  _JAXErrorMixin._error_page: _JAXErrorMixin#_error_page.
  _JAXErrorMixin._module_name: _JAXErrorMixin#_module_name.
---
# Module: [`jax/_src/errors.py`](../../../../../../raw/code/jax/jax/_src/errors.py)

## Classes
### `ConcretizationTypeError`  ·  implements/extends JAXTypeError
- def: [`jax/_src/errors.py:46`](../../../../../../raw/code/jax/jax/_src/errors.py#L46)
- doc: This error occurs when a JAX Tracer object is used in a context where a
- signature: `class ConcretizationTypeError(JAXTypeError):`
- protocol/private: `__init__`[`L130`](../../../../../../raw/code/jax/jax/_src/errors.py#L130)
- uses (calls/refs, reference-scoped): [`Tracer`](core.md#Tracer), [`_error_repr`](core.md#Tracer._error_repr), [`JAXTypeError`](errors.md#JAXTypeError), [`_origin_msg`](core.md#Tracer._origin_msg), [`export`](errors.md#export), [`__init__`](errors.md#_JAXErrorMixin.__init__), [`TracerBoolConversionError`](errors.md#TracerBoolConversionError)
- used by: [`while_loop`](lax/control_flow/loops.md#while_loop), [`concrete_or_error`](core.md#concrete_or_error), [`JAXTypeError`](errors.md#JAXTypeError), [`concretization_function_error`](core.md#concretization_function_error), [`_trace_to_jaxpr`](ad_checkpoint.md#_trace_to_jaxpr), [`_infer_scan_length`](lax/control_flow/loops.md#_infer_scan_length), [`TracerBoolConversionError`](errors.md#TracerBoolConversionError), [`__dlpack__`](core.md#Tracer.__dlpack__), [`addressable_data`](core.md#Tracer.addressable_data), [`addressable_shards`](core.md#Tracer.addressable_shards), [`committed`](core.md#Tracer.committed), [`delete`](core.md#Tracer.delete), [`devices`](core.md#Tracer.devices), [`global_shards`](core.md#Tracer.global_shards), [`is_deleted`](core.md#Tracer.is_deleted), [`is_fully_addressable`](core.md#Tracer.is_fully_addressable), [`is_fully_replicated`](core.md#Tracer.is_fully_replicated), [`on_device_size_in_bytes`](core.md#Tracer.on_device_size_in_bytes), [`tobytes`](core.md#Tracer.tobytes), [`tolist`](core.md#Tracer.tolist), [`traceback`](core.md#Tracer.traceback), [`unsafe_buffer_pointer`](core.md#Tracer.unsafe_buffer_pointer), [`__reduce__`](core.md#Tracer.__reduce__)

### `JAXIndexError`  ·  implements/extends IndexError, _JAXErrorMixin
- def: [`jax/_src/errors.py:41`](../../../../../../raw/code/jax/jax/_src/errors.py#L41)
- doc: JAX-specific `IndexError`
- signature: `class JAXIndexError(_JAXErrorMixin, IndexError):`
- uses (calls/refs, reference-scoped): [`export`](errors.md#export), [`NonConcreteBooleanIndexError`](errors.md#NonConcreteBooleanIndexError), [`_JAXErrorMixin`](errors.md#_JAXErrorMixin)
- used by: [`NonConcreteBooleanIndexError`](errors.md#NonConcreteBooleanIndexError), [`_JAXErrorMixin`](errors.md#_JAXErrorMixin)

### `JAXTypeError`  ·  implements/extends TypeError, _JAXErrorMixin
- def: [`jax/_src/errors.py:36`](../../../../../../raw/code/jax/jax/_src/errors.py#L36)
- doc: JAX-specific `TypeError`
- signature: `class JAXTypeError(_JAXErrorMixin, TypeError):`
- uses (calls/refs, reference-scoped): [`ConcretizationTypeError`](errors.md#ConcretizationTypeError), [`KeyReuseError`](errors.md#KeyReuseError), [`UnexpectedTracerError`](errors.md#UnexpectedTracerError), [`export`](errors.md#export), [`TracerArrayConversionError`](errors.md#TracerArrayConversionError), [`TracerIntegerConversionError`](errors.md#TracerIntegerConversionError), [`_JAXErrorMixin`](errors.md#_JAXErrorMixin)
- used by: [`ConcretizationTypeError`](errors.md#ConcretizationTypeError), [`KeyReuseError`](errors.md#KeyReuseError), [`UnexpectedTracerError`](errors.md#UnexpectedTracerError), [`__init__`](errors.md#TracerBoolConversionError.__init__), [`TracerArrayConversionError`](errors.md#TracerArrayConversionError), [`TracerIntegerConversionError`](errors.md#TracerIntegerConversionError), [`_JAXErrorMixin`](errors.md#_JAXErrorMixin)

### `KeyReuseError`  ·  implements/extends JAXTypeError
- def: [`jax/_src/errors.py:660`](../../../../../../raw/code/jax/jax/_src/errors.py#L660)
- doc: This error occurs when a PRNG key is reused in an unsafe manner.
- signature: `class KeyReuseError(JAXTypeError):`
- uses (calls/refs, reference-scoped): [`JAXTypeError`](errors.md#JAXTypeError), [`export`](errors.md#export)
- used by: [`jaxpr_type_signature`](../experimental/key_reuse/_core.md#jaxpr_type_signature), [`JAXTypeError`](errors.md#JAXTypeError), [`_scan_key_type_signature`](../experimental/key_reuse/_core.md#_scan_key_type_signature), [`_while_key_type_signature`](../experimental/key_reuse/_core.md#_while_key_type_signature), [`key_reuse_error_with_source_traceback`](../experimental/key_reuse/_core.md#key_reuse_error_with_source_traceback)

### `NonConcreteBooleanIndexError`  ·  implements/extends JAXIndexError
- def: [`jax/_src/errors.py:137`](../../../../../../raw/code/jax/jax/_src/errors.py#L137)
- doc: This error occurs when a program attempts to use non-concrete boolean indices
- signature: `class NonConcreteBooleanIndexError(JAXIndexError):`
- protocol/private: `__init__`[`L223`](../../../../../../raw/code/jax/jax/_src/errors.py#L223)
- uses (calls/refs, reference-scoped): [`Tracer`](core.md#Tracer), [`export`](errors.md#export), [`__init__`](errors.md#_JAXErrorMixin.__init__), [`JAXIndexError`](errors.md#JAXIndexError)
- used by: [`expand_bool_indices`](numpy/indexing.md#NDIndexer.expand_bool_indices), [`JAXIndexError`](errors.md#JAXIndexError)

### `TracerArrayConversionError`  ·  implements/extends JAXTypeError
- def: [`jax/_src/errors.py:229`](../../../../../../raw/code/jax/jax/_src/errors.py#L229)
- doc: This error occurs when a program attempts to convert a JAX Tracer object into
- signature: `class TracerArrayConversionError(JAXTypeError):`
- protocol/private: `__init__`[`L310`](../../../../../../raw/code/jax/jax/_src/errors.py#L310)
- uses (calls/refs, reference-scoped): [`Tracer`](core.md#Tracer), [`_error_repr`](core.md#Tracer._error_repr), [`JAXTypeError`](errors.md#JAXTypeError), [`_origin_msg`](core.md#Tracer._origin_msg), [`export`](errors.md#export), [`__init__`](errors.md#_JAXErrorMixin.__init__)
- used by: [`JAXTypeError`](errors.md#JAXTypeError), [`__array__`](core.md#Tracer.__array__)

### `TracerBoolConversionError`  ·  implements/extends ConcretizationTypeError
- def: [`jax/_src/errors.py:413`](../../../../../../raw/code/jax/jax/_src/errors.py#L413)
- doc: This error occurs when a traced value in JAX is used in a context where a
- signature: `class TracerBoolConversionError(ConcretizationTypeError):`
- protocol/private: `__init__`[`L518`](../../../../../../raw/code/jax/jax/_src/errors.py#L518)
- uses (calls/refs, reference-scoped): [`Tracer`](core.md#Tracer), [`ConcretizationTypeError`](errors.md#ConcretizationTypeError), [`_error_repr`](core.md#Tracer._error_repr), [`JAXTypeError`](errors.md#JAXTypeError), [`_origin_msg`](core.md#Tracer._origin_msg), [`export`](errors.md#export), [`__init__`](errors.md#_JAXErrorMixin.__init__)
- used by: [`ConcretizationTypeError`](errors.md#ConcretizationTypeError), [`error`](core.md#concretization_function_error.error)

### `TracerIntegerConversionError`  ·  implements/extends JAXTypeError
- def: [`jax/_src/errors.py:317`](../../../../../../raw/code/jax/jax/_src/errors.py#L317)
- doc: This error can occur when a JAX Tracer object is used in a context where a
- signature: `class TracerIntegerConversionError(JAXTypeError):`
- protocol/private: `__init__`[`L406`](../../../../../../raw/code/jax/jax/_src/errors.py#L406)
- uses (calls/refs, reference-scoped): [`Tracer`](core.md#Tracer), [`_error_repr`](core.md#Tracer._error_repr), [`JAXTypeError`](errors.md#JAXTypeError), [`_origin_msg`](core.md#Tracer._origin_msg), [`export`](errors.md#export), [`__init__`](errors.md#_JAXErrorMixin.__init__)
- used by: [`JAXTypeError`](errors.md#JAXTypeError), [`concretization_function_error`](core.md#concretization_function_error)

### `UnexpectedTracerError`  ·  implements/extends JAXTypeError
- def: [`jax/_src/errors.py:525`](../../../../../../raw/code/jax/jax/_src/errors.py#L525)
- doc: This error occurs when you use a JAX value that has leaked out of a function.
- signature: `class UnexpectedTracerError(JAXTypeError):`
- protocol/private: `__init__`[`L655`](../../../../../../raw/code/jax/jax/_src/errors.py#L655)
- uses (calls/refs, reference-scoped): [`JAXTypeError`](errors.md#JAXTypeError), [`export`](errors.md#export), [`__init__`](errors.md#_JAXErrorMixin.__init__)
- used by: [`__call__`](hijax.md#custom_vjp3.__call__), [`_decorator`](lax/lax.md#composite._decorator), [`JAXTypeError`](errors.md#JAXTypeError), [`_trace_composite_to_jaxpr`](lax/lax.md#_trace_composite_to_jaxpr), [`get_data`](array.md#make_array_from_callback.get_data), [`escaped_tracer_error`](core.md#escaped_tracer_error), [`_check_for_tracers`](custom_derivatives.md#_check_for_tracers), [`_check_for_tracers`](custom_partitioning.md#_check_for_tracers)

### `_JAXErrorMixin`
- def: [`jax/_src/errors.py:22`](../../../../../../raw/code/jax/jax/_src/errors.py#L22)
- doc: Mixin for JAX-specific errors
- signature: `class _JAXErrorMixin:`
- protocol/private: `__init__`[`L27`](../../../../../../raw/code/jax/jax/_src/errors.py#L27), `_error_page`[`L24`](../../../../../../raw/code/jax/jax/_src/errors.py#L24), `_module_name`[`L25`](../../../../../../raw/code/jax/jax/_src/errors.py#L25)
- uses (calls/refs, reference-scoped): [`JAXTypeError`](errors.md#JAXTypeError), [`JAXIndexError`](errors.md#JAXIndexError)
- used by: [`JAXTypeError`](errors.md#JAXTypeError), [`__init__`](errors.md#TracerBoolConversionError.__init__), [`__init__`](errors.md#ConcretizationTypeError.__init__), [`__init__`](errors.md#TracerArrayConversionError.__init__), [`__init__`](errors.md#TracerIntegerConversionError.__init__), [`JAXIndexError`](errors.md#JAXIndexError), [`__init__`](errors.md#NonConcreteBooleanIndexError.__init__), [`__init__`](errors.md#UnexpectedTracerError.__init__)

## Module values
- `export` — [`L19`](../../../../../../raw/code/jax/jax/_src/errors.py#L19)

