---
title: 'Module: jax/_src/pallas/mosaic/error_handling.py'
type: catalog
provenance: extracted
module: jax/_src/pallas/mosaic/error_handling.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax._src.pallas.mosaic.error_handling`/
symbols:
  mlir_error_to_verification_error: mlir_error_to_verification_error().
  _handle_xla_runtime_error: _handle_xla_runtime_error().
  parse_location_string: parse_location_string().
  MosaicError: MosaicError#
  VerificationError: VerificationError#
  mlir_error_to_verification_error._get_diagnostic_message: mlir_error_to_verification_error()._get_diagnostic_message().
  traceback_from_raw_frames: traceback_from_raw_frames().
  RawFrame: RawFrame.
  redact_locations: redact_locations().
  VerificationError.__init__: VerificationError#__init__().
  LOCATION_PATTERN: LOCATION_PATTERN.
  FRAME_PATTERN: FRAME_PATTERN.
  MLIR_ERR_PREFIX: MLIR_ERR_PREFIX.
---
# Module: [`jax/_src/pallas/mosaic/error_handling.py`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/error_handling.py)

## Classes
### `MosaicError`  ·  implements/extends Exception
- def: [`jax/_src/pallas/mosaic/error_handling.py:47`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/error_handling.py#L47)
- doc: Error thrown by Pallas when re-raising a Mosaic internal error.
- signature: `class MosaicError(Exception):`
- uses (calls/refs, reference-scoped): [`VerificationError`](error_handling.md#VerificationError)
- used by: [`_handle_xla_runtime_error`](error_handling.md#_handle_xla_runtime_error), [`VerificationError`](error_handling.md#VerificationError)

### `VerificationError`  ·  implements/extends MosaicError
- def: [`jax/_src/pallas/mosaic/error_handling.py:51`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/error_handling.py#L51)
- doc: Error thrown by Pallas when re-raising a verification error.
- signature: `class VerificationError(MosaicError):`
- protocol/private: `__init__`[`L54`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/error_handling.py#L54)
- uses (calls/refs, reference-scoped): [`MosaicError`](error_handling.md#MosaicError), [`MLIR_ERR_PREFIX`](error_handling.md#MLIR_ERR_PREFIX)
- used by: [`mlir_error_to_verification_error`](error_handling.md#mlir_error_to_verification_error), [`MosaicError`](error_handling.md#MosaicError)

## Functions
- `_get_diagnostic_message(diagnostic)` — [`L86`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/error_handling.py#L86)
- `_handle_xla_runtime_error(base_err: _jax.JaxRuntimeError)` — [`L58`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/error_handling.py#L58) — Reformats JaxRuntimeError to include a Python traceback.
- `mlir_error_to_verification_error(base_err: ir.MLIRError)` — [`L81`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/error_handling.py#L81) — Reformats MLIRError to include a Python traceback.
- `parse_location_string(location_string: str)` — [`L112`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/error_handling.py#L112) — Parses a serialized MLIR location.
- `redact_locations(err_msg: str)` — [`L99`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/error_handling.py#L99) — Removes location strings from an error message.
- `traceback_from_raw_frames(frames: list[RawFrame])` — [`L154`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/error_handling.py#L154) — Constructs a traceback from a list of RawFrame objects.

## Module values
- `FRAME_PATTERN` — [`L33`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/error_handling.py#L33)
- `LOCATION_PATTERN` — [`L30`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/error_handling.py#L30)
- `MLIR_ERR_PREFIX` — [`L38`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/error_handling.py#L38)
- `RawFrame` — [`L44`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/error_handling.py#L44)

