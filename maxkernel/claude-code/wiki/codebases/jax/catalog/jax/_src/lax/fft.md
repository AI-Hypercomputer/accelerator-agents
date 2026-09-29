---
title: 'Module: jax/_src/lax/fft.py'
type: catalog
provenance: extracted
module: jax/_src/lax/fft.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax._src.lax.fft`/
symbols:
  fft_p: fft_p.
  fft: fft().
  _rfft_transpose: _rfft_transpose().
  _irfft_transpose: _irfft_transpose().
  FftType: FftType#
  _fft_transpose_rule: _fft_transpose_rule().
  _str_to_fft_type: _str_to_fft_type().
  fft_abstract_eval: fft_abstract_eval().
  _fft_lowering_gpu: _fft_lowering_gpu().
  FftType.IRFFT: FftType#IRFFT.
  _fft_batching_rule: _fft_batching_rule().
  FftType.RFFT: FftType#RFFT.
  _fft_lowering: _fft_lowering().
  _fft_impl: _fft_impl().
  FftType.IFFT: FftType#IFFT.
  _complex_dtype: _complex_dtype.
  _real_dtype: _real_dtype.
  FftType.FFT: FftType#FFT.
  __all__: __all__.
---
# Module: [`jax/_src/lax/fft.py`](../../../../../../../raw/code/jax/jax/_src/lax/fft.py)

## Classes
### `FftType`  ·  implements/extends IntEnum
- def: [`jax/_src/lax/fft.py:39`](../../../../../../../raw/code/jax/jax/_src/lax/fft.py#L39)
- doc: Describes which FFT operation to perform.
- signature: `class FftType(enum.IntEnum):`
- members:
  - `FFT` — [`L42`](../../../../../../../raw/code/jax/jax/_src/lax/fft.py#L42)
  - `IFFT` — [`L45`](../../../../../../../raw/code/jax/jax/_src/lax/fft.py#L45)
  - `IRFFT` — [`L51`](../../../../../../../raw/code/jax/jax/_src/lax/fft.py#L51)
  - `RFFT` — [`L48`](../../../../../../../raw/code/jax/jax/_src/lax/fft.py#L48)
- used by: [`fft`](fft.md#fft), [`_rfft_transpose`](fft.md#_rfft_transpose), [`_irfft_transpose`](fft.md#_irfft_transpose), [`_fft_transpose_rule`](fft.md#_fft_transpose_rule), [`_str_to_fft_type`](fft.md#_str_to_fft_type), [`fft_abstract_eval`](fft.md#fft_abstract_eval), [`_fft_lowering_gpu`](fft.md#_fft_lowering_gpu)

## Functions
- `_fft_batching_rule(batched_args, batch_dims, fft_type, fft_lengths)` — [`L239`](../../../../../../../raw/code/jax/jax/_src/lax/fft.py#L239)
- `_fft_impl(x, fft_type, fft_lengths)` — [`L88`](../../../../../../../raw/code/jax/jax/_src/lax/fft.py#L88)
- `_fft_lowering(ctx, x, *, fft_type, fft_lengths)` — [`L130`](../../../../../../../raw/code/jax/jax/_src/lax/fft.py#L130)
- `_fft_lowering_gpu(ctx, x, *, fft_type, fft_lengths)` — [`L143`](../../../../../../../raw/code/jax/jax/_src/lax/fft.py#L143)
- `_fft_transpose_rule(t, operand, fft_type, fft_lengths)` — [`L230`](../../../../../../../raw/code/jax/jax/_src/lax/fft.py#L230)
- `_irfft_transpose(t, fft_lengths)` — [`L206`](../../../../../../../raw/code/jax/jax/_src/lax/fft.py#L206)
- `_rfft_transpose(t, fft_lengths)` — [`L184`](../../../../../../../raw/code/jax/jax/_src/lax/fft.py#L184)
- `_str_to_fft_type(s: str)` — [`L55`](../../../../../../../raw/code/jax/jax/_src/lax/fft.py#L55)
- `fft(x, fft_type: FftType | str, fft_lengths: Sequence[int])` — [`L68`](../../../../../../../raw/code/jax/jax/_src/lax/fft.py#L68)
- `fft_abstract_eval(x, fft_type, fft_lengths)` — [`L94`](../../../../../../../raw/code/jax/jax/_src/lax/fft.py#L94)

## Module values
- `__all__` — [`L34`](../../../../../../../raw/code/jax/jax/_src/lax/fft.py#L34)
- `_complex_dtype` — [`L91`](../../../../../../../raw/code/jax/jax/_src/lax/fft.py#L91)
- `_real_dtype` — [`L92`](../../../../../../../raw/code/jax/jax/_src/lax/fft.py#L92)
- `fft_p` — [`L245`](../../../../../../../raw/code/jax/jax/_src/lax/fft.py#L245)

