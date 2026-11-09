# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Which `jnp` and `lax` operations Mosaic can lower inside a TPU kernel.

The set is smaller than NumPy and smaller than XLA, and the failure mode is a
compile error at the end of an afternoon rather than a slow kernel. This module
compiles one tiny kernel per operation and records whether it lowered, so the
reference files can carry a table instead of a guess.

A `FAIL` here is a finding, not a broken probe: it means the operation does not
lower and the reference file should say so with the error Mosaic actually gave.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import harness

_GROUP = "lowering"

# Kept small so a survey of ~50 operations stays inside a couple of minutes.
# 8x128 is one native f32 tile, so nothing here is also testing tail handling.
_ROWS = 8
_COLS = 128


def _compiles(
  build_body: Callable[..., None], *, dtype_name: str = "float32"
) -> str:
  """Compiles and runs a one-operation kernel; returns a short success note.

  Raises whatever Mosaic raised, so the caller's verdict machinery can record
  the real error text. Runs the kernel rather than only lowering it, because
  a few operations lower and then fail at run time.
  """
  import jax
  import jax.numpy as jnp
  from jax.experimental import pallas as pl

  dtype = getattr(jnp, dtype_name)
  source = jnp.arange(_ROWS * _COLS, dtype=dtype).reshape(_ROWS, _COLS) % 17

  result = pl.pallas_call(
    build_body,
    out_shape=jax.ShapeDtypeStruct((_ROWS, _COLS), dtype),
  )(source)
  jax.block_until_ready(result)
  return f"lowered and ran on {_ROWS}x{_COLS} {dtype_name}"


def _register_unary_survey(
  name: str,
  question: str,
  apply: Callable[[object], object],
  *,
  dtype_name: str = "float32",
) -> None:
  """Registers a probe that applies one operation inside a kernel."""

  def body() -> str:
    def kernel(source_ref, output_ref) -> None:
      output_ref[...] = apply(source_ref[...])

    return _compiles(kernel, dtype_name=dtype_name)

  harness.probe(_GROUP, name, question)(body)


def _lazy(expression: str) -> Callable[[object], object]:
  """Builds a unary lambda from source, so an absent symbol fails per-probe.

  Importing `jnp` at module scope and referencing e.g. `jnp.cumsum` directly
  would make a missing attribute an import-time crash for the whole suite.
  """

  def apply(operand):
    import jax
    import jax.numpy as jnp
    from jax import lax
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    return eval(  # noqa: S307 — the expressions are literals in this file.
      expression,
      {"jnp": jnp, "lax": lax, "jax": jax, "pl": pl, "pltpu": pltpu},
      {"x": operand},
    )

  return apply


_ELEMENTWISE: Sequence[tuple[str, str]] = (
  ("exp", "jnp.exp(x)"),
  ("exp2", "jnp.exp2(x)"),
  ("log", "jnp.log(jnp.abs(x) + 1.0)"),
  ("tanh", "jnp.tanh(x)"),
  ("erf", "jax.scipy.special.erf(x)"),
  ("rsqrt", "lax.rsqrt(jnp.abs(x) + 1.0)"),
  ("reciprocal", "pl.reciprocal(x + 1.0)"),
  ("pow", "jnp.power(jnp.abs(x) + 1.0, 2.5)"),
  ("sin", "jnp.sin(x)"),
  ("floor_divide", "x // 3.0"),
  ("remainder", "jnp.remainder(x, 3.0)"),
  ("sign", "jnp.sign(x)"),
  ("clip", "jnp.clip(x, 1.0, 5.0)"),
  ("where", "jnp.where(x > 3.0, x, 0.0)"),
  ("logistic", "jax.nn.sigmoid(x)"),
  ("silu", "jax.nn.silu(x)"),
  ("gelu_tanh", "jax.nn.gelu(x, approximate=True)"),
  ("gelu_exact", "jax.nn.gelu(x, approximate=False)"),
  ("softmax_last_axis", "jax.nn.softmax(x, axis=-1)"),
)

_REDUCTION: Sequence[tuple[str, str]] = (
  (
    "sum_lane_axis",
    "jnp.broadcast_to(jnp.sum(x, axis=-1, keepdims=True), x.shape)",
  ),
  (
    "sum_sublane_axis",
    "jnp.broadcast_to(jnp.sum(x, axis=-2, keepdims=True), x.shape)",
  ),
  (
    "max_lane_axis",
    "jnp.broadcast_to(jnp.max(x, axis=-1, keepdims=True), x.shape)",
  ),
  (
    "argmax_lane_axis",
    "jnp.broadcast_to(jnp.argmax(x, axis=-1, keepdims=True), x.shape).astype(x.dtype)",
  ),
  ("cumsum_lane_axis", "jnp.cumsum(x, axis=-1)"),
  ("cumsum_sublane_axis", "jnp.cumsum(x, axis=-2)"),
  ("cummax_lane_axis", "jnp.cummax(x, axis=-1)"),
  ("sort_lane_axis", "jnp.sort(x, axis=-1)"),
  ("top_k_lane_axis", "jnp.broadcast_to(jax.lax.top_k(x, 1)[0], x.shape)"),
  ("searchsorted", "jnp.searchsorted(x[0], x).astype(x.dtype)"),
)

_SHAPE: Sequence[tuple[str, str]] = (
  ("reshape_flatten_roundtrip", "x.reshape(-1).reshape(x.shape)"),
  ("transpose_2d", "x.reshape(8, 128).T.reshape(8, 128) * 1.0"),
  ("flip_lane_axis", "jnp.flip(x, axis=-1)"),
  ("roll_jnp", "jnp.roll(x, 3, axis=-1)"),
  ("roll_pltpu", "pltpu.roll(x, 3, 1)"),
  ("repeat_pltpu", "pltpu.repeat(x[:, :64], 2, 1)"),
  ("concatenate_lane", "jnp.concatenate([x[:, :64], x[:, :64]], axis=-1)"),
  ("pad_then_slice", "jnp.pad(x, ((0, 0), (0, 8)))[:, :128]"),
  (
    "iota_like",
    "jnp.broadcast_to(jax.lax.broadcasted_iota(x.dtype, x.shape, 1), x.shape)",
  ),
  (
    "take_along_axis",
    "jnp.take_along_axis(x, jnp.zeros(x.shape, jnp.int32), axis=-1)",
  ),
  (
    "one_hot_via_compare",
    "(jax.lax.broadcasted_iota(jnp.int32, x.shape, 1) == 3).astype(x.dtype)",
  ),
  ("dynamic_slice_static_size", "jax.lax.dynamic_slice(x, (0, 0), (8, 128))"),
)

_CONTROL_FLOW: Sequence[tuple[str, str]] = (
  ("fori_loop", "jax.lax.fori_loop(0, 4, lambda i, acc: acc + 1.0, x)"),
  ("scan", "jax.lax.scan(lambda c, _: (c + 1.0, None), x, None, length=4)[0]"),
  (
    "cond",
    "jax.lax.cond(x[0, 0] > 0, lambda v: v + 1.0, lambda v: v - 1.0, x)",
  ),
  (
    "while_loop",
    "jax.lax.while_loop(lambda s: s[0, 0] < 3.0, lambda s: s + 1.0, x)",
  ),
  ("select_n", "jax.lax.select_n((x > 3.0).astype(jnp.int32), x, x * 2.0)"),
)

for _name, _expression in (*_ELEMENTWISE, *_REDUCTION, *_SHAPE, *_CONTROL_FLOW):
  _register_unary_survey(
    _name,
    f"Does `{_expression}` lower inside a Mosaic TPU kernel?",
    _lazy(_expression),
  )


@harness.probe(
  _GROUP,
  "integer_ops_int32",
  "Which integer operations lower on an int32 block?",
)
def _integer_ops_int32() -> str:
  import jax.numpy as jnp

  expressions = {
    "add": "x + 1",
    "mul": "x * 3",
    "floordiv_const": "x // 3",
    "mod_const": "x % 3",
    "shift_left": "x << 2",
    "bitwise_and": "x & 7",
    "popcount": "jax.lax.population_count(x)",
    "cumsum_lane": "jnp.cumsum(x, axis=-1)",
    "max_reduce": "jnp.broadcast_to(jnp.max(x, axis=-1, keepdims=True), x.shape)",
  }

  outcomes = {}
  for name, expression in expressions.items():
    apply = _lazy(expression)

    def kernel(source_ref, output_ref, apply=apply) -> None:
      output_ref[...] = apply(source_ref[...])

    try:
      _compiles(kernel, dtype_name="int32")
    except Exception as error:  # noqa: BLE001 — recording the failure is the point.
      outcomes[name] = f"NO({type(error).__name__})"
    else:
      outcomes[name] = "yes"
  del jnp
  return " ".join(f"{name}={outcome}" for name, outcome in outcomes.items())


@harness.probe(
  _GROUP,
  "dot_dtype_matrix",
  "Which operand dtype combinations does an in-kernel jnp.dot accept?",
)
def _dot_dtype_matrix() -> str:
  """A dtype that does not lower here forces an upcast the caller must budget."""
  import jax
  import jax.numpy as jnp
  from jax.experimental import pallas as pl

  candidates = [
    ("float32", "float32"),
    ("bfloat16", "float32"),
    ("bfloat16", "bfloat16"),
    ("float16", "float32"),
    ("int8", "int32"),
    ("float8_e4m3fn", "float32"),
    ("float8_e5m2", "float32"),
    ("int4", "int32"),
  ]

  outcomes = {}
  for operand_name, accumulate_name in candidates:
    operand_dtype = getattr(jnp, operand_name, None)
    accumulate_dtype = getattr(jnp, accumulate_name)
    if operand_dtype is None:
      outcomes[f"{operand_name}->{accumulate_name}"] = "dtype-absent"
      continue

    def kernel(lhs_ref, rhs_ref, output_ref, acc=accumulate_dtype) -> None:
      output_ref[...] = jnp.dot(
        lhs_ref[...], rhs_ref[...], preferred_element_type=acc
      )

    lhs = jnp.ones((256, 256), dtype=operand_dtype)
    rhs = jnp.ones((256, 256), dtype=operand_dtype)
    try:
      result = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((256, 256), accumulate_dtype),
      )(lhs, rhs)
      jax.block_until_ready(result)
    except Exception as error:  # noqa: BLE001 — recording the failure is the point.
      outcomes[f"{operand_name}->{accumulate_name}"] = (
        f"NO({type(error).__name__})"
      )
    else:
      outcomes[f"{operand_name}->{accumulate_name}"] = "yes"

  return " ".join(f"{key}={value}" for key, value in outcomes.items())
