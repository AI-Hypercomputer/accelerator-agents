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
"""What changing the algorithm is worth, and what it costs in accuracy.

Supplies answers for `reference/design/algorithm.md`. Two kinds of change live here and
they are not interchangeable: an *exact restructuring* costs nothing in
accuracy and can be taken on the timing alone, while an *approximation* has to
carry the error it introduces and the threshold that error is judged against.
Every probe here reports both the time and the error, so the reference file
cannot quote one without the other.
"""

from __future__ import annotations

import harness

_GROUP = "algorithm"


@harness.probe(
  _GROUP,
  "flash_vs_materialized_attention",
  "What does the online-softmax restructuring of attention buy over materializing the score matrix?",
)
def _flash_vs_materialized_attention() -> str:
  """The canonical exact restructuring: same answer, different memory profile.

  The Pallas arm is checked against the XLA arm before any timing is quoted.
  An attention kernel that drops the rescaling term is both wrong and fast.
  """
  import math

  import jax
  import jax.numpy as jnp
  from jax.experimental import pallas as pl
  from jax.experimental.pallas import tpu as pltpu

  head_dim = 128
  block_queries = 512
  block_keys = 512
  # A sentinel rather than -inf: a running maximum initialised to -inf makes
  # the first rescaling factor exp(-inf - m) and invites NaN if any block is
  # entirely masked. -1e30 underflows exp() to exactly 0 without that risk.
  negative_sentinel = -1e30

  def materialized(queries, keys, values, scale):
    scores = jnp.einsum(
      "qd,kd->qk", queries, keys, preferred_element_type=jnp.float32
    )
    weights = jax.nn.softmax(scores * scale, axis=-1)
    return jnp.einsum(
      "qk,kd->qd",
      weights.astype(values.dtype),
      values,
      preferred_element_type=jnp.float32,
    )

  def flash_kernel(
    queries_ref,
    keys_ref,
    values_ref,
    output_ref,
    running_max_ref,
    running_sum_ref,
    accumulator_ref,
    *,
    scale: float,
  ) -> None:
    key_step = pl.program_id(1)
    last_key_step = pl.num_programs(1) - 1

    @pl.when(key_step == 0)
    def _initialize() -> None:
      running_max_ref[...] = jnp.full(
        running_max_ref.shape, negative_sentinel, jnp.float32
      )
      running_sum_ref[...] = jnp.zeros_like(running_sum_ref)
      accumulator_ref[...] = jnp.zeros_like(accumulator_ref)

    scores = (
      jnp.dot(
        queries_ref[...],
        keys_ref[...].T,
        preferred_element_type=jnp.float32,
      )
      * scale
    )
    block_max = jnp.max(scores, axis=-1, keepdims=True)
    updated_max = jnp.maximum(running_max_ref[...], block_max)
    rescale = jnp.exp(running_max_ref[...] - updated_max)
    probabilities = jnp.exp(scores - updated_max)

    running_sum_ref[...] = running_sum_ref[...] * rescale + jnp.sum(
      probabilities, axis=-1, keepdims=True
    )
    accumulator_ref[...] = accumulator_ref[...] * rescale + jnp.dot(
      probabilities.astype(values_ref.dtype),
      values_ref[...],
      preferred_element_type=jnp.float32,
    )
    running_max_ref[...] = updated_max

    @pl.when(key_step == last_key_step)
    def _finalize() -> None:
      output_ref[...] = accumulator_ref[...] / running_sum_ref[...]

  def build_flash(sequence_length: int, scale: float):
    import functools

    return jax.jit(
      pl.pallas_call(
        functools.partial(flash_kernel, scale=scale),
        grid=(
          sequence_length // block_queries,
          sequence_length // block_keys,
        ),
        in_specs=[
          pl.BlockSpec((block_queries, head_dim), lambda q, k: (q, 0)),
          pl.BlockSpec((block_keys, head_dim), lambda q, k: (k, 0)),
          pl.BlockSpec((block_keys, head_dim), lambda q, k: (k, 0)),
        ],
        out_specs=pl.BlockSpec((block_queries, head_dim), lambda q, k: (q, 0)),
        out_shape=jax.ShapeDtypeStruct(
          (sequence_length, head_dim), jnp.float32
        ),
        scratch_shapes=[
          pltpu.VMEM((block_queries, 1), jnp.float32),
          pltpu.VMEM((block_queries, 1), jnp.float32),
          pltpu.VMEM((block_queries, head_dim), jnp.float32),
        ],
      )
    )

  findings = []
  for sequence_length in (2048, 8192):
    key = jax.random.key(0)
    queries = jax.random.normal(
      key, (sequence_length, head_dim), dtype=jnp.bfloat16
    )
    keys = jax.random.normal(
      jax.random.key(1), (sequence_length, head_dim), dtype=jnp.bfloat16
    )
    values = jax.random.normal(
      jax.random.key(2), (sequence_length, head_dim), dtype=jnp.bfloat16
    )
    scale = 1.0 / math.sqrt(head_dim)

    materialized_call = jax.jit(
      lambda q, k, v, scale=scale: materialized(q, k, v, scale)
    )
    flash_call = build_flash(sequence_length, scale)

    reference = materialized_call(queries, keys, values)
    try:
      produced = flash_call(queries, keys, values)
    except Exception as error:  # noqa: BLE001
      findings.append(
        f"S={sequence_length}: flash arm did not run "
        f"({type(error).__name__}: {str(error)[:200]})"
      )
      continue

    max_error = float(jnp.max(jnp.abs(produced - reference)))
    scale_of_output = float(jnp.max(jnp.abs(reference)))
    relative = max_error / (scale_of_output + 1e-30)
    if relative > 5e-2:
      findings.append(
        f"S={sequence_length}: INVALID, flash arm disagrees "
        f"(relative_max_error={relative:.3e}); timing withheld"
      )
      continue

    # At short S both arms can sit near the host round trip, and their ratio
    # is then a ratio between two dispatch measurements rather than a
    # finding that flash and materialized attention cost the same at short
    # sequence. The short-S arm is exactly the one this probe exists to
    # report, so it is the one that has to clear the floor.
    materialized_timing = harness.time_call(
      lambda: materialized_call(queries, keys, values),
      warmups=3,
      repeats=10,
      batch=32,
    )
    flash_timing = harness.time_call(
      lambda: flash_call(queries, keys, values),
      warmups=3,
      repeats=10,
      batch=32,
    )
    harness.require_above_dispatch_floor(
      materialized_timing, what=f"the S={sequence_length} materialized arm"
    )
    score_matrix_mib = sequence_length * sequence_length * 4 / 2**20
    findings.append(
      f"S={sequence_length}: materialized={materialized_timing.median * 1e6:.0f}us "
      f"flash={flash_timing.median * 1e6:.0f}us "
      f"flash/materialized={flash_timing.median / materialized_timing.median:.2f}x "
      f"(the score matrix XLA materializes is {score_matrix_mib:.0f}MiB in f32) "
      f"relative_max_error={relative:.2e}"
    )

  return (
    f"bf16 q/k/v, head_dim={head_dim}, non-causal, single head, "
    f"blocks {block_queries}x{block_keys}, f32 accumulation. "
    + " || ".join(findings)
  )


@harness.probe(
  _GROUP,
  "accumulator_precision",
  "What does accumulating a long reduction in bf16 instead of f32 cost, in time and in error?",
)
def _accumulator_precision() -> str:
  """An approximation, so it arrives with its error or it does not arrive."""
  import jax
  import jax.numpy as jnp
  from jax.experimental import pallas as pl

  rows = 1024
  contraction = 16384
  columns = 1024
  block_contraction = 512

  key = jax.random.key(0)
  lhs = jax.random.normal(key, (rows, contraction), dtype=jnp.bfloat16)
  rhs = jax.random.normal(
    jax.random.key(1), (contraction, columns), dtype=jnp.bfloat16
  )

  exact = jnp.dot(
    lhs.astype(jnp.float32),
    rhs.astype(jnp.float32),
    preferred_element_type=jnp.float32,
    precision=jax.lax.Precision.HIGHEST,
  )
  exact_scale = float(jnp.max(jnp.abs(exact)))

  def build(accumulate_name: str):
    accumulate_dtype = getattr(jnp, accumulate_name)

    def matmul_kernel(lhs_ref, rhs_ref, output_ref) -> None:
      step = pl.program_id(0)

      @pl.when(step == 0)
      def _initialize() -> None:
        output_ref[...] = jnp.zeros_like(output_ref)

      output_ref[...] += jnp.dot(
        lhs_ref[...], rhs_ref[...], preferred_element_type=accumulate_dtype
      ).astype(accumulate_dtype)

    return jax.jit(
      pl.pallas_call(
        matmul_kernel,
        grid=(contraction // block_contraction,),
        in_specs=[
          pl.BlockSpec((rows, block_contraction), lambda k: (0, k)),
          pl.BlockSpec((block_contraction, columns), lambda k: (k, 0)),
        ],
        out_specs=pl.BlockSpec((rows, columns), lambda k: (0, 0)),
        out_shape=jax.ShapeDtypeStruct((rows, columns), accumulate_dtype),
      )
    )

  findings = []
  for accumulate_name in ("float32", "bfloat16"):
    try:
      call = build(accumulate_name)
      produced = call(lhs, rhs).astype(jnp.float32)
      timing = harness.time_call(
        lambda call=call: call(lhs, rhs), warmups=3, repeats=10, batch=24
      )
    except Exception as error:  # noqa: BLE001
      findings.append(f"{accumulate_name}=NO({type(error).__name__})")
      continue
    if accumulate_name == "float32":
      harness.require_above_dispatch_floor(
        timing, what="the f32-accumulator arm"
      )
    max_error = float(jnp.max(jnp.abs(produced - exact)))
    mean_error = float(jnp.mean(jnp.abs(produced - exact)))
    findings.append(
      f"{accumulate_name}: {timing.median * 1e6:.0f}us "
      f"relative_max_error={max_error / exact_scale:.3e} "
      f"relative_mean_error={mean_error / exact_scale:.3e}"
    )

  return (
    f"bf16 operands, {rows}x{contraction}x{columns}, {contraction // block_contraction} "
    f"accumulation steps, reference is an f32 HIGHEST-precision matmul "
    f"(|max| = {exact_scale:.3f}). "
    + " | ".join(findings)
    + " || an accumulator choice is an approximation: quote the error next "
    "to the time or do not quote the time"
  )


@harness.probe(
  _GROUP,
  "cumsum_alternatives",
  "Does `jnp.cumsum` lower here, and what does a triangular-matmul substitute cost?",
)
def _cumsum_alternatives() -> str:
  """Turns a lowering gap into a costed technique, or kills the substitute."""
  import jax
  import jax.numpy as jnp
  from jax.experimental import pallas as pl

  rows = 512
  length = 512

  key = jax.random.key(0)
  source = jax.random.normal(key, (rows, length), dtype=jnp.float32)
  reference = jnp.cumsum(source.astype(jnp.float32), axis=-1)
  reference_scale = float(jnp.max(jnp.abs(reference)))

  def direct_kernel(source_ref, output_ref) -> None:
    output_ref[...] = jnp.cumsum(source_ref[...], axis=-1)

  def triangular_kernel(source_ref, triangle_ref, output_ref) -> None:
    # cumsum along the last axis is x @ upper_triangular_ones.
    output_ref[...] = jnp.dot(
      source_ref[...], triangle_ref[...], preferred_element_type=jnp.float32
    )

  def scan_kernel(source_ref, output_ref) -> None:
    output_ref[...] = jax.lax.associative_scan(
      jnp.add, source_ref[...], axis=-1
    )

  triangle = (
    jax.lax.broadcasted_iota(jnp.int32, (length, length), 0)
    <= jax.lax.broadcasted_iota(jnp.int32, (length, length), 1)
  ).astype(jnp.float32)

  findings = []
  arms = {
    "jnp.cumsum": (direct_kernel, (source,)),
    "triangular_matmul": (triangular_kernel, (source, triangle)),
    "associative_scan": (scan_kernel, (source,)),
  }
  for label, (kernel, operands) in arms.items():
    try:
      call = jax.jit(
        pl.pallas_call(
          kernel,
          out_shape=jax.ShapeDtypeStruct((rows, length), jnp.float32),
        )
      )
      produced = call(*operands)
      jax.block_until_ready(produced)
    except Exception as error:  # noqa: BLE001 — an absent lowering is the finding.
      findings.append(f"{label}=DOES-NOT-LOWER({type(error).__name__})")
      continue
    max_error = float(jnp.max(jnp.abs(produced - reference)))
    # A small triangular matmul is short next to the host round trip; timed
    # alone it reports the round trip, not the substitute's cost, which is
    # the number this probe exists to produce. Let the measurement pick the
    # batch.
    timing = harness.time_above_floor(
      lambda call=call, operands=operands: call(*operands),
      what=f"the {label} arm",
      batch=48,
      repeats=20,
    )
    harness.require_above_dispatch_floor(timing, what=f"the {label} arm")
    findings.append(
      f"{label}: {timing.median * 1e6:.1f}us "
      f"relative_max_error={max_error / reference_scale:.2e}"
    )

  return (
    f"f32 {rows}x{length}, cumulative sum along the lane axis, three ways. "
    + " | ".join(findings)
    + f" || the triangular arm does {2.0 * rows * length * length / 1e9:.2f} "
    "GFLOP of matmul to replace a scan, so it is a trade of VPU work for "
    "MXU work rather than a free substitution"
  )


@harness.probe(
  _GROUP,
  "split_contraction_accuracy",
  "Does splitting a reduction into partial sums change the answer, and by how much?",
)
def _split_contraction_accuracy() -> str:
  """An exact restructuring in exact arithmetic, not in floating point.

  Split-k is usually taken as free. It is free in accuracy only to the extent
  that a tree of partial sums is no worse than a serial chain -- which is
  generally true, and worth having the number for rather than the intuition.
  """
  import jax
  import jax.numpy as jnp

  length = 1 << 20
  key = jax.random.key(0)
  values = jax.random.normal(key, (length,), dtype=jnp.float32)

  exact = float(
    jnp.sum(
      values.astype(jnp.float64) if jax.config.jax_enable_x64 else values,
      dtype=jnp.float32,
    )
  )
  high_precision = float(
    jnp.sum(values.reshape(-1, 1024).astype(jnp.float32), dtype=jnp.float32)
  )

  def serial_sum(chunk_count: int) -> float:
    chunks = values.reshape(chunk_count, -1)
    partials = jnp.sum(chunks, axis=-1, dtype=jnp.float32)
    total = jnp.float32(0.0)
    for index in range(chunk_count):
      total = total + partials[index]
    return float(total)

  findings = []
  for chunk_count in (1, 8, 64, 512):
    findings.append(f"{chunk_count}chunks={serial_sum(chunk_count):.6f}")

  spread = max(
    abs(serial_sum(count) - high_precision) for count in (1, 8, 64, 512)
  )
  return (
    f"f32 sum of {length} standard-normal values. "
    f"single_reduce={exact:.6f} chunked_reduce={high_precision:.6f} | "
    + " ".join(findings)
    + f" || largest deviation across the split counts = {spread:.3e}, i.e. "
    f"{spread / (abs(high_precision) + 1e-30):.2e} relative. This is XLA, "
    "not a kernel: it bounds how much the *ordering* alone can move a "
    "reduction of this length in f32."
  )
