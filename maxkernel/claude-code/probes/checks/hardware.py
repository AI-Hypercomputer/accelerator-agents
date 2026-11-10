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
"""The chip's fixed facts, and the hard limits a kernel either respects or dies on.

Supplies answers for `reference/tpu_hardware_facts.md`. Everything in here is a
capability or a numerical identity, never a throughput: these probes answer
"what is true of this chip" rather than "how fast is this", so none of them
goes near `harness.require_above_dispatch_floor` and none of them is worth
re-running to confirm a number that moved.

The group exists because the other six modules are organized around a *design
decision* — what to fuse, how to partition — and a constant like "the default
scoped VMEM limit is half of physical" belongs to none of them while being
load-bearing for all of them.
"""

from __future__ import annotations

import harness

_GROUP = "hardware"


# --------------------------------------------------------------------------
# What the chip reports, and what those numbers are per
# --------------------------------------------------------------------------


@harness.probe(
  _GROUP,
  "tpu_info_scaling_denominator",
  "Are get_tpu_info's capacities and rates per chip or per TensorCore?",
)
def _tpu_info_scaling_denominator() -> str:
  """Establishes the denominator every `get_tpu_info()` figure is quoted in.

  This is the single most consequential thing about that struct and it is not
  in any field: a gen-7 chip carries two physical TensorCores, and the table
  reports the slice one kernel sees. Getting it wrong is a clean factor of
  two in either direction — too small for a roofline to look absurd, too
  large for a result to be usable.

  Rather than assert it, this reads the constructor out of the installed
  jax and reports both the divisor and the un-divided chip figures, so a
  reader can quote whichever they mean and say which.
  """
  import inspect

  from jax._src.pallas.mosaic import tpu_info as tpu_info_module
  from jax.experimental.pallas import tpu as pltpu

  info = pltpu.get_tpu_info()
  cores_per_chip = info.chip_version.num_physical_tensor_cores_per_chip

  source = inspect.getsource(tpu_info_module)
  divided_fields = sorted(
    {
      line.split("=", 1)[0].strip()
      for line in source.splitlines()
      if "// tensor_cores_per_chip" in line and "=" in line
    }
  )

  per_chip = {
    "hbm_capacity_bytes": info.hbm_capacity_bytes * cores_per_chip,
    "mem_bw_bytes_per_second": info.mem_bw_bytes_per_second * cores_per_chip,
    "bf16_ops_per_second": info.bf16_ops_per_second * cores_per_chip,
  }

  return (
    f"chip_version={info.chip_version} generation={info.generation} "
    f"num_physical_tensor_cores_per_chip={cores_per_chip} "
    f"num_cores={info.num_cores} is_split_chip={info.is_split_chip} "
    f"is_megacore={info.is_megacore} || fields the constructor divides by "
    f"tensor_cores_per_chip: {divided_fields} || so the reported figures "
    f"are PER TENSORCORE; multiplied back to per chip: "
    + " ".join(f"{name}={value:.4g}" for name, value in per_chip.items())
  )


@harness.probe(
  _GROUP,
  "native_matmul_dtype_matrix",
  "Which operand dtype pairs does the MXU multiply natively, without a cast?",
)
def _native_matmul_dtype_matrix() -> str:
  """Settles what a zero in an `ops_per_second` field means.

  `int8_ops_per_second = 0` invites the reading "the table is incomplete",
  and the in-kernel `dot_dtype_matrix` probe encourages it further by showing
  that an int8 x int8 -> int32 dot *compiles and runs*. Both can be true at
  once only if the int8 operands are being widened before they reach the MXU,
  which is exactly what this predicate reports.
  """
  import itertools

  import jax.numpy as jnp
  from jax.experimental.pallas import tpu as pltpu

  info = pltpu.get_tpu_info()

  candidates = [
    "float32",
    "bfloat16",
    "float16",
    "float8_e4m3fn",
    "float8_e5m2",
    "int8",
    "uint8",
    "int4",
    "uint4",
  ]
  available = [name for name in candidates if hasattr(jnp, name)]

  supported: list[str] = []
  for lhs_name, rhs_name in itertools.product(available, repeat=2):
    lhs = jnp.dtype(getattr(jnp, lhs_name))
    rhs = jnp.dtype(getattr(jnp, rhs_name))
    if info.is_matmul_supported(lhs, rhs):
      supported.append(f"{lhs_name}x{rhs_name}")

  integer_names = {"int8", "uint8", "int4", "uint4"}
  integer_pairs = [
    pair for pair in supported if pair.split("x")[0] in integer_names
  ]

  return (
    f"is_matmul_supported over {len(available)}x{len(available)} dtype "
    f"pairs on generation={info.generation}: {len(supported)} native pairs "
    f"= {supported} || native integer pairs: {len(integer_pairs)} "
    f"{integer_pairs} || reported rates: "
    f"bf16={info.bf16_ops_per_second:.4g} fp8={info.fp8_ops_per_second:.4g} "
    f"int8={info.int8_ops_per_second:.4g} int4={info.int4_ops_per_second:.4g}"
  )


@harness.probe(
  _GROUP,
  "sublane_tiling_is_a_formula",
  "Is get_sublane_tiling a measurement of the chip or a computed heuristic?",
)
def _sublane_tiling_is_a_formula() -> str:
  """Checks the reported tiling against the closed form the source uses.

  `sublane_tiling_by_dtype` in `checks/layout.py` measures what the chip does
  with narrow dtypes. This one asks the different question of how much
  authority the *reported* number carries, because a value computed as
  `num_sublanes * (32 // bitwidth)` and documented as depending on XLA flags
  is not an oracle, and a kernel that tiles by it is trusting a heuristic.
  """
  import jax.numpy as jnp
  from jax._src import dtypes as jax_dtypes
  from jax.experimental.pallas import tpu as pltpu

  info = pltpu.get_tpu_info()

  rows = []
  matches_formula = True
  for name in ("float32", "bfloat16", "int8", "int4"):
    dtype = jnp.dtype(getattr(jnp, name))
    reported = info.get_sublane_tiling(dtype)
    # Must be `itemsize_bits`, not `itemsize * 8`: a sub-byte dtype has an
    # itemsize of 1, so int4 would predict a bitwidth of 8 and the probe
    # would report a mismatch that is its own arithmetic error rather than
    # jax's. This is the same call the function under test makes.
    bitwidth = jax_dtypes.itemsize_bits(dtype)
    predicted = info.num_sublanes * (32 // bitwidth)
    matches_formula = matches_formula and reported == predicted
    rows.append(
      f"{name}(bits={bitwidth}) reported={reported} formula={predicted}"
    )

  docstring = (info.get_sublane_tiling.__doc__ or "").strip().replace("\n", " ")

  return (
    f"num_sublanes={info.num_sublanes} (the field, unchanged by dtype) || "
    + " ".join(rows)
    + f" || matches num_sublanes*(32//bitwidth) for every dtype: "
    f"{matches_formula} || jax's own words: {docstring!r}"
  )


# --------------------------------------------------------------------------
# Precision: what DEFAULT actually computes
# --------------------------------------------------------------------------


@harness.probe(
  _GROUP,
  "f32_default_is_one_bf16_pass",
  "What precision does `precision=DEFAULT` on f32 operands actually compute in?",
)
def _f32_default_is_one_bf16_pass() -> str:
  """Proves the DEFAULT precision story numerically instead of from timings.

  Hypothesis: DEFAULT on f32 operands rounds them to bf16 and does one MXU
  pass. A rate comparison (f32 DEFAULT throughput close to the bf16 rate) can
  suggest this, but a timing can be explained away. An error that matches a
  bf16 reference to several digits cannot.

  The reference is numpy float64 on the host, which is strictly better than
  every arm — the rule `reference/design/algorithm.md` insists on for any
  accuracy gate.
  """
  import jax
  import jax.numpy as jnp
  import numpy as np

  size = 1024
  generator = np.random.default_rng(0)
  left_host = generator.standard_normal((size, size), dtype=np.float32)
  right_host = generator.standard_normal((size, size), dtype=np.float32)

  exact = np.dot(left_host.astype(np.float64), right_host.astype(np.float64))
  scale = float(np.max(np.abs(exact)))

  left = jnp.asarray(left_host)
  right = jnp.asarray(right_host)
  left_bf16 = left.astype(jnp.bfloat16)
  right_bf16 = right.astype(jnp.bfloat16)

  def relative_error(result) -> float:
    host = np.asarray(jax.block_until_ready(result), dtype=np.float64)
    return float(np.max(np.abs(host - exact))) / scale

  arms = {
    "f32_DEFAULT": relative_error(
      jnp.dot(left, right, precision=jax.lax.Precision.DEFAULT)
    ),
    "f32_HIGH": relative_error(
      jnp.dot(left, right, precision=jax.lax.Precision.HIGH)
    ),
    "f32_HIGHEST": relative_error(
      jnp.dot(left, right, precision=jax.lax.Precision.HIGHEST)
    ),
    "bf16_operands_f32_accum": relative_error(
      jnp.dot(left_bf16, right_bf16, preferred_element_type=jnp.float32)
    ),
  }

  ratio = arms["f32_DEFAULT"] / arms["bf16_operands_f32_accum"]

  return (
    f"{size}^3 matmul, relative max error against a numpy float64 exact: "
    + " ".join(f"{name}={value:.4e}" for name, value in arms.items())
    + f" || f32_DEFAULT / bf16_operands = {ratio:.3f}x -- a ratio near 1.0 "
    f"means DEFAULT threw the f32 mantissa away before the MXU saw it, and "
    f"HIGHEST is the only arm that did not"
  )


# --------------------------------------------------------------------------
# Hard limits: what the allocator and the pipeliner will refuse
# --------------------------------------------------------------------------


@harness.probe(
  _GROUP,
  "vmem_limit_bytes_raises_ceiling",
  "Does CompilerParams(vmem_limit_bytes=...) actually move the scoped limit?",
)
def _vmem_limit_bytes_raises_ceiling() -> str:
  """Distinguishes a knob that works from one that is quietly ignored.

  Hypothesis: the default scoped VMEM limit is well below physical
  `vmem_capacity_bytes` (measure it with `vmem_scratch_bisect`), so the
  question is whether the remainder is reachable through
  ``vmem_limit_bytes`` or is reserved. The answer decides whether a tile
  budget is written against the default scoped limit or against physical
  capacity. The request size is chosen to sit between the two on a chip
  whose default is half of physical; compare it with the capacity printed.
  """
  import jax
  import jax.numpy as jnp
  from jax.experimental import pallas as pl
  from jax.experimental.pallas import tpu as pltpu

  info = pltpu.get_tpu_info()
  request_mib = 48
  rows = request_mib * 2**20 // (128 * 4)

  def kernel(output_ref, scratch_ref) -> None:
    step = pl.program_id(0)
    scratch_ref[pl.ds(step, 8), :] = jnp.full((8, 128), step + 1.0, jnp.float32)
    output_ref[...] = scratch_ref[pl.ds(step + 8, 8), :]

  def attempt(compiler_params) -> str:
    try:
      jax.block_until_ready(
        pl.pallas_call(
          kernel,
          grid=(2,),
          out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32),
          scratch_shapes=[pltpu.VMEM((rows, 128), jnp.float32)],
          compiler_params=compiler_params,
        )()
      )
    except Exception as error:  # noqa: BLE001 — the rejection is the finding.
      first_line = str(error).strip().splitlines()[0][:160]
      return f"REJECTED({type(error).__name__}: {first_line})"
    return "ACCEPTED"

  default_arm = attempt(None)
  raised_arm = attempt(pltpu.CompilerParams(vmem_limit_bytes=128 * 2**20))

  return (
    f"a {request_mib} MiB f32 VMEM scratch, written and read at different "
    f"runtime offsets: default={default_arm} "
    f"vmem_limit_bytes=128MiB={raised_arm} || physical "
    f"vmem_capacity_bytes={info.vmem_capacity_bytes / 2**20:.0f} MiB; "
    f"run vmem_scratch_bisect for the default scoped limit on this chip"
  )


@harness.probe(
  _GROUP,
  "buffer_count_cap_by_position",
  "How deep can pl.Buffered go, and does the answer depend on where it sits?",
)
def _buffer_count_cap_by_position() -> str:
  """Prices the pipelining ladder's first real rung.

  Buffer depth past two is the reason to nest an `emit_pipeline` inside an
  `ANY` `pallas_call` rather than a matter of taste, so where the cap applies
  is the finding, not the cap itself. Reporting one without the other has
  produced a flatly wrong rule elsewhere.
  """
  import jax
  import jax.numpy as jnp
  from jax.experimental import pallas as pl

  steps = 32
  shape = (steps * 8, 128)

  def kernel(source_ref, output_ref) -> None:
    output_ref[...] = source_ref[...] * 2.0

  def attempt(count: int) -> str:
    try:
      result = pl.pallas_call(
        kernel,
        grid=(steps,),
        in_specs=[
          pl.BlockSpec(
            (8, 128), lambda i: (i, 0), pl.Buffered(buffer_count=count)
          )
        ],
        out_specs=pl.BlockSpec((8, 128), lambda i: (i, 0)),
        out_shape=jax.ShapeDtypeStruct(shape, jnp.float32),
      )(jnp.ones(shape, jnp.float32))
      correct = bool(jnp.allclose(jax.block_until_ready(result), 2.0))
    except Exception as error:  # noqa: BLE001 — the rejection is the finding.
      first_line = str(error).strip().splitlines()[0][:150]
      return f"REJECTED({type(error).__name__}: {first_line})"
    return f"ACCEPTED(correct={correct})"

  arms = {count: attempt(count) for count in (1, 2, 3, 4)}

  return (
    "pl.Buffered(buffer_count=N) as the pipeline_mode of a TOP-LEVEL "
    "pallas_call grid BlockSpec: "
    + " ".join(f"N={count}:{verdict}" for count, verdict in arms.items())
    + " || the position matters: this is the outer grid, not an inner "
    "pltpu.emit_pipeline BlockSpec"
  )


@harness.probe(
  _GROUP,
  "block_sublane_legality_by_dtype",
  "Is the divisible-by-8 block rule the same for f32, bf16 and int8?",
)
def _block_sublane_legality_by_dtype() -> str:
  """Asks which block shapes compile, holding the kernel body inert.

  Deliberately a pure copy. `sublane_tiling_by_dtype` established that Mosaic
  will not do arithmetic on an int8 vector at all, so any int8 arm that does
  work conflates two rejections; the copy isolates the shape question.

  The rule under test is the one every tiling guide states as "(8, 128)".
  It is stated for f32 and repeated for everything else, and the tiling
  formula says the sublane tile widens as the dtype narrows, so the two
  cannot both be right at 8 bits.
  """
  import jax
  import jax.numpy as jnp
  from jax.experimental import pallas as pl
  from jax.experimental.pallas import tpu as pltpu

  info = pltpu.get_tpu_info()
  total_rows = 256

  def kernel(source_ref, output_ref) -> None:
    output_ref[...] = source_ref[...]

  def attempt(dtype, sublanes: int) -> bool:
    steps = total_rows // sublanes
    shape = (total_rows, 128)
    jax.block_until_ready(
      pl.pallas_call(
        kernel,
        grid=(steps,),
        in_specs=[pl.BlockSpec((sublanes, 128), lambda i: (i, 0))],
        out_specs=pl.BlockSpec((sublanes, 128), lambda i: (i, 0)),
        out_shape=jax.ShapeDtypeStruct(shape, dtype),
      )(jnp.ones(shape, dtype))
    )
    return True

  findings = []
  for name in ("float32", "bfloat16", "int8"):
    dtype = jnp.dtype(getattr(jnp, name))
    accepted, rejected = [], []
    first_error = ""
    for sublanes in (8, 16, 32, 64):
      try:
        attempt(dtype, sublanes)
      except Exception as error:  # noqa: BLE001 — a rejection is the finding.
        rejected.append(sublanes)
        if not first_error:
          first_error = (
            f"{type(error).__name__}: "
            f"{str(error).strip().splitlines()[0][:150]}"
          )
      else:
        accepted.append(sublanes)
    note = f" first_rejection[{first_error}]" if first_error else ""
    findings.append(
      f"{name}(tiling={info.get_sublane_tiling(dtype)}): "
      f"ok={accepted} rejected={rejected}{note}"
    )

  return "block sublane dim over a (256,128) array, pure copy: " + " || ".join(
    findings
  )


# --------------------------------------------------------------------------
# DMA: what a runtime value may and may not be used for
# --------------------------------------------------------------------------


def _dma_offset_kernel_factory(rows: int, prove_multiple_of: int | None):
  """Builds a kernel that DMAs `rows` rows out of HBM at a runtime offset."""
  from jax.experimental import pallas as pl
  from jax.experimental.pallas import tpu as pltpu

  def kernel(
    offset_ref, source_ref, output_ref, scratch_ref, semaphore
  ) -> None:
    offset = offset_ref[0]
    if prove_multiple_of is not None:
      offset = pl.multiple_of(offset, prove_multiple_of)
    copy = pltpu.make_async_copy(
      source_ref.at[pl.ds(offset, rows)], scratch_ref, semaphore
    )
    copy.start()
    copy.wait()
    output_ref[...] = scratch_ref[...]

  return kernel


@harness.probe(
  _GROUP,
  "traced_dma_offset_by_dtype",
  "May a DMA offset be a runtime value, and does the dtype change the answer?",
)
def _traced_dma_offset_by_dtype() -> str:
  """Separates "the offset is misaligned" from "the offset is unprovable".

  Every arm here is passed an offset that *is* legally aligned. If a rejection
  still lands, the objection cannot be about the value, and the whole rule
  changes shape: it stops being arithmetic the caller can satisfy by choosing
  a better number and becomes a proof obligation the caller has to discharge.
  """
  import jax
  import jax.numpy as jnp
  from jax.experimental import pallas as pl
  from jax.experimental.pallas import tpu as pltpu

  rows = 64
  total_rows = 512

  def attempt(dtype, offset: int, prove_multiple_of: int | None) -> str:
    shape = (total_rows, 128)
    source = jnp.arange(total_rows * 128, dtype=jnp.float32).reshape(shape)
    source = source.astype(dtype)
    try:
      result = pl.pallas_call(
        _dma_offset_kernel_factory(rows, prove_multiple_of),
        in_specs=[
          pl.BlockSpec(memory_space=pltpu.SMEM),
          pl.BlockSpec(memory_space=pl.ANY),
        ],
        out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
        out_shape=jax.ShapeDtypeStruct((rows, 128), dtype),
        scratch_shapes=[
          pltpu.VMEM((rows, 128), dtype),
          pltpu.SemaphoreType.DMA,
        ],
      )(jnp.array([offset], jnp.int32), source)
      host = jax.block_until_ready(result)
    except Exception as error:  # noqa: BLE001 — the rejection is the finding.
      return f"REJECTED({str(error).strip().splitlines()[0][:170]})"
    expected = source[offset : offset + rows]
    correct = bool(jnp.array_equal(host, expected))
    return f"ok(correct={correct})"

  findings = []
  for name in ("float32", "bfloat16"):
    dtype = jnp.dtype(getattr(jnp, name))
    # Every offset below is a multiple of 64, so it is aligned for any
    # sublane tiling this chip uses. A rejection here is about provability.
    plain = {
      offset: attempt(dtype, offset, None) for offset in (0, 64, 128, 320)
    }
    proved = attempt(dtype, 128, 64)
    findings.append(
      f"{name}: "
      + " ".join(
        f"offset{offset}={verdict}" for offset, verdict in plain.items()
      )
      + f" | with pl.multiple_of(offset, 64) at offset 128: {proved}"
    )

  return (
    "a 64-row DMA out of an HBM ref at an offset read from SMEM: "
    + " || ".join(findings)
  )


@harness.probe(
  _GROUP,
  "traced_dma_length",
  "May the number of rows a DMA transfers be a runtime value?",
)
def _traced_dma_length() -> str:
  """Tests the transferred length, which is a different thing from the shape.

  "DMA sizes must be static" and "block shapes must be static" get stated as
  one rule, and they are not one rule. If a length may be traced, a ragged
  tail needs no power-of-two decomposition and no masking — the copy just
  moves fewer rows.
  """
  import jax
  import jax.numpy as jnp
  from jax.experimental import pallas as pl
  from jax.experimental.pallas import tpu as pltpu

  total_rows = 512
  capacity = 64

  def kernel(
    length_ref, source_ref, output_ref, scratch_ref, semaphore
  ) -> None:
    scratch_ref[...] = jnp.zeros_like(scratch_ref)
    length = length_ref[0]
    copy = pltpu.make_async_copy(
      source_ref.at[pl.ds(0, length)],
      scratch_ref.at[pl.ds(0, length)],
      semaphore,
    )
    copy.start()
    copy.wait()
    output_ref[...] = scratch_ref[...]

  shape = (total_rows, 128)
  source = jnp.arange(total_rows * 128, dtype=jnp.float32).reshape(shape)

  def attempt(length: int) -> str:
    try:
      result = pl.pallas_call(
        kernel,
        in_specs=[
          pl.BlockSpec(memory_space=pltpu.SMEM),
          pl.BlockSpec(memory_space=pl.ANY),
        ],
        out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
        out_shape=jax.ShapeDtypeStruct((capacity, 128), jnp.float32),
        scratch_shapes=[
          pltpu.VMEM((capacity, 128), jnp.float32),
          pltpu.SemaphoreType.DMA,
        ],
      )(jnp.array([length], jnp.int32), source)
      host = jax.block_until_ready(result)
    except Exception as error:  # noqa: BLE001 — the rejection is the finding.
      return f"REJECTED({str(error).strip().splitlines()[0][:170]})"
    moved = bool(jnp.array_equal(host[:length], source[:length]))
    untouched = bool(jnp.all(host[length:] == 0.0))
    return (
      f"ok(moved_rows_correct={moved} rows_past_length_still_zero={untouched})"
    )

  arms = {length: attempt(length) for length in (8, 40, 64)}

  return (
    f"a DMA of a runtime number of rows into a fixed {capacity}-row VMEM "
    "scratch, length read from SMEM: "
    + " ".join(f"length{length}={verdict}" for length, verdict in arms.items())
    + " || the scratch SHAPE is static in every arm; only the transferred "
    "extent is traced"
  )


@harness.probe(
  _GROUP,
  "dynamic_index_lane_versus_sublane",
  "Is a runtime index allowed on the lane axis, the sublane axis, or both?",
)
def _dynamic_index_lane_versus_sublane() -> str:
  """Measures the asymmetry that forces lane-axis gathers into a one-hot.

  The two trailing axes are described as "sublanes and lanes" as though they
  were a pair. For a runtime index they are not remotely a pair, and which
  one you are indexing decides whether the natural code compiles or has to be
  rewritten as a matmul against a one-hot mask.
  """
  import jax
  import jax.numpy as jnp
  from jax.experimental import pallas as pl
  from jax.experimental.pallas import tpu as pltpu

  rows, columns = 256, 512

  def sublane_kernel(index_ref, source_ref, output_ref) -> None:
    output_ref[...] = source_ref[pl.ds(index_ref[0], 8), :]

  def lane_kernel(index_ref, source_ref, output_ref) -> None:
    output_ref[...] = source_ref[:, pl.ds(index_ref[0], 128)]

  def attempt(kernel, out_shape, index: int) -> str:
    source = jnp.arange(rows * columns, dtype=jnp.float32).reshape(
      rows, columns
    )
    try:
      result = pl.pallas_call(
        kernel,
        in_specs=[
          pl.BlockSpec(memory_space=pltpu.SMEM),
          pl.BlockSpec(memory_space=pltpu.VMEM),
        ],
        out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
        out_shape=jax.ShapeDtypeStruct(out_shape, jnp.float32),
      )(jnp.array([index], jnp.int32), source)
      jax.block_until_ready(result)
    except Exception as error:  # noqa: BLE001 — the rejection is the finding.
      return f"REJECTED({str(error).strip().splitlines()[0][:170]})"
    return "ACCEPTED"

  # 8 and 128 are the respective tile extents, so both indices are aligned.
  sublane_arm = attempt(sublane_kernel, (8, columns), 16)
  lane_arm = attempt(lane_kernel, (rows, 128), 128)

  return (
    f"an f32 ({rows},{columns}) VMEM ref sliced at an index read from SMEM: "
    f"sublane axis, source[ds(i,8), :] at i=16 -> {sublane_arm} || "
    f"lane axis, source[:, ds(i,128)] at i=128 -> {lane_arm}"
  )


# --------------------------------------------------------------------------
# Destructive: these can halt the Mosaic core and poison the process
# --------------------------------------------------------------------------


@harness.probe(
  _GROUP,
  "zero_size_dma_needs_no_wait",
  "Is a zero-size DMA a true no-op, or does it still owe a wait?",
  destructive=True,
)
def _zero_size_dma_needs_no_wait() -> str:
  """Licenses — or refuses — predication by size instead of by branch.

  If a zero-length copy is genuinely nothing, a grid step that should do no
  work can issue a copy of zero rows rather than branch around it, which
  keeps every step's control flow identical. If it is not, the same idiom
  leaves a semaphore owed and takes the core down.

  Destructive because the failing side of that question is a core halt, and
  a halt wedges the device for every probe queued behind it in this process.
  """
  import jax
  import jax.numpy as jnp
  from jax.experimental import pallas as pl
  from jax.experimental.pallas import tpu as pltpu

  rows = 64
  shape = (rows, 128)

  def kernel(source_ref, output_ref, scratch_ref, semaphore) -> None:
    scratch_ref[...] = jnp.zeros_like(scratch_ref)
    copy = pltpu.make_async_copy(
      source_ref.at[pl.ds(0, 0)], scratch_ref.at[pl.ds(0, 0)], semaphore
    )
    copy.start()
    # Deliberately no wait(). If a zero-size copy is a real DMA this leaves
    # the semaphore un-drained and the kernel should not survive exit.
    output_ref[...] = scratch_ref[...]

  result = pl.pallas_call(
    kernel,
    in_specs=[pl.BlockSpec(memory_space=pl.ANY)],
    out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
    out_shape=jax.ShapeDtypeStruct(shape, jnp.float32),
    scratch_shapes=[pltpu.VMEM(shape, jnp.float32), pltpu.SemaphoreType.DMA],
  )(jnp.ones(shape, jnp.float32))
  host = jax.block_until_ready(result)

  return (
    "a zero-row async copy was started and never waited, and the kernel "
    f"still completed: zero-size behaves as a no-op. output all zero: "
    f"{bool(jnp.all(host == 0.0))}"
  )


@harness.probe(
  _GROUP,
  "undrained_semaphore_halts_core",
  "What does leaving a real DMA un-waited actually do?",
  destructive=True,
)
def _undrained_semaphore_halts_core() -> str:
  """Establishes what the balance rule costs when it is broken.

  "Semaphores must balance before the kernel exits" is stated everywhere and
  the consequence almost nowhere, which leaves a reader unable to recognize
  the failure when they cause it. The failure mode is the deliverable here,
  not the rule.

  Written so that the rule holding produces `FAIL`: if this probe returns
  normally, an un-drained semaphore was tolerated.
  """
  import jax
  import jax.numpy as jnp
  from jax.experimental import pallas as pl
  from jax.experimental.pallas import tpu as pltpu

  rows = 64
  shape = (rows, 128)

  def kernel(source_ref, output_ref, scratch_ref, semaphore) -> None:
    copy = pltpu.make_async_copy(source_ref, scratch_ref, semaphore)
    copy.start()
    # No wait(). The semaphore is left holding a completion.
    output_ref[...] = jnp.zeros_like(output_ref)

  result = pl.pallas_call(
    kernel,
    in_specs=[pl.BlockSpec(memory_space=pl.ANY)],
    out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
    out_shape=jax.ShapeDtypeStruct(shape, jnp.float32),
    scratch_shapes=[pltpu.VMEM(shape, jnp.float32), pltpu.SemaphoreType.DMA],
  )(jnp.ones(shape, jnp.float32))
  jax.block_until_ready(result)

  return (
    "a started-but-never-waited full-size DMA completed without error -- "
    "the semaphore balance rule was NOT enforced at runtime here"
  )


@harness.probe(
  _GROUP,
  "xla_reference_vs_f64_oracle",
  "Is an f32 XLA reference more accurate than the Pallas kernel it is judging?",
)
def _xla_reference_vs_f64_oracle() -> str:
  """Checks the instrument, not the kernel.

  A kernel is normally judged against "the same operation in ordinary JAX".
  On TPU that reference is not automatically the accurate side: an f32
  ``einsum`` / ``dot_general`` at default precision lowers to the MXU and
  multiplies in bf16. A Pallas kernel that contracts on the VPU
  (``jnp.sum(a * b)``) stays in true f32 and is therefore *more* accurate
  than the thing grading it — but the mismatch is reported against the
  kernel, so the tempting fix is to loosen the tolerance and ship a false
  accuracy claim.

  Every arm here is measured against a numpy float64 oracle, which is the
  only way to say which side is wrong rather than argue about it. Read the
  detail string for the diagnostic pattern: **the VPU kernel disagrees with
  the XLA reference by far more than it disagrees with the oracle.** When a
  real kernel shows that pattern, the reference is what needs fixing.

  Three arms and what each one is for:

  * ``xla_default`` — the reference as usually written. Expect bf16 error.
  * ``xla_highest`` — the same reference with ``Precision.HIGHEST`` pinned.
    This is the fix, and this probe is also the check that the fix lands at
    XLA level on this stack, which is not guaranteed.
  * ``pallas_vpu`` — the contraction done as a multiply and a reduction
    inside a Pallas kernel, never touching the MXU.

  Note this cannot be reproduced under CPU interpret mode: there is no MXU
  there, so every arm is f32 and the whole effect disappears. A green local
  run says nothing about it.
  """
  import jax
  import jax.numpy as jnp
  import numpy as np
  from jax.experimental import pallas as pl

  rows, cols, depth = 64, 64, 128
  generator = np.random.default_rng(0)
  left_host = generator.standard_normal((rows, depth), dtype=np.float32)
  right_host = generator.standard_normal((cols, depth), dtype=np.float32)

  # out[i, j] = sum_k left[i, k] * right[j, k]
  exact = np.einsum(
    "ik,jk->ij",
    left_host.astype(np.float64),
    right_host.astype(np.float64),
  )
  scale = float(np.max(np.abs(exact)))

  left = jnp.asarray(left_host)
  right = jnp.asarray(right_host)

  def to_host(result) -> np.ndarray:
    return np.asarray(jax.block_until_ready(result), dtype=np.float64)

  def relative_to_oracle(result) -> float:
    return float(np.max(np.abs(to_host(result) - exact))) / scale

  def vpu_kernel(left_ref, right_ref, output_ref) -> None:
    # A multiply and a reduction: the VPU, in true f32, no dot_general.
    product = left_ref[...][:, None, :] * right_ref[...][None, :, :]
    output_ref[...] = jnp.sum(product, axis=-1)

  arms: dict[str, float] = {}
  results: dict[str, np.ndarray] = {}

  def record(name: str, result) -> None:
    arms[name] = relative_to_oracle(result)
    results[name] = to_host(result)

  record(
    "xla_default",
    jnp.einsum("ik,jk->ij", left, right, precision=jax.lax.Precision.DEFAULT),
  )
  record(
    "xla_highest",
    jnp.einsum("ik,jk->ij", left, right, precision=jax.lax.Precision.HIGHEST),
  )

  try:
    pallas_call = jax.jit(
      pl.pallas_call(
        vpu_kernel,
        out_shape=jax.ShapeDtypeStruct((rows, cols), jnp.float32),
      )
    )
    record("pallas_vpu", pallas_call(left, right))
  except Exception as error:  # noqa: BLE001 — a refusal is the finding.
    raise harness.Unsupported(
      f"the VPU contraction did not lower, so the comparison this probe "
      f"exists to make cannot be built: {type(error).__name__}: "
      f"{str(error).strip()[:160]}"
    ) from error

  kernel_vs_reference = (
    float(np.max(np.abs(results["pallas_vpu"] - results["xla_default"])))
    / scale
  )
  f32_eps = float(np.finfo(np.float32).eps)
  bf16_like = arms["xla_default"] / f32_eps

  verdict = (
    "the reference is the inaccurate side"
    if arms["pallas_vpu"] < arms["xla_default"]
    else "the reference is at least as accurate as the kernel here"
  )

  return (
    f"contraction ik,jk->ij at rows={rows} cols={cols} depth={depth}, f32 "
    f"operands, relative max error against a numpy float64 oracle: "
    + " ".join(f"{name}={value:.4e}" for name, value in arms.items())
    + f" || kernel-vs-reference disagreement {kernel_vs_reference:.4e}, "
    f"which is what a test would report and would report against the "
    f"kernel || f32 eps {f32_eps:.2e}; xla_default sits at "
    f"{bf16_like:.0f}x f32 eps || {verdict}. The diagnostic is a kernel "
    f"that fails the reference and matches the oracle: fix the reference "
    f"with Precision.HIGHEST, do not loosen the tolerance"
  )


@harness.probe(
  _GROUP,
  "input_output_alias_in_place",
  "When a pallas_call aliases an input to its output, does the caller pay a "
  "copy of the whole operand, with and without buffer donation?",
)
def _input_output_alias_in_place() -> str:
  """Prices `input_output_aliases` the way MaxKernel's harness calls a kernel.

  Hypothesis: aliasing an input to an output lets a kernel update a few rows
  of a large buffer in place, but only if the caller donates the buffer. The
  MaxKernel test harness jits `computation` without `donate_argnums`, so XLA
  must keep the caller's array intact; if it does that by copying the whole
  operand before the kernel writes into it, an "in-place" kernel pays a cost
  that grows with the buffer, not with the update, and the copy shows up
  outside the kernel in the profile.

  Method: one kernel writes a single (8, cols) block of a (rows, cols) f32
  buffer through an aliased output. Three arms at two buffer sizes:
  `fresh_output` (no alias, the kernel alone: the control), `aliased` (jit
  without donation, the harness case) and `aliased_donated` (jit with
  `donate_argnums=0`). Each arm is checked for correctness (written block
  updated, an untouched block read at a different offset preserved), the
  compiled HLO is searched for a copy of the full operand, and the
  per-call time is compared across the two sizes. A cost that scales with
  the buffer means a full copy is paid.
  """
  import re

  import jax
  import jax.numpy as jnp
  import numpy as np
  from jax.experimental import pallas as pl

  if jax.default_backend() != "tpu":
    raise harness.Unsupported("needs a TPU: the copy is decided by XLA:TPU")

  cols = 1024
  sizes = (16384, 65536)  # rows; the second is 4x the first

  def kernel(x_ref, o_ref) -> None:
    o_ref[...] = x_ref[...] + 1.0

  def build(rows: int, alias: bool):
    block = pl.BlockSpec((8, cols), lambda i: (0, 0))
    out_rows = rows if alias else 8
    return pl.pallas_call(
      kernel,
      grid=(1,),
      in_specs=[block],
      out_specs=block,
      out_shape=jax.ShapeDtypeStruct((out_rows, cols), jnp.float32),
      input_output_aliases={0: 0} if alias else {},
    )

  def full_copies(compiled_text: str, rows: int) -> int:
    shape = f"f32[{rows},{cols}]"
    return sum(
      1
      for line in compiled_text.splitlines()
      if shape in line.split("=", 1)[-1][:80] and re.search(r"\bcopy", line)
    )

  report = []
  times: dict[str, list[float]] = {}
  for rows in sizes:
    for arm in ("fresh_output", "aliased", "aliased_donated"):
      alias = arm != "fresh_output"
      call = build(rows, alias)
      fn = (
        jax.jit(call, donate_argnums=0)
        if arm == "aliased_donated"
        else jax.jit(call)
      )
      base = jnp.arange(rows * cols, dtype=jnp.float32).reshape(rows, cols)
      base = base / (rows * cols)
      expected_block = np.asarray(base[:8]) + 1.0
      untouched = np.asarray(base[8:16])
      copies = full_copies(fn.lower(base).compile().as_text(), rows)

      out = fn(base)
      jax.block_until_ready(out)
      ok = np.allclose(np.asarray(out[:8]), expected_block)
      if alias:
        ok = ok and np.allclose(np.asarray(out[8:16]), untouched)
      if not ok:
        raise harness.Invalid(f"{arm} at rows={rows} computed a wrong result")

      # Chain the output back in, so the donated arm always has a live
      # buffer and every arm does the same work per call.
      state = {"buf": out if alias else base}

      def step(fn=fn, state=state, alias=alias):
        result = fn(state["buf"])
        if alias:
          state["buf"] = result
        return result

      timing = harness.time_above_floor(step, what=f"{arm} rows={rows}")
      harness.require_above_dispatch_floor(timing, what=f"{arm} rows={rows}")
      times.setdefault(arm, []).append(timing.median)
      report.append(
        f"{arm} rows={rows}: {timing.median * 1e6:.1f}us "
        f"full_operand_copies_in_hlo={copies}"
      )

  scaling = {
    arm: values[1] / values[0] for arm, values in times.items() if values[0]
  }
  return (
    " | ".join(report)
    + " || time ratio when the buffer grows 4x (about 1 = no copy, about 4 ="
    + " a full-operand copy): "
    + " ".join(f"{arm}={ratio:.2f}" for arm, ratio in scaling.items())
  )
