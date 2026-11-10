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
"""What is computed once, where it is kept, and what recomputing it costs.

Supplies answers for `reference/design/precomputation.md`. Every probe here is
about
the same axis: a quantity that does not change over the grid, over the layers,
or over the decode steps, and the levels it can be hoisted to.

Three things this module has to establish, in order, because a design cannot
use the third without the first two:

* **The mechanism exists.** An SMEM scratch survives a grid step; a prefetched
  scalar reaches an `index_map`; the dtypes tokamax downcasts its metadata to
  are actually accepted. These are `PASS`/`FAIL`, not timings.
* **The compiler does not already do it.** If Mosaic hoists grid-invariant
  work out of the grid loop by itself, every design decision below is wasted
  effort. `grid_invariant_recompute_cost` is written so that "the compiler
  handles it" shows up as a ratio of 1.0.
* **The hoist is cheaper than the recompute.** Precomputation is not free:
  building metadata in-kernel costs a scalar loop, and building it outside
  costs an XLA op and an SMEM transfer. Both are measured against the thing
  they replace.

Every timing arm here reports the arm it is a ratio against, measured in the
same process, and every kernel that allocates a buffer also reads it back — see
`probes/README.md` for why both of those are non-negotiable.
"""

from __future__ import annotations

import harness

_GROUP = "precomputation"


# --------------------------------------------------------------------------
# Dry-run plumbing
# --------------------------------------------------------------------------
#
# `JAX_PLATFORMS=cpu PROBE_DRY_RUN=1 python run_probes.py --group precomputation`
# runs these kernels under `pltpu.InterpretParams()` to catch a typo, a wrong
# `index_map` arity or a bad `scratch_shapes` signature in seconds, on the
# machine, instead of after a wait in the queue.
#
# Interpret mode does not invoke Mosaic, so it cannot answer any question this
# module actually asks: not whether an SMEM dtype is accepted, not what a
# recompute costs. It answers them backwards — a kernel with no TPU lowering
# runs happily on CPU. So a CPU run without the opt-in raises `Invalid` before
# touching a device, and a CPU run *with* the opt-in stamps every detail string
# `DRY-RUN(cpu-interpret)` so no number from it can be quoted by accident.


def _dry_run() -> bool:
  """True on the opted-in CPU dry run; raises on an accidental CPU run."""
  import jax

  if jax.devices()[0].platform == "tpu":
    return False
  if not harness.env_flag("PROBE_DRY_RUN"):
    raise harness.Invalid(
      "not running on a TPU and PROBE_DRY_RUN is unset. Interpret mode "
      "does not invoke Mosaic, so it cannot answer this question and "
      "would answer it backwards. No verdict."
    )
  return True


def _interpret(dry: bool):
  """The `interpret=` argument for `pallas_call`, or None on hardware."""
  if not dry:
    return None
  from jax.experimental.pallas import tpu as pltpu

  return pltpu.InterpretParams()


def _stamp(dry: bool, detail: str) -> str:
  """Marks a dry-run detail string so its contents cannot be quoted."""
  return f"DRY-RUN(cpu-interpret) {detail}" if dry else detail


# --------------------------------------------------------------------------
# 1. Does the mechanism exist?
# --------------------------------------------------------------------------


@harness.probe(
  _GROUP,
  "scratch_persists_across_grid_steps",
  "Does a scratch written at grid step 0 still hold its value at step N?",
)
def _scratch_persists_across_grid_steps() -> str:
  """The precondition for filling metadata once and reading it every step.

  `gmm_v2` fills an SMEM `MetadataRef` at the top of `kernel_main` and then
  every `index_map` reads it on every grid step. That only works if scratch
  is allocated once for the whole `pallas_call` rather than per step, and
  nothing in the API says which. If scratch were per-step, the pattern would
  not fail loudly — it would read whatever was in the buffer, which is often
  the previous step's values and therefore often nearly right.

  Both memory spaces are tested because the answer need not be the same, and
  a design hoisting a *table* rather than a *scalar* needs the VMEM answer.
  """
  import jax
  import jax.numpy as jnp
  from jax.experimental import pallas as pl
  from jax.experimental.pallas import tpu as pltpu

  dry = _dry_run()
  steps = 8
  sentinel = 12345.0

  def smem_kernel(out_ref, scratch_ref) -> None:
    step = pl.program_id(0)

    @pl.when(step == 0)
    def _fill() -> None:
      # Written once, at step 0 only. Any later step reading the right
      # value read it from a buffer that outlived its own grid step.
      for index in range(steps):
        scratch_ref[index] = index * 3 + 1

    out_ref[step] = scratch_ref[step]

  smem_out = pl.pallas_call(
    smem_kernel,
    grid=(steps,),
    out_shape=jax.ShapeDtypeStruct((steps,), jnp.int32),
    out_specs=pl.BlockSpec(memory_space=pltpu.SMEM),
    scratch_shapes=[pltpu.SMEM((steps,), jnp.int32)],
    interpret=_interpret(dry),
  )()
  smem_expected = [index * 3 + 1 for index in range(steps)]
  smem_observed = [int(value) for value in jax.device_get(smem_out)]
  smem_ok = smem_observed == smem_expected

  def vmem_kernel(out_ref, scratch_ref) -> None:
    step = pl.program_id(0)

    @pl.when(step == 0)
    def _fill() -> None:
      scratch_ref[...] = jnp.full((8, 128), sentinel, jnp.float32)

    # The step index is added *after* the read, so a scratch that did not
    # persist shows up as a wrong base value rather than a wrong offset.
    out_ref[...] = scratch_ref[...] + step.astype(jnp.float32)

  vmem_out = pl.pallas_call(
    vmem_kernel,
    grid=(steps,),
    out_shape=jax.ShapeDtypeStruct((steps * 8, 128), jnp.float32),
    out_specs=pl.BlockSpec((8, 128), lambda step: (step, 0)),
    scratch_shapes=[pltpu.VMEM((8, 128), jnp.float32)],
    interpret=_interpret(dry),
  )()
  vmem_host = jax.device_get(vmem_out)
  vmem_ok = all(
    bool((vmem_host[step * 8 : step * 8 + 8, :] == sentinel + step).all())
    for step in range(steps)
  )

  detail = (
    f"a scratch written only under pl.when(program_id==0) and read at every "
    f"one of {steps} grid steps: "
    f"SMEM int32 scratch persists={smem_ok} (read back {smem_observed}) || "
    f"VMEM f32 scratch persists={vmem_ok} (sentinel {sentinel:g} survived every step) "
    f"|| a False here would mean scratch is per-step and in-kernel metadata "
    f"has to be rebuilt every step"
  )
  if not (smem_ok and vmem_ok):
    return harness.Verdict.FAIL, _stamp(dry, detail)
  return _stamp(dry, detail)


@harness.probe(
  _GROUP,
  "scalar_prefetch_drives_index_map",
  "Can a prefetched scalar array choose which block each grid step loads?",
)
def _scalar_prefetch_drives_index_map() -> str:
  """The zero-per-step-cost way to spend precomputed metadata.

  Metadata that reaches the `index_map` moves the DMA rather than the
  compute: the block a step reads is chosen by a scalar the kernel body never
  sees and never pays for. This is what `ragged_dot` buys with
  `PrefetchScalarGridSpec(num_scalar_prefetch=2)` and what splash attention
  buys with 6.

  The permutation is deliberately not the identity, so a kernel that ignored
  the prefetched scalars entirely would produce the un-permuted answer and
  fail rather than pass.
  """
  import jax
  import jax.numpy as jnp
  from jax.experimental import pallas as pl
  from jax.experimental.pallas import tpu as pltpu

  dry = _dry_run()
  blocks = 4
  permutation = jnp.array([3, 1, 2, 0], jnp.int32)

  def kernel(_perm_ref, x_ref, out_ref) -> None:
    out_ref[...] = x_ref[...]

  def source_index_map(step, perm_ref):
    return (perm_ref[step], 0)

  def out_index_map(step, _perm_ref):
    return (step, 0)

  source = jnp.repeat(
    jnp.arange(blocks, dtype=jnp.float32)[:, None], 8 * 128, axis=1
  ).reshape(blocks * 8, 128)

  result = pl.pallas_call(
    kernel,
    grid_spec=pltpu.PrefetchScalarGridSpec(
      num_scalar_prefetch=1,
      grid=(blocks,),
      in_specs=[pl.BlockSpec((8, 128), source_index_map)],
      out_specs=pl.BlockSpec((8, 128), out_index_map),
    ),
    out_shape=jax.ShapeDtypeStruct((blocks * 8, 128), jnp.float32),
    interpret=_interpret(dry),
  )(permutation, source)

  host = jax.device_get(result)
  observed = [float(host[step * 8, 0]) for step in range(blocks)]
  expected = [float(value) for value in jax.device_get(permutation)]
  correct = observed == expected

  detail = (
    f"PrefetchScalarGridSpec(num_scalar_prefetch=1), index_map reading the "
    f"prefetched int32[{blocks}] to permute which block each step loads: "
    f"requested={expected} observed={observed} correct={correct} "
    f"|| the identity would have given {list(range(blocks))}, so this arm "
    f"distinguishes the scalars being used from being ignored"
  )
  if not correct:
    return harness.Verdict.FAIL, _stamp(dry, detail)
  return _stamp(dry, detail)


@harness.probe(
  _GROUP,
  "smem_metadata_dtype_support",
  "Which dtypes may metadata use in SMEM, as a scratch and as an operand?",
)
def _smem_metadata_dtype_support() -> str:
  """Whether the SMEM budget can be bought down by narrowing the metadata.

  splash attention downcasts its mask metadata to the narrowest integer that
  holds the range before putting it in SMEM (`downcast_smem_data=True`,
  int8 for a 0/1/2 block mask, int8/int16/int32 chosen by range for the
  rest), and tokamax's GDN puts plain `bool` predicates there
  (`p_id_is_first_tile`). Both are claims about what SMEM accepts, and
  neither has been checked on this chip.

  Each arm writes at one index and reads back at a different one, so a dtype
  that is "accepted" because the whole buffer was eliminated shows up as a
  wrong value rather than a pass.
  """
  import jax
  import jax.numpy as jnp
  from jax.experimental import pallas as pl
  from jax.experimental.pallas import tpu as pltpu

  dry = _dry_run()
  elements = 64
  candidates = {
    "int32": jnp.int32,
    "int16": jnp.int16,
    "int8": jnp.int8,
    "uint32": jnp.uint32,
    "bool": jnp.bool_,
    "float32": jnp.float32,
  }

  scratch_results = {}
  operand_results = {}

  for label, dtype in candidates.items():
    is_bool = dtype is jnp.bool_

    def scratch_kernel(
      out_ref, scratch_ref, _dtype=dtype, _is_bool=is_bool
    ) -> None:
      for index in range(elements):
        value = (index % 2 == 0) if _is_bool else (index % 7)
        scratch_ref[index] = jnp.array(value, _dtype)
      # Read back at an index other than the last one written.
      out_ref[0] = scratch_ref[elements // 2].astype(jnp.int32)
      out_ref[1] = scratch_ref[elements // 2 + 1].astype(jnp.int32)

    try:
      written = pl.pallas_call(
        scratch_kernel,
        out_shape=jax.ShapeDtypeStruct((2,), jnp.int32),
        out_specs=pl.BlockSpec(memory_space=pltpu.SMEM),
        scratch_shapes=[pltpu.SMEM((elements,), dtype)],
        interpret=_interpret(dry),
      )()
      observed = [int(value) for value in jax.device_get(written)]
      half = elements // 2
      expected = (
        [int(half % 2 == 0), int((half + 1) % 2 == 0)]
        if is_bool
        else [half % 7, (half + 1) % 7]
      )
      scratch_results[label] = (
        f"ok(correct={observed == expected})"
        if observed == expected
        else f"WRONG(got={observed} want={expected})"
      )
    except Exception as error:  # noqa: BLE001 — the rejection is the measurement.
      scratch_results[label] = (
        f"REJECTED({type(error).__name__}: {str(error).strip()[:90]})"
      )

    def operand_kernel(meta_ref, out_ref) -> None:
      out_ref[0] = meta_ref[elements // 2].astype(jnp.int32)

    try:
      values = (
        jnp.arange(elements) % 2 == 0
        if is_bool
        else (jnp.arange(elements) % 7).astype(dtype)
      )
      read = pl.pallas_call(
        operand_kernel,
        out_shape=jax.ShapeDtypeStruct((1,), jnp.int32),
        in_specs=[pl.BlockSpec(memory_space=pltpu.SMEM)],
        out_specs=pl.BlockSpec(memory_space=pltpu.SMEM),
        interpret=_interpret(dry),
      )(values)
      want = int(jax.device_get(values)[elements // 2])
      got = int(jax.device_get(read)[0])
      operand_results[label] = (
        f"ok(correct={got == want})"
        if got == want
        else f"WRONG(got={got} want={want})"
      )
    except Exception as error:  # noqa: BLE001
      operand_results[label] = (
        f"REJECTED({type(error).__name__}: {str(error).strip()[:90]})"
      )

  scratch_text = " ".join(f"{k}={v}" for k, v in scratch_results.items())
  operand_text = " ".join(f"{k}={v}" for k, v in operand_results.items())
  bytes_per = {k: jnp.dtype(v).itemsize for k, v in candidates.items()}
  detail = (
    f"an SMEM buffer of {elements} elements, written at every index and read "
    f"back at index {elements // 2}: "
    f"as a scratch: {scratch_text} || as an input operand: {operand_text} "
    f"|| numpy itemsize per dtype: {bytes_per} -- the question this answers is "
    f"whether downcasting metadata buys SMEM, since SMEM capacity is small"
  )
  return _stamp(dry, detail)


# --------------------------------------------------------------------------
# 2. Does the compiler already do it?
# --------------------------------------------------------------------------


@harness.probe(
  _GROUP,
  "grid_invariant_recompute_cost",
  "Does Mosaic hoist grid-invariant work out of the grid, or do you pay it every step?",
)
def _grid_invariant_recompute_cost() -> str:
  """The probe the whole area stands on.

  Three arms compute the same output from the same grid-invariant table:

  * `recompute` builds the table and reduces it, every step;
  * `vmem_hoist` builds the table once into a VMEM scratch at step 0, and
    reduces it every step;
  * `smem_hoist` builds the table *and* its reduction once at step 0, keeps
    the single scalar in SMEM, and reads that scalar every step.

  The ladder is what makes the answer readable. If Mosaic performed loop-
  invariant code motion on the grid loop, all three arms would land on the
  same number and manual hoisting would be pure cost. If only the third arm
  is fast, the table build *and* its reduction both have to be hoisted by
  hand, which is the shape of the advice the reference file gives.

  The table is consumed through a full `jnp.sum`, not a slice of it: a slice
  invites the compiler to narrow the exponential to the elements actually
  read, which would measure a table 1/512th the size while looking correct.
  """
  import jax
  import jax.numpy as jnp
  from jax.experimental import pallas as pl
  from jax.experimental.pallas import tpu as pltpu

  dry = _dry_run()
  table = 8 if dry else 256
  steps = 4 if dry else 512
  batch = 1 if dry else 8

  def build_table() -> jax.Array:
    # `tpu.iota` refuses a float result type — an f32 iota fails Mosaic
    # verification with "must be vector of integer or index values". Build
    # it in int32 and convert.
    rows = jax.lax.broadcasted_iota(jnp.int32, (table, table), 0).astype(
      jnp.float32
    )
    columns = jax.lax.broadcasted_iota(jnp.int32, (table, table), 1).astype(
      jnp.float32
    )
    # Transcendental work, so the table is expensive enough to see, and
    # genuinely independent of program_id, so it is legally hoistable.
    return jnp.exp(-0.0001 * (rows - columns) * (rows - columns))

  def recompute_kernel(x_ref, out_ref) -> None:
    out_ref[...] = x_ref[...] + jnp.sum(build_table())

  def vmem_hoist_kernel(x_ref, out_ref, table_ref) -> None:
    @pl.when(pl.program_id(0) == 0)
    def _fill() -> None:
      table_ref[...] = build_table()

    out_ref[...] = x_ref[...] + jnp.sum(table_ref[...])

  def smem_hoist_kernel(x_ref, out_ref, total_ref) -> None:
    @pl.when(pl.program_id(0) == 0)
    def _fill() -> None:
      total_ref[0] = jnp.sum(build_table())

    out_ref[...] = x_ref[...] + total_ref[0]

  source = jnp.arange(steps * 8 * 128, dtype=jnp.float32).reshape(
    steps * 8, 128
  )
  block = pl.BlockSpec((8, 128), lambda step: (step, 0))
  shared = dict(
    grid=(steps,),
    in_specs=[block],
    out_specs=block,
    out_shape=jax.ShapeDtypeStruct((steps * 8, 128), jnp.float32),
    interpret=_interpret(dry),
  )

  arms = {
    "recompute": jax.jit(pl.pallas_call(recompute_kernel, **shared)),
    "vmem_hoist": jax.jit(
      pl.pallas_call(
        vmem_hoist_kernel,
        scratch_shapes=[pltpu.VMEM((table, table), jnp.float32)],
        **shared,
      )
    ),
    "smem_hoist": jax.jit(
      pl.pallas_call(
        smem_hoist_kernel,
        scratch_shapes=[pltpu.SMEM((1,), jnp.float32)],
        **shared,
      )
    ),
  }

  outputs = {name: jax.device_get(call(source)) for name, call in arms.items()}
  reference = outputs["recompute"]
  disagreement = {
    name: float(abs(value - reference).max()) for name, value in outputs.items()
  }
  if max(disagreement.values()) > 1e-2:
    raise harness.Invalid(
      "the three arms do not compute the same thing, so no timing is "
      f"reportable: max |arm - recompute| = {disagreement}"
    )

  if dry:
    return _stamp(
      dry,
      f"all three arms agree to {max(disagreement.values()):.3g}; "
      "no timing is taken under interpret mode",
    )

  timings = {}
  for name, call in arms.items():
    timing = harness.time_call(lambda c=call: c(source), batch=batch)
    harness.require_above_dispatch_floor(timing, what=f"{name} arm")
    timings[name] = timing

  baseline = timings["smem_hoist"].median
  ratios = {name: timing.median / baseline for name, timing in timings.items()}
  per_step_extra = (
    (timings["recompute"].median - timings["smem_hoist"].median) / steps * 1e9
  )
  detail = (
    f"a grid-invariant {table}x{table} f32 exp table consumed through a full sum, "
    f"over a {steps}-step grid: "
    + " || ".join(f"{name}: {timing}" for name, timing in timings.items())
    + " || ratio against smem_hoist: "
    + " ".join(f"{name}={ratio:.2f}x" for name, ratio in ratios.items())
    + f" || recompute costs {per_step_extra:.0f}ns per grid step more than the "
    f"hoisted scalar. A ratio near 1.0 on every arm would mean Mosaic "
    f"hoists grid-invariant work by itself and none of this is worth doing"
  )
  return detail


@harness.probe(
  _GROUP,
  "metadata_cse_across_pallas_calls",
  "Does XLA compute shared metadata once when several pallas_calls ask for it?",
)
def _metadata_cse_across_pallas_calls() -> str:
  """Checks the assumption tokamax's GDN wrapper leaves in a comment.

  `causal_conv1d_gated_delta_rule/wrapper.py` computes its sequence metadata
  inside the per-call helper and notes: "Will be executed multiple times
  per-layer but will be CSEed by compiler." A transformer calls that helper
  once per layer with identical arguments, so if the compiler does *not*
  common those subgraphs, the metadata is rebuilt tens of times per token and
  the comment is a bug.

  The control arm is what makes this readable: the same graph with metadata
  that genuinely differs per call cannot be CSEd, so the gap between the two
  instruction counts is the size of the deduplicated work. Without it, a
  single instruction count says nothing.
  """
  import jax
  import jax.numpy as jnp
  from jax.experimental import pallas as pl
  from jax.experimental.pallas import tpu as pltpu

  dry = _dry_run()
  calls = 4
  groups = 512

  def kernel(meta_ref, x_ref, out_ref) -> None:
    out_ref[...] = x_ref[...] + meta_ref[groups - 1].astype(jnp.float32)

  call_kernel = pl.pallas_call(
    kernel,
    in_specs=[
      pl.BlockSpec(memory_space=pltpu.SMEM),
      pl.BlockSpec(memory_space=pltpu.VMEM),
    ],
    out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
    out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32),
    interpret=_interpret(dry),
  )

  def build_metadata(sizes, salt):
    # Deliberately the shape of real metadata: a sort and a prefix sum over
    # a group-sizes vector, both of which survive into HLO as recognizable
    # instructions.
    ordered = jnp.sort(sizes + salt)
    return jnp.cumsum(ordered).astype(jnp.int32)

  def shared(sizes, x):
    metadata = build_metadata(sizes, 0)
    total = jnp.zeros((8, 128), jnp.float32)
    for _ in range(calls):
      total = total + call_kernel(metadata, x)
    return total

  def distinct(sizes, x):
    total = jnp.zeros((8, 128), jnp.float32)
    for index in range(calls):
      total = total + call_kernel(build_metadata(sizes, index), x)
    return total

  def once(sizes, x):
    return call_kernel(build_metadata(sizes, 0), x)

  sizes = jnp.arange(groups, dtype=jnp.int32)
  x = jnp.ones((8, 128), jnp.float32)

  def instruction_count(function) -> tuple[int, int, int]:
    text = jax.jit(function).lower(sizes, x).compile().as_text()
    lines = text.splitlines()
    return (
      len(lines),
      sum(line.count(" sort(") + line.count("=sort(") for line in lines),
      text.count("custom-call"),
    )

  shared_lines, shared_sorts, shared_calls = instruction_count(shared)
  distinct_lines, distinct_sorts, distinct_calls = instruction_count(distinct)
  once_lines, once_sorts, _ = instruction_count(once)

  detail = (
    f"{calls} pallas_calls in one jit, metadata = sort+cumsum over int32[{groups}]: "
    f"shared metadata: {shared_lines} HLO lines, {shared_sorts} sort(s), "
    f"{shared_calls} custom-call mentions || "
    f"per-call distinct metadata (the control, cannot be CSEd): "
    f"{distinct_lines} HLO lines, {distinct_sorts} sort(s) || "
    f"one call only: {once_lines} HLO lines, {once_sorts} sort(s) || "
    f"CSE happened if shared sorts ({shared_sorts}) is near the one-call "
    f"count ({once_sorts}) rather than the control ({distinct_sorts})"
  )
  return _stamp(dry, detail)


# --------------------------------------------------------------------------
# 3. What does the hoist itself cost?
# --------------------------------------------------------------------------


@harness.probe(
  _GROUP,
  "smem_scalar_fill_cost",
  "What does building metadata in-kernel cost, per scalar written to SMEM?",
)
def _smem_scalar_fill_cost() -> str:
  """Prices `gmm_v2`'s `fill_metadata` loop.

  Building metadata inside the kernel means a sequential scalar loop on the
  scalar core: `fill_metadata` writes two SMEM entries per gm tile, and the
  tile count grows with the batch. Whether that is free or whether it eats
  the kernel depends on a per-scalar cost nobody has measured, and the answer
  decides how large a problem the in-kernel variant can address before the
  out-of-kernel variant wins.

  Reported as a slope between the two largest arms rather than as an absolute
  at any one size, because the fixed cost of a launch is in every arm and the
  slope cancels it.
  """
  import jax
  import jax.numpy as jnp
  from jax.experimental import pallas as pl
  from jax.experimental.pallas import tpu as pltpu

  dry = _dry_run()
  ladder = (64, 256) if dry else (0, 1024, 4096, 16384)
  batch = 1 if dry else 256

  def make(elements: int):
    allocated = max(elements, 1)

    def kernel(out_ref, scratch_ref) -> None:
      def body(index, carry):
        scratch_ref[index] = carry + index
        return carry + 1

      if elements:
        jax.lax.fori_loop(0, elements, body, 0)
      else:
        scratch_ref[0] = 0
      # Read back somewhere other than the last index written.
      out_ref[0] = scratch_ref[allocated // 2]

    return jax.jit(
      pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((1,), jnp.int32),
        out_specs=pl.BlockSpec(memory_space=pltpu.SMEM),
        scratch_shapes=[pltpu.SMEM((allocated,), jnp.int32)],
        interpret=_interpret(dry),
      )
    )

  arms = {elements: make(elements) for elements in ladder}
  for elements, call in arms.items():
    observed = int(jax.device_get(call())[0])
    expected = (elements // 2) * 2 if elements else 0
    if elements and observed != expected:
      raise harness.Invalid(
        f"the fill loop wrote the wrong value at N={elements}: "
        f"got {observed}, want {expected}. No timing is reportable."
      )

  if dry:
    return _stamp(
      dry, f"fill loops at N={list(ladder)} all wrote correct values"
    )

  timings = {}
  for elements, call in arms.items():
    # Extra warmups: with few, the first timed region can still include
    # one-time warmup cost, which shows up as an outlier and a huge spread.
    timing = harness.time_call(call, batch=batch, warmups=10, repeats=30)
    timings[elements] = timing

  largest, second = ladder[-1], ladder[-2]
  harness.require_above_dispatch_floor(
    timings[largest], what=f"N={largest} arm"
  )
  slope_ns = (
    (timings[largest].median - timings[second].median)
    / (largest - second)
    * 1e9
  )
  fixed_us = timings[ladder[0]].median * 1e6
  detail = (
    "a lax.fori_loop writing N int32 scalars into an SMEM scratch, one kernel "
    "launch per call: "
    + " || ".join(
      f"N={elements}: {timing}" for elements, timing in timings.items()
    )
    + f" || slope between N={second} and N={largest}: {slope_ns:.2f}ns per scalar "
    f"written; the N={ladder[0]} arm is the launch floor at {fixed_us:.1f}us and "
    f"is the denominator the others should be read against"
  )
  return detail


@harness.probe(
  _GROUP,
  "precompute_outside_versus_inside",
  "Is metadata cheaper built in XLA outside the kernel or by a scalar loop inside it?",
)
def _precompute_outside_versus_inside() -> str:
  """The fork tokamax takes both ways, priced.

  `ragged_dot` and GDN build their metadata in XLA before the `pallas_call`
  and hand it in through SMEM; `gmm_v2` hands in the raw group sizes and
  builds the same thing inside the kernel with `fill_metadata`. Both are
  defensible and the source says nothing about which is faster, so the two
  are measured here at the same problem size, end to end — the XLA op and the
  kernel together, since moving work across the boundary moves cost, not
  work.

  Both arms are checked to produce the same output before either is timed.
  """
  import jax
  import jax.numpy as jnp
  from jax.experimental import pallas as pl
  from jax.experimental.pallas import tpu as pltpu

  dry = _dry_run()
  groups = 64 if dry else 1024
  batch = 1 if dry else 256

  def consume_kernel(prefix_ref, x_ref, out_ref) -> None:
    out_ref[...] = x_ref[...] + prefix_ref[groups - 1].astype(jnp.float32)

  outside_call = pl.pallas_call(
    consume_kernel,
    in_specs=[
      pl.BlockSpec(memory_space=pltpu.SMEM),
      pl.BlockSpec(memory_space=pltpu.VMEM),
    ],
    out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
    out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32),
    interpret=_interpret(dry),
  )

  def inside_kernel(sizes_ref, x_ref, out_ref, prefix_ref) -> None:
    def body(index, carry):
      running = carry + sizes_ref[index]
      prefix_ref[index] = running
      return running

    jax.lax.fori_loop(0, groups, body, 0)
    out_ref[...] = x_ref[...] + prefix_ref[groups - 1].astype(jnp.float32)

  inside_call = pl.pallas_call(
    inside_kernel,
    in_specs=[
      pl.BlockSpec(memory_space=pltpu.SMEM),
      pl.BlockSpec(memory_space=pltpu.VMEM),
    ],
    out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
    out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32),
    scratch_shapes=[pltpu.SMEM((groups,), jnp.int32)],
    interpret=_interpret(dry),
  )

  sizes = (jnp.arange(groups, dtype=jnp.int32) % 5) + 1
  x = jnp.ones((8, 128), jnp.float32)

  outside = jax.jit(
    lambda s, v: outside_call(jnp.cumsum(s).astype(jnp.int32), v)
  )
  inside = jax.jit(inside_call)

  outside_out = jax.device_get(outside(sizes, x))
  inside_out = jax.device_get(inside(sizes, x))
  gap = float(abs(outside_out - inside_out).max())
  if gap != 0.0:
    raise harness.Invalid(
      f"the two arms disagree by {gap}, so they are not the same "
      "computation and no timing is reportable"
    )

  if dry:
    return _stamp(dry, f"both arms agree exactly at groups={groups}")

  timings = {
    # See `smem_scalar_fill_cost` for why the warmups are raised: warmup
    # cost leaking into the first timed region inflates the spread.
    "outside_xla": harness.time_call(
      lambda: outside(sizes, x), batch=batch, warmups=10, repeats=30
    ),
    "inside_kernel": harness.time_call(
      lambda: inside(sizes, x), batch=batch, warmups=10, repeats=30
    ),
  }
  for name, timing in timings.items():
    harness.require_above_dispatch_floor(timing, what=f"{name} arm")
  ratio = timings["inside_kernel"].median / timings["outside_xla"].median
  detail = (
    f"a prefix sum over int32[{groups}] feeding one pallas_call, timed end to end: "
    + " || ".join(f"{name}: {timing}" for name, timing in timings.items())
    + f" || inside_kernel / outside_xla = {ratio:.2f}x "
    f"|| both arms produced identical output; the XLA arm pays an extra op "
    f"launch and an SMEM transfer, the in-kernel arm pays a {groups}-iteration "
    f"scalar loop"
  )
  return detail


@harness.probe(
  _GROUP,
  "dynamic_grid_from_smem_skips_work",
  "Does a grid length read from SMEM actually skip the steps it excludes?",
)
def _dynamic_grid_from_smem_skips_work() -> str:
  """What the metadata is ultimately spent on.

  GDN's `outer_kernel` reads `num_tiles` out of SMEM and passes it straight
  to `pltpu.emit_pipeline(grid=(num_tiles,))`. The whole point of computing
  the metadata was to make that number small when the batch is ragged — but
  only if the pipeline genuinely runs fewer steps rather than running the
  static maximum and predicating. A padded-out grid would be correct and
  would make every metadata computation upstream of it pointless.

  Both arms are the same compiled kernel called with a different scalar, so
  a difference cannot be a compilation difference.
  """
  import jax
  import jax.numpy as jnp
  from jax.experimental import pallas as pl
  from jax.experimental.pallas import tpu as pltpu

  dry = _dry_run()
  if dry:
    raise harness.Unsupported(
      "pltpu.emit_pipeline raises 'Unsupported TPU device kind: cpu' and "
      "cannot be dry-run at all"
    )

  full, few = 64, 8
  rows, columns = 256, 512
  # The *short* arm is the constraint: the grid=8 arm does an eighth of the
  # grid=64 arm's work. `batch` is the first guess; `harness.time_above_floor`
  # grows it if an arm's timed region misses the dispatch floor.
  batch = 128

  def outer_kernel(steps_ref, x_hbm, out_hbm) -> None:
    def body(x_ref, out_ref) -> None:
      value = x_ref[...]
      for _ in range(4):
        value = value * 1.0009765625 + 1.0
      out_ref[...] = value

    pltpu.emit_pipeline(
      body,
      grid=(steps_ref[0],),
      in_specs=[pl.BlockSpec((rows, columns), lambda step: (step, 0))],
      out_specs=pl.BlockSpec((rows, columns), lambda step: (step, 0)),
    )(x_hbm, out_hbm)

  call = jax.jit(
    pl.pallas_call(
      outer_kernel,
      in_specs=[
        pl.BlockSpec(memory_space=pltpu.SMEM),
        pl.BlockSpec(memory_space=pl.ANY),
      ],
      out_specs=pl.BlockSpec(memory_space=pl.ANY),
      out_shape=jax.ShapeDtypeStruct((full * rows, columns), jnp.float32),
    )
  )

  source = jnp.ones((full * rows, columns), jnp.float32)
  timings = {}
  for label, steps in (("grid=64", full), ("grid=8", few)):
    argument = jnp.array([steps], jnp.int32)
    timing = harness.time_above_floor(
      lambda a=argument: call(a, source), what=f"{label} arm", batch=batch
    )
    harness.require_above_dispatch_floor(timing, what=f"{label} arm")
    timings[label] = timing

  ratio = timings["grid=64"].median / timings["grid=8"].median
  ideal = full / few
  detail = (
    f"one compiled pltpu.emit_pipeline whose grid length is read from an SMEM "
    f"scalar, {rows}x{columns} f32 blocks: "
    + " || ".join(f"{label}: {timing}" for label, timing in timings.items())
    + f" || grid=64 / grid=8 = {ratio:.2f}x against an ideal of {ideal:.0f}x "
    f"|| a ratio near 1.0 would mean the short grid still walked all {full} "
    f"steps and the metadata upstream of it bought nothing"
  )
  return detail
