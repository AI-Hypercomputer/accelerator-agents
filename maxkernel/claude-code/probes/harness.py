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
"""Registry and reporting for the TPU probe suite.

A *probe* answers one question about the hardware or about the Pallas/Mosaic
version in front of it, and prints an answer that can be quoted without the
surrounding conversation. Probes exist because a claim like "there is no
``jnp.cumsum`` lowering" is true of a *version*, not of TPUs, and a stale claim
costs more than a missing one.

Three rules the reporting enforces:

* **A verdict is owned, never annotated.** A probe that suspects its own result
  returns ``INVALID`` and *withholds* the tempting figure. A caveat printed
  beside a confident number gets quoted without the caveat.
* **A failure is a result.** Probes named after a claim are written so that the
  claim being true shows up as ``FAIL``. Do not "fix" a failing probe.
* **Absolutes drift, ratios do not.** Throughput moves run to run on identical
  hardware. Probes that report a time should also report the arm they are a
  ratio against.
"""

from __future__ import annotations

import dataclasses
import enum
import os
import statistics
import time
import traceback
from collections.abc import Callable, Iterable, Sequence
from typing import Any


class Verdict(enum.StrEnum):
  """How a probe turned out."""

  PASS = "PASS"
  FAIL = "FAIL"
  UNSUPPORTED = "UNSUPPORTED"
  INVALID = "INVALID"
  SKIPPED = "SKIPPED"


@dataclasses.dataclass(frozen=True)
class Probe:
  """One registered question."""

  group: str
  name: str
  question: str
  body: Callable[[], Any]
  destructive: bool


@dataclasses.dataclass(frozen=True)
class Result:
  """What a probe printed."""

  probe: Probe
  verdict: Verdict
  detail: str
  seconds: float


_REGISTRY: list[Probe] = []


def probe(
  group: str,
  name: str,
  question: str,
  *,
  destructive: bool = False,
) -> Callable[[Callable[[], Any]], Callable[[], Any]]:
  """Registers a probe.

  Args:
    group: Coarse bucket used by ``--group``/``--skip-group``. Group by the
      reference file the answer lands in, not by the JAX symbol involved.
    name: Unique, stable identifier. It is what ``--pick`` matches and what
      the reference files cite, so renaming one orphans a citation.
    question: The question in one line, phrased so the answer is meaningful
      on its own. This is printed above the verdict.
    destructive: True if the probe can halt the Mosaic core. A halt wedges
      the device for the rest of the *process* — every later probe then
      reports ``FAILED_PRECONDITION: the program continuator has halted``,
      which is indistinguishable from a real failure. Destructive probes are
      excluded by default and the runner gives each its own interpreter. If
      you are unsure whether a probe can halt the core, assume it can.

  Returns:
    The decorator, which registers and returns the function unchanged.
  """

  def register(body: Callable[[], Any]) -> Callable[[], Any]:
    _REGISTRY.append(
      Probe(
        group=group,
        name=name,
        question=question,
        body=body,
        destructive=destructive,
      )
    )
    return body

  return register


class Unsupported(Exception):
  """Raised by a probe when the symbol or feature is absent on this version.

  This lands as ``UNSUPPORTED`` rather than ``FAIL``: "this JAX does not have
  it" and "it exists and misbehaves" are different findings and must not
  share a verdict.
  """


class Invalid(Exception):
  """Raised by a probe that no longer trusts its own measurement.

  Raising this *suppresses* whatever number the probe had computed. That is
  the point: a guard that annotates a number instead of owning the verdict
  does not stop the number being quoted.
  """


def registered_probes() -> Sequence[Probe]:
  """Returns every probe registered so far, in declaration order."""
  return tuple(_REGISTRY)


def _classify(error: BaseException) -> tuple[Verdict, str]:
  """Maps an exception raised by a probe body onto a verdict and detail."""
  if isinstance(error, Unsupported):
    return Verdict.UNSUPPORTED, str(error)
  if isinstance(error, Invalid):
    return Verdict.INVALID, str(error)
  if isinstance(error, (AttributeError, ImportError, NotImplementedError)):
    return Verdict.UNSUPPORTED, f"{type(error).__name__}: {error}"
  message = str(error).strip().replace("\n", " | ")
  if len(message) > 600:
    message = message[:600] + " …[truncated]"
  return Verdict.FAIL, f"{type(error).__name__}: {message}"


def run_probe(target: Probe, *, show_traceback: bool = False) -> Result:
  """Runs one probe and captures its verdict without letting it escape."""
  started = time.perf_counter()
  try:
    returned = target.body()
  except BaseException as error:  # noqa: BLE001 — a probe may raise anything.
    verdict, detail = _classify(error)
    if show_traceback and verdict is Verdict.FAIL:
      traceback.print_exc()
  else:
    if isinstance(returned, tuple) and len(returned) == 2:
      verdict, detail = Verdict(returned[0]), str(returned[1])
    else:
      verdict, detail = Verdict.PASS, str(returned)
  return Result(
    probe=target,
    verdict=verdict,
    detail=detail,
    seconds=time.perf_counter() - started,
  )


def report(result: Result) -> None:
  """Prints one result in the suite's grep-able three-line form."""
  print(f"\n[{result.probe.group}] {result.probe.name}")
  print(f"    Q: {result.probe.question}")
  print(f"    {result.verdict} ({result.seconds:.2f}s): {result.detail}")


def report_summary(results: Iterable[Result]) -> int:
  """Prints the closing tally. Returns the number of ``FAIL`` verdicts."""
  collected = list(results)
  counts: dict[Verdict, int] = {}
  for result in collected:
    counts[result.verdict] = counts.get(result.verdict, 0) + 1

  print("\n" + "=" * 72)
  print("SUMMARY")
  print("=" * 72)
  for verdict in Verdict:
    if verdict in counts:
      print(f"  {verdict:<12} {counts[verdict]}")
  for result in collected:
    print(f"  {result.verdict:<12} [{result.probe.group}] {result.probe.name}")
  print("=" * 72)
  return counts.get(Verdict.FAIL, 0)


# --------------------------------------------------------------------------
# Timing
# --------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Timing:
  """A per-call measurement, in seconds, with the spread that qualifies it."""

  median: float
  minimum: float
  maximum: float
  repeats: int
  batch: int = 1

  @property
  def spread_pct(self) -> float:
    """Peak-to-peak spread as a percentage of the median."""
    if self.median == 0.0:
      return float("inf")
    return 100.0 * (self.maximum - self.minimum) / self.median

  @property
  def region_median(self) -> float:
    """Median wall time of one timed region — ``batch`` calls and one block.

    This, not `median`, is the quantity that has to clear the host round
    trip for the measurement to be about the device at all.
    """
    return self.median * self.batch

  def __str__(self) -> str:
    batch_note = f" batch={self.batch}" if self.batch != 1 else ""
    return (
      f"median={self.median * 1e6:.1f}us "
      f"min={self.minimum * 1e6:.1f}us "
      f"max={self.maximum * 1e6:.1f}us "
      f"spread={self.spread_pct:.0f}% n={self.repeats}{batch_note}"
    )


def time_call(
  call: Callable[[], Any],
  *,
  warmups: int = 3,
  repeats: int = 20,
  batch: int = 1,
) -> Timing:
  """Times a device call, blocking on its result, and returns per-call time.

  The caller is responsible for making ``call`` block — return a JAX array
  and this function will ``block_until_ready`` it. A probe that forgets to
  block times dispatch, which on TPU is microseconds against a kernel's
  milliseconds and reads as an implausibly fast result. Check any timing
  against a known rate before quoting it.

  Args:
    call: The thing to time. Must return something ``block_until_ready``
      accepts.
    warmups: Untimed iterations, to pay for compilation.
    repeats: Timed iterations.
    batch: How many dispatches to put inside one timed region. **This is the
      knob that decides whether a small kernel is measurable at all.** JAX
      dispatch is asynchronous, so ``batch`` calls followed by one block pay
      the host round trip once and the device cost ``batch`` times; the
      returned figure is per call. With ``batch=1`` nothing below the round
      trip (see `dispatch_floor_seconds`) can be resolved at all, and every
      arm of a sweep then reports the same number while looking like a real
      result.
      Size it so ``batch * expected_device_time`` clears the floor severalfold.

  Returns:
    Per-call timing. The spread is across timed regions, not across the
    calls inside one.
  """
  import jax

  if batch < 1:
    raise ValueError(f"batch must be >= 1, got {batch}")

  if env_flag("PROBE_CPU_DRYRUN"):
    # A dry run is asking "does this script execute", not "how long does it
    # take", and its arms run in Pallas interpret mode where a single call
    # is seconds rather than microseconds. Timing one call each keeps a
    # whole group to minutes; leaving the real counts in place does not
    # finish. Nothing timed under the flag is a result — see the flag's
    # note in the probe modules that honour it.
    warmups, repeats, batch = 0, 1, 1

  def timed_region() -> None:
    result = None
    for _ in range(batch):
      result = call()
    jax.block_until_ready(result)

  for _ in range(warmups):
    timed_region()

  samples: list[float] = []
  for _ in range(repeats):
    started = time.perf_counter()
    timed_region()
    samples.append((time.perf_counter() - started) / batch)

  return Timing(
    median=statistics.median(samples),
    minimum=min(samples),
    maximum=max(samples),
    repeats=repeats,
    batch=batch,
  )


_DISPATCH_FLOOR_CACHE: list[float] = []


def dispatch_floor_seconds() -> float:
  """Returns the host round trip of one trivial blocking device call.

  Measured once per process, on first use, and cached. The method is the
  single-call arm of the `dispatch_floor` probe in ``checks/partitioning.py``:
  a jitted ``pallas_call`` that copies one ``(8, 128)`` float32 tile, timed
  with `time_call` (``batch=1``), median taken. The kernel does no work worth
  measuring, so the figure is host overhead. Any single-dispatch measurement
  near it is measuring the round trip and not the kernel.

  The value depends on the host, the runtime and the TPU type, so it is never
  hardcoded. Under ``PROBE_CPU_DRYRUN`` the call runs in Pallas interpret
  mode and the figure means nothing; no verdict comes from a dry run.
  """
  if _DISPATCH_FLOOR_CACHE:
    return _DISPATCH_FLOOR_CACHE[0]

  import jax
  import jax.numpy as jnp
  from jax.experimental import pallas as pl

  def copy_kernel(source_ref, output_ref) -> None:
    output_ref[...] = source_ref[...]

  kwargs = {}
  if env_flag("PROBE_CPU_DRYRUN"):
    from jax.experimental.pallas import tpu as pltpu

    kwargs["interpret"] = pltpu.InterpretParams()
  source = jnp.ones((8, 128), dtype=jnp.float32)
  single_call = jax.jit(
    pl.pallas_call(
      copy_kernel,
      out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32),
      **kwargs,
    )
  )
  timing = time_call(lambda: single_call(source), warmups=5, repeats=50)
  _DISPATCH_FLOOR_CACHE.append(timing.median)
  return timing.median


def require_above_dispatch_floor(
  timing: Timing,
  *,
  what: str,
  multiple: float = 5.0,
) -> None:
  """Raises ``Invalid`` if a timing sits too close to the host round trip.

  A probe that sweeps a parameter and reports every arm at 1.00x has usually
  not discovered that the parameter does not matter. It has discovered that
  every arm was shorter than the dispatch floor. That reads as a confident
  null result, which is worse than no result at all, so it is a verdict
  rather than a footnote.

  The test is against `Timing.region_median` — ``batch`` calls and one block
  — because that is the wall time the single round trip is amortized over. A
  per-call figure far *below* the floor is the intended outcome of batching,
  not a defect, and checking `median` instead would reject exactly the probes
  that had been fixed.

  The fix inside the probe is almost always ``batch=`` on `time_call`, or a
  larger shape. Suppressing this check is not one of the options.

  Args:
    timing: The candidate measurement.
    what: What was being timed, named for the message.
    multiple: How far above the floor the timed region has to sit.

  Raises:
    Invalid: If the region median is below
      ``multiple * dispatch_floor_seconds()``.
  """
  floor = dispatch_floor_seconds()
  threshold = multiple * floor
  if timing.region_median < threshold:
    raise Invalid(
      f"{what}: timed region is {timing.region_median * 1e6:.1f}us "
      f"({timing.batch} call(s) x {timing.median * 1e6:.2f}us), below the "
      f"{multiple:g}x dispatch-floor threshold of {threshold * 1e6:.1f}us "
      f"({floor * 1e6:.0f}us round trip, measured in this process). That is host "
      f"overhead, not device time, so the number is withheld. Raise "
      f"`batch=` on time_call, or enlarge the shape."
    )


def time_above_floor(
  call: Callable[[], Any],
  *,
  what: str,
  batch: int = 8,
  repeats: int = 20,
  multiple: float = 5.0,
  max_batch: int = 8192,
) -> Timing:
  """Times ``call``, growing ``batch`` once if the region misses the floor.

  A fixed ``batch=`` guess is wrong whenever the arm's cost is not what was
  predicted, and both directions hurt: too small and
  `require_above_dispatch_floor` withholds the result, too large and a slow
  arm takes minutes. Arms whose per-call cost is not linear in the shape make
  it worse, because no single batch is right on both sides of a cliff.

  So time once and let the measurement pick the batch: if the timed region is
  under ``multiple * dispatch_floor_seconds()``, scale ``batch`` to target
  twice that threshold (capped at ``max_batch``) and time again. Two attempts
  at most. The caller still gets a `Timing` and should still hand it to
  `require_above_dispatch_floor`, which owns the verdict.

  Args:
    call: The thing to time; see `time_call`.
    what: What is being timed, named in the retry message.
    batch: The first attempt's batch.
    repeats: Timed iterations per attempt.
    multiple: How far above the floor the timed region has to sit.
    max_batch: Cap on the grown batch. An arm that needs more dispatches than
      this to become measurable has the wrong shape, and ``INVALID`` is the
      right outcome rather than a ten-minute probe.

  Returns:
    The last attempt's timing.
  """
  import math

  threshold = multiple * dispatch_floor_seconds()
  timing = time_call(call, batch=batch, repeats=repeats)
  if timing.region_median >= threshold:
    return timing
  grow = math.ceil(2.0 * threshold / max(timing.region_median, 1e-9))
  batch = min(max_batch, batch * grow)
  print(
    f"      . {what}: region {timing.region_median * 1e6:.0f}us under floor, "
    f"retrying at batch={batch}",
    flush=True,
  )
  return time_call(call, batch=batch, repeats=repeats)


def env_flag(name: str, default: bool = False) -> bool:
  """Reads a boolean environment variable."""
  raw = os.environ.get(name)
  if raw is None:
    return default
  return raw.strip().lower() in ("1", "true", "yes", "on")
