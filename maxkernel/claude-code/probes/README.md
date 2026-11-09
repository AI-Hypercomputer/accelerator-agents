# `probes/`: questions answered on the target TPU

A probe answers one question about the hardware or the toolchain in front of it (does this op
lower, what does this primitive cost relative to a control, what does this API do on this chip) and
prints an answer that can be quoted without the surrounding conversation. Most facts worth knowing
here are true of a *toolchain and chip*, not of TPUs in general, so they come from the target, not
from memory. The reference files name the probe that answers each such question; which module
supplies which reference file, and every probe's question, is in
[`../reference/probe_catalog.md`](../reference/probe_catalog.md).

## Layout

```
harness.py         registry, verdicts, timing, dispatch-floor guard. Every probe goes through it.
run_probes.py      selection and reporting. The entry point on a TPU host.
run_probes_test.py registry sanity test (skips without jax).
checks/            one module per reference area; see probe_catalog.md for the pairing
```

`checks/` is grouped by where the answer lands, not by what it measures. A probe that answers a
question two reference files ask has one home, and the other file cites it by name.

## How agents run probes

Agents never call `run_probes.py` directly. They go through the MaxKernel tool, which submits the
job through `tools/tpu_client.py` to the TPU server on a GCE TPU VM:

```bash
{{VENV_PYTHON}} {{MAXKERNEL_ROOT}}/tools/run_probe.py --list
{{VENV_PYTHON}} {{MAXKERNEL_ROOT}}/tools/run_probe.py --pick <name> --run_dir <run_dir>
```

Each run is stamped with the live TPU type and toolchain. Results are cached under
`workspace/_probe_cache/`, keyed by that TPU type and toolchain stamp, so a different chip or
toolchain is a cache miss and the probe re-runs. The cache is local state: never commit it.

## Running directly on a TPU host

On a TPU VM, or inside a GKE TPU pod, from this directory:

```bash
python -u run_probes.py --list
python -u run_probes.py --group <group>
python -u run_probes.py --pick <name>
```

Always pass `-u`. Python block-buffers stdout when it is not a tty, and a process that hangs never
flushes, so a wedged run leaves an empty log: no banner, no traceback, nothing to debug from.

| Flag | Effect |
| --- | --- |
| `--group NAME` | run one group's probes; repeatable |
| `--pick NAME` | run the probe named exactly `NAME`; repeatable; also the only way to run a destructive probe |
| `--include-destructive` | include destructive probes in a group selection |
| `--list` | print the registry and exit without touching a device (works on a CPU-only machine with jax installed) |
| `--traceback` | print the full traceback under a `FAIL` |

**`--pick` is an exact match, not a substring, and a miss is silent.** A `--pick` that matches
nothing selects nothing: the run prints `probe suite: 0 of N probes`, an empty summary, and exits
cleanly, having measured nothing. Check `--list` for the full name first, and read the
`probe suite: X of N` line before anything else in the log.

## Reading the output

Three lines per probe: group and name, the question, and the verdict with elapsed wall time and a
detail string. The detail string is the deliverable: it carries the numbers and names the arm each
ratio is against. Five verdicts, and the distinctions are load-bearing:

| Verdict | Means |
| --- | --- |
| `PASS` | the probe ran and its answer is in the detail string |
| `FAIL` | the probe ran and the thing it checked is not true, or an arm raised |
| `UNSUPPORTED` | the symbol or feature is absent on this toolchain or chip; a different finding from `FAIL` |
| `INVALID` | the probe does not trust its own measurement and has withheld the number |
| `SKIPPED` | destructive, and not opted into |

`INVALID` exists because a caveat printed beside a confident number gets quoted without the caveat.
The most common cause is the dispatch floor: `harness.require_above_dispatch_floor` withholds any
timing whose timed region is too close to the host round trip, which `harness.dispatch_floor_seconds()`
measures once per process on the live host. Correctness-gated timing probes also raise `Invalid`
when the kernel computes the wrong answer; no timing survives a wrong result.

## Destructive probes

Four probes are registered with `destructive=True`: `zero_size_dma_needs_no_wait`,
`undrained_semaphore_halts_core`, `sc_multi_subcore_reduction` and `sc_subcore_barrier_minimal`.
They can halt or deadlock the Mosaic core, and a halted core wedges the device for the rest of the
process: every later probe then reports a precondition failure indistinguishable from a real
finding. Run each one alone, in its own process, under a wall-clock timeout, never beside results
you want back. `tools/run_probe.py` refuses them by default; agents do not opt in.

## Writing a probe

```python
@harness.probe(_GROUP, "some_name", "The question, as a question?")
def _some_name() -> str:
  ...
  return f"detail with the numbers and the arm they are a ratio against"
```

- **The name is the citation key.** Reference files quote it and `--pick` matches it. Renaming a
  probe orphans a citation; add a new one instead.
- **Report a ratio and name its denominator.** Absolute rates move run to run on identical hardware.
  Measure the control arm in the same process and report the ratio; `Timing.spread_pct` says
  whether a gap is real.
- **Block on the result.** `harness.time_call` does this. A probe that times dispatch instead of
  execution reads as implausibly fast. Do not guess `batch=` from a cost model: use
  `harness.time_above_floor`, which times once and re-times at a larger batch if the region is
  under the floor, then hand the timing to `require_above_dispatch_floor`.
- **Run with `python -u` and print before any call that can hang.** Emit `> starting <arm>` before
  dispatching, so a surviving log names the boundary the device died on.
- **Read back what you allocate, at a different offset than you wrote.** Reading nothing invites
  dead-code elimination; reading the same offset invites store-to-load forwarding. Both make a
  buffer look free.
- **Include the negative control in the same measurement.** A hand-rolled arm that beats the
  managed one, measured in a different process, is usually a dispatch-floor artifact.
- **A failing probe is a result.** Do not "fix" it. Probes are named after claims and written so the
  outcome is visible either way.
- **Destructive means its own process.** If you are unsure whether a probe can halt the core,
  assume it can and pass `destructive=True`.
- **Derive the verdict from how many arms produced a number, never from the absence of an
  exception.** A probe whose arms all fail returns `UNSUPPORTED`; a single surviving arm still has
  to clear the floor. When every arm fails identically, the defect is upstream of all of them
  (shared scaffolding, a bad output store), not a property of the hardware.
- **If an error names a compiler flag, retry with that flag before recording an absence.** An error
  recorded verbatim is read, not just its verdict.
- **A docstring is a hypothesis.** State the hypothesis and the method; when the measurement
  contradicts it, the docstring is what changes. Docstrings carry no past numbers or outcomes.

## Dry-running on CPU

Most TensorCore kernels can be checked on a CPU machine before spending TPU time:

```bash
JAX_PLATFORMS=cpu PROBE_CPU_DRYRUN=1 python -u run_probes.py --pick <name>
```

Probes that honour `PROBE_CPU_DRYRUN` pass `interpret=pltpu.InterpretParams()` to `pallas_call`,
and `harness.time_call` drops to one call per arm with no warmups. `InterpretParams()` models
TPU memory spaces and DMAs, so it catches a wrong `index_map` arity, a bad `scratch_shapes`
signature, a mismatched block shape and an incorrect algorithm. Plain `interpret=True` is not a
substitute for TPU memory spaces.

Limits: some APIs refuse the CPU device outright (for example `pltpu.emit_pipeline` may raise
`Unsupported TPU device kind`), some interpret-mode features are incomplete (overlapping
`pl.Element` windows may not be implemented), and SparseCore kernels cannot be dry-run at all
(probe `sc_kernel_cannot_dry_run`). Interpret mode never invokes Mosaic, so it cannot say whether an
op lowers, what it costs or what fits in VMEM, and it answers lowering questions backwards: an op
with no TPU lowering runs happily on CPU. **A dry run never produces a verdict**; its timings and
`PASS`es mean only "the script executes".
