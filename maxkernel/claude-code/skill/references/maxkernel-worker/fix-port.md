# Repair the port

`maxkernel-worker` reads this in Phase 0.8 to repair the run's baseline.
`tools/verify_port.py` has compared
`<run_dir>/base.py` against golden values captured by running the user's own
PyTorch module on CPU, and they disagree.

**The golden values are right and `base.py` is wrong.** That is not an
assumption — the golden values came from executing the user's actual source.
Do not "fix" the comparison, do not loosen the tolerance, and do not conclude
the discrepancy is acceptable. Your only move is to change `base.py` so it
computes what the source computes.

--------------------------------------------------------------------------------

## Scope — you may edit exactly one file

`<run_dir>` is the worker's own. While following this reference:

*   You may edit: `<run_dir>/base.py`
*   You must NOT touch: `<run_dir>/torch_golden.npz`, `<run_dir>/torch_golden.json`,
    `<run_dir>/test_kernel.py`, `<run_dir>/get_inputs.py`, any
    `<run_dir>/iter<n>/` artifact, or anything under `<run_dir>/ref/`. Leave
    `<run_dir>/state.json` alone until you are back in Phase 0.8, which
    records the verdict.
*   You must NOT change `state.atol` / `state.rtol`. The tolerances are the
    user's (SKILL.md); a port that only passes at a loosened tolerance has not
    been fixed.

## Inputs

*   `<run_dir>/port_verification.json` — the failure report, with per-output
    failing fractions, the worst offending index, and a diagnosis
*   `<run_dir>/base.py` — the port to repair
*   `<run_dir>/torch_context.md` — **§2 is the specification and §3 is the
    argument contract.** These define what `base.py` must compute and the
    signature it must expose.
*   `<run_dir>/torch_golden.json` — measured shapes, dtypes and argnums
*   `state.primary.source_path` — the user's original module, if you need it

--------------------------------------------------------------------------------

## Step 1: Read the failure shape before reading the code

`port_verification.json` tells you what *kind* of bug this is. Start there —
it narrows the search enormously, and the three shapes have almost disjoint
causes.

**Output count or shape mismatch.** Not a numerical bug at all. The port's
output structure differs: a tuple was flattened or merged, an output was
transposed, a `keepdims` was dropped, or `computation` returns one tensor where
the source returns two. Check §2's output list and the golden manifest's
`outputs` array.

**Nearly every element wrong (failing_fraction > 0.95).** A wholesale semantic
difference, not accumulated rounding. In rough order of likelihood:
*   a reduction over the wrong axis;
*   a missing or extra scale factor (the `1/sqrt(d)` in attention, the `1/N`
    in a mean);
*   mean-vs-RMS, or epsilon inside vs. outside the square root;
*   a transposed operand in a matmul;
*   an activation applied in the wrong place or not at all;
*   a parameter bound to the wrong argument position — check §3's table
    against the golden manifest's `argnum` values.

**A thin sliver wrong (failing_fraction < 0.02).** Boundary handling. The bulk
of the computation is right and the edges are not:
*   a mask applied with the wrong comparison, or off by one row/column;
*   a ragged tail where the length is not divisible by a tile;
*   padding included in a reduction that should have excluded it;
*   `-inf` vs. a large negative number in a masked softmax;
*   the first or last element of a scan.
The report gives `worst_index` — look at what is special about that
coordinate.

**A broad middle (2%–95%).** Usually precision, not algorithm. Check the
accumulator dtype first: if the source accumulates a bf16 input in fp32, the
port must too (`preferred_element_type=jnp.float32`). Then check reduction
order and any place the port computes in higher precision than the source did.
Only after both come back clean should you suspect the algorithm.

## Step 2: Fix the cause, not the symptom

Make the smallest change that addresses the diagnosed cause. Do not rewrite
`base.py` wholesale — a rewrite loses the parts that were already verified
correct and usually trades one bug for another.

`base.py` must remain:

*   **pure JAX** — no Pallas, no `pallas_call`, no `torch`. `verify_port.py`
    runs it on the CPU backend, which only works because it is plain JAX, and
    the baseline is by definition what you write *without* a custom kernel;
*   **signature-stable** — the same `computation(...)` argument order and the
    same `argnum` for every entry as `torch_context.md` §3. If you believe the
    signature itself is wrong, that is the fix, but record it explicitly in
    `<run_dir>/maxkernel_debug_history.md` because the harness and `get_inputs()` depend on it;
*   **dtype-faithful** — same storage dtypes in and out as the manifest
    records;
*   **unoptimized** — do not make it faster. It is the denominator, and a
    baseline that has been quietly improved makes every later speedup smaller
    and the comparison dishonest.

## Step 3: Re-verify before you finish

```bash
{{VENV_PYTHON}} {{MAXKERNEL_ROOT}}/tools/verify_port.py \
  <run_dir>/base.py <run_dir>/torch_golden.npz <run_dir>/torch_golden.json \
  --atol <state.atol> --rtol <state.rtol> \
  --out <run_dir>/port_verification.json
```

This runs on CPU and costs no TPU time, so run it. Iterate until it passes or
until you are confident it cannot be fixed by editing `base.py`. One pass
through this reference is one of Phase 0.8's three attempts.

If the failing fraction went **down** but is not zero, you have found one of
several bugs — keep going.

If you genuinely cannot make it pass, stop and say so plainly, naming what you
changed, what the residual disagreement looks like, and your best hypothesis.
Do NOT declare success, do not loosen tolerances, and do not edit the golden
file. Phase 0.8 stops the run after three attempts, which is the correct
outcome: a run whose baseline is wrong should not spend five iterations
optimizing against it.

## Before you return to Phase 0.8

Append to `<run_dir>/maxkernel_debug_history.md`, in 2–4 sentences: the failure
shape you diagnosed, the cause you found, the change you made, and the final
`verify_port.py` verdict with its numbers. Then go back to Phase 0.8 step 2
and branch on that verdict.

If `<run_dir>/port_verification.json`, the golden files or `base.py` itself
was missing, there was nothing to diagnose from. Do not re-derive the
diagnosis by running the comparison from scratch; the phase that owed the
report failed, and that is what needs fixing.

Treating a port that did not pass as repaired is the specific mistake this
gate exists to prevent: it would let five iterations run against a baseline
known to be wrong.
