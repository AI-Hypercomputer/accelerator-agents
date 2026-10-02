# Write the jnp reference

`maxkernel-worker` reads this **once per run**, in Phase 0.7 of `torchax`
reference mode. Following it, you write exactly one file: `<run_dir>/base.py`,
a readable pure-JAX restatement of the user's PyTorch module.

The run already has a mechanically-converted JAX function — torchax produced it
in Phase 0.2 and it serves as the correctness oracle. **You do not get to see
it.** You are writing a *readable* equivalent from the jaxpr and the original
source, and your output is checked against the hidden oracle afterwards.

That hiding is deliberate. It is not about information — you have the PyTorch
source and could in principle reconstruct the oracle. It is about **anchoring**:
an agent shown a working implementation transcribes it, and a transcription is
not what the run needs. The run needs a readable reference whose correctness has
been *established*, not assumed.

--------------------------------------------------------------------------------

## Inputs and outputs

`<run_dir>` is the worker's own.

*   State file: `<run_dir>/state.json`
*   Original source: `state.primary.source_path`
*   **The jaxpr**: `state.jaxpr_path` (`<run_dir>/ref.jaxpr.txt`) — your primary specification
*   **Structured facts**: `state.jaxpr_facts_path` (`<run_dir>/ref.facts.json`)
*   Optimized HLO: `state.hlo_path` — useful context, not a specification
*   The only output: `<run_dir>/base.py`

While following this reference, do NOT read `<run_dir>/ref/` (reference
kernels, a different chain of custody) and do NOT write anything but
`base.py`.

--------------------------------------------------------------------------------

## Step 1: Read the jaxpr as the specification

The jaxpr is what JAX will actually compile. It is more precise than the
PyTorch source, and it settles the questions the source leaves open. A real
example, for an RMSNorm:

```
{ lambda ; a:f32[2048] b:f32[4,256,2048]. let
    c:f32[4,256,2048] = pow b 2.0:f32[]
    d:f32[4,256] = reduce_sum[axes=(2,)] c
    e:f32[4,256,1] = broadcast_in_dim[broadcast_dimensions=(0, 1)] d
    f:f32[4,256,1] = div e 2048.0:f32[]
    g:f32[4,256,1] = add f 9.999999747378752e-06:f32[]
    h:f32[4,256,1] = rsqrt g
    i:f32[4,256,2048] = mul b h
    j:f32[1,1,2048] = broadcast_in_dim[broadcast_dimensions=(2,)] a
    k:f32[4,256,2048] = mul i j
  in (k,) }
```

Read it for the things the source hides:

*   **The reduction axis is explicit** — `axes=(2,)`, not "the last dimension".
*   **`.mean(-1)` is a `reduce_sum` followed by `div 2048.0`.** Reproduce the
    operation, not the spelling.
*   **Constants are materialized at their real precision** — the epsilon is
    `9.999999747378752e-06`, the fp32 value of `1e-5`. Use the source's `eps`
    argument; do not hard-code this literal.
*   **Broadcast structure is explicit**, including which axes are expanded.
*   **The dtype ladder is explicit** — every intermediate carries its dtype, so
    any upcast or downcast the source performs is visible.

`ref.facts.json` gives the same content structured: the operation census, the
reduction axes, the contraction dimension numbers, and the input/output shapes.

## Step 2: Note the argument order — it is not what you would guess

The jaxpr's parameters are **states first, then forward inputs**. In the example
above `a` is the `weight` parameter and `b` is the input `x`, in that order,
because that is how `torchax.extract_jax` flattens: `jax_func(states, args)`.

Your `computation(...)` must use **the same order**, because
`tools/jaxpr_compare.py` feeds both functions the same flattened argument list.
`state.primary.golden_meta_path` records it explicitly under
`calling_convention.flat_signature`. Read that field rather than inferring the
order from the jaxpr's variable names.

(Be aware that `tools/capture_torch_golden.py`, used in the *other* reference
mode, records the opposite order — forward inputs first, then parameters. The
two conventions genuinely differ and guessing silently mis-binds every
argument.)

## Step 3: Write `<run_dir>/base.py`

Requirements:

1.  **Pure JAX.** `jax`, `jax.numpy`, `jax.lax`. **No Pallas, no
    `pallas_call`, no `torch`, no `torchax`, no custom calls.** The baseline is
    what a competent engineer writes *without* a custom kernel, and the loop's
    whole job is to beat it.
2.  **A module-level `def computation(...)`.** Hard contract:
    `{{MAXKERNEL_ROOT}}/tools/assemble_test_harness.py` binds it as
    `base_computation` and refuses the file without it.
3.  **The argument order from Step 2**, exactly.
4.  **The dtypes the jaxpr shows**, in and out. If it accumulates fp32 from a
    bf16 input, do the same (`preferred_element_type=jnp.float32`). A baseline
    that quietly computes in higher precision makes every later correctness
    check meaningless.
5.  **Readable.** This file exists *because* a jaxpr is not readable. Name
    intermediates after what they are, and shape-comment every one:
    `# Shape: (batch, seq, hidden)`.
6.  **Not optimized.** Do not fuse, do not reorder for speed, do not pick a
    cleverer formulation. It is the denominator; making it faster makes every
    reported speedup smaller and the comparison dishonest.
7.  **Preserve the user's problem dimensions exactly** (general_rules #6).

## Step 4: Check your own work before finishing

```bash
{{VENV_PYTHON}} {{MAXKERNEL_ROOT}}/tools/jaxpr_compare.py \
  <state.primary.source_path> <run_dir>/base.py \
  --entry computation --seed <state.seed or 1024> \
  --atol <state.atol> --rtol <state.rtol> --strict
```

This runs on CPU and costs no TPU time, so run it. It checks two independent
things and you must satisfy both:

*   **The value gate** — your output against the hidden oracle. Failing this
    means the reference is simply wrong.
*   **The structural verdict** — your jaxpr's normalized operation census and
    reduction signature against torchax's.

**Why the structural half matters, in numbers.** A reference that drops the
epsilon differs from the oracle by only `7.8e-05` — it *passes* a 1e-2 value
gate and is caught solely as `ref-only={'ADD': 1}`. A reference that reduces
over the wrong axis produces a census *identical* to the oracle and is caught
solely by the reduction signature. Neither check subsumes the other.

Reading the verdict:

*   **`identical`** — done. Your restatement matches operation for operation.
*   **`equivalent`** — differences are known-equivalent spellings only
    (`x**2` / `jnp.square(x)` / `x*x` all normalize to one class). Also fine.
*   **`divergent`** — look at `surviving diff` and `reductions match`. A
    mismatched reduction signature, or a missing operation, is a bug: fix it. A
    reformulation such as `x / sqrt(v)` where the reference uses
    `x * rsqrt(v)` also reports `divergent` — prefer the reference's
    formulation, since you are writing a restatement, not an improvement.

Iterate until both gates pass. If you genuinely cannot make
them pass, say so plainly, naming the surviving difference and your hypothesis.
Do not loosen the tolerances and do not declare success.

## Before you return to Phase 0.7

Record in `<run_dir>/maxkernel_debug_history.md`, in 2–4 sentences: the entry
point signature you committed to, the verdict from `jaxpr_compare.py` with its
numbers, and any difference you accepted and why. Then go back to Phase 0.7
step 3, which runs the gate for the record.

If the jaxpr, the facts file or the source was missing or unreadable, there
was nothing to write a reference *from*. Phase 0.3 owed those files; a missing
one is a bug upstream, not something to work around by reading the PyTorch
source instead.

Phase 0.7 allows three passes through this reference and then stops the run —
which is right, because every number the run would report is measured against
this file.
