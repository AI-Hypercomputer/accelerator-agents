# The MaxKernel job file

A job file is the single declaration that starts a run. The orchestrator reads
it, validates it, and derives every branch from it — which oracle to build,
whether a JAX reference is needed and who writes it, whether the advisory spine
runs, and how performance gets measured.

It exists to replace inference. `tools/classify_inputs.py` has to guess which of
four `.py` files in a directory is the thing being converted, and refuses
(exit 3) when it cannot tell. A job file says so outright.

```bash
{{VENV_PYTHON}} tools/validate_job.py examples/jobs/02-pytorch-to-pallas.json
```

---

## The four supported shapes

| Shape | `input.type` | `has_reference` | `needs_jax_conversion` | `jax_conversion.method` |
| --- | --- | --- | --- | --- |
| JAX → Pallas | `jax` | `false` | `false` | `none` |
| PyTorch → Pallas | `pytorch` | `false` | `true` | `torchax` or `llm_port` |
| PyTorch + CUDA ref → Pallas | `pytorch` | `true` | `true` | `torchax` or `llm_port` |
| CUDA → Pallas | `cuda` | `false` | `true` | `llm_port` only |
| *no source* → Pallas | `specification` | either | depends on `synthesize_as` | `torchax`/`llm_port` when synthesizing PyTorch |

The third row is the one worth being precise about: **PyTorch is the input and
CUDA is only a reference.** The CUDA is never executed, never measured, and
never becomes `base.py`. It reaches the planner solely as triaged entries in
`ideas_ledger.json`.

---

## Fields

### `input` — required

| Field | Meaning |
| --- | --- |
| `type` | `"jax"`, `"pytorch"` or `"cuda"`. The thing being converted. |
| `path` | File or directory. Relative paths resolve against the job file. |
| `entry_point` | The class or function being converted, e.g. `"Model"`. **Optional but strongly advised** — it is what removes the ambiguity that makes `classify_inputs.py` refuse to guess. |
| `inputs_fn` | Name of the function returning forward arguments, usually `"get_inputs"`. |
| `init_inputs_fn` | Name of the function returning constructor arguments, usually `"get_init_inputs"`. May return a flat list or `(args, kwargs)`; both conventions are handled. |

### `input.specification` — when there is no source file

Set `input.type: "specification"`, leave `path` null, and describe the
computation. Phase 0.1 synthesizes a baseline from it and **stops for your
approval** before optimizing.

| Field | Meaning |
| --- | --- |
| `description` | **Required.** The computation, in a sentence or two. |
| `operation` | Short name, e.g. `"swiglu_ffn"`. |
| `synthesize_as` | `"pytorch"` (default) or `"jax"`. |
| `tensors[]` | **Required.** Each needs `name`, `role` (`input`/`output`/`parameter`), `shape`, `dtype`. These are the problem — an agent may never substitute them. |
| `constants` | Scalars: `eps`, head counts, scale factors. |
| `numerics` | Accumulation dtypes and where precision matters. |
| `edge_cases` | Masking, ragged tails, divisibility. |
| `reference_formula` | The mathematics, if you have it. |
| `known_implementations` | A named equivalent, e.g. "LLaMA's FeedForward". |
| `user_confirms_baseline` | **Required, must be `true`.** Acknowledges that the approval step exists. It is *not* the approval — the run still stops and waits. |

> **Why this mode is weaker than every other.** Everywhere else the user's code
> is the specification, the baseline and the thing measured against, and none
> of those roles is in question because the user wrote it. Here all three come
> from an agent. Write it wrong and the run optimizes the wrong computation
> with every gate passing; write it slowly and every speedup is inflated by the
> size of the mistake. Neither is detectable from inside the loop, which is why
> the phase ends at a human.
>
> Prefer `synthesize_as: "pytorch"`. The run then proceeds as an ordinary
> PyTorch input: torchax converts it mechanically, that conversion is checked
> against eager PyTorch, and the whole gate chain applies. `"jax"` skips all of
> it — the baseline becomes `base.py` directly and nothing independent ever
> checks it.

`has_reference` works normally here: a CUDA kernel can still be supplied as an
advisory reference for a computation you only described.

### `has_reference` and `references` — required

`has_reference` repeats what `len(references)` already says. That redundancy is
deliberate: it is a **checksum**, not duplication. A job that declares a
reference and forgets the path is a mistake worth catching at validation rather
than discovering as a silently reference-less run four phases later. The
validator fails on any disagreement between the two.

Each entry in `references`:

| Field | Meaning |
| --- | --- |
| `type` | `"cuda"`, `"triton"`, `"pallas"`, `"jax"`, `"pytorch"` |
| `path` | File or directory |
| `role` | `"donor"` — a source of ideas. The only role today. |
| `relationship` | Your claim about how it relates to the input: `"same_operation"`, `"same_family"`, `"unknown"`. **A hint, not a fact** — Phase 0.6 reconciles it independently and its verdict wins. Saying `"same_operation"` does not make it so. |
| `notes` | Free text passed to the reconciler. Use it for what you already know differs. |

### `needs_jax_conversion` and `jax_conversion` — required

"Conversion to JAX" means an **idiomatic, non-Pallas JAX reference** — not JAX
in general. Pallas *is* JAX (`pl.pallas_call` is a JAX primitive), so no run
ever leaves JAX. What this switch controls is whether the run gets a plain-JAX
`base.py` in between.

| `jax_conversion.method` | What happens |
| --- | --- |
| `"torchax"` | `torchax.extract_jax` converts the module mechanically. No LLM, no transcription, no mistranslation risk. Costs a torch/torchax version pin. **PyTorch input only** — torchax traces PyTorch modules, not CUDA. |
| `"llm_port"` | An agent hand-writes `base.py`, gated in Phase 0.8 against golden values. No extra dependency. |
| `"none"` | Only valid when `needs_jax_conversion` is false. |

`jax_conversion.emit_readable_reference` (default `true`) decides whether a
readable `base.py` is produced in torchax mode.

> **The constraint that bites if you ignore it.**
> `tools/assemble_test_harness.py` binds `base_computation` from a file defining
> `computation`. **No `base.py` means the internal paired-timing harness cannot
> run.** That happens when `needs_jax_conversion` is false for a non-JAX input,
> or when `method` is `torchax` with `emit_readable_reference` false. Both are
> legitimate intentions, so the validator lets them through — but only if you
> also set `evaluation.compare_against_source: true`, which measures externally
> instead. Otherwise it is an error, raised before a TPU is touched rather than
> four phases in.

### Everything else — optional, with defaults

| Field | Default | Meaning |
| --- | --- | --- |
| `correctness.atol` / `.rtol` | `1e-2` | Tolerances. The validator warns above `0.5`: a tolerance that loose accepts almost any output. |
| `correctness.seed` | `1024` | Re-set before *each* model construction, which is how two modules come out with identical random weights without anything being copied. |
| `correctness.num_trials` | `5` | Input sets per correctness check. |
| `target.tpu_version` | from `tpu_config.json` | e.g. `"TPU v6e"`. |
| `loop.max_iterations` | `5` | Optimization iterations. |
| `evaluation.compare_against_source` | `false` | Also time the kernel against the **original PyTorch** on the TPU via `evaluation/compare_kernel.py`. Invalid for a CUDA input — a TPU host cannot execute CUDA. |
| `constraints.preserve_shapes` | `true` | Never substitute problem dimensions (general_rules #6). |
| `name`, `description`, `notes` | — | `notes` is passed through to the planner. |

---

## What the validator checks

Beyond required fields and enum membership, it rejects seven contradictions
that are each a real failure mode:

1. `has_reference: true` with no `references` — a silently reference-less run.
2. `has_reference: false` with references listed — silently ignored input.
3. `torchax` on a CUDA input — torchax traces PyTorch, not CUDA C++.
4. `needs_jax_conversion: false` on a non-JAX input — no `base.py`, so no
   internal harness.
5. `torchax` + `emit_readable_reference: false` — same, unless external
   evaluation is declared, in which case it downgrades to a warning.
6. `needs_jax_conversion: true` on a JAX input — already the reference.
7. `evaluation.compare_against_source: true` on a CUDA input — nothing to
   compare against on a TPU.

On success it prints the plan the job implies, so you can see what will happen
before it happens:

```
JOB VALID: rmsnorm-torch
  input        : pytorch  (../../evaluation/examples/rmsnorm/ref.py)
  references   : 0
  oracle       : tools/torchax_oracle.py (mechanical, with step-0 check vs eager)
  base.py      : written from the jaxpr in Phase 0.7  [maxkernel-worker, write-jnp-reference]
  harness      : internal (paired timing)
  external eval: True
  phases       : 0 read state -> 0.2 oracle -> 0.3 jaxpr + HLO export
                 -> 0.7 write jnp reference + jaxpr_compare gate -> 0.9 harness
                 -> 1-6 plan/implement/compile/test/autotune/profile
```

Exit `0` valid · `1` invalid · `2` valid but a declared path does not exist.

---

## Examples

`examples/jobs/` holds one runnable job per supported shape:

| File | Shape |
| --- | --- |
| `01-jax-to-pallas.json` | JAX → Pallas |
| `02-pytorch-to-pallas.json` | PyTorch → Pallas, torchax |
| `03-pytorch-with-cuda-reference.json` | PyTorch + CUDA reference → Pallas |
| `04-cuda-to-pallas.json` | CUDA → Pallas, `llm_port` |
| `05-specification-to-pallas.json` | No source file — baseline synthesized from a spec, then approved |
