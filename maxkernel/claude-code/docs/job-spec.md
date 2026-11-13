# The MaxKernel job file

A job file is the single declaration that starts a run. The orchestrator reads
it, validates it, and derives every branch from it — which oracle to build,
whether a JAX reference is needed and who writes it, whether the advisory spine
runs, and how performance gets measured.

It exists to replace inference. `tools/classify_inputs.py` has to guess which of
four `.py` files in a directory is the thing being converted, and refuses
(exit 3) when it cannot tell. A job file says so outright.

```bash
{{VENV_PYTHON}} tools/validate_job.py path/to/job.json
```

---

## The supported shapes

| Shape | `input.type` | `has_reference` | `needs_jax_conversion` | `jax_conversion.method` |
| --- | --- | --- | --- | --- |
| JAX → Pallas | `jax` | `false` | `false` | `none` |
| PyTorch → Pallas | `pytorch` | `false` | `true` | `torchax` or `llm_port` |
| PyTorch + CUDA ref → Pallas | `pytorch` | `true` | `true` | `torchax` or `llm_port` |
| CUDA → Pallas | `cuda` | `false` | `true` | `llm_port` only |
| Triton / cuTile / CuTe DSL / other GPU → Pallas | `triton`, `cutile`, `cutedsl`, `gpu` | either | `true` | `llm_port` only |
| PyTorch codebase (package, GPU-only deps) → Pallas | `pytorch` + `input.extraction` | either | `true` | `llm_port` advised |
| JAX codebase (package, module-library parameters) → Pallas | `jax` + `input.extraction` | either | `false` | `none` |
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
| `type` | `"jax"`, `"pytorch"`, or a GPU dialect: `"cuda"`, `"triton"`, `"cutile"`, `"cutedsl"`, `"gpu"` (any other). The thing being converted. Only `triton` among the GPU dialects gets golden values (Triton's CPU interpreter); the others are ported unverified. |
| `path` | File or directory. Relative paths resolve against the job file. |
| `entry_point` | The class or function being converted, e.g. `"Model"`. **Optional but strongly advised** — it is what removes the ambiguity that makes `classify_inputs.py` refuse to guess. |
| `inputs_fn` | Name of the function returning forward arguments, usually `"get_inputs"`. |
| `init_inputs_fn` | Name of the function returning constructor arguments, usually `"get_init_inputs"`. May return a flat list or `(args, kwargs)`; both conventions are handled. |
| `init_args` / `init_kwargs` | Constructor arguments as JSON values, for a source with no `get_init_inputs()`. Mutually exclusive with `init_inputs_fn`. A constructor that takes an enum needs a source that accepts its string value. |
| `configs` | Forward inputs, for a source with no `get_inputs()` — see below. Mutually exclusive with `inputs_fn`. |
| `extraction` | The source cannot run as is; Phase 0.15 extracts a standalone one — see below. |

`entry_point`, `inputs_fn`, `init_inputs_fn` and the spec fields reach the
capture tools as `--entry-point`, `--inputs-fn`, `--init-inputs-fn` and
`--input-spec <run_dir>/input_spec.json`. Declared, they replace discovery
outright; undeclared, the tools fall back to the KernelBench conventions.

### `input.configs` — when the source ships no `get_inputs()`

Code from the wild rarely has an input factory, and general_rules #6 forbids an
agent from choosing shapes. So you declare them, one entry per config to
capture:

```json
"init_kwargs": {"hidden_size": 1024, "num_heads": 16, "activation": "gelu"},
"configs": [
  {"name": "seq2048", "inputs": [
    {"name": "x", "shape": [4, 2048, 1024], "dtype": "bfloat16"},
    {"name": "positions", "shape": [4, 2048], "dtype": "int32",
     "init": "randint", "low": 0, "high": 2048}
  ]}
]
```

| Field of `configs[].inputs[]` | Meaning |
| --- | --- |
| `shape`, `dtype` | A tensor argument. `dtype` is a torch dtype name (`bfloat16`, `float32`, `int32`, …). |
| `init` | `randn` (default for floats), `rand`, `randint` (needs `low` and `high`), `zeros`, `ones`. **Integer tensors have no default**: they are usually indices, and their valid range is part of the problem. |
| `scale` | Multiplies the sampled values. |
| `value` | A non-tensor (static) argument instead of a tensor. |

Each declared config is captured once (`--num-configs` does not apply), and
`get_inputs()` in the generated harness is built from the capture. Constructor
arguments are shared by every config: one run is one model. To compare model
sizes, run one job per size.

Golden capture stores every parameter and buffer, once, however many configs
you declare, plus each config's inputs and outputs. Large real-world parameter
sets — a model's full weights, embedding tables — can exceed the
capture's default 256 MB cap, and the capture then degrades to an unverified
port. Do not shrink the config to fit: the declared dimensions are the problem.
Raise the cap with `correctness.max_capture_bytes` instead. The tool's error
gives a lower bound on the bytes needed (it stops at the first config that
does not fit), so leave headroom.

### `input.extraction` — code that cannot run as is

A repository is not a benchmark file. It has relative imports, depends on
things the run cannot load, and ships no input factory. `extraction` has an
agent (Phase 0.15) write one self-contained `source.py` in the input's own
framework — with your constructor arguments and inputs — by tracing your entry
point through the codebase, and then **stops for your approval** like a
synthesized baseline. It applies to `input.type: "pytorch"` and `"jax"`:

| | PyTorch | JAX |
| --- | --- | --- |
| Why the code cannot be used as is | It imports CUDA-only libraries (CuTe DSL, Triton, custom extensions) or touches `torch.cuda`, and the capture tools execute one self-contained file on a TPU host's CPU. | A JAX input becomes `base.py` unchanged, and `base.py` is one self-contained file with a module-level `computation`. A package that keeps its parameters in a module library (Flax, Equinox, Haiku) is neither. |
| Agent | `maxkernel-extract-source` | `maxkernel-extract-jax-source` |
| The extracted `source.py` | A `Model` class, driven by your declared arguments | A `computation` function taking inputs and parameters as arguments, plus a `get_inputs()` that builds your declared configs. Only `jax`, `numpy` and the standard library are imported. |
| After approval | An ordinary PyTorch run on that file: golden capture, port, port check | Copied to `base.py`. An ordinary JAX run: no oracle, no port. |

| Field | Meaning |
| --- | --- |
| `entry_point` | **Required.** What to reproduce, as `file.py:Name`, where the file path is **relative to `input.path`** (not to its parent or the repository root). For a repository cloned to `my-repo/` with the class in `my-repo/pkg/layers.py`: with `input.path: "my-repo"` it is `pkg/layers.py:Layer`; with `input.path: "my-repo/pkg"` it is `layers.py:Layer`. When `input.path` is a single file, write just `Name`. The validator resolves it, checks the file defines `Name`, and suggests the right spelling when the path is off by a directory. |
| `notes` | Which path to reproduce when `forward()` (or the JAX call method) has several: backend, inference vs training, what to return. Without it the agent takes the default-argument path in inference mode and records that choice. |
| `user_confirms_extraction` | **Required, must be `true`.** Acknowledges the approval step exists. It is not the approval. |

Rules the validator enforces: `extraction` needs `input.type: "pytorch"` or
`"jax"`, and `input.configs`; `input.entry_point`, `inputs_fn` and
`init_inputs_fn` cannot be set alongside it (the extracted file has a fixed
entry point); and a PyTorch or JAX `input.path` that is a **directory**
requires it. A JAX extraction keeps `needs_jax_conversion: false`, like any
JAX input.

What you get at the gate: `source.py`, `extraction_rationale.md` (the path
reproduced, the paths not reproduced, and assumptions; plus, for PyTorch,
divergences between the repository's own torch path and its kernel path, and
for JAX, each library call that was re-expressed in plain `jax`), and
`extraction_check.py`, which compares the extraction with your original.

- **PyTorch.** The original cannot run on the TPU host, so you run
  `extraction_check.py` **on a machine where it does**; that is the only check
  of the link from your code to `source.py`. From approval onwards the run is
  an ordinary PyTorch run on the extracted file, and every later gate applies
  to it.
- **JAX.** The agent tries to run `extraction_check.py` on the host, with the
  original's own parameters loaded into the extracted `computation`, and the
  rationale opens with the result. If your package's dependencies are not
  installed there, it could not run, and the check is yours to run. Either
  way, **no later gate looks at the extracted file again**: it is the
  baseline, and a JAX baseline has no oracle. Your approval and that script
  are the whole check.

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

`references` holds at most one entry; the validator rejects more. Its fields:

| Field | Meaning |
| --- | --- |
| `type` | The reference's dialect: `"cuda"`, `"triton"`, `"cutile"`, `"cutedsl"`, `"gpu"`, or `"pallas"`, `"jax"`, `"pytorch"` (see below) |
| `path` | File or directory |
| `role` | `"donor"` — a source of ideas. The only role today. |
| `relationship` | Your claim about how it relates to the input: `"same_operation"`, `"same_family"`, `"unknown"`. **A hint, not a fact** — Phase 0.6 reconciles it independently and its verdict wins. Saying `"same_operation"` does not make it so. |
| `notes` | Free text passed to the reconciler. Use it for what you already know differs. |

`type` becomes `references[].kind` in `state.json` and tells the analyzer which
translation table applies: `"cuda"`, `"triton"`, `"cutile"` or `"cutedsl"` for
a reference in exactly that dialect, and `"gpu"` for anything else: several
dialects mixed, CUTLASS/CuTe C++, or another language. `"pallas"`, `"jax"` and
`"pytorch"` are accepted but get no dedicated handling.

Phase 0.5 always runs `gpu_static_facts.py`, which detects each file's dialect
itself, so the CUDA, Triton, cuTile and CuTe DSL parts of any reference get
extracted facts. Whatever it cannot extract the analyzer reads from source,
marking those numbers as not extractor-verified. `path` may be a directory
holding a multi-file kernel project.

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

### `backward` — optional

By default a run optimizes the input as written. Add a `backward` object to
optimize for training instead, and say where the backward comes from:

| `backward.from` | `input` is | What happens |
| --- | --- | --- |
| `"source"` | a **backward** implementation, of any input type | No extra phase: the backward goes through the same steps as any input of its type. The result is a backward kernel, timed on the backward alone. |
| `"autodiff"` | the **forward** | The forward is ported and checked as usual, then Phase 0.85 (`tools/make_vjp_base.py`) rewrites `base.py` using `jax.vjp` as `computation(cotangent, *forward_args) -> (forward_output, gradients)`, or `-> gradients` when the forward stays fixed. No agent writes the backward baseline; it is as trustworthy as the forward port. |

| Field | Meaning |
| --- | --- |
| `backward.optimize_forward` | `autodiff` only. `true` (default): both passes are optimized, and the result is `forward_pass(*forward_args) -> (forward_output, saved)` and `backward_pass(saved, cotangent) -> gradients`, a ready `jax.custom_vjp` pair. `false`: your forward stays fixed and the result is a backward kernel returning `gradients`. |
| `backward.wrt` | `autodiff` only. Names of the forward's arguments to differentiate, as named in your source. Default: every floating-point array argument and, for a PyTorch module, every trainable parameter; buffers are never differentiated. |
| `backward.forward_context` | `source` only. Path to the forward code, read when generating inputs so the saved forward values a backward consumes (activations, gather indices, routing scores) are consistent rather than independent random arrays. |

Use `source` when your backward is JAX or Pallas (it becomes `base.py` as is) or
PyTorch or Triton (its port is golden-checked). Use `autodiff` when you have
only the forward, or when your backward is CUDA, CuTe DSL or another GPU
language, which could never be checked: supply that backward as the job's
reference instead and the planner still gets its ideas. The gradient is only
checked if the forward port is, so this helps when the forward is JAX,
PyTorch or Triton.

Two things to know about an `autodiff` run:

- What is timed. By default, forward + backward together, as one training
  step. The two passes are kept apart as in training: the backward sees only
  what the forward saved and the cotangent, and no work on the cotangent can be
  merged into the forward. With `optimize_forward: false`, the backward alone:
  it receives only the forward inputs (all a fixed forward is guaranteed to
  hand it) and recomputes the rest, exactly as the `jax.vjp` baseline does. Use
  it as the backward of a `jax.custom_vjp` whose forward saves its inputs.
- It assumes your source's gradient is the true gradient of its forward. If
  the source defines its own (`autograd.Function`, a gradient hook), you have
  a backward: use `source`. Phase 0.85 stops if it finds one.

### Everything else — optional, with defaults

| Field | Default | Meaning |
| --- | --- | --- |
| `correctness.atol` / `.rtol` | `1e-2` | Tolerances. The validator warns above `0.5`: a tolerance that loose accepts almost any output. |
| `correctness.seed` | `1024` | Re-set before *each* model construction, which is how two modules come out with identical random weights without anything being copied. |
| `correctness.num_trials` | `5` | Input sets per correctness check. |
| `correctness.max_capture_bytes` | `256000000` | Ceiling, in bytes, on what the Phase 0.2 capture stores: parameters and buffers once, plus every config's inputs and outputs. Raise it when real weights exceed the default, rather than shrinking the config. A larger capture costs disk and host memory. |
| `target.tpu_version` | from `tpu_config.json` | e.g. `"TPU v6e"`. |
| `loop.max_iterations` | `5` | Optimization iterations. |
| `evaluation.compare_against_source` | `false` | Also time the kernel against the **original PyTorch** on the TPU via `evaluation/compare_kernel.py`. Invalid for a CUDA input — a TPU host cannot execute CUDA. |
| `constraints.preserve_shapes` | `true` | Never substitute problem dimensions (general_rules #6). |
| `name`, `description`, `notes` | — | `notes` is passed through to the planner. |

---

## What the validator checks

Beyond required fields and enum membership, it rejects sixteen contradictions
that are each a real failure mode:

1. `has_reference: true` with no `references` — a silently reference-less run.
2. `has_reference: false` with references listed — silently ignored input.
3. `torchax` on a GPU input (`cuda`, `triton`, `cutile`, `cutedsl`, `gpu`) — torchax
   traces PyTorch, not GPU kernels.
4. `needs_jax_conversion: false` on a non-JAX input — no `base.py`, so no
   internal harness.
5. `torchax` + `emit_readable_reference: false` — same, unless external
   evaluation is declared, in which case it downgrades to a warning.
6. `needs_jax_conversion: true` on a JAX input — already the reference.
7. `evaluation.compare_against_source: true` on a GPU input — nothing to
   compare against on a TPU.
8. More than one entry in `references` — a run takes at most one.
9. A `backward` object without a valid `from`, or with an unknown key — the
   intent is ambiguous.
10. `backward.wrt` or `backward.optimize_forward` with `from: "source"`, or
    `backward.forward_context` with `from: "autodiff"` — each would be silently
    ignored.
11. `from: "autodiff"` with torchax and `emit_readable_reference: false` —
    there is no `base.py` to differentiate.
12. `from: "autodiff"` with `evaluation.compare_against_source: true` — it
    would compare a backward kernel against the forward source.
13. `input.configs` with `input.inputs_fn`, or `init_args`/`init_kwargs` with
    `init_inputs_fn` — two sources for the same arguments.
14. A malformed config: a tensor without a valid shape or dtype, an integer
    tensor without an `init` range, `value` together with `shape`.
15. `input.extraction` on an input that is neither PyTorch nor JAX, without
    `input.configs`, without `entry_point`, without
    `user_confirms_extraction`, or alongside
    `input.entry_point` / `inputs_fn` / `init_inputs_fn`.
16. A PyTorch or JAX `input.path` that is a directory, without
    `input.extraction` — the PyTorch capture tools cannot execute a package,
    and a JAX input has to be the one file that becomes `base.py`.

On success it prints the plan the job implies, so you can see what will happen
before it happens:

```
JOB VALID: rmsnorm-torch
  input        : pytorch  (../../evaluation/examples/rmsnorm/ref.py)
  references   : 0
  oracle       : tools/torchax_oracle.py (mechanical, with step-0 check vs eager)
  base.py      : written from the jaxpr in Phase 0.7  [maxkernel-write-jnp-reference]
  harness      : internal (paired timing)
  external eval: True
  phases       : 0 read state -> 0.2 oracle -> 0.3 jaxpr + HLO export
                 -> 0.7 write jnp reference + jaxpr_compare gate -> 0.9 harness
                 -> 1-6 plan/implement/compile/test/autotune/profile
```

Exit `0` valid · `1` invalid · `2` valid but a declared path does not exist.