# Synthesize a baseline

`maxkernel-worker` reads this in Phase 0.1. The user did not supply a source
file. They described what they want computed, and you write the reference
implementation of it.

## Read this before you write a line

**You are writing the thing this run will be scored against.**

Every other route into MaxKernel starts from code the user wrote. That code is
the specification, the baseline, and the thing the speedup is measured against
— and because the user wrote it, none of those roles is in question. Here, all
three come from you.

That creates two failure modes with no downstream detector:

*   **Write it wrong and the whole run optimizes the wrong computation.** There
    is no user source to check against, so no oracle can catch it. Every gate
    downstream compares the kernel to *your* baseline; if your baseline is
    wrong, they all pass and the answer is wrong.
*   **Write it slowly and every speedup is inflated.** A baseline that
    materializes an unnecessary intermediate, or loops where it could batch,
    makes the optimized kernel look better by exactly the margin of your
    mistake. You would be grading your own work with a curve you set.

Neither is a hypothetical to be careful about. They are the reason this phase
ends with a **human approval gate**: you write the baseline, the run stops, and
the user reads it before anything is optimized. Write it to be read.

--------------------------------------------------------------------------------

## Inputs and outputs

`<run_dir>` is the worker's own.

*   State file: `<run_dir>/state.json`
*   The specification: `<run_dir>/job.json`, under `input.specification`
*   Your outputs, both required:
    *   `<run_dir>/source.py` — the baseline implementation
    *   `<run_dir>/baseline_rationale.md` — what you wrote and why, for the
        user's review

Do not write `<run_dir>/base.py`. That name belongs to the JAX reference and is
produced later by the normal route, from `source.py`, exactly as it would be for
a user-supplied file. Leave `state.json` alone while following this reference;
Phase 0.1 records the approval once the user gives it.

--------------------------------------------------------------------------------

## Step 1: Read the specification and find what is missing

`input.specification` gives you:

| field | what it fixes |
| --- | --- |
| `description` | the computation, in prose |
| `operation` | a short name |
| `synthesize_as` | `"pytorch"` (default) or `"jax"` |
| `tensors[]` | every tensor's `name`, `role`, `shape`, `dtype` — **the problem dimensions, which you may never change** |
| `constants` | scalars such as `eps`, head counts, scale factors |
| `numerics` | accumulation dtypes, where precision matters |
| `edge_cases` | masking, ragged tails, divisibility |
| `reference_formula` | the mathematics, when the user gave it |
| `known_implementations` | a named equivalent, e.g. "LLaMA's FeedForward" |

**Where the specification is ambiguous, do not resolve it silently.** Pick the
reading you believe is intended, implement that, and record the ambiguity and
your choice prominently in `baseline_rationale.md`. An assumption the user can
see and correct at the approval gate costs one round trip; one they cannot see
costs the whole run.

If the specification is too thin to implement at all — no formula, no named
equivalent, and a description that could mean several different computations —
STOP and report to your caller exactly what you need. Do not invent a plausible operation.

## Step 2: Write `<run_dir>/source.py`

Target `synthesize_as`:

*   **`"pytorch"`** (the default, and the better choice) — write a
    `torch.nn.Module` named `Model`, plus `get_init_inputs()` and
    `get_inputs()`. The run then proceeds exactly as a user-supplied PyTorch
    input: torchax converts it mechanically, that conversion is checked against
    eager PyTorch in step 0, and `base.py` comes out of the verified route. You
    get the whole gate chain for free.
*   **`"jax"`** — write a module-level `computation(...)`, pure JAX. This skips
    the conversion, and with it every check: nothing independent exists to
    verify the baseline against. Only use it when the user asked for it.

Follow `{{MAXKERNEL_ROOT}}/evaluation/templates/reference_model.py.template`
for the PyTorch shape. The contract:

```python
class Model(nn.Module):          # the computation
def get_init_inputs():           # constructor args — list, or (args, kwargs)
def get_inputs():                # forward args, fresh per call
def get_input_groups():          # optional — explicit edge cases worth checking
```

### What "plain" means, concretely

Write what a competent engineer writes when they are not thinking about
performance yet:

*   **Use the framework's own operators.** `torch.nn.functional.silu`, not a
    hand-rolled `x * torch.sigmoid(x)`. `@` for matmuls. The idiomatic call is
    both clearer and the one a reader can check against the formula.
*   **One statement per step of the mathematics**, named after what it is, with
    a shape comment: `# Shape: (batch, seq, intermediate)`.
*   **Do not fuse, tile, chunk, or reorder for speed.** Not in the baseline.
    Every fusion you perform here is one the Pallas kernel cannot claim later,
    and the run will report a smaller speedup that is no less real — which
    means you have hidden work from the measurement.
*   **Do not pessimize either.** No unnecessary `.contiguous()`, no
    materializing an intermediate you do not need, no Python loop over a batch
    dimension that vectorizes cleanly. That inflates the speedup, which is
    worse than useless — it is a number that looks like success.
*   **Honour `numerics` exactly.** If the spec says accumulate in fp32 from
    bf16 storage, do that. Precision the baseline gives away silently becomes
    precision the kernel is not required to keep.
*   **Preserve every shape and dtype from `tensors[]` verbatim** (general_rules
    #6). They are the problem. If a shape seems wrong to you, say so in the
    rationale — do not change it.

`get_input_groups()` is worth writing when `edge_cases` names anything: a
ragged tail, a zero row, a non-divisible size. Without it the harness only ever
samples the typical case.

## Step 3: Write `<run_dir>/baseline_rationale.md`

This is what the user reads at the approval gate. Assume they will spend two
minutes on it, and that those two minutes are the only defence this run has.

```markdown
# Synthesized baseline: <operation>

## What I implemented
The computation in one paragraph, as equations over the named tensors. A reader
should be able to check this against their own intent without reading Python.

## Assumptions I made
Every place the specification was ambiguous and I chose a reading. State the
alternative I did not take. **If this list is empty, say so explicitly** — an
absent section reads as an oversight, a stated "none" reads as a claim.

## Shapes and dtypes
The table from the specification, as implemented, so a transcription error is
visible side by side.

## Numerics
Where accumulation happens and in what precision, and any place the framework's
default differs from what the specification asked for.

## Why this is a fair baseline
The honest part. Name the operations a Pallas kernel is likely to fuse or
avoid, and confirm you did NOT pre-fuse them. Name anything that might look
gratuitously slow and explain why it is the idiomatic form rather than a
handicap.

## What I did not implement
Anything in the specification I left out, and why.
```

## Step 4: Check it runs before handing it over

```bash
{{VENV_PYTHON}} -c "
import ast, sys
src = open('<run_dir>/source.py').read()
ast.parse(src)
assert 'def get_inputs' in src
print('parses, has get_inputs')
"
```

Then execute it once on CPU — construct the model with `get_init_inputs()`,
call it with `get_inputs()`, and confirm the output shape matches the `output`
row of `tensors[]`. A baseline that does not run wastes the user's approval
round trip.

## What you must not do

*   **No Pallas.** Not in `source.py`, not anywhere. The loop's entire job is
    to beat this file.
*   **No optimization.** See above — every trick you apply here is one the
    kernel cannot be credited for.
*   **No changing the user's shapes, dtypes or constants.**
*   **No proceeding past the gate.** You write the files and go back to Phase
    0.1 step 3, which verifies them and then stops the run for the user's
    approval. You do not approve it yourself and you do not advance the loop.

## Before you return to Phase 0.1

Both files must exist and the smoke test must have produced the specified
output shape. Written here means "awaiting approval", not "accepted": the
approval is the user's and happens after the worker returns.

Keep, for the report Phase 0.1 step 4 sends up, 3–5 sentences: the operation
you implemented, the one-line formula, the number of assumptions you recorded
(and the most consequential one), and confirmation that the file parses and
runs with the specified output shape. State plainly that the baseline is
awaiting user approval and that no Pallas was written and nothing was
optimized.

If the baseline would not run, do not hand a broken file to the approval gate:
fix it, or stop and report the error to your caller.
