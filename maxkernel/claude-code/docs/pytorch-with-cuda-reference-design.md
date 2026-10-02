# Design: PyTorch → Pallas, with a CUDA kernel as a reference donor

Status: implemented on branch `pytorch_cuda`
Scope: extends the MaxKernel loop (`skill/SKILL.md`, `agents/maxkernel-*.md`)

---

## 1. What is being asked for, precisely

Two changes, and they are **not** the same change:

1. **A new primary source language.** The loop must accept a PyTorch module as
   the thing being converted, the way it accepts JAX today.
2. **A new *kind* of input.** Alongside that PyTorch source, the user hands the
   loop a CUDA kernel that is *not* the thing being converted. It is a second,
   independent implementation of (roughly) the same computation, written by
   someone who already thought hard about it. The loop should mine it for
   design ideas during planning.

The existing CUDA support conflates these. Today `input_language` is a single
enum — `jax | cuda | pytorch` — and CUDA means "this is the source, port it".
That is the wrong shape for a reference kernel. A reference is advisory: it
must never define the semantics, never define the baseline, and never be the
thing we measure against.

So the central design move is:

> **Split the single `source` slot into two slots with different trust levels:
> a `primary` that defines truth, and zero or more `references` that only
> propose hypotheses.**

Everything else follows from that split.

### 1.1 Prerequisite: repair the existing non-JAX path

At HEAD, `skill/SKILL.md` step 3 invokes `tools/detect_input_language.py` and
`agents/maxkernel-worker.md` Phase 0.25 dispatches `maxkernel-analyze-source`.
**Neither exists.** `157fc9b` reverted the agent; the tool was never committed.
A `cuda` or `pytorch` run today fails at the first non-JAX step. The work below
assumes these are restored — the new agents replace `maxkernel-analyze-source`
outright, so restoring it verbatim is not necessary, but the missing detector
tool is on the critical path.

---

## 2. The two spines

The clearest way to hold the new workflow in your head is that it has **two
parallel chains of custody**, and the whole design is about not letting them
touch except at one controlled junction.

**The semantics spine (load-bearing).** PyTorch source → golden values →
`base.py` JAX port → test harness → every correctness check and every speedup
number. If anything here is wrong, every number the run reports is meaningless.
Nothing from the CUDA reference is ever allowed into this chain.

**The advisory spine (non-load-bearing).** CUDA reference → structural brief →
reconciliation against the primary → a ledger of borrowable ideas → the
planner's hypothesis list. If this chain is wrong, the run is merely *no better
than it would have been without the reference*. It cannot make the run
incorrect.

The junction is a single document — the ideas ledger — and it carries a trust
level that the planner is required to respect.

```
  SEMANTICS SPINE (truth)                    ADVISORY SPINE (hypotheses)
  ───────────────────────                    ───────────────────────────
  source_torch.py                            reference.cu
        │                                          │
        │ analyze-torch-source                     │ analyze-cuda-reference
        ▼                                          ▼
  torch_context.md ──────────┐          ┌──── ref_cuda_context.md
        │                    │          │
        │ (port)             └──► reconcile-reference ◄─┘
        ▼                              │
     base.py                           ▼
        │                        reference_alignment.md
        │                        ideas_ledger.json   ← trust: aligned|partial|
        │ verify vs golden              │              divergent|rejected
        ▼                               │
  torch_golden.npz  ── GATE ──►  base.py is trustworthy
        │                               │
        ▼                               ▼
  test_kernel.py ──────────────►  plan-kernel  ◄── roofline + profile
                                        │
                                        ▼
                                 implement-kernel  (never sees reference.cu)
```

---

## 3. The new workflow, phase by phase

Phases 1–6 of `maxkernel-worker.md` are unchanged. Everything new is in the
run-once front end, which grows from one phase (0.25) into five.

### Phase 0.1 — Intake and slot assignment *(orchestrator, `SKILL.md`)*

The orchestrator stops classifying "the input" and starts classifying **each
input**. Inputs arrive as a list: inline code blocks, file paths, directories.

```bash
{{VENV_PYTHON}} {{MAXKERNEL_ROOT}}/tools/classify_inputs.py \
  <run_dir>/inbox/ --json <run_dir>/inputs.json
```

`classify_inputs.py` is deterministic marker matching (the generalization of
the missing `detect_input_language.py`). Per file it emits
`{path, language, has_entry_point, evidence}`. The orchestrator then assigns
slots by this rule, in order:

1. If the user named the roles explicitly ("convert this, use this as
   reference"), believe the user. Always.
2. Exactly one non-CUDA file with an entry point → `primary`. CUDA files →
   `reference`.
3. Multiple candidates for `primary` → **ask the user.** Do not guess which of
   two Python files is the thing to convert. A wrong guess here silently
   benchmarks the wrong computation for five iterations.
4. Zero CUDA files → `reference_sources` is empty and the whole advisory spine
   is skipped. The PyTorch path must work standalone.

A CUDA file *can* still be the primary, as today — that is the case where
`references` is empty and `primary.language == "cuda"`. The two-slot model is a
superset of the current behaviour, not a replacement for it.

### Phase 0.2 — Golden capture *(new; the single highest-value addition)*

`maxkernel-worker.md` Phase 0.5 currently contains this admission:

> *"It binds `base.py` as both sides, so it proves the harness runs and the
> shapes are consistent — it compares the baseline against itself and therefore
> can never detect a mistranslated port."*

That hole is tolerable for CUDA, which cannot run here. It is **not** tolerable
for PyTorch, because PyTorch *runs on the CPU in the existing venv*. We can
simply execute the user's reference and keep the answers.

```bash
{{VENV_PYTHON}} {{MAXKERNEL_ROOT}}/tools/capture_torch_golden.py \
  <run_dir>/source.py --out <run_dir>/torch_golden.npz \
  --meta <run_dir>/torch_golden.json --device cpu --dtype-policy preserve
```

Produces:
- `torch_golden.npz` — the exact input tensors (seeded) and the reference
  outputs, plus an fp64 recomputation of the same graph where the ops support
  it, so tolerance can be argued from the source's own conditioning rather than
  guessed.
- `torch_golden.json` — shapes, dtypes, argument order, the entry point's
  signature, `get_init_inputs()` / `get_inputs()` if the file is KernelBench-shaped,
  and the observed fp32-vs-fp64 divergence of the reference itself.

This is a deterministic tool, not an agent. It runs the user's file in a
subprocess with `CUDA_VISIBLE_DEVICES=""`; if the module is hard-bound to CUDA
and will not run on CPU, it exits non-zero and the run degrades gracefully to
the current self-comparison behaviour, with the degradation recorded in state.

Everything downstream gets better because this file exists:
- the port can be *verified* instead of assumed (Phase 0.4),
- `get_inputs()` can be generated from real recorded shapes instead of inferred
  from prose in a context brief,
- `atol`/`rtol` can be defended from the reference's own numerical spread.

### Phase 0.25a — Analyze the primary *(`maxkernel-analyze-torch-source`)*

Same contract as the reverted `maxkernel-analyze-source`, narrowed to PyTorch
and with the golden file available. Writes `torch_context.md` and `base.py`.

The important sections for PyTorch differ from CUDA, and the prompt should say
so rather than reusing the CUDA section list:

- **§2 What It Computes** — the op graph in `forward()` in order, with every
  intermediate's shape and dtype. This is the specification.
- **§3 Module Contract** — constructor parameters vs forward arguments;
  buffers and parameters vs activations; what is `static_argnums` on the JAX
  side. (Getting this wrong is the most common PyTorch port failure: a
  `self.weight` is a runtime array, a `self.eps` is a trace-time constant.)
- **§4 Materialization Map** — which intermediates hit HBM, which XLA would
  fuse anyway, and where the fusion boundaries are. This is the PyTorch
  analogue of the CUDA memory-hierarchy section and it is what tells the
  planner where a Pallas kernel can actually earn its keep.
- **§5 Numerics** — autocast, `.half()`/`.bfloat16()` placement, accumulation
  dtype of each reduction, any `torch.backends` flag that changes precision.
- **§6 Dynamic-shape and control-flow hazards** — `nonzero`, `masked_select`,
  boolean indexing, `.item()`, data-dependent loops. These have no clean TPU
  form and must be surfaced before planning, not discovered at compile time.
- **§8 Port Notes / risk list** — as today, but now it may cite the golden
  comparison instead of speculating.

### Phase 0.25b — Analyze the reference *(`maxkernel-analyze-cuda-reference`)*

This is *not* the same agent as 0.25a with a different input, and making it one
is the mistake to avoid. Its differences are contractual:

- It **must not write `base.py`.** It has no `Write` access to anything outside
  `<run_dir>/ref/`. The reference must never become the measured baseline —
  that is the whole point of the slot split.
- It **must not write a JAX port at all.** No port means no temptation to
  measure against it.
- It reads a deterministic facts file first:
  ```bash
  {{VENV_PYTHON}} {{MAXKERNEL_ROOT}}/tools/cuda_static_facts.py \
    <run_dir>/ref/*.cu --json <run_dir>/ref/cuda_facts.json
  ```
  which extracts `__global__` signatures, launch configurations, shared-memory
  byte counts, `#define`d tile constants and `wmma` fragment shapes by parsing,
  not by reading comprehension. Grid dimensions are load-bearing evidence about
  the decomposition; an LLM misreading `dim3 grid((n+255)/256)` poisons every
  inference built on it.

Output: `<run_dir>/ref/ref_cuda_context.md`, structurally the sections 1, 4, 5,
6, 7 and 9 of the reverted `maxkernel-analyze-source` brief — inventory,
parallel decomposition, memory hierarchy, synchronization/numerics, CUDA→TPU
translation table, and opportunities. Sections 2, 3 and 8 (specification, host
contract, port notes) are **dropped**: the reference does not get to specify
anything, and it is not being ported.

### Phase 0.3 — Reconcile the reference *(`maxkernel-reconcile-reference`)*

The new linchpin. It answers one question: **does the CUDA reference actually
compute what the PyTorch source computes?**

This is not a formality. A reference kernel plucked from another project
routinely differs in ways that quietly invalidate everything borrowed from it:
a fused bias or activation epilogue the torch version does separately; a
causal mask where the torch version is bidirectional; fp16 accumulation the
torch version does in fp32; a transposed layout; a power-of-two-only fast path;
a different head dimension. If the planner borrows a tiling strategy justified
by a fused epilogue that this problem does not have, it will produce a slower
kernel and nobody will know why.

Inputs: `torch_context.md` §2/§3/§5, `ref_cuda_context.md`, `cuda_facts.json`,
`torch_golden.json`.

Output 1 — `<run_dir>/reference_alignment.md`, whose header is a single verdict:

| verdict | meaning | effect on the loop |
| --- | --- | --- |
| `aligned` | same computation, same numerics contract, same or compatible shapes | ledger ideas usable as primary hypotheses |
| `partial` | same core computation, enumerated differences (extra epilogue, different masking, different dtype policy) | each ledger idea is tagged with which difference it depends on; ideas that depend on a difference are demoted |
| `divergent` | same *family* (both are attention, both are GEMM) but materially different problem | ledger is inspiration only; every idea must be independently justified by roofline before adoption |
| `rejected` | different operation entirely, or unparseable | ledger discarded; run proceeds as pure PyTorch→Pallas |

The verdict is written once and never revised mid-run, exactly like
`input_language`.

Output 2 — `<run_dir>/ideas_ledger.json`, the actual deliverable. One record
per borrowable idea:

```json
{
  "id": "LEDGER-003",
  "claim": "Keeps the softmax accumulator unnormalized across the K-loop and
            rescales once at the end, avoiding a second pass over S.",
  "evidence": "ref/flash.cu:112-148, running m_i/l_i registers, BLOCK_N=64",
  "class": "ALGORITHMIC",
  "depends_on_difference": null,
  "tpu_translation": "Carry (m, l, acc) in VMEM scratch across the Pallas grid's
                      kv axis; rescale in the pl.when(last) epilogue.",
  "mechanism": "Removes one full read of the S matrix from HBM: 2*B*H*Sq*Sk*2B
                bytes saved per call.",
  "falsifiable_as": "HBM bytes/call in the XProf trace drop by >=40% vs iter N-1
                     with no change in FLOPs.",
  "status": "proposed",
  "adopted_in": null,
  "verdict": null
}
```

**The portability triage — `class` — is the intellectual core.** Every idea is
sorted into exactly one of three buckets, and the planner treats each bucket
completely differently:

- **`ALGORITHMIC`** — survives the hardware change intact. Online/streaming
  softmax recurrences; the choice of what to recompute vs. store; which operand
  is held resident across the loop; the fusion boundary the author picked; a
  numerical reformulation (log-sum-exp rescaling, unnormalized accumulators);
  exploited sparsity or block structure; loop ordering that changes reuse.
  *These are the reason to read a CUDA kernel at all.* They are statements
  about the problem, not about the GPU.

- **`STRUCTURAL`** — a real decision, but its *value* is hardware-specific and
  must be re-derived. Tile shapes, block counts, unroll factors, K-loop
  chunking, double buffering. The CUDA's `BLOCK_M=128, BLOCK_N=64` is evidence
  about the working set's shape and about what the author believed fit in
  fast memory — it is **not** a value to copy into a `BlockSpec`. The ledger
  records the *ratio and the reason*, and the planner re-solves the tile sizing
  from the 16 MB VMEM budget and the 128×128 MXU.

- **`NON_PORTABLE`** — record it and drop it. Warp shuffles, `__syncthreads`,
  bank-conflict padding, per-thread register blocking, occupancy targets,
  `atomicAdd`-as-reduction, `__ldg`/`__restrict__`, stream overlap,
  `cp.async`. These are listed explicitly *so the planner can see they were
  considered and discarded*, which is a much stronger defence against
  transliteration than silence.

The reconciler is also required to emit a **negative finding** section: things
the CUDA does that the TPU should *not* do, with the reason. "The author
spends 40 lines on a warp-level transpose because shared-memory bank conflicts
punish the naive layout; on TPU the DMA moves whole tiles and this work has no
analogue — do not port it." Naming the trap is what stops a planner walking
into it.

### Phase 0.4 — Verify the port against golden *(new gate)*

```bash
{{VENV_PYTHON}} {{MAXKERNEL_ROOT}}/tools/verify_port.py \
  <run_dir>/base.py <run_dir>/torch_golden.npz \
  --atol <atol> --rtol <rtol> --out <run_dir>/port_verification.json
```

Binds `base.py`'s `computation` against the recorded golden inputs and compares
to the recorded golden outputs.

**Implementation note — CPU by default, TPU when the data fits.** The original
plan said "submit through `tpu_client.py --action correctness_test`" without
qualification. That is only conditionally possible: `tpu_client.py` opens the
code file in text mode and submits the source, so there is no channel for a
binary `.npz`, and the golden arrays must travel base64-inlined inside the
submitted script.

The shipped tool therefore takes `--device cpu | tpu | auto`. CPU is the
default and is the better check on the merits — `base.py` is contractually pure
JAX, a semantic check should not be entangled with TPU rounding, and it costs
no queue time. `tpu` submits the inlined script for the stronger claim that the
port is right on the hardware the run measures on; `auto` does that when the
payload fits under `--max-embed-bytes` and falls back to CPU, recording the
fallback, when it does not. An explicit `--device tpu` that does not fit exits
5 rather than submitting a source file of tens of megabytes.

- **Pass** → write `"port_verified": true` into state and continue. The run's
  denominator is now evidence-backed rather than assumed.
- **Fail** → dispatch `maxkernel-fix-port` (a narrow repair agent: it may edit
  `base.py` and nothing else, and it is told the elementwise diff and the
  worst-offending index, not just "mismatch"). Up to 3 attempts, then **stop
  the run**. A run whose baseline is wrong should not spend five iterations
  optimizing against it — that is worse than no run at all, because it produces
  a confident, wrong speedup number.
- **Skipped** (golden capture failed) → `"port_verified": false` with a reason,
  surfaced in the final report.

This gate is what makes PyTorch input *more* trustworthy than CUDA input, not
less, and it is worth building even if the CUDA-reference feature is deferred.

### Phase 0.5 — Harness *(existing, with two changes)*

- `maxkernel-generate-test-file` reads `torch_golden.json` for shapes and dtypes
  in preference to prose in the context brief. Recorded tensors beat described
  tensors.
- The assembled harness gains an optional golden-comparison mode so that every
  iteration's optimized kernel is checked against **both** `base.py` (the paired
  timing comparison, as today) and the golden values (an absolute correctness
  anchor). Today a port error that survives Phase 0.4 could still be masked if
  an optimized kernel reproduces the *port's* error; anchoring to golden closes
  that.

### Phase 1 — Planning *(`maxkernel-plan-kernel`, extended)*

The planner gains a fourth input alongside `base.py`, the harness and the
profile: the ledger. Its prompt changes in three specific ways.

**(a) Read order is fixed and it matters.** Roofline first, ledger second. The
planner must produce its own bound classification and VMEM arithmetic *before*
opening `ideas_ledger.json`. This ordering is the anti-anchoring mechanism: a
planner that reads a clever CUDA trick first will rationalize a plan around it;
a planner that has already written "this is memory-bandwidth bound, the naive
form moves 4.2 GB, the floor is 0.9 GB" will evaluate the same trick against a
number.

**(b) Every hypothesis carries a provenance tag.**

```markdown
### Hypothesis 2 — fuse the rescale into the KV loop
Provenance: LEDGER-003 (ALGORITHMIC, trust=aligned)
Mechanism: removes one HBM pass over S; predicted 4.2 GB -> 2.8 GB per call
Falsifiable as: XProf HBM bytes/call drop >= 30% vs iter 2
```

Allowed provenance values: `LEDGER-<id>`, `ROOFLINE`, `PROFILE-iter<k>`,
`WIKI-<page>`. This makes the reference's contribution *measurable*: at the end
of the run we can say which ledger entries actually produced speedups, rather
than asserting that the CUDA reference helped.

**(c) Trust-conditional rules.** If `reference_trust == "divergent"`, a
`LEDGER-*` provenance is not sufficient on its own — the hypothesis must also
carry an independent `ROOFLINE` justification. If `rejected`, the ledger is not
read at all. `STRUCTURAL` ideas may never be adopted as literal constants; the
plan must show the VMEM/MXU derivation that arrives at its own tile sizes, and
may cite the CUDA's ratio only as corroboration.

### Phase 2–5 *(unchanged, with one restriction and one addition)*

**Restriction:** `maxkernel-implement-kernel` and
`maxkernel-fix-kernel-compilation` **never read `<run_dir>/ref/`.** The
implementer works from the plan and the ledger entries the plan cites, which
have already been translated into TPU terms. This is enforceable, not just
advisory: `hooks/workspace-guard.py` already gates tool use by path, so add a
rule denying those two agent types read access under `<run_dir>/ref/`. Left
unenforced, the single likeliest failure mode of this whole feature is an
implementer "helpfully" consulting the CUDA and transliterating a warp-level
reduction into `jnp` scalar ops.

**Addition:** `maxkernel-summarize-profile` adjudicates the ledger. For each
entry with `status == "adopted"` in this iteration, it checks the falsifiable
claim against the trace and writes `verdict: "confirmed" | "refuted" |
"inconclusive"` plus one line of evidence, via `tools/ledger.py` (deterministic
status transitions — agents propose, the tool writes). A refuted entry is not
re-proposed by the next iteration's planner. This is the loop *learning* from
the CUDA reference rather than merely reading it.

### Finish *(reporting changes)*

The final report gains a second table:

```
Reference contribution (reference_trust = partial)
  LEDGER-001  unnormalized accumulator     adopted iter2  confirmed   -38% HBM
  LEDGER-003  KV-major grid order          adopted iter3  refuted     +4% time
  LEDGER-004  fused epilogue               not adopted    (depends on difference D2)
  LEDGER-007  warp-shuffle reduction       NON_PORTABLE   dropped
```

And the existing baseline-honesty paragraph is extended: for a PyTorch primary,
speedups are against the JAX port of the torch module, now *verified against
torch's own CPU outputs*; the CUDA reference was never executed and contributes
no number.

---

## 4. State schema changes

`<run_dir>/state.json` gains:

```json
{
  "primary": {
    "language": "pytorch",
    "source_path": "<abs>/source.py",
    "context_path": "<abs>/torch_context.md",
    "golden_path": "<abs>/torch_golden.npz",
    "golden_meta_path": "<abs>/torch_golden.json",
    "port_verified": true
  },
  "references": [
    {
      "kind": "cuda",
      "source_path": "<abs>/ref/flash.cu",
      "facts_path": "<abs>/ref/cuda_facts.json",
      "context_path": "<abs>/ref/ref_cuda_context.md"
    }
  ],
  "reference_trust": "partial",
  "reference_alignment_path": "<abs>/reference_alignment.md",
  "ideas_ledger_path": "<abs>/ideas_ledger.json"
}
```

Backward compatibility: keep `input_language` and `source_context_path` as
aliases of `primary.language` and `primary.context_path`, written on every
state update, so the seven agents that already branch on them keep working
unmodified. New agents read the structured fields. Resuming a run that predates
this schema fills `primary` from the flat fields and sets `references: []`.

---

## 5. Inventory of new components

### Tools (deterministic — no model judgement)

| Tool | Purpose | Why deterministic |
| --- | --- | --- |
| `classify_inputs.py` | per-file language + entry point + slot candidacy | generalizes the missing `detect_input_language.py`; classification must be reproducible across resumes |
| `capture_torch_golden.py` | run torch on CPU, record inputs/outputs/fp64 spread | executing code is not a judgement call |
| `cuda_static_facts.py` | extract launch config, tile constants, `wmma` shapes from `.cu` | grid dims are evidence; an LLM must not paraphrase them |
| `verify_port.py` | assemble + submit the golden check for `base.py` | gate must be mechanical |
| `ledger.py` | read/append/transition `ideas_ledger.json` | status transitions are a state machine, not prose |

### Subagents

| Agent | Writes | Explicitly cannot |
| --- | --- | --- |
| `maxkernel-analyze-torch-source` | `torch_context.md`, `base.py` | write Pallas; read `<run_dir>/ref/` |
| `maxkernel-analyze-cuda-reference` | `ref/ref_cuda_context.md` | write `base.py` or anything outside `ref/`; write Pallas |
| `maxkernel-reconcile-reference` | `reference_alignment.md`, `ideas_ledger.json` | write code of any kind |
| `maxkernel-fix-port` | `base.py` only | touch the optimized kernel or the harness |

### Extended subagents

| Agent | Change |
| --- | --- |
| `maxkernel-plan-kernel` | roofline-before-ledger ordering; provenance tags; trust-conditional adoption rules |
| `maxkernel-implement-kernel` | hard prohibition on reading `ref/` |
| `maxkernel-generate-test-file` | prefer `torch_golden.json` shapes over brief prose |
| `maxkernel-summarize-profile` | adjudicate adopted ledger entries against the trace |

### Hook

`hooks/workspace-guard.py` — deny reads under a **sealed** `<run_dir>/ref/`.

**Implementation note — the rule is phrased on paths, not on agents.** The
original plan said "deny reads for `maxkernel-implement-kernel` and
`maxkernel-fix-kernel-compilation`". A `PreToolUse` payload carries
`tool_name`, `tool_input` and `cwd` and does not identify the calling subagent,
so that rule could not actually be enforced. The shipped mechanism is a seal
file: the worker drops `<run_dir>/ref/.sealed` at the end of Phase 0.3, and the
guard refuses every read under a `ref/` directory containing it. This is
enforceable with the information the hook has and expresses the same intent —
the raw reference is readable exactly while it is being distilled, and never
afterwards. Every consumer that must not read it (planner, implementer,
compilation fixer) runs after Phase 0.3 by construction.

---

## 6. Why this shape, and what it is defending against

| Failure mode | Where it would happen | Defence in this design |
| --- | --- | --- |
| Reference silently becomes the measured baseline | analyzer writes `base.py` from the CUDA | the reference analyzer has no write access outside `ref/` |
| Borrowed idea is justified by a computation we aren't doing | planner reads brief directly | Phase 0.3 reconciliation + `depends_on_difference` tagging |
| Transliterated SIMT structure | implementer reads `.cu` | `ref/` sealed after Phase 0.3 + explicit `NON_PORTABLE` bucket |
| CUDA's tile constants copied into `BlockSpec` | planner treats numbers as values | `STRUCTURAL` class requires re-derivation from VMEM/MXU |
| Mistranslated torch port makes all numbers wrong | Phase 0.5's self-comparison can't see it | Phase 0.2 golden + Phase 0.4 gate |
| Anchoring: plan rationalized around a clever trick | planner reads ledger first | enforced read order, roofline written before ledger is opened |
| Reference credited for speedups it didn't cause | final report | provenance tags + profile adjudication |
| Precision quietly taken from the CUDA | numerics section of the wrong brief | torch `§5` is authoritative unless reconciliation proved agreement |

The asymmetry throughout is deliberate: **the primary source is trusted by
default and verified anyway; the reference is distrusted by default and earns
influence one adjudicated claim at a time.**

---

## 7. Sequencing

1. **Repair.** Commit `tools/detect_input_language.py` (or go straight to
   `classify_inputs.py`) and restore a working primary-source analyzer. Without
   this nothing non-JAX runs at all.
2. **PyTorch primary, standalone.** `analyze-torch-source`,
   `capture_torch_golden.py`, `verify_port.py`, `fix-port`, the Phase 0.4 gate.
   This is independently valuable and ships without any CUDA-reference work.
3. **Reference slot, inert.** Two-slot state schema, `classify_inputs.py`,
   `analyze-cuda-reference`, `cuda_static_facts.py`. Brief is produced but the
   planner does not read it yet. Lets the brief's quality be judged before it
   can affect a plan.
4. **Reconciliation and ledger.** `reconcile-reference`, `ledger.py`, planner
   provenance tags, guard-hook restriction.
5. **Closing the learning loop.** Profile adjudication, refuted-idea
   suppression, the reference-contribution table, and — the natural extension —
   promoting confirmed `ALGORITHMIC` entries into `wiki/concepts/` so the next
   run on the same kernel family starts with them.

Step 5 is where this stops being a translation feature and becomes what the
wiki is already built for: a loop that accumulates transferable knowledge about
what makes a kernel fast, sourced from every good implementation it is shown.
