# The torchax oracle pipeline

How a PyTorch module becomes a Pallas kernel without a hand-written JAX port in
the middle, and how every step is checked.

All numbers below are measured, on `evaluation/examples/rmsnorm/ref.py`.

---

## The five steps

```
                    ┌─────────────────────────────────────────┐
  source.py  ──────▶│ 0. torchax_oracle.py                    │
  (PyTorch)         │    extract_jax(model) -> (states, fn)   │
                    │    STEP 0: fn(x) vs eager model(x)      │──▶ oracle.npz
                    └─────────────────────────────────────────┘    oracle.json
                                      │
                                      │  the agent NEVER sees jax_fn
                                      ▼
                    ┌─────────────────────────────────────────┐
  source.py  ──────▶│ 1. jaxpr_export.py                      │──▶ ref.jaxpr.txt
                    │    jaxpr  = what was asked for          │──▶ ref.hlo.txt
                    │    HLO    = what XLA actually did       │──▶ ref.facts.json
                    └─────────────────────────────────────────┘
                                      │
                    ┌─────────────────┴───────────────────────┐
                    │   given to the agent as reference       │
                    ▼                                         ▼
       ┌─────────────────────────┐              ┌──────────────────────────┐
       │ 2. agent writes jnp     │   OPTIONAL   │ 3. agent writes Pallas   │
       │    reference            │              │    kernel                │
       └─────────────────────────┘              └──────────────────────────┘
                    │                                         │
        ┌───────────┴───────────┐                             │
        ▼                       ▼                             ▼
  allclose vs oracle    jaxpr_compare.py                allclose vs oracle
                        (structural)                    ONLY. never structural.
```

Step 2 is optional and user-controlled. Step 3 is not.

---

## Step 0 — the oracle, and why it is itself verified

`torchax.extract_jax(model)` converts the module to JAX **mechanically**. No
LLM, no transcription, so no mistranslation risk — which is the whole reason to
prefer it over a hand-written `base.py`.

But torchax is a translation too, and translations can be wrong. Measured:

```
STEP 0  torchax vs eager PyTorch: max|diff| = 1.907e-06   rel = 3.528e-07
```

Small, real, **not zero**. So `torchax_oracle.py` runs the comparison before
anything downstream trusts `jax_fn`, and **exits 4** if it fails. An oracle that
was never checked is an assumption wearing a lab coat.

The agent never receives `jax_fn`. Hiding it is not about information — the
agent has the PyTorch source and could in principle reconstruct it. It is about
**anchoring**: an agent shown a working JAX implementation produces a
transliteration of it rather than a TPU-native design.

### Where the oracle runs

`torchax_oracle.py` runs on the host. `capture_torch_golden.py` — the
`llm_port`-mode oracle — takes `--device {cpu,xla}`, and `verify_port.py` takes
`--device {cpu,tpu,auto}`, so either half of that comparison can run on the
accelerator.

**Default to CPU on both.** The gate asks whether the port is *semantically*
faithful, and on CPU a disagreement is attributable to the port rather than to
matmul precision, reduction order or accumulation width. It also costs no TPU
queue time. Use `--device xla` / `--device tpu` when you want the stronger
claim — that the port is right on the hardware the run measures on — and keep
the two halves on the same device. `verify_port.py` warns when they differ,
because a mismatched pair turns a device difference into what reads as a port
bug.

### Calling convention — the trap

`extract_jax` returns `jax_func(states, args_tuple, kwargs=None)` where `states`
is a flat dict of parameters and buffers. Flattened for tracing, the signature
is **states first**:

```
{ lambda ; a:f32[2048] b:f32[4,256,2048]. ...   # a = weight, b = x
```

`tools/capture_torch_golden.py` records the **opposite** order — forward inputs
first, then parameters. Both are written explicitly into the oracle manifest
under `flat_signature` and `golden_compatible_order`. Never assume one.

---

## Step 1 — two IRs, because they answer different questions

**jaxpr** is pre-optimization. It says what was asked for, and it is better spec
material than the PyTorch source:

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

It shows what the source hides: `.mean(-1)` decomposed into `reduce_sum` +
`div 2048.0`, the reduction axis as `axes=(2,)`, the exact broadcast structure,
and eps as its real fp32 value `9.999999747378752e-06` rather than `1e-5`.

**Optimized HLO** is post-XLA. It says what XLA *did*. For this same RMSNorm:

```
HLO fusions: 2    kinds: {'kLoop': 2, 'kCustom': 1}
```

Two fusion regions — so XLA left an HBM round trip between them. **That is the
opportunity, and it is invisible in the jaxpr.** Planning from jaxpr alone
produces proposals to fuse things XLA already fused.

### Scale

| target | equations | ~tokens |
| --- | ---: | ---: |
| RMSNorm | 9 | 120 |
| one transformer block | 76 | 1,550 |

Fine per fusion target. A 32-layer model unrolled would be ~50k tokens, so
`--max-eqns` warns rather than silently handing that to a planner.

---

## Step 2 — the optional jnp reference, and how it is checked

User-controlled. Worth knowing what you give up by skipping it: the jnp
reference is also the **fallback artifact** (if Pallas will not compile — static
shapes, data-dependent control flow — there is nothing to fall back on) and the
**secondary baseline** that separates the kernel's contribution from torchax's
lowering quality.

When produced, it is checked **twice**, and neither check alone is sufficient.

### Value alone is too weak

Measured, an agent that drops the epsilon entirely:

```
missing_eps   value vs oracle: max|diff| = 7.820e-05
```

That **passes** a 1e-2 `allclose`. The bug is invisible to the value check and
is caught only structurally, as `ref-only={'ADD': 1}`.

### Structure alone is too strict

Measured, an agent writing `jnp.mean(jnp.square(x), -1)` — a **bit-identical**
implementation, max|diff| exactly 0.0 — has a jaxpr that differs from the
reference, because torch's `.pow(2)` emits `pow` while `jnp.square` emits
`square`. Naive structural equality rejects a perfect answer.

So `jaxpr_compare.py` normalizes before diffing. Squaring has three common
spellings (`x ** 2`, `jnp.square(x)`, `x * x`) and all three collapse to
`SQUARE`; `x * rsqrt(v)` and `x / sqrt(v)` are registered as a benign swap.
Verdicts:

| verdict | meaning |
| --- | --- |
| `identical` | normalized census and reduction signature both match |
| `equivalent` | differs only through known-equivalent spellings |
| `divergent` | a real structural difference — worth a look |

Measured on four candidates:

| candidate | verdict | caught by | value diff |
| --- | --- | --- | ---: |
| faithful (`mean`+`square`) | `identical` | — | 0.000e+00 |
| reformulated (`/sqrt`) | `divergent` | census `ref-only={MUL:1}` | 1.907e-06 |
| wrong reduction axis | `divergent` | **reduction signature** | 1.164e+00 |
| epsilon dropped | `divergent` | **census** `ref-only={ADD:1}` | 7.820e-05 |

Note the last two rows: one is caught only by the reduction signature (its
census is *identical* to the reference), the other only by the census. Both
checks are load-bearing.

The calibration is deliberate: catch real bugs, never reject faithful code, and
report a reformulation as `divergent` with a one-line legible diff rather than
absorbing it. Over-normalizing to make reformulations pass would also absorb
the dropped epsilon.

---

## Step 3 — the Pallas kernel, checked by value only

**Never run `jaxpr_compare.py` against a Pallas kernel.**

A good Pallas kernel deliberately changes the structure — online softmax,
unnormalized accumulators, a fused epilogue, a different loop order. Structural
divergence there is the goal, not a defect. Mechanically it would not work
either: `make_jaxpr` on a `pallas_call` yields one opaque primitive with the
body nested inside, so the census is apples to oranges by construction.

The Pallas kernel is checked against the oracle by value, and by nothing else.

---

## The torch/torchax pin, and why it caps the chain

`requirements.txt` pins **`torch==2.9.0`** and **`torchax==0.0.13`**, and the
pin is load-bearing in both directions.

torchax 0.0.13 does **not** import against torch 2.14: it builds autocast and
decomposition tables at import time keyed on aten overloads that newer torch
removed — `aten.prod.dim_Dimname`, `aten.cholesky`, `aten.all.dimname` — and
some of those tables are built lazily when the `Environment` is first
constructed, so the failure can surface well after the import line.
`tools/torchax_oracle.py` catches it and reports a version conflict rather than
letting an `AttributeError` naming an operator masquerade as a missing package.

On torch 2.9 those operators still exist and torchax imports cleanly. What sets
2.9 rather than something newer is `torch_xla`, whose latest release must equal
the torch minor version — so `torch_xla` caps the whole chain, and `jax[tpu]`'s
hard pin on `libtpu` anchors everything underneath. torchax also imports `flax`
without declaring it, which is why that appears in `requirements.txt` too.

## Step-0 tolerance is a property of the backend, not of the conversion

Measured, same comparison, two devices:

| backend | torchax vs eager PyTorch | why |
| --- | --- | --- |
| CPU | **~1.9e-06** | fp32 throughout; the two differ only in reduction order |
| TPU | **~1.0e-3** | XLA's default matmul precision on TPU is bf16, so any matmul-bearing module diverges from eager fp32 by roughly bf16 epsilon |

*(TPU figure measured on a v6e-8 VM, Linear+GELU, torch 2.9 / torchax 0.0.13.)*

A single fixed tolerance is therefore wrong on one of the two. `--atol`/`--rtol`
default to the backend's own floor — `DEFAULT_TOLERANCE` in
`tools/torchax_oracle.py`, `1e-4` on CPU and `1e-3` on an accelerator — and the
resolved value is printed and recorded in the manifest.

The accelerator floor sits **at** the measured divergence, not above it. That
is deliberate and it has a cost: a module that lands just the wrong side of
bf16 epsilon will fail step 0 and need an explicit `--atol`. The alternative is
worse — a loose step 0 admits an oracle that does not reproduce the source, and
every check afterwards is measured against it.

This matters because the failure is misleading. At `1e-4` a TPU run fails step 0
and falls back to `llm_port`, reporting *"torchax does not reproduce eager
PyTorch"* — which is **false**, and sends you hunting a conversion bug that is
really just bf16. The step-0 failure message now says so explicitly when the
divergence is near that scale.
