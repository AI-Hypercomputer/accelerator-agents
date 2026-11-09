---
name: pallas-code-style
description: >-
  Prescriptive rules for how a Pallas TPU kernel is written down: dimension names, shape comments,
  signature sections, body order, naming, comments, errors, file layout, config classes and reading
  hardware constants. Read before implementing or revising a kernel file.
---

# Pallas TPU kernel code style

How to write a Pallas/Mosaic TPU kernel so the next reader can follow it. Every section is a rule,
with the good form and the form it replaces. It governs how a kernel is *written*, never what it
*decides* — block shapes, pipelining and precision belong to the design files under
[`design/`](design/).

Nothing here is a performance claim. Three rules exist because of the hardware and name the probe
that checks them (§12, §13); everything else is a readability rule.

## 0. The idea behind all of it

A Pallas kernel removes two things ordinary Python relies on.

- **Types are gone.** Every kernel parameter is an untyped `Ref`. Eight refs are eight positional
  slots, and swapping two of them is a bug no checker catches.
- **Locality is gone.** The shape the body sees is decided by a `BlockSpec` and `index_map`
  elsewhere. A reader of `x_ref[...]` cannot see whether it is one tile or the whole tensor.

So style has one job: **put the shape and the memory space back in front of the reader wherever the
framework erased them.** Every rule below is that idea applied somewhere.

## 1. Name the dimensions once, at the top of the file

Open every kernel file with a one-line docstring and a dimension glossary.

```python
"""Fused RMSNorm + output projection, Pallas TPU."""

# Dimensions.
#   B       batch
#   T       tokens per sequence;  T_pad = T rounded up to tt
#   D       model dim (contraction)
#   N       output features
#   tt, td, tn   tile extents of T, D, N seen by the kernel body
#   nt, nd, nn   grid extents: nd = D // td, ...
```

| Form | Means | Example |
| --- | --- | --- |
| `UPPERCASE` | full logical extent | `D`, `N` |
| `t` + letter | tile extent of that dimension | `td` |
| `n` + letter | grid extent over that dimension | `nd = D // td` |

Three names, one dimension: `tn` tiles `N` and there are `nn` of them.

```python
# Bad: four unrelated names for two dimensions
blk = cfg.block_k
num_blocks = contraction // blk
KDIM = x.shape[-1]
kk = pl.program_id(2)

# Good
td = cfg.tile_d
nd = D // td
d_i = pl.program_id(2)  # which D tile this step handles
```

**A padded extent gets its own name.** `T` and `T_pad` are different numbers and the bugs live
where one is used for the other.

## 2. Every ref carries a trailing shape comment

**Every ref parameter has a trailing `# [shape] dtype` comment, in every kernel, however short.**
This replaces the type annotation §4 tells you to omit.

```python
def kernel(
    # Inputs.
    x_ref,            # [tt, td] bf16
    w_ref,            # [td, tn] bf16
    scale_ref,        # [1, tn] f32
    # Outputs.
    o_ref,            # [tt, tn] bf16
    # Scratch.
    acc_scratch,      # [tt, tn] f32   carried across the D grid axis
):
```

Rules for the brackets:

- **Glossary letters, not numbers.** `[tt, td]` says which dimensions; `[256, 512]` goes stale when
  the config changes. The exception is a fixed hardware extent, e.g. `# [8, 128] one native f32
  tile`, and then say so.
- **Say which shape it is.** A `BlockSpec` ref shows the *block*; a `memory_space=pl.ANY` ref shows
  the *whole array*. The code does not distinguish them, and confusing them is how a DMA copies far
  more than intended.

  ```python
  (x_hbm_ref,)  # [B, T, D] f32    whole array, pl.ANY
  (x_vmem_ref,)  # [tt, td] f32     one block, copied from x_hbm_ref
  ```

- **Dtype whenever it is not obvious, and always on scratch.** An f32 accumulator under bf16 inputs
  is a decision; the comment records it was deliberate.
- **Note what the `index_map` did that the body cannot see**: permuted, padded, aliased.

  ```python
  (w_ref,)  # [td, tn]  transposed by index_map; caller passes [N, D]
  (mask_ref,)  # [tt, T_pad]  padded extent, not T
  ```

No `Args:` docstring restating parameter names: the shape is the information, and a docstring is
too far from the parameter to stay current.

## 3. Derive extents from the refs, not the config

Inside the body, read shapes off the refs. The config is a second source of truth and the two
drift.

```python
# Bad: lies silently if a BlockSpec disagrees with the config
tt, tn = cfg.tile_t, cfg.tile_n

# Good
tt, tn = o_ref.shape
```

Compute grid extents (`nd = D // td`) once, in `computation`, and close over them. **No bare shape
indices in arithmetic** — unpack and name them first.

```python
# Bad
nbytes = x.shape[2] * x.shape[3] * 2

# Good
_, _, H, Dh = x.shape
nbytes = H * Dh * jnp.dtype(x.dtype).itemsize
```

## 4. Shape-check at the boundary, bare refs inside

The annotation style inverts at the kernel boundary.

**Outside — `computation` and any public wrapper — annotate arrays**, with jaxtyping using the
glossary's letters where it is available, or `jax.Array` plus an explicit shape check. This is
where a caller's wrong shape can still produce a readable error.

```python
def computation(
    x: Float[Array, "*B T D"],
    w: Float[Array, "D N"],
    scale: Float[Array, "#D N"],
) -> Float[Array, "*B T N"]:
```

jaxtyping: `*B` = any leading batch dims; a lowercase letter = an independent extent of the same
axis (`"T t"`: query and key length); `#` = broadcastable against its namesake.

**Inside — the kernel body — leave refs unannotated** and let the shape comment carry the type.

```python
# Bad: a ref is not an array; this misleads
def kernel(x_ref: jax.Array, o_ref: jax.Array): ...
# Bad: correct but adds nothing the name did not
def kernel(x_ref: jax.Ref, o_ref: jax.Ref): ...


# Good: bare refs with shape comments, as in section 2
```

Past roughly eight refs, positional slots stop being safe even with comments; group related refs
(e.g. per-operand tuples unpacked at the top) rather than growing the flat list further.

## 5. Naming

### Single-character budget

A single character is allowed only for:

- a loop index — `i`, `j`;
- a math tensor whose letter appears in a shape annotation — `q`, `k`, `v` under
  `Float[Array, "*B T H D"]`;
- an uppercase static extent from the glossary — `D`, `N`;
- gradient shorthand `d<x>` — `dq`, `dkv`.

**Anything computed gets a word**, grid indices included:

```python
# Bad: `k` is already the contraction extent
k = pl.program_id(2)

# Good
k_i = pl.program_id(2)  # which K tile
n_start = pl.program_id(1) * tn
```

### Ref suffixes carry the memory space

| Suffix | Use for |
| --- | --- |
| `_ref` | space unambiguous within this signature |
| `_hbm_ref` / `_vmem_ref` / `_smem_ref` | explicit memory space |
| `_scratch` | scratch persisting across grid steps |
| `_slot_ref` | one slot of a multi-buffered scratch |
| `_sem` | a semaphore |
| `_ptr` | **never** — these are shaped, typed references, not pointers |

**Name the memory space whenever more than one appears in the same signature.** All-VMEM kernels
say `x_ref`; a kernel that copies HBM→VMEM says `x_hbm_ref` and `x_vmem_ref`, because the
difference *is* the function.

## 6. The kernel signature is sectioned

Three comments, in every kernel: `# Inputs.`, `# Outputs.`, `# Scratch.` (scalar-prefetch refs
first, under `# Scalar prefetch.`, when present).

```python
def kernel(
    # Scalar prefetch.
    group_ids_smem_ref,   # [nt] i32
    # Inputs.
    x_hbm_ref,            # [B, T, D] bf16   whole array, pl.ANY
    # Outputs.
    o_hbm_ref,            # [B, T, N] bf16
    # Scratch.
    acc_vmem_ref,         # [tt, tn] f32
    copy_sem,
):
```

The input/output/scratch boundary in a Pallas signature is invisible and its order is a contract
with `pallas_call` that nothing validates. Getting it wrong traces, runs, and writes the wrong
buffer.

## 7. The kernel body has one order

1. **Grid indices** — every `pl.program_id`, at the top.
2. **Shape constants** read off the refs (§3).
3. **Semaphore aliasing** — name semaphore slices before any DMA code.
4. **`@pl.when` init guard** — zero or seed the accumulator.
5. **Loads / DMA starts.**
6. **Compute into the scratch accumulator.**
7. **Finalizing store, guarded on the last step.**
8. **Epilogue** — activation, residual, cast.

Init and finalize sit at fixed positions, so a reader finds them without tracing control flow.

```python
def kernel(x_ref, w_ref, o_ref, acc_scratch):  # shape comments elided here only
  d_i = pl.program_id(2)
  tt, tn = o_ref.shape

  @pl.when(d_i == 0)
  def _():
    acc_scratch[...] = jnp.zeros((tt, tn), jnp.float32)

  acc_scratch[...] += jnp.dot(
    x_ref[...], w_ref[...], preferred_element_type=jnp.float32
  )

  @pl.when(d_i == nd - 1)
  def _():
    o_ref[...] = acc_scratch[...].astype(o_ref.dtype)
```

- **`nd` is a static Python int closed over from `computation`**, not `pl.num_programs(2)`. Both
  work; the static value is the one the glossary and comments use.
- **Fusing the zeroing into the first accumulation** (`=` under `d_i == 0`, `+=` under `d_i != 0`)
  saves a pass at the cost of a second guard. Do it deliberately and comment why.

**Every conditional is an optimization barrier**, so control flow is a style concern too. Splitting
the body into statically specialized branches (including an explicit no-op branch), or replacing
`pl.when` with a `lax.cond` nest on first/last step, can let the compiler overlap more within each
branch. It costs readability; make that trade deliberately and comment it, never by default.

## 8. Comments name the compromise

Write noticeably more comments in a kernel body than in the code around it. In a kernel the *why* is
unrecoverable from the code: every line is a compromise with the hardware, and the code records only
which side won.

```python
# Good
# f32 accumulator: the D reduction is long and bf16 accumulation drifts.
# Zero-stride load: broadcasts across sublanes without a copy.
# No-op branch spelled out: a folded guard is an optimization barrier.

# Bad
# Load x.
x = x_ref[...]
# Add to accumulator.
acc_scratch[...] += y
```

One-line module docstring per file (glossary below it as comments); no docstrings on nested
closures such as `def _():` under `pl.when`; one consistent `# TODO` form.

## 9. Unsupported inputs raise `NotImplementedError`

A fallback chain catches `NotImplementedError` and nothing else, so the exception type decides
between a graceful fallback and a crash.

```python
# Bad: turns a fallback into a crash
if bias is not None:
  raise ValueError("Pallas TPU path does not support bias.")

# Good
if bias is not None:
  raise NotImplementedError("Pallas TPU path does not support bias.")
```

`ValueError` means *the caller's arguments are wrong*. `NotImplementedError` means *the arguments
are fine and this implementation cannot serve them*.

## 10. Files

### The kernel file

A MaxKernel kernel is one file with a module-level `kernel` (the body passed to `pallas_call`), a
module-level `computation` (builds and invokes the `pallas_call`), and `main()`. Lay it out
top-to-bottom by **what each function may touch**:

| Section | May touch | Why |
| --- | --- | --- |
| docstring + glossary | — | §1 |
| tiling helpers | Python ints only: block and grid extents, VMEM arithmetic | checkable with no device |
| array helpers | `jax.Array` in, `jax.Array` out — **no refs** | callable from plain JAX, unit-testable against the reference |
| ref / DMA helpers | refs, `make_async_copy`, semaphores | the only code that can deadlock |
| `kernel` | refs | the body, in §7's order |
| `computation` | arrays; the `pallas_call`, `BlockSpec`s, index maps | the only place `pallas_call` appears |
| `main()` | — | demo invocation |

**The code that decides how big the blocks are is not the code that uses them.** A block-size
heuristic is VMEM arithmetic with nothing to say about the math; keep it in the tiling helpers, not
interleaved with the body. Make "does this function take a `Ref`?" a visible boundary: annotated
array functions above, bare ref functions below. A function that takes only arrays can be called
from XLA, a test, or inside the kernel unchanged.

Upstream instances, by path (look up with `tools/search_api.py`):
`tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_mask.py` (mask algebra in pure
numpy) feeding `splash_attention_mask_info.py` (host objects to device arrays); and
`tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_kernel.py` (array-only helpers above ref-taking
`gmm` / `tgmm`). In a short kernel the sections collapse; the order does not change.

## 11. Config classes

When a kernel carries a config (tile sizes and similar knobs), make it immutable and keyword-only:

```python
@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class Config:
  tile_m: int = 128
  tile_n: int = 512

  def __post_init__(self):
    if self.tile_m % 8 or self.tile_n % 128:
      raise ValueError("tile_m must be a multiple of 8 and tile_n of 128")
```

- `frozen`, `kw_only`, `slots`. Use pydantic dataclasses instead where the config is written to and
  read from a cache and must round-trip through serialization.
- **Single-field constraints in the annotation** (`pydantic.PositiveInt`, `Annotated[...]`).
- **Cross-field invariants in `__post_init__`, raising `ValueError`** — "`tile_k` divides `K`" is
  not expressible in an annotation.
- Hardware-dependent legality (e.g. the minimum sublane tile for the dtype) is read at run time, not
  baked into the check (§13). Whether the divisible-by-8 rule changes with dtype: probe
  `block_sublane_legality_by_dtype`.

## 12. Correctness is judged by the shared harness

MaxKernel owns the test harness: `tools/assemble_test_harness.py` builds it once per run from
`tools/test_harness_template.py`, the reference `base.py`, and a generated `get_inputs()`. It binds
the reference's `computation` as `base_computation` and yours as `opt_computation` in separate
namespaces, and compares them with `allclose` at a tolerance the tool sets. Do not write your own
test file, golden file or tolerance. What this means for the kernel code:

- **`computation` takes exactly what `get_inputs()` produces** (shapes, dtypes, static arguments)
  and returns what the reference returns. Read `<run_dir>/test_kernel.py` before writing the
  signature.
- **State precision explicitly on every f32 contraction**, in the kernel and in any helper:
  `precision=` on `jnp.dot` / `dot_general`, with a comment saying why. Unset or `DEFAULT` on f32
  operands may lower to a single bf16 MXU pass, so an f32-looking contraction can be a bf16 one;
  verify on the target with probe `f32_default_is_one_bf16_pass`.
- **When a tolerance fails, suspect the reference before loosening anything.** A kernel that
  contracts at higher precision than a `DEFAULT`-precision reference can be *more* accurate than
  the thing judging it. Check both against a float64 oracle (probe `xla_reference_vs_f64_oracle`)
  and report which side is wrong. CPU interpret mode has no MXU and cannot reproduce this.
- Keep array-only helpers (§10) free of refs so they can be checked in plain JAX against the
  reference when debugging a mismatch.

## 13. Read hardware constants, never hardcode them

**Every hardware constant comes from `pltpu.get_tpu_info()` at run time.** Hardcoded chip constants
were right when written and stay silently wrong on the next chip.

```python
# Bad: true on some chip, wrong on others
NUM_SUBLANES = 8
VMEM_BUDGET = 16 * 1024 * 1024

# Good
info = pltpu.get_tpu_info()
vmem_budget = info.vmem_capacity_bytes
sublanes = info.get_sublane_tiling(x.dtype)
```

The three hardware facts behind this rule, each with the probe that checks it on the target:

- **Sublane tiling widens for narrow dtypes.** `get_sublane_tiling` is a formula,
  `8 * (32 // bitwidth)` — 8 for f32, 16 for bf16, 32 for int8 — not a per-chip measurement. Read it
  rather than writing `8`; whether the hardware honours it is probe `sublane_tiling_by_dtype` (and
  `sublane_tiling_is_a_formula`).
- **VMEM capacity varies by chip.** A fixed budget under-fills a larger chip or overflows a smaller
  one. Read `vmem_capacity_bytes`; see what the target reports with probe
  `tpu_info_reported_constants`.
- **`get_tpu_info()` may describe one TensorCore, not the whole chip.** Check the scaling
  denominator with probe `tpu_info_scaling_denominator` before using any capacity or rate in
  arithmetic; see [`tpu_hardware_facts.md`](tpu_hardware_facts.md) §1.

The one literal that is fine is a native tile extent inside a shape comment, where it documents
rather than computes: `# [8, 128]`.

## Checklist

- [ ] One-line module docstring, then a dimension glossary: uppercase logical, `t`-prefixed tile,
      `n`-prefixed grid. Padded extents named separately.
- [ ] Every ref has a trailing `# [shape] dtype` comment in glossary letters, saying block or
      whole array, noting permute/pad/alias.
- [ ] Extents read off refs; grid extents computed once in `computation`; no bare `x.shape[2]` in
      arithmetic.
- [ ] Arrays annotated at the boundary, refs bare inside. Signature sectioned (§6).
- [ ] Body order: indices → shapes → semaphores → init guard → loads → compute → final store →
      epilogue.
- [ ] Single characters only per §5. Memory space in ref names where two share a signature; no
      `_ptr`.
- [ ] Comments name the compromise, not the operation.
- [ ] Unsupported inputs raise `NotImplementedError`, not `ValueError`.
- [ ] File sections ordered ints → arrays → refs/DMA → `kernel` → `computation` → `main`;
      `pallas_call` only in `computation`.
- [ ] Config frozen, kw_only, slots; cross-field checks in `__post_init__`.
- [ ] `computation` matches `get_inputs()`; explicit `precision=` on every f32 contraction; no
      private test harness or tolerance.
- [ ] Hardware constants from `pltpu.get_tpu_info()` / `get_sublane_tiling()`, never hardcoded.
