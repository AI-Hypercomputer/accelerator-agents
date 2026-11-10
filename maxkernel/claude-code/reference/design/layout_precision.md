---
name: layout-precision
description: >-
  Design rules for block shapes, the lane/sublane asymmetry, transposes, masking, dtype choice and
  in-kernel precision in Pallas TPU kernels, plus the arrangements under which a relayout is never
  generated. Read it when choosing block shapes or dtypes, or when a design contains a transpose,
  expansion, concatenation or narrow matmul.
---

# Layout and precision

The two trailing axes of a TPU array are not alike: the last maps to lanes, the second-minor to
sublanes. Most surprising layout costs trace back to that asymmetry, or to a permutation the design
asked for without noticing. Precision belongs here because choosing a dtype is choosing how many
bytes cross HBM.

Probes run with
`tools/run_probe.py --pick <probe_name> --run_dir <run_dir>`.

## Questions and probes

| question | probe(s) (`layout`, `layout_arrange`) |
| --- | --- |
| What does a block smaller than one native tile cost per useful element? | `native_tile_granularity` |
| Does the reported per-dtype sublane tiling show up in measured time? | `sublane_tiling_by_dtype` |
| Is reducing over lanes cheaper or dearer than over sublanes? | `lane_vs_sublane_reduction` |
| What does an in-kernel transpose cost against one outside / in `dot_general`? | `transpose_cost` |
| Is masking a ragged tail cheaper than padding the array? | `mask_versus_pad` |
| What does an in-kernel dtype conversion cost per element? | `dtype_conversion_cost` |
| Do `pltpu.bitcast` and sub-byte dtypes work in a kernel? | `bitcast_and_subbyte` |
| Is expansion cost set by the axis or by blocked vs interleaved spelling? | `expand_blocked_vs_interleaved`, `expand_axis_leading_vs_sublane_vs_lane` |
| Do `n` narrow matmuls cost more than one over the lane-concatenated operand? | `concat_to_widen_narrow_matmul` |
| Does concatenating varying operands that share a stationary one pay? | `weight_stationary_lhs_concat` |
| Is expanding a matmul operand before the contraction paid for in MACs? | `expand_before_vs_after_matmul` |
| Does a lane split cost anything off a 128 multiple? | `lane_split_alignment` |
| Does the compiler hoist elementwise work out of an expansion? | `elementwise_before_vs_after_expand` |
| Does a select cost more when the index width mismatches the data? | `mask_index_dtype_match` |
| Does a sliding window want its windowed axis leading with sublane extent 1? | `sliding_window_compact_vs_sublane` |
| Is a masked reduction cheaper than a real transpose for a degenerate axis? | `degenerate_transpose_via_masked_sum` |

Related probes in other modules: `block_sublane_legality_by_dtype`, `sublane_tiling_is_a_formula`,
`native_matmul_dtype_matrix`, `f32_default_is_one_bf16_pass`, `xla_reference_vs_f64_oracle`
(hardware); `integer_ops_int32`, `dot_dtype_matrix` (lowering); `mxu_peak_by_dtype`,
`mxu_block_rows_cost` (compute_units).

---

## The block-shape rule

The Pallas TPU lowering requires the last two block dims to be divisible by 8 and 128 respectively,
or equal to the corresponding array dims. 8 sublanes by 128 lanes; read `num_sublanes` /
`num_lanes` from `pltpu.get_tpu_info()`. A violating block is a trace-time `ValueError`.

**Necessary, not sufficient, and dtype-dependent.** Narrow integer dtypes can demand their own
sublane multiple: an int8 block may be refused below its own sublane tiling (`Failed to set window
params`) while f32 and bf16 follow the rule as written. Toolchain-dependent; verify with
`block_sublane_legality_by_dtype` for every dtype the kernel blocks, including fp8 and int4.

**The `or equal to` escape** is how a small trailing extent passes: reshape so the small axis spans
the whole array. A rank-3 `(steps, sublanes, lanes)` array blocked `(1, sublanes, lanes)` is legal
for any `sublanes` and `lanes`. It is also how you write a legal kernel with a catastrophically bad
block.

**Tiling applies to the last two dims, so rank decides which axis is constrained.** On a rank-2
memref the leading dim is tiled too; at rank ≥ 3 only the trailing two are. So a per-row DMA
destination should be the leading axis of a rank-≥3 scratch. Flattening `[NB, K, C]` to
`[NB, K*C]` makes `NB` a tiled leading dim of a rank-2 memref and is refused unless it divides
the tile. When a rank reduction is refused, check which axis became tiled before assuming the shape
is wrong.

## Lane and sublane are not symmetric

- **Lanes:** hypothesis: below a threshold of a couple of lane widths, the lane extent is
  effectively free, because a small block's cost is dominated by the fixed per-step cost rather
  than by its bytes. Whatever you did not ask for, you paid for anyway: a narrow trailing dim does
  not save time. Find the threshold on your chip with `native_tile_granularity`.
- **Sublanes:** a short sublane extent wastes less per element, but still pays the per-step cost.
- **Moves:** what costs is the permutation a move generates, not the axis it names. See
  "Arranging so the relayout never exists" below.

### Small blocks are not cheap

`native_tile_granularity` sweeps lane and sublane extents with the step count held fixed, so data
volume grows with the block. Hypothesis (verify on the target): total time stays nearly flat while
the data grows by a large factor, on both axes, more extremely on lanes. The per-step fixed cost
dominates every small block.

The conclusion is **"make blocks big", not "match the native tile".** The tile is 8×128; the
bandwidth knee is far larger. Size blocks to the knee from `block_size_bandwidth_curve` and the step
model in [`pipelining.md`](pipelining.md).

## Reported versus effective sublane tiling

`get_tpu_info()` reports a per-dtype sublane tiling, `8 * (32 // bitwidth)` (8 f32, 16 bf16, 32
int8, 64 int4), while common code hardcodes 8 for all dtypes. It is a formula, not a measurement of
the chip: verify with `sublane_tiling_is_a_formula`.

Does it bind? `sublane_tiling_by_dtype` runs an elementwise add over `(steps, sublanes, 128)` with
total rows fixed, per dtype, relative to each dtype's tallest arm. If bf16 padded to 16 sublanes,
its 8-sublane arm would sit apart from f32's. Hypothesis: both dtypes track the step count, not the
reported tiling, so **do not size a block to the reported tiling; size it to the bandwidth knee.**

The same probe exposes a lowering limit: **int8 elementwise vector arithmetic may not lower**
(`Only vector<i16> and vector<i32> are supported ... Please cast`). Holding int8 in a ref and doing
arithmetic on it are separate questions. Plan the cast to i16/i32 into the kernel; verify with
`integer_ops_int32` and `bitcast_and_subbyte`.

## Reduction axis

`lane_vs_sublane_reduction` sums a large f32 array over axis −1 versus −2, then broadcasts back.
Hypothesis: no measurable difference. **Reduce over whichever axis the algorithm wants.** This
measures the reduction only, not a transpose to enable one.

## Transposes: absorbed into `dot_general` versus a real permute

- **A transpose that feeds `dot_general` is absorbed into the dimension numbers.** `transpose_cost`
  compares no transpose, `rhs.T` in-kernel and contraction via dimension numbers on a bf16 matmul.
  Hypothesis: all three are indistinguishable. Do not design around an RHS transpose; pick
  dimension numbers from the layout you have (NN for `[M,K]@[K,N]`, NT when the RHS is stored
  `[N,K]`) instead of moving the array to fit a fixed contraction.
- **A move on a value that never reaches the MXU is a real permute** — a `swapaxes` on an
  activation, a reduction landing on the wrong axis, a repeat on a trailing axis. It costs real
  per-element time. Costs and the probes that measure them are in
  [`../tpu_hardware_facts.md`](../tpu_hardware_facts.md) (relayout). Free in one case and expensive in
  the other is the whole rule: **a transpose you can hand to `dot_general` is free; one you cannot is
  not.**
- **The LHS transpose is the expensive operand case.** `lhs.T @ rhs` cannot feed the MXU directly
  and goes through the XLU; upstream grouped-matmul kernels spend an extra VMEM buffer on the LHS
  (`num_buffers + 1` instead of `num_buffers`) to cache the transposed tile rather than recompute it.
  Treat an LHS transpose as expensive until measured; `transpose_cost` covers only the RHS.
- **Rolls cost by tile alignment, not axis**: a roll by a whole tile is cheap on either axis, a roll
  by one element costs on either. A strided in-kernel read may not compile at all
  (toolchain-dependent). See [`../tpu_hardware_facts.md`](../tpu_hardware_facts.md).

## Masking versus padding

`mask_versus_pad` runs a ragged f32 array (rows not dividing the block) three ways: masked tail,
same grid unmasked (control, isolates the iota-and-select cost), and padded to a multiple.
Hypothesis: masking and padding are within noise of each other and of the unmasked control.
**Mask**: it is about free and does not require owning the allocation. Do not restructure a call site to pad.

Masking idioms:

- **Build masks with `jax.lax.broadcasted_iota` at the rank you need.** Reshaping a 1-D bool to
  `(N, 1)` is an i1 shape cast Mosaic may not lower.
- **Mask values are a large finite negative** (a fraction of f32 max), never `-inf`; `-inf` gives NaN
  under softmax rescaling, and the headroom keeps the value from overflowing when added to a logit.
  Apply before the exponential.
- **Index and iota dtype is int32 or int16.** An int8 compare may be refused (`Target does not
  support this comparison`); verify with `mask_index_dtype_match`.
- The boundary-tile idiom (mask the store with a select rather than narrow the block) is in
  [`partitioning.md`](partitioning.md).

## Dtype conversion is bandwidth, not arithmetic

`dtype_conversion_cost` runs f32→f32, int32→f32, bf16→f32, f32→bf16, bf16→bf16 and int8→bf16 on a
large blocked array. int32→f32 (an equal-width conversion) is the control: hypothesis, it costs the
same as f32→f32, so **the conversion itself is free and the ratios track the bytes.**

Hypothesis: the savings are sublinear. Halving the dtype buys less than half the time, because the
per-step fixed cost does not shrink with bytes; it shrinks at larger blocks.

**Rule: keep HBM narrow, widen per tile in VMEM.** Store bf16 (or narrower) in HBM and cast slices
in the kernel; a full-tensor cast in HBM doubles traffic.

## Which dtypes work

`bitcast_and_subbyte` checks `pltpu.bitcast` f32→i32 and blocks of int4, uint4, float8_e4m3fn,
float8_e5m2 and int8. That answers **storage** only. Arithmetic is a separate question (int8
elementwise may not lower, above; `integer_ops_int32`), and so is matmul support: check
`dot_dtype_matrix` for what an in-kernel `jnp.dot` accepts and `native_matmul_dtype_matrix` for what
multiplies without a cast. The chip's reported native matmul pairs may omit pairs that work; treat
the list as a lower bound. Speed per dtype is `mxu_peak_by_dtype`.

## Precision inside the kernel

- `DotAlgorithmPreset` is unavailable in Pallas TPU. Use `preferred_element_type` and
  `jax.lax.Precision`. For f32 operands: DEFAULT is one bf16 pass, HIGH a 3-pass and HIGHEST a
  6-pass bf16 algorithm, at XLA level.
- **f32 at DEFAULT is bf16-accurate**, identical to bf16 operands with an f32 accumulator
  (`f32_default_is_one_bf16_pass`). Layout consequence: **an f32 array in HBM usually buys nothing**
  for a DEFAULT matmul.
- **`Precision.HIGH` does not compile in a Pallas `dot_general`** (`Unsupported dot precision`;
  toolchain-dependent). The in-kernel ladder is DEFAULT and HIGHEST. Writing HIGH is a build
  failure, not a slow path.
- **HIGHEST is honoured in-kernel**: error drops from bf16-level to near f32 eps. Its cost is a large
  multiple of DEFAULT; the XLA-level ratio comes from `mxu_peak_by_dtype` and differs from the
  in-kernel ratio at small shapes. Quote the one matching the level you design at.
- **The same request at XLA level may not be honoured.** An f32 `einsum` at HIGHEST can return
  bf16-accurate results, so an f32 XLA reference may be the inaccurate side of a comparison. Check
  with `xla_reference_vs_f64_oracle`.
- Some upstream matmul wrappers accept a `precision` argument and discard it. Verify that a
  precision knob lands before trusting it.

---

## Arranging so the relayout never exists

Everything above prices a move you already have. This section is the half that changes a design:
arrangements under which the move is never generated. Apply it while choosing which axis each value
is produced on; afterwards it can only justify a cost that is already there. The general principle —
a mismatch you cannot resolve here is an algebra question — is in
[`cross_area_moves.md`](cross_area_moves.md) and [`algorithm.md`](algorithm.md).

### Expansion cost is set by spelling, not axis

**The rule:** an expansion (grouped-head fan-out, tiling over batch, fanning a scalar over a block)
costs according to whether it **interleaves elements**, not which axis it runs along.

- **Blocked** — broadcast into a new axis, then reshape (`x[:, None, :]` broadcast to
  `(a, n, b)` then reshape) — is near-free on lanes and on sublanes alike.
- **Interleaved** — an element-repeating `jnp.repeat(x, n, axis=...)` — is expensive on both axes.
- **The worst spelling is `jnp.repeat(x, n, axis=1)` on a rank-3 array**, nominally the sublane axis.
  A rule phrased over axes cannot see this; one phrased over spellings can. On a rank-2 block the
  same call can be cheap, so this is not a statement about `jnp.repeat` in general. If you find one
  in a kernel, rewrite it as broadcast-then-reshape and re-measure.
- **Materialising a blocked expansion can beat avoiding it.** Broadcasting inside the contraction is
  not free either. "Never materialise the expansion" is not safe advice; measure before contorting
  an algorithm.

Verify on the target with `expand_blocked_vs_interleaved` (axis versus spelling, identical bytes)
and `expand_axis_leading_vs_sublane_vs_lane`. Any rule elsewhere justified by "it avoids the lane
axis" should be re-read as a question about the permutation it generates.

### Arrangements that pay

| arrangement | probe | when it applies |
| --- | --- | --- |
| **Expand blocked, never interleaved** | `expand_blocked_vs_interleaved` | any expansion |
| **Widen a narrow contraction by concatenating siblings.** Where a head or feature dim `w` is narrower than the MXU wants, one matmul over the concatenation beats `n` over the parts | `concat_to_widen_narrow_matmul` | penalty grows roughly as `128 / w`; the largest lever in this file |
| **Concatenate the varying operands of matmuls sharing a stationary one**, issue one matmul | `weight_stationary_lhs_concat` | only once the varying operand is small (well under a tile of rows); nothing on wide operands |
| **Apply an expansion after a matmul, not before**, where the contraction permits | `expand_before_vs_after_matmul` | real, but pays less than the MAC count says; do not budget the saving at the expansion factor |

The two concatenation rows are one fact from two directions: one shrinks output width, the other the
row count, and both are flat until the operand drops below roughly half a tile, then climb steeply.
The quantity is how far under the MXU's granularity the varying dimension has fallen
([`compute_units.md`](compute_units.md), `mxu_block_rows_cost`).

They do not cover every "one matmul where there were `n`". A body issuing several full-tile matmuls,
one per head, reads clean here but can still pay for the *number of matmul ops*. Giving the head axis
a leading dim and a `dot_general` batch dim removes that cost with every shape, FLOP and byte
unchanged; that lives in [`algorithm.md`](algorithm.md) and [`compute_units.md`](compute_units.md).
On the way out, recombining a batched `[NV, C, DV]` result with `transpose(1, 0, 2).reshape(...)`
is a real sublane/untiled relayout; slice on the untiled axis and lane-concatenate instead.

### Arrangements that do not pay (hypotheses, re-check at your shape)

Each is defensible from the tiling model and each is expected to measure about 1.0 or backwards. They
are recorded so they are not proposed again without a probe.

| arrangement | hypothesis | probe |
| --- | --- | --- |
| Split/concatenate lanes only at multiples of 128 | no measurable effect; the compiler handles an unaligned lane cut | `lane_split_alignment` |
| Do elementwise work before an expansion, not after | the compiler already hoists it | `elementwise_before_vs_after_expand` |
| Match a mask/iota/index itemsize to the selected data | no measurable repack (int16 vs int32) | `mask_index_dtype_match` |
| Give a sliding window a windowed axis of extent 1 (leading-axis select instead of sublane roll) | unresolved at small tap counts; arms do not separate | `sliding_window_compact_vs_sublane` |
| Express a degenerate-axis transpose `(w,1)→(1,w)` as a masked reduction | backwards: the real permute is cheaper | `degenerate_transpose_via_masked_sum` |

The pattern: arrangements that save a *data movement* (aligned cut, hoist, narrow mask, cheaper
window) are already done by the compiler. The ones that pay **feed the MXU a wide enough operand** or
**avoid asking for an element-interleaved permutation**. The compiler handles layout bookkeeping; it
will not restructure a contraction for you.

Read the tables the other way too: where a design has a move it cannot delete, ask which row it
fails — an interleaved expansion, a contraction narrower than the MXU wants — because the fix is one
spelling or one concatenation upstream, not a cheaper mechanism at the point of cost.

---

## Deciding

1. **Make blocks big.** Size to the bandwidth knee and step model in [`pipelining.md`](pipelining.md),
   not to the native tile.
2. **Check legality:** trailing two block dims divisible by 8 and 128 or equal to the array's;
   per-dtype sublane minimums via `block_sublane_legality_by_dtype`. Use a rank-3 reshape for a small
   trailing extent; keep per-row DMA destinations on a leading axis of a rank-≥3 scratch.
3. **Reduce over the natural axis.**
4. **Mask the ragged tail with `broadcasted_iota`**; do not pad the array. Finite mask values, int32 or
   int16 indices.
5. **Store narrow in HBM, widen in VMEM.** Expect less than the full byte saving.
6. **Hand transposes to `dot_general`** by choosing dimension numbers from the layout you have. Treat
   an LHS transpose and any non-operand move as a real cost.
7. **Price rolls by tile alignment and expansions by spelling,** never by axis. Rewrite
   `jnp.repeat` on a rank-3 block as broadcast-then-reshape.
8. **Widen narrow matmuls by concatenation** when the varying dim is well under a tile.
9. **Pick in-kernel precision from two rungs**, DEFAULT or HIGHEST. f32 in HBM at DEFAULT buys
   nothing. Check any f32 XLA reference against an f64 oracle.
10. **A strided block is many small transfers**; a layout change that suits compute can move you to
    the expensive end of the step-cost curve ([`pipelining.md`](pipelining.md)).

## Gaps (open questions with probes)

- **Does int8 or int4 sublane tiling bind?** `sublane_tiling_by_dtype` cannot test int8 (arithmetic
  does not lower) and does not sweep int4; a cast-to-i32 variant would measure the cast too.
- **Does the LHS transpose cost what upstream assumes?** `transpose_cost` covers only the RHS. Probe
  to build: an `lhs.T @ rhs` arm against a pre-transposed LHS.
- **Non-operand relayout in bf16.** Relayout costs in
  [`../tpu_hardware_facts.md`](../tpu_hardware_facts.md) are f32 evidence; bf16 is unmeasured.
- **How does the interleave penalty scale with lane count?** Run `expand_blocked_vs_interleaved` at
  more than one lane width with the spelling held fixed.
- **Does "materialising a blocked expansion beats avoiding it" survive a DMA-bound loop or a wider
  expansion factor?** Measured on one contraction with constant `index_map`s.
- **The sliding-window arrangement is unresolved, not refuted.** Larger tap counts, or taps crossing
  a tile boundary: `sliding_window_compact_vs_sublane` at those shapes.
- **The concatenation rows were swept in `n` at one shape each**, bf16 only; the `128 / w` reading is
  an interpolation, not a fitted law. Sweep contraction depth and dtype with
  `concat_to_widen_narrow_matmul` and `weight_stationary_lhs_concat`.
- **Does the masked-reduction substitute ever win on a narrower dtype?** Tested on f32 only:
  `degenerate_transpose_via_masked_sum`.
- **Sub-byte arithmetic, `pltpu.bitcast` on sub-byte types, and layout changes across a kernel
  boundary** are unmeasured.
- **In-kernel HIGHEST cost versus shape** has no registered probe.
