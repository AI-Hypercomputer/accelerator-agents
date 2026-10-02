# TPU Optimization Blueprint — generalizable playbook

**Generic** — applies across model families. Distilled from hundreds of experiments across multiple model families (Gemma 4, Llama 3 8B, Llama 4, DeepSeek V3, and Flux), this is the **procedural complement** to [`model-optimization-index.md`](model-optimization-index.md):

- **Index** answers _"for topic X (CE / sharding / VMEM / flags / …), what's the mechanism and what's been refuted?"_
- **Blueprint** answers _"in what order do I apply optimizations, what compounds, and what trap will bite me?"_

Both are read together at hypothesis time by [`/formulate-hypothesis`](../.claude/skills/formulate-hypothesis/SKILL.md).

---

## TL;DR — the ladder

A single ladder, cheap → expensive, applied **after** capturing a clean profile (see [Phase 0](#phase-0--substrate-sanity-mandatory-pre-flight)). Steady-state numbers from the data: top of the ladder buys ~+5–15 % MFU at any given variant; bottom of the ladder ([Phase 3](#phase-3--kernel--structural-code-work-ml-effort-biggest-unitary-wins)) regularly buys 2–30×. Always [verify the change took effect](concepts/silent-noop-audit.md) before drawing perf conclusions.

| Phase | Effort | Typical lift | Examples (with citations) |
|---|---|---|---|
| **0. Substrate sanity** | trivial | unlocks the rest | profiler-off; megacore on; libtpu compat; fp32 master; loss trajectory check |
| **1. Cheap memory + precision** | S | ~+0.5–17 % per lever | AC=selective (+16.8 % on a 3B MoE); adamw_bf16_state; fp32 master + bf16 compute |
| **2. Known-good high-yield flag bundle** | S | ~+8–11 % | `xla_tpu_scoped_vmem_limit_kib` (**near-mandatory on v6e+**, sweep 65k–131k per generation, starting at 81920 — +8.8 % on a 3B MoE at v5p-16); MaxText `HOST_OFFLOAD_FLAGS + DATA_PARALLEL_OVERLAP + LAYOUT_FOR_ALL_REDUCE_SCATTER + CF_FOR_ALL_GATHER`; v5p `enhanced_launch_barrier=false` |
| **3. Kernel + structural code work** | M–L | **2–30×** | Tokamax CE port (+19.4 % on a 3B MoE); drop Pallas at low batch (4.4× in a dispatch-bound regime); gate+up MoE fusion (frontier mover); scalar-gather → bulk index_select (30× MFU); compile-region firing (+19 %) |
| **4. Batch + sequence amortization** | S–M | ~+0.9 pp / doubling | lbs-doubling at FSDP ≥ 8 until HBM-cliff; total batch multiple of 1024 |
| **5. Cross-variant + cross-generation transfer** | M | replicates the stack | port + re-tune block sizes per gen; re-verify per [Knobs translation matrix](SCHEMA.md#model-wikimodels-architecture-lane-md) |
| **6. Exploratory flag sweep (last resort)** | S | typically < 1 pp; often 0 | Only **after** Phase 1–5 saturate. Single-flag flips that aren't in the known-good bundle are inert at the current op-point in ≥ 8 sampled experiments — see [`inert-flag-flip-trap`](concepts/inert-flag-flip-trap.md) |

Skip the ladder only when the profile points squarely at one topic (see [model-optimization-index TLDR](model-optimization-index.md#tldr--fast-decision-aid)). Otherwise, work top-to-bottom — the higher rungs unlock or remove blocking constraints for the lower ones.

**The "flag sweep last" rule, restated**: empirically, the highest-yield flag work is applying *known-good bundles* — that's Phase 2. Discovering new flag wins by sweeping the libtpu / XLA namespace is the lowest-yield activity in the ledger (most sweeps are flat or refuted) and belongs at the bottom. The single exception is when a profile signature concretely implicates a specific flag (e.g., `dynamic-slice` dominant → re-check `xla_tpu_use_continuation_fusion`); in that case skip to it directly.

---

## Phase 0 — substrate sanity (mandatory pre-flight)

These cost ~zero, but a missing one invalidates every downstream measurement. **Always run before formulating a hypothesis.**

| Check | Why | Reference |
|---|---|---|
| Profiler **off** during measurement | Profiler can floor MFU (7 % artifact in one v5p-16 series — real value was 19.68 %) | [`observations/jax-v5p16-profiler-2x67-overhead.md`](observations/jax-v5p16-profiler-2x67-overhead.md) |
| Megacore **on** (v5p only) | Toggling megacore off doubled apparent MFU at bit-identical compute (a bogus 2× lift) | [`concepts/mfu.md`](concepts/mfu.md) |
| Skip steps 1–3 in MFU calc | JIT / XLA compile distorts; profile at steady state | [`observations/jax-v174-hlo-invariant-to-measurement.md`](observations/jax-v174-hlo-invariant-to-measurement.md) |
| libtpu version registers your flags | MaxText DeepSeek bundle was rejected as unknown on April-13-2026 libtpu image; ~5–8 experiments wasted on inert flags | `feedback_libtpu_version_xla_flag_compatibility` (memory) |
| fp32 master weights (non-LoRA) | bf16 master destroys convergence; this is a Phase-0 invariant, not a Phase-1 lever | [`model-optimization-index.md`](model-optimization-index.md#dimension-alignment--dtype) |
| Loss trajectory ≥ 10 steps within bf16 noise | Frozen-loss / max-entropy attractor masquerading as MFU win — 5 torchax experiments hit `ln(262000) = 12.4761` with measurable MFU but zero learning | [`concepts/numerics-trajectory-validation.md`](concepts/numerics-trajectory-validation.md) (new) |
| The change actually took effect | tyro-plumbing silent skips; `enable_cpu_offload` overrides AC mode; torch.compile silent no-op unlocked a hidden 19 % | [`concepts/silent-noop-audit.md`](concepts/silent-noop-audit.md) (new) |
| FSDP divisibility | `lbs % FSDP_degree == 0` — `lbs+1` cliff burned ~3 experiments per model family | [Phase 4](#phase-4--batch--sequence-amortization-s-m-effort) |

**The cheapest Phase-0 lift on record**: dropping the profiler from the measurement loop — same code, +0.5 pp MFU, sometimes much more.

---

## Phase 1 — cheap memory + precision (S effort, broadly applicable)

These are the levers that compound the cleanest across model families. Apply them all unless one is already on by default.

| Lever | When | Typical lift | Cross-model evidence |
|---|---|---|---|
| **AC = selective** | HBM headroom > 20 % | +5–17 % at small variant; OOMs at large variant | +16.8 % on a 3B MoE, Gemma 4 torchax exp5 (+8.7 % step time), Llama 3 jax frontier; refuted at a 24B MoE on v5p-32 (HBM-tight) — [`concepts/rematerialization.md`](concepts/rematerialization.md) |
| **adamw_bf16_state** | always (frees 8 → 4 bytes/param) | enables larger batch; perf-neutral itself | compounds with AC on a 3B MoE; first MFU at a 24B MoE on v5p-32 (torchax) |
| **fp32 master + bf16 compute (AMP)** | always (non-LoRA) | semantically required; perf-neutral | Universal: DeepSeek V3 v005d/v007b, Llama 4 v003, Gemma 4 exp52, Llama 3 8B JAX |
| **bf16 CE / drop spurious fp32 upcast** | when LM-head writes f32 logits | +5–19 % | Gemma 4 torchax exp12 (+5.8 %), tokamax CE port (+19.4 % on a 3B MoE) — see [Phase 3](#phase-3--kernel--structural-code-work-ml-effort-biggest-unitary-wins) |
| **Sequence × batch shape sweep** | once Phase 1 stable | +0.9 pp / doubling | See [Phase 4](#phase-4--batch--sequence-amortization-s-m-effort) |

**Cross-cite**: [Dimension alignment + dtype](model-optimization-index.md#dimension-alignment--dtype) for the underlying mechanism. The "always cast bf16 master" anti-pattern lives in that section's refuted-pattern list.

---

## Phase 2 — known-good high-yield flag bundle (S effort)

**This phase is `known-good bundles only`**. Exploratory single-flag sweeps live in [Phase 6](#phase-6--exploratory-flag-sweep-last-resort). The known-good bundles are the ones that have repeatedly cleared the multi-model bar; they are also the ones that are [refuted in isolation](#refuted-pattern--flag-flip-in-isolation) (sub-flags tested alone are flat) and supported only as a bundle.

### Universal pre-flight (every lane, every generation)

```
LIBTPU_INIT_ARGS=
  --xla_tpu_use_enhanced_launch_barrier=false    # mandatory on multi-host eager dispatch; harmless on JAX
```

### On JAX-stack (jax / torchax-thick / maxtext)

This is where the *big* flag wins live.

```
LIBTPU_INIT_ARGS=
  --xla_tpu_scoped_vmem_limit_kib=81920                       # tunable; see VMEM section below
  --xla_tpu_reduce_scatter_collective_matmul_mode=none        # V2 enum, beats default
  --xla_tpu_all_gather_collective_matmul_mode=none            # V2 enum
```

Then layer the MaxText bundles (see [`benchmarks/xla_flags_library.py`](codebases/maxtext.md)):

- `HOST_OFFLOAD_FLAGS` — host-offload friendly scheduler features (**the single biggest lever** on Llama 3 — +11.1 %).
- `DATA_PARALLEL_OVERLAP` — DP / FSDP collective overlap (in every JAX-stack frontier).
- `LAYOUT_FOR_ALL_REDUCE_SCATTER` — layout pin for RS (DeepSeek V3 v007b, Llama 4 v003).
- `CF_FOR_ALL_GATHER` + `CF_FOR_ALL_REDUCE` — continuation fusion presets.
- `DISABLE_BUNDLE_AWARE_COST_MODEL` — fixed a 3× backward regression in MoE (`<internal-bug>`).

**On `torchax-thin` (`Compile(enable=False) + call_jax`)**: the MaxText bundles are *largely inert* — most of these flags are consumed by torch.compile graph passes that don't run on thin. Set the VMEM flag and `enhanced_launch_barrier=false`; defer the rest until you switch to a graph-bearing path. See `project_torchtitan_parallelism_flags_inert_on_torchax_thin`.

### `xla_tpu_scoped_vmem_limit_kib` — near-mandatory flag; the **value** is a hyperparameter

The flag itself is one of the few that is a universal must-set on JAX-stack. The **value** is a hyperparameter, sweepable in the **65k–131k range** depending on generation and workload:

| Generation | VMEM hardware cap | Sweep range to try | Starting value | Notes |
|---|---|---|---|---|
| **v6e** (96 MB) | 98304 KiB | **65536 → 98304** | 81920 | Unlocks the larger Pallas heuristics (`tokamax v_block=2048`, splash `block_kv=2048`) that don't fit on v5p. Above 98304 driver-rejects. The biggest single flag lever on v6e+ workloads, but the optimum within the range is workload-dependent. |
| **v7x** (~128 MB inferred) | ~131072 KiB | **65536 → 131072** | 81920 | Sweep with even more headroom than v6e; verify the upper bound against the bundled libtpu before fully extending. |
| **v5p** (65472 KiB / 64 MB hard cap on libtpu 0.0.41+) | 65472 KiB | **fixed at 65472** | 81920 (silently clamped to 65472) | Anything > 65472 is silently clamped; values < 65472 measurably regress. +8.8 % MFU on a 3B MoE at v5p-16 because the *default* is below 65472. |
| **v8i** (384 MB) | ~393216 KiB | TBD | TBD | Not yet measured; will likely need a higher upper bound. |

**Sweep procedure** (Phase 2 — *not* Phase 6, despite being a value-sweep — because the lift is large and the search is narrow):

1. Start at 81920 on the first run.
2. If the workload is HBM-bound at frontier, sweep up by 16k increments (`81920 → 98304 → 114688 → 131072` on v6e+, capping at the hardware limit).
3. If the workload regresses at higher values (Gemma 4 jax exp41 saw `VMEM=131072` give −2.9 %; Llama 3 jax exp48 saw `81920` itself give −1.6 % at MoE level), step back down.
4. **The sweep is workload × generation specific**; don't carry a value over from another model without re-checking. The 81920 starting point holds across the ledger but the optimum doesn't.

Above the per-gen hardware cap is silently clamped on v5p, **hard-rejected** on v6e+ (driver error). This is the only Phase-2 value that should **always** be swept on the first port of a recipe to a new generation, and never deferred to Phase 6.

### What is **definitively refuted** (every lane)

| Lever | Status | Source |
|---|---|---|
| `xla_tpu_enable_windowed_einsum_for_*` (V1 bool) | DEPRECATED on libtpu v160+, silently rejected | [Index → FSDP / collective optimization](model-optimization-index.md#fsdp--collective-optimization) |
| `--xla_jf_spmd_threshold_for_windowed_einsum_mib` | Silently ignored on libtpu v160+ | ibid. |
| `xla_tpu_use_megacore=false` on v5p | Regresses; megacore is mandatory | [Hardware envelope](model-optimization-index.md#tpu-hardware-envelope-essential-without-lookup) |
| `enable_collective_matmul` (V1) | −14.7 % on Llama 3 jax (HARD refute) | Llama 3 8B JAX exp51 |
| `sa_use_fused_bwd_kernel=True` on multi-host v5p | Deadlock at step 1 — universal lane-wide refute | Multi-model JAX-stack survey |
| `xla_tpu_enable_aggressive_loop_fusion_layout_opt=true` | Harmful on at least one MoE workload | MoE workload survey |
| VMEM **above** per-gen hardware cap | Driver rejects on v6e+; on v5p silently clamped to 65472 KiB | `project_v5p16_vmem_cap_universal` |
| VMEM value **carried over** from another model without re-sweeping | The optimum is workload × generation specific (Gemma 4 jax exp41 `=131072` −2.9 %; Llama 3 jax exp48 `=81920` at MoE level −1.6 %); sweep 65k–131k per port | Multi-model survey |

### Universal trap: **the flag-sweep trap**

[`/formulate-hypothesis`](../.claude/skills/formulate-hypothesis/SKILL.md) already enforces the precedent check; the survey adds quantitative evidence. See [`concepts/inert-flag-flip-trap.md`](concepts/inert-flag-flip-trap.md) — and the [refuted patterns](#anti-playbook--top-5-traps-that-repeated) section below. The actionable rule: in Phase 2 apply the bundle and the VMEM flag; defer all *other* flag changes to [Phase 6](#phase-6--exploratory-flag-sweep-last-resort), after Phase 3–5 saturate.

---

## Phase 3 — kernel + structural code work (M–L effort, biggest unitary wins)

**This is where the top-10 wins live.** Eight of the top ten single-experiment lifts in the data are kernel work or structural model refactors. The mechanism is *lane-specific* — JAX-stack wins through **kernel ports + structural refactors + scan**, while eager-dispatch lanes win through **kernel removals + dispatch reduction + compile-region firing**. Don't mix the two ladders.

> **Read the [Structural rewrite pattern catalog](#structural-rewrite-pattern-catalog) section below before touching model code.** It enumerates the seven model-level rewrite patterns invoked here — op-count reduction, weight fusion + matmul folding, reshape elimination, loop→scan, sharding-aware fusion, numerical fidelity, and the documented anti-patterns. Each pattern lists the experiments that proved it.

### On JAX-stack (jax / torchax-thick / maxtext)

#### 3A.JAX — kernel **ports** (biggest single lever)

Adopt a known-good kernel from the [Pallas kernel catalog](analyses/2026-04-23-pallas-kernel-directory.md). Walk the catalog **before** writing a new one.

| Win | Mechanism | Refs |
|---|---|---|
| **Tokamax CE port** | `linear_softmax_cross_entropy_loss/pallas_mosaic_tpu_kernel.py` + canonical sum-reduction `shard_map` wrap (`in_specs=(P('fsdp', None), P('fsdp'), P(None, None))` + `psum('fsdp')` + `out_specs=P()`) | +19.4 % / +4.18 pp on a 3B MoE at v5p-16 (tokamax CE port); +1.61 pp on the 24B variant at v5p-32 after FSDP-sharding `fused_weight` — [`concepts/memory-efficient-cross-entropy.md`](concepts/memory-efficient-cross-entropy.md) |
| **Splash attention** | `sa_block_q = sa_block_kv = 1024` (cross-model default); 2048 on v6e+ where VMEM allows | Universal: Gemma 4, Llama 3 (`bkv=2048` matches MaxText), DeepSeek V3 v007b (all 8 blocks @ 2048), Llama 4 — [Index → Splash attention](model-optimization-index.md#splash-attention) |
| **Custom MoE kernel (Pallas SwiGLU expert FFN)** | When `expert_count ≥ 16` and per-track matmuls are large | a 3B MoE (jax); DeepSeek V3 `megablox=True + sparse_matmul=True` (v005d) — [`concepts/segment-matmul-pallas-kernel.md`](concepts/segment-matmul-pallas-kernel.md) |

#### 3C.JAX — structural model refactors

| Refactor | Mechanism | Refs |
|---|---|---|
| **Gate + up MoE FFN fusion** | Single `gate_up_weight [T,E,H,2*ffn]` weight | +0.76 pp on a 3B MoE (jax); independent of kernel choice |
| **FSDP-sharding rule for kernel-coupled tensors** | Add explicit FSDP rule for fused weight tensors (e.g., `fused_weight`) so kernel ports work at larger variants | unblocked tokamax CE on a 24B MoE at v5p-32 |

#### 3D.JAX — compilation strategy

| Strategy | Mechanism | Refs |
|---|---|---|
| **Scan over layers** | `jax.lax.scan` / `torchprime.scan_layers` / `torchax.ScannedModule` / Flax `nn.scan` — compile O(N) → O(1); supports the [DeepSeek V3 `scan_layers=True + custom_sort_vjp`](experiments/deepseek_v3_autoresearch_optimization/maxtext/) frontier | ~18 % MFU on a 24B MoE (scan over layers) — [`concepts/scan-over-layers.md`](concepts/scan-over-layers.md) |
| **JAX compilation cache (3-env-var pattern)** | `JAX_COMPILATION_CACHE_DIR + JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=1 + JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0` | Gemma 4 jax exp45 (**6.67× faster total wall** on second run) — `feedback_jax_compilation_cache` |

#### 3E.JAX — torchax-thin lever (Compile(enable=False) + call_jax)

| Refactor | Mechanism | Refs |
|---|---|---|
| **Single `call_jax` per layer** | Wrap `Layer.forward` in one `interop.call_jax` → one PT-trace per layer instead of per-op | up to 22× MFU (0.23 % → 5.00 %) on a 3B MoE torchax lane (single call_jax per layer) |
| **fp32 cast around freeze-sensitive kernels** | Cast q/k/v to fp32 before `pallas.ops.tpu.flash_attention` / `splash_attention` | a 3B MoE torchax lane (first valid seq=8192 frontier; +0.47 pp) — `project_bf16_freeze_heads_per_chip` |

### On eager-dispatch lanes

Eager-dispatch PyTorch lanes have a **fundamentally different cost model** from the JAX stack. Eager dispatch is expensive; XLA's compile-time fusion only fires inside compile regions. The winning moves are structural rather than flag-based: (a) make the compile region actually fire (verify it isn't a [silent no-op](concepts/silent-noop-audit.md)); (b) **remove** Pallas kernels at low batch where dispatch tax exceeds the kernel benefit; (c) structural model refactors that shrink kernel-launch count. These levers are enumerated as abstract patterns in the [Structural rewrite pattern catalog](#structural-rewrite-pattern-catalog) below — op-count reduction (Pattern 1), drop-Pallas-at-low-batch (Pattern 1), gate+up fusion (Pattern 2), and scalar-gather → bulk index_select (Pattern 3). Up to **30× MFU** (0.18 % → 5.39 %) on a 3B MoE has come from the scalar-gather fix alone in a dispatch-bound regime.

---

## Phase 4 — batch + sequence amortization (S–M effort)

Apply **after** Phase 3 — the kernel/structural changes change the HBM budget, which changes where the lbs cliff lives.

| Rule | Why | Evidence |
|---|---|---|
| At FSDP ≥ 8, double lbs until OOM cliff | Collectives amortize over more tokens (~+0.9 pp MFU / doubling) | torchax bs-amortization curve on a 3B MoE: **+1.95 pp cumulative** over 7 doublings; +0.96 pp on the 24B variant, +0.89 pp on the 3B variant (jax) |
| Total batch multiple of 1024 | MXU efficiency; per-chip ≥ 8, prefer 128 | [Dimension alignment](model-optimization-index.md#dimension-alignment--dtype) |
| Always re-check FSDP divisibility | `lbs % FSDP != 0` is a compile error | burned twice on a 3B MoE (jax) |
| Always re-check per-chip dim alignment | Total batch can grow while per-chip drops below 128/256 cliff | Multi-model |
| `lbs+1` past HBM cliff fails *hard* | Often GBs short, not MBs — no "tune the kv block" rescue. `sa_block_kv` is a throughput knob, not a fit knob | a 24B MoE has ~30 MB HBM headroom — *any* static cost shift flips fit |

**Sequence sweep** is a separate axis. At fixed per-shard tokens (e.g., 8192) the (bs, seq) curve is monotone in bs/decreasing in seq through the heads-per-chip × seq freeze threshold (per `project_bf16_freeze_heads_per_chip`).

---

## Phase 5 — cross-variant + cross-generation transfer

Once you have a stack that works on one row of the [variant matrix](SCHEMA.md#model-wikimodels-architecture-lane-md), it's tempting to clone it to the next row. **This step has its own trap surface.**

| What to do | Why it's not automatic |
|---|---|
| **Update the Knobs translation matrix** in the model page | The single best place to encode "this is universal" vs "this needs re-tuning per row" |
| **Re-test [AC=selective](model-optimization-index.md#activation-checkpointing-ac) per-variant** | On a MoE, AC=selective OOMs at a 24B variant on v5p-32 despite +0.77 pp at the 3B variant on v5p-16. **AC=selective is not universal.** |
| **Re-tune Pallas block sizes per generation** | v5p VMEM = 65 MB → v6e = 96 MB → v7x ≈ 128 MB. Tokamax `v_block=2048` fits on v6e, fails on v5p |
| **Re-test splash kernel choice** | `sa_block_kv=1024` is universal default but `sa_block_kv=2048` only fits on v6e+ |
| **Verify consumer-site actually reached on the new lane** | The cross-lane transfer trap — not config-field-set; see below |
| **lane → lane is rarely 1-to-1** | Llama 3 8B: JAX 43.3 % MFU vs torchax 36.8 % (**+17 % gap, +7.8 pp**) at the same recipe. The gap is dispatch overhead at the framework level — addressable only by lane switch |

---

## Phase 6 — exploratory flag sweep (last resort)

Once Phase 0–5 have saturated, you can sweep individual XLA / libtpu flags that aren't in the [Phase 2 known-good bundle](#phase-2--known-good-high-yield-flag-bundle-s-effort). **In the ledger this phase has the lowest yield**: most flag flips outside the known-good bundle are flat or refuted at the current op-point. Spending cluster slots here before Phase 3 saturates is what makes the [flag-sweep trap](concepts/inert-flag-flip-trap.md) the most populous refuted-pattern category.

### When to actually sweep

Only enter Phase 6 when:

1. **Phase 0–5 are at their respective ceilings** for the current variant row. The model page's *Open hyps* count should be near 0 for non-flag categories.
2. **A profile signature concretely implicates a specific flag** — e.g., a dominant `dynamic-slice` bucket implicates continuation-fusion flags; a high `idle` collective-wait implicates async-collective flags. Use the [model-optimization-index TLDR table](model-optimization-index.md#tldr--fast-decision-aid).
3. **The libtpu version registers the flag** (verify against `raw/code/maxtext/benchmarks/xla_flags_library.py` annotations; see `feedback_libtpu_version_xla_flag_compatibility`).
4. **You have a falsifiable hypothesis** — "flip flag X to value Y to remove bucket Z worth N % of step." If you can't state the bucket, don't sweep.

### When *not* to sweep

- The flag is in the [definitively-refuted list](#what-is-definitively-refuted-every-lane) above (V1 collective-matmul, `megacore=false`, `enable_collective_matmul`, fused-bwd on multi-host).
- The flag is XLA-default-on for libtpu v160+ (async-collective fusion flags — flipping them is a no-op).
- You haven't captured a fresh profile that points at the bucket the flag targets.
- You're porting flags from a recipe at a different libtpu version (the MaxText DeepSeek bundle was rejected as unknown on the April-13-2026 libtpu image).

### What a Phase-6 sweep looks like (pattern)

A useful sweep is **profile-anchored, narrow, and bundle-aware**:

1. Capture a fresh profile on the current frontier — record top-3 buckets.
2. Pick the flag(s) named by the [TLDR table](model-optimization-index.md#tldr--fast-decision-aid) for that profile signature. **One flag at a time** within the sweep; bundle effects belong in Phase 2.
3. Run baseline + flag-flipped on the same cluster within a tight window (avoid drift).
4. Validate with HLO diff (changed HLO indicates the flag had a compiler effect) and step-time delta with bf16 noise band.
5. If supported, promote to the Phase-2 bundle for *this lane × this generation*. If refuted, add to the [per-model refuted-patterns layer](SCHEMA.md).

This is also the right place to do **shape-anchored** sweeps that look flag-like but are really kernel-tuning — e.g., re-tuning splash block sizes per generation, re-checking VMEM-budget headroom after a structural change. They're cheap, they target a specific bucket, and they belong after the structural work in Phase 3 has stabilized the HBM budget.

---

## Structural rewrite pattern catalog

A catalog of **structural model-code rewrites** that produced measurable TPU performance wins in the autoresearch ledger. These are model-level patterns (changes to `model.py` and friends) — not kernel-library work, not flag flips, not training-loop infrastructure. They are the [Phase 3 levers](#phase-3--kernel--structural-code-work-ml-effort-biggest-unitary-wins) that compose with kernel ports / removals.

Six pattern categories plus a documented anti-pattern list, each linked to the experiments that proved it and the [model-optimization-index](model-optimization-index.md) topic for the underlying mechanism. Read this catalog before opening `model.py`.

### Pattern 1 — op-count reduction (shrink kernel-launch count)

Eager-dispatch lanes are dispatch-bound; the JAX lane has compile cost ∝ HLO module count. Both lanes win from refactors that turn N small ops into 1 batched op.

| Pattern | What you change | Wins | Mechanism / when |
|---|---|---|---|
| **Single trace boundary per layer (torchax)** | Wrap `Layer.forward()` in one `interop.call_jax` instead of letting per-op dispatch fall through PT trace | up to **22× MFU** (0.23 % → 5.00 %) on a 3B MoE torchax lane | Each PT op → JAX op → trace edge is expensive. Collapsing the layer into one tracing window removes per-op overhead at every call. |
| **vmap over tracks (JAX)** | Fold a track dimension into the matmul `M` dim so one batched `ragged_dot` replaces a Python `for t in tracks:` loop of single matmuls | frontier mover on a 3B MoE in a dispatch-bound regime | Replaces N small matmuls with 1 large one; XLA can then schedule + fuse across the whole batch. |
| **Per-track MoE dispatch refactor** | `moe_dispatch_combine` batched-over-T → per-track inner loop with cleaner shapes | +0.18 pp / +9 % on a 3B MoE (eager-dispatch lane) | Counter-intuitive: a *narrower* inner loop with shape-stable dispatch was cheaper than the wider batched version that launched many tiny ops. |
| **Drop Pallas at low-batch dispatch-bound regimes** | Remove `segment_matmul` Pallas kernel → masked dense `matmul` fallback (XLA fuses, no kernel-launch tax) | **4.4× MFU** (2.21 % → 9.74 %) on a 3B MoE at low batch; chunked-loss variant +0.49 pp | The mirror image of kernel-porting. When dispatch overhead × layer-call count exceeds the per-call kernel benefit, the dense fallback wins. See [`cross-lane-transfer-trap`](concepts/cross-lane-transfer-trap.md). |

**Diagnostic**: when the profile shows `custom-call` or `data formatting` > 15 % with many small calls and small per-call timings, the bottleneck is dispatch count, not per-call cost. Op-count reduction is the right lever class.

**Reference**: [`segment-matmul-pallas-kernel`](concepts/segment-matmul-pallas-kernel.md).

### Pattern 2 — weight fusion + matmul folding

Two adjacent matmuls that share an input can almost always be folded into one. XLA does some of this automatically; doing it in source lets you (a) wire one `segment_matmul` instead of two, (b) shrink the optimizer state by one parameter group, (c) eliminate the cross-matmul reshape entirely.

| Pattern | What you change | Wins | Mechanism / when |
|---|---|---|---|
| **Gate + up MoE FFN fusion** | Separate `gate_weight [T, E, H, ffn]` + `up_weight [T, E, H, ffn]` → single `gate_up_weight [T, E, H, 2*ffn]`; one `segment_matmul`; `chunk(2)` to split the output | frontier 10.42 % MFU at lbs=4 seq=8192 on v5p-8 for a 3B MoE (4.7× session lift over a 2.21 % start); +0.76 pp on a 3B MoE (jax). **Independent of kernel choice** — works with both masked-fallback and segment_matmul. | One matmul + one chunk beats two matmuls + a sum/concat at the SwiGLU consumer. Halves the gate/up roundtrip through HBM. |
| **QKV fusion** | When Q/K/V share the input, materialize one combined `qkv_weight` and `split` the output | Universal — most modern model.py files do this already; check it's actually fused in your fork | Same logic as gate+up: one matmul + cheap split beats three matmuls. |
| **FSDP-sharding rule for the fused tensor** | After fusing gate+up, the new weight tensor needs its own FSDP sharding rule (otherwise the kernel port that consumes it OOMs at scale) | unblocked tokamax CE on a 24B MoE at v5p-32 — without the rule, fused_weight is replicated, eats 30 MB of headroom on v5p-32 which is razor-thin | Whenever you fuse a weight, **audit the sharding spec generator** — a fresh tensor with no rule defaults to replicated, which is silently wrong at scale. |

**Trap**: the fused init *must* match the unfused init at the RNG level if you want bit-identical loss across the refactor — see [Pattern 6 — numerical fidelity](#pattern-6--numerical-fidelity-during-refactors).

**Reference**: [`gated-linear-unit`](concepts/gated-linear-unit.md), [`fsdp`](concepts/fsdp.md), [Index → MoE expert dispatch](model-optimization-index.md#moe-expert-dispatch).

### Pattern 3 — reshape / data-formatting elimination

The `data formatting` xprof bucket is layout bridges — copies, transposes, broadcasts that XLA inserted to reconcile mismatched layouts at fusion boundaries. When it's > 5 % of step time, there's a reshape pair somewhere that you can eliminate at the model level.

| Pattern | What you change | Wins | Mechanism / when |
|---|---|---|---|
| **Scalar-gather → bulk `index_select`** | A dropless dispatcher's `take_along_dim` → `index_select`; `torch.gather` → flat `index_select` | **30× MFU** (0.18 % → 5.39 %) on a 3B MoE; a later variant — 5.5× per-step / 11× wall | `take_along_dim` lowered through scalar-gather emission for each routed token (one HLO call per element). Bulk index-select issues one HLO `gather` with row-shaped slices and a clean data-movement pattern. See [`observations/tamm-dropless-dispatcher-take-along-dim-routes-through-scalar-gather`](observations/tamm-dropless-dispatcher-take-along-dim-routes-through-scalar-gather.md). |
| **Eliminate sequence-axis transpose pairs** | When two adjacent ops want different `(batch, seq, hidden)` orderings, find the upstream layout pin and propagate it through, instead of inserting a transpose | Multi-model — splash kernel wants SEQ_MINOR; the rest of the stack often wants SEQ_MAJOR. Drop the transpose by pinning the upstream weight layout. | Each transpose materializes the tensor in HBM. With seq=8192 hidden=4096 bf16 the tensor is 64 MB — paying for that twice per layer per step is large. |
| **Drop spurious `fp32` upcast in LM-head / loss path** | If the LM-head writes `f32` logits before the bf16-tiled CE kernel, the f32 logits transient is huge and the cast on read is wasted | Gemma 4 torchax exp12 +5.8 %; Llama 3 8B torchax chunked_xla CE; tokamax CE port +19.4 % on a 3B MoE | The `[N, V]` logits in bf16 at V=128 k cost 256 KB/token; in fp32 they cost 512 KB/token. CE kernel reads them once per chunk. Keeping them bf16 halves HBM I/O on the dominant bucket. |
| **Layout-pin at the kernel input boundary** | Source-level layout constraint on splash QKV inputs to skip XLA's `LayoutAssignment` insertion of a copy | Open hypothesis under JAX explicit sharding + the newer `Format` API. Refuted under GSPMD-mode propagation (the enum doesn't affect HLO layout). | See [`jax-experimental-layout`](concepts/jax-experimental-layout.md), [`sharding-in-types`](concepts/sharding-in-types.md). |

**Diagnostic**: when `bitcast` / `copy` / `broadcast` appears at the top of HLO Op Stats, or when `data formatting` is > 5 % of step in xprof Overview, walk the HLO neighborhood of the largest copy and find which two ops disagree on layout. Often a single source-level reshape (or its removal) cuts both copies.

**Reference**: [`decode-profile-signature`](concepts/decode-profile-signature.md), [`hlo-dumping-and-diffing`](concepts/hlo-dumping-and-diffing.md), [Index → HBM transient memory + I/O](model-optimization-index.md#hbm-transient-memory--io).

### Pattern 4 — loop / unroll → scan and fold

Python `for` loops over identical layer modules generate O(N) HLO modules at JIT time. Replacing them with `scan` collapses compile to O(1) and lets XLA fuse-along the layer boundary.

| Pattern | What you change | Wins | Mechanism / when |
|---|---|---|---|
| **Scan over identical layers (JAX-stack)** | Python `for layer in self.layers:` → `jax.lax.scan` / `torchprime.scan_layers` / `torchax.ScannedModule` / Flax `nn.scan` | ~18 % MFU on a 24B MoE (unblocks a compile-OOM); DeepSeek V3 v007b — `scan_layers=True + custom_sort_vjp` is part of the 27 % MFU stack; canonical pattern on every JAX-stack model with > 24 identical layers | Compile time drops O(N) → O(1). HLO module count typically shrinks ~85 % (e.g., 30 modules vs 204–206 baseline on a 24B MoE). XLA can also fuse forward + checkpoint + backward across the scan body. |
| **Sequence-parallel `shard_as` propagation through scan backward** | When sequence parallelism is on, the scan body's backward needs explicit `shard_as` annotations or it OOMs reconstructing the per-layer activation residual | Required for the DeepSeek V3 and Llama 4 stacks | Scan + 2-D sharding has a known backward OOM trap if you don't propagate sharding into the carry. |
| **Coalesce per-token loops via bulk ops** | `for tok in batch: gather(W, tok)` → `index_select(W, all_tokens)` (same family as the scalar-gather fix above) | 30× on a 3B MoE; fat-collective coalescing on another MoE | Same principle as scan but at the inner loop. |

**On eager-dispatch lanes specifically**: a native scan-over-layers is **not yet a working production lever** there — the Python-loop backward breaks fusion and an autograd-preserving wrap can un-shard FSDP DTensor params, regressing throughput at production shape. Use scan only on the JAX-stack lanes today; on eager-dispatch lanes, rely on compile-region firing + the structural refactors in this catalog for compile-time wins.

**Reference**: [`scan-over-layers`](concepts/scan-over-layers.md), [`multi-shard-sequence-parallel-correction`](concepts/multi-shard-sequence-parallel-correction.md).

### Pattern 5 — sharding-aware structural patterns

Some refactors are correct in eager mode but break under FSDP / SPMD propagation. The rewrite either has to *add* a sharding rule, or be written in a form that the partitioner can handle.

| Pattern | What you change | Wins | Mechanism / when |
|---|---|---|---|
| **`shard_map` wrap around Pallas kernels** | Mosaic kernels under GSPMD raise `NotImplementedError` on auto-partition. Wrap with explicit `in_specs` / `out_specs` / `check_rep=False` + `jax.lax.psum` on reductions. The canonical sum-reduction pattern is `in_specs=(P('fsdp', None), P('fsdp'), P(None, None)) + psum('fsdp') + out_specs=P()`. | tokamax CE port +19.4 % on a 3B MoE; universal for every Pallas-port experiment | [`pallas-mosaic-gspmd-requires-shard-map`](observations/pallas-mosaic-gspmd-requires-shard-map.md) is the canonical observation. |
| **FSDP rule for a freshly-fused weight** | Adding a fused tensor without an FSDP rule replicates it silently. At memory-tight variants (a 24B MoE on v5p-32 has ~30 MB headroom) this OOMs the port. | unblocked tokamax CE on a 24B MoE at v5p-32 | Whenever you fuse weights (gate+up, QKV) audit the sharding spec generator; new tensors default to replicated. |
| **Dedicated TP axis for kernel-internal collectives** | When a `shard_map`'s inner `all_gather` shares an axis with the outer FSDP `all_gather`, multi-host nests the two collectives on the same axis and deadlocks. Use a separate `'tp'` axis. | Universal — applies to every vocab-parallel / TP-inside-kernel pattern | See [`cross-lane-transfer-trap`](concepts/cross-lane-transfer-trap.md). |

**Reference**: [`fsdp`](concepts/fsdp.md), [`pallas-kernel`](concepts/pallas-kernel.md), [Index → Sharding strategy](model-optimization-index.md#sharding-strategy).

### Pattern 6 — numerical fidelity during refactors

Many of the rewrites above pass step-1 bit-identical but **drift over the trajectory** because they shift RNG consumption, reduction order, or attention-kernel precision. The refactor must include a numerics check or it ships as `invalid`.

| Pattern | What you change | When required |
|---|---|---|
| **fp32 cast around freeze-sensitive kernels** | Cast `q`, `k`, `v` to fp32 before `pallas.ops.tpu.flash_attention` / `splash_attention` when heads/chip × seq exceeds the multi-block freeze threshold | first valid seq=8192 frontier on a 3B MoE torchax lane (14.49 → 14.96 % MFU). Without the cast, the loss is stuck at `ln(vocab_size)` — the max-entropy attractor. See [`numerics-trajectory-validation`](concepts/numerics-trajectory-validation.md). |
| **Match RNG consumption across fused init** | A fused `gate_up_weight` initialized in one RNG draw produces different values from two unfused draws even at the same seed | a fused gate-up on a 3B MoE showed +5.2 % step-7 loss drift relative to baseline. Either match the unfused draw pattern (init each half separately, then concatenate) or treat the divergence as expected and re-baseline. |
| **Watch reduction-order shifts** | FSDP-allgather + DTensor cast + offload_dot pipelines can change the order of summations across shards. Within bf16 noise across 10+ steps is the gate. | offload_dot on a 3B MoE — +0.21 step-7 loss drift (5 % relative). Required ≥ 10-step trajectory validation before declaring a refactor `supported`. |
| **Reduction-convention check at kernel boundaries** | When dropping a Pallas loss kernel for a chunked-XLA fallback, the chunk-vs-full reduction conventions can disagree | a chunked-XLA CE fallback diverged the loss 8000× at step 1 because it summed differently than the Pallas kernel it replaced. |

**The gate**: every supported verdict on a structural refactor requires ≥ 10-step trajectory within bf16 noise of baseline (not just step-1 bit-identical). For RNG-shifting or reduction-order-shifting refactors, extend to a 50-step parity. See [`numerics-trajectory-validation`](concepts/numerics-trajectory-validation.md).

### Pattern 7 — anti-patterns (model rewrites the ledger rejected)

Documented for citation in hypothesis rationales. **Don't propose these without new evidence.**

| Anti-pattern | Why it fails | Refs |
|---|---|---|
| **Replace XLA-fused norm/matmul with Pallas custom-call** | Custom-call boundary breaks the existing `loop_fusion` across the norm → matmul seam. Adds dispatch overhead. Net negative when the original op is < 10 % of step time. | Gemma 4 torchax exp33 −8.1 %; Gemma 4 jax exp47 levanter CE −5.61 %; Llama 3 jax exp53 tokamax mosaic_tpu CE −4.4 %; fused RMSNorm+QKV on a 3B MoE −0.21 pp. **Rule: only port a Pallas kernel when the target op is > 10 % of step time AND the boundary cost can amortize.** |
| **Pallas kernel drop-in without re-tuning per generation** | v5p heuristics (VMEM 64 MB) and v6e (96 MB) differ; `v_block=2048` fits v6e but compile-hangs on v5p | Observed on a 3B MoE (jax). Always re-tune block sizes when porting across TPU generations. |
| **TP=2 on small-E narrow-attention MoE / small chip count** | TP collectives dominate at 2 chips/axis; FSDP-only beats it | Gemma 4 torchax exp32 −2.4×; Llama 3 torchax exp25 −14 %. TP is only worth it after FSDP capacity is exhausted. |
| **Pure DP / ZeRO-1 (replicated weights, no FSDP shard)** | GSPMD's replicated-matmul codegen inflates conv-fusion bytes | Multi-model refute. Use FSDP. |
| **Loss kernel without matching reduction convention** | Drop-in produces step-1 loss divergence | a chunked-XLA CE fallback. Always match reduction convention. |
| **AC wrapping that breaks the compile region's type-name match** | The layer-iteration helper looks for a specific class name; an AC wrapper changes it → the compile region silently no-ops (cost a hidden ~19 % MFU on a 3B MoE) | Verify the compile region is actually firing (see [`silent-noop-audit`](concepts/silent-noop-audit.md)). |

### How to choose which pattern to apply

Walk the patterns in this order, anchored to the profile:

1. **Phase-0 audit first** — the rewrite premise has to be measured on the current frontier, not a memory of how things looked five experiments ago. Capture a fresh profile. Verify no [silent no-ops](concepts/silent-noop-audit.md).
2. **Match the dominant xprof bucket to a pattern category** using [`decode-profile-signature`](concepts/decode-profile-signature.md):
   - `custom-call` / `data formatting` > 15 % and many small calls → Pattern 1 op-count reduction
   - Two adjacent matmuls sharing an input → Pattern 2 weight fusion
   - `bitcast` / `copy` / `broadcast` at top of Op Stats, or `data formatting` > 5 % → Pattern 3 reshape elimination
   - High HLO module count or compile time > 30 s → Pattern 4 scan / loop fold
   - OOM at scale despite reasonable static budget → Pattern 5 sharding-aware (often missing FSDP rule)
3. **Always pair the rewrite with the Pattern 6 numerics gate** — at least 10-step trajectory, monotone-decreasing loss, no max-entropy attractor.
4. **Cross-check against Pattern 7 anti-patterns** before committing.

The largest cumulative wins in the ledger came from **stacking patterns** (e.g., on a 3B MoE eager-dispatch lane: gate+up fusion (Pattern 2) + drop Pallas segment_matmul (Pattern 1) + scalar-gather fix (Pattern 3) compounding to 10.42 % MFU from a 2.21 % start — a 4.7× session lift through three model-rewriting categories applied in sequence). See [`compounding-optimizations`](concepts/compounding-optimizations.md).

---

## Universal levers — confirmed multi-model

These are the patterns that **transfer cleanly across model families** when the lane matches.

1. **MaxText XLA flag bundle on JAX stack** — single biggest cross-model lever; bundle-only (not sub-flag) — Phase 2.
2. **fp32 master + bf16 compute (AMP)** — universal, semantics-required, perf-neutral — Phase 0/1.
3. **Splash attention `block_q = block_kv ∈ {1024, 2048}` with SEQ_MINOR + fused_bwd (single-host only)** — universal default — Phase 3A.
4. **Selective remat (`checkpoint_dots_with_no_batch_dims`)** — beats full or none when HBM headroom allows — Phase 1.
5. **bf16 CE + drop spurious fp32 upcast** — Gemma 4, Llama 3, and a 3B MoE all showed it — Phase 1 → 3A.
6. **adamw_bf16_state** — universal optimizer-state shrink — Phase 1.
7. **JAX compilation cache (3-env-var pattern)** — wall-time win, not MFU — Phase 3D.

See [`concepts/compounding-optimizations.md`](concepts/compounding-optimizations.md) (new) for the compounding analysis: applied together these walked a 3B MoE at v5p-16 on the JAX lane from ~18 % → ~26 % MFU over ~30 experiments, with no single move exceeding 4 pp.

---

## Anti-playbook — top 5 traps that repeated

These are the failure modes that show up across model families. Each has a concept page for citation in hypothesis rationale.

### 1. Inert-flag-at-this-op-point — [`concepts/inert-flag-flip-trap.md`](concepts/inert-flag-flip-trap.md)

Most populous failure category — at least 8 sampled experiments wasted. XLA / libtpu flag gets accepted but produces bit-identical HLO or zero MFU delta.

**Fix**: before launching, verify (a) the kernel/path consuming the flag is actually invoked, (b) the regime matches the flag's lever (compute-bound vs HBM-bound vs dispatch-bound), (c) the libtpu version registers the flag. Examples: `splash sa_block_kv=1024` inert when the invocation path skips the CLI knob; `xla_tpu_scoped_vmem_limit_kib=81920` inert on torchax in a dispatch-bound regime; all collective-scheduling flags inert at a compute-bound frontier.

### 2. Cross-lane single-axis port — [`concepts/cross-lane-transfer-trap.md`](concepts/cross-lane-transfer-trap.md)

JAX-lane wins assumed to transfer to other PyTorch lanes without recognizing co-dependencies.

**Fix**: any cross-lane port must verify the **consumer site** is reached, not just the **config field** is set. Examples: on a 3B MoE torchax lane, AC=selective was silently overridden by `enable_cpu_offload`; `dots_saveable` lowered to the same HLO as `nothing_saveable` in a dispatch-bound regime; `splash 1024` was bit-identical across two v5p-16 clusters. **Pallas segment_matmul is a +19 % win on JAX and a 4.4× loss on an eager-dispatch lane** — same kernel family, opposite verdict, decided by framework dispatch overhead.

### 3. OOM walls treated as flag-tunable

8+ sampled experiments. Treating an OOM as "tune the flag and retry" rather than "this is a static cost that needs a code-level change."

**Fix**: classify OOM source — static (FSDP weights + optimizer state + activations) vs runtime kernel buffer — before proposing a fix. Examples: `sa_block_kv` does not free HBM (it's a VMEM/throughput knob); `lbs+1` dies in CE VJP because tile sizes aren't shape-aware; AC=selective is 3B-specific. **On a 24B MoE at v5p-32 the HBM headroom is ~30 MB — any static cost shift flips fit.**

### 4. Numerics drift masquerading as MFU win — [`concepts/numerics-trajectory-validation.md`](concepts/numerics-trajectory-validation.md)

Frozen loss / step-1 bit-identical is **insufficient**. The trajectory over ≥10 steps must stay within bf16 noise.

**Fix**: mandatory ≥10-step monotone-loss check + step-N (N≥10) loss-vs-baseline check. Examples: 5 torchax experiments hit the max-entropy uniform CE attractor (loss = 12.4761 = ln(262000)) with measurable MFU; offload_dot on a 3B MoE → +0.21 step-7 loss drift; a fused gate-up RNG-init mismatch (which also caught a real refactor bug); a scan + mask refactor was a 4.84× MFU win at step 1, but its 50-step trajectory diverged 2.7 %.

### 5. Splash + multi-host + secondary knob → deadlock

Each of (`fused_bwd kernel`, scan-FSDP wrap, FSDP2 with a newer model variant, splash + tracing flags + seq=8192) triggers a class of hang invariant across the secondary knob.

**Fix**: maintain a per-lane "do not touch on multi-host with splash" list. Examples: `sa_use_fused_bwd_kernel=True` step-1 deadlock on JAX (universal lane-wide); a splash compile-finalize hang on an eager-dispatch lane independent of block-size, fused-bwd-toggle, and profiler-toggle (4 reproductions); FSDP2 on a newer model variant OOMs at step 2 on an eager-dispatch lane (cross-variant); SimpleFSDP hangs step 2 on the same variant.

### Honorable mentions

- **Pallas drop-in for ops < 10 % of step time** — custom-call boundary breaks XLA's tight loop_fusion across norm/matmul seam. Refuted on Gemma 4 (exp33 −8.1 %, exp47 −5.61 %), Llama 3 (exp53 tokamax mosaic_tpu CE −4.4 %). **Rule**: only port a Pallas kernel when the target op is > 10 % of step time AND its boundary cost can amortize.
- **TP=2 at small chip counts** — Gemma 4 torchax exp32 (−2.4×), Llama 3 torchax exp25 (−14 %). **Rule**: TP is only worth it after FSDP capacity is exhausted. See [Index → Tensor parallelism](model-optimization-index.md#tensor-parallelism-tp).
- **Image / launch infrastructure failures classed as refuted** — 8+ experiments wasted because the change didn't actually take effect. Always verify; see [`concepts/silent-noop-audit.md`](concepts/silent-noop-audit.md).

---

## The compounding principle

The walk from baseline to frontier is almost never a single 4×; it's a chain of 10–30 small wins applied in [Phase 1–4](#tldr--the-ladder) order. See [`concepts/compounding-optimizations.md`](concepts/compounding-optimizations.md) for the multiplicative analysis.

**Canonical examples**:

| Walk | Span | Mechanisms applied | Net |
|---|---|---|---|
| 3B MoE, JAX lane, v5p-16 | 18.7 % → 26.61 % MFU | AC=selective → CE kernel → adamw_bf16 → VMEM=81920 → splash 1024 → Collective Matmul V2 → tokamax CE → lbs=16 | +7.9 pp / +42 % over ~30 experiments |
| 3B MoE, eager-dispatch lane, v5p-16 | 0.18 % → 14.24 % MFU | scalar-gather fix → drop Pallas segment_matmul → gate+up fusion → compile-region firing | **~80× MFU** over ~7 supported wins |
| Llama 3 8B jax v6e-8 | baseline → 43.3 % MFU | MaxText XLA bundle (+11.1 %) → tokamax chunked_xla CE → splash bkv=2048 → SparseCore offload of RS+AG+AR → nothing_saveable remat → bs=4 | **beats MaxText reference** (+8.9 % per chip) |
| DeepSeek V3 v5p-512 | 7.04 % → 27.04 % MFU | FSDP=256 (no EP) → `scan_layers=True` → `custom_sort_vjp` → `HOST_OFFLOAD_FLAGS` (11 flags) → `LAYOUT_FOR_ALL_REDUCE_SCATTER` → `DATA_PARALLEL_OVERLAP` | 3.84× lift |

**Lesson**: don't wait for a 4× hypothesis — the data shows 4× lifts when they exist are usually [silent-no-op unlocks](concepts/silent-noop-audit.md) or [structural model refactors](#structural-rewrite-pattern-catalog). Routine work is +0.5–2 pp at a time, compounded.

---

## Lane-specific caveats

Several execution lanes show up in the data with different ground rules.

### JAX (torchtitan-JAX, MaxText, or native)

- Highest MFU ceiling per the data (Llama 3 8B: 43.3 % beats MaxText's 44.6 % per-chip).
- All Phase-2 flag bundles apply.
- Pallas kernel **ports** are the biggest unitary lever (Phase 3A).
- `splash sa_use_fused_bwd_kernel=True` deadlocks on multi-host v5p — universal refute.
- Compilation cache (Phase 3D) buys 6.67× wall on second run with zero MFU change.

### torchax (call_jax bridge from PyTorch)

- Phase-1 levers transfer cleanly. Phase-2 flag bundles **mostly inert** in dispatch-bound regime.
- `Compile(enable=False) + call_jax` graph-construction means flags consumed by torch.compile graph passes (AC mode, `fsdp_reshard`, async-TP, splash-internal knobs) are inert. See `project_torchtitan_parallelism_flags_inert_on_torchax_thin`.
- The biggest structural lever is **wrap a whole layer in one `call_jax`** (up to 22× MFU on a 3B MoE).
- bf16 freeze threshold is **heads/chip × seq** — multi-block kernels freeze at large per-shard heads × large seq. fp32 cast around splash/flash kernels is the bypass.
- TP=2 unconditionally regresses at small chip counts (Gemma 4, Llama 3 both −14% / −2.4×).

### MaxText (reference stack — `raw/code/maxtext/`)

- Treat as **upper-bound reference**, not a porting target — its canonical recipes (DeepSeek V3 v007b 27 % MFU, Llama 4 Maverick v003 35.59 % MFU) are achievable.
- Mine `benchmarks/xla_flags_library.py` for flag bundles — every JAX-stack frontier in the data uses these bundles.
- Mine `MaxText/configs/` for recipes (e.g., `llama4_maverick_dropless_v5p_256`, DeepSeek `scan_layers=True + custom_sort_vjp`).

---

## Where to look next — open research questions

These are gaps the data flagged as worth investing in next.

1. **Zig-Zag (load-balanced causal) ring attention** — only confirmed-absent kernel in the 2026-04-23 Pallas-kernel-source-survey. Writing it (vs porting) is the only justifying case for fresh kernel work per [pallas kernel patterns](model-optimization-index.md#pallas-kernel-patterns-lane-wide).
2. **TP=2 code path for a very large MoE at 256 chips** — the structural wall at the largest variant needs TP=2 or a larger slice (256 → 512 chips).
3. **MaxText Llama 4 Maverick's EP-via-collective-permute-decomposition pattern** — 217K CP-start/done pairs overlap conv-fusion. Port target for any large MoE (`xla_tpu_decompose_all_to_all_for_overlap=true` + `xla_tpu_enable_async_collective_permute=true`).
4. **Sharding-in-types + JAX explicit-sharding** — open hypothesis to re-test source-level layout pins around splash (refuted under GSPMD; deterministic propagation under explicit sharding may flip it).
5. **v7x / v8 generational re-tuning** — Pallas block sizes need re-tuning per generation; v5p heuristics don't transfer.
6. **Per-model `refuted-patterns.md` layer** — does not yet exist for any model; the refutation classifications here would seed one cleanly.

---

## How to use this blueprint

In the autoresearch loop:

1. At the start of an optimization session, [capture a clean profile](concepts/profile-capture.md) (Phase 0).
2. Walk the [TL;DR ladder](#tldr--the-ladder) top-to-bottom, skipping a phase only if it's already at frontier on this variant.
3. For each candidate hypothesis, before launching: cross-check against the [anti-playbook](#anti-playbook--top-5-traps-that-repeated) and the per-topic [refuted-pattern principles in the index](model-optimization-index.md#optimization-topic-index).
4. After the verdict, update the model page's [Knobs translation matrix](SCHEMA.md) row for the technique (universal vs needs re-tuning vs lane-specific).
5. If the lesson generalizes, promote it to this blueprint as a new universal lever or new trap (see [maintenance](#how-to-maintain-this-file)).

[`/formulate-hypothesis`](../.claude/skills/formulate-hypothesis/SKILL.md) walks both this file and [`model-optimization-index.md`](model-optimization-index.md) at proposal time; the index gives the topic depth, this blueprint gives the procedural ordering.

---

## How to maintain this file

- **Don't add per-experiment data here** — that goes in the model page and the per-model refuted-patterns layer (when one exists).
- **Do add a Phase-1–5 lever** when a new optimization shows up in supported experiments across ≥ 2 model families with the same mechanism.
- **Do add a new trap** when a refuted pattern repeats across ≥ 2 model families with the same failure mode.
- **Do update the compounding-walk table** when a session brings a variant matrix row across a milestone (e.g., a new 2× session lift).
- **Don't duplicate concept depth** — link to [`model-optimization-index.md`](model-optimization-index.md) for mechanism, link to `concepts/*.md` for definitions.
- **Promote shared findings from per-model wikis to here** when the principle (not the experiment ID) generalizes.

---

## See also

- [`model-optimization-index.md`](model-optimization-index.md) — topic-keyed depth pages + per-topic refuted-pattern principles
- [`SCHEMA.md`](../SCHEMA.md) — page types, wiki structure, the variant matrix + Knobs translation matrix definitions
- [`wiki/experiments/program.md`](experiments/program.md) — procedural loop spec
- [`wiki/index.md`](index.md) — everything-index
- [`.claude/skills/formulate-hypothesis/SKILL.md`](../.claude/skills/formulate-hypothesis/SKILL.md) — consumer of this file at hypothesis time
- [`.claude/skills/edit-model-code/SKILL.md`](../.claude/skills/edit-model-code/SKILL.md) — consumer at code-edit time

## Sources

Distilled from the following corpora:

- Internal MoE autoresearch — hundreds of experiments across the `jax/` and `torchax/` lanes (a large fraction supported, the rest refuted/inconclusive/invalid)
- DeepSeek V3 autoresearch — 8 experiments on maxtext lane
- Gemma 4 autoresearch — 19 jax + 33 torchax + 2 maxtext
- Llama 3 8B autoresearch — 62 jax + 12 torchax + 2 maxtext
- Llama 4 autoresearch — 5 maxtext (Scout + Maverick)
- Flux autoresearch — 6 maxdiffusion
- Public-wiki Gemma 4 + Llama 3 mirrors

Reference codebases (linked in [`model-optimization-index.md`](model-optimization-index.md#canonical-reference-stacks)): MaxText (`benchmarks/xla_flags_library.py`), tokamax (CE / splash kernels), tpu-recipes, scaling-book, xprof-mcp playbook, JAX, torchtitan, ultrascale-playbook.
