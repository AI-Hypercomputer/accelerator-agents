# Model-Optimization Index — TPU performance auto-optimization

**Generic** — applies across model families. This is the lookup table the [`/formulate-hypothesis`](../.claude/skills/formulate-hypothesis/SKILL.md) skill walks before proposing a hypothesis. It contains (a) reference data essential at hypothesis time that you should not have to look up, and (b) a topic-keyed pointer set into the depth pages.

**Per-model state lives elsewhere** — this file is intentionally model-agnostic:
- Per-model frontier history → `wiki/models/<model>-<lane>.md` variant matrix
- Per-model refuted experiments (with v-IDs) → `wiki/experiments/<model>_autoresearch_optimization/refuted-patterns.md` (per-model layer; create when first model adopts it)
- Cross-model lessons that *generalize* → promoted here as a "principle" in the relevant topic

**How `/formulate-hypothesis` uses this file**:
1. Read the **TLDR** (always) — pick the topic(s) implicated by the recent profile + frontier signal.
2. Read **only the relevant topic sections** below — not the whole file. The skill should pick 1–3 topics and skip the rest.
3. From each topic, follow the cited concept + observation page links via the Explore subagent.
4. Apply the **Generic refuted-pattern principles** from the topic before proposing.
5. Pull in the per-model refuted-patterns layer (if it exists) for the specific model + variant in scope.

---

## TLDR — fast decision aid

If you have a recent xprof bucket-attribution for the current frontier, use this table to pick the topic. The "first-line hypothesis class" is a starting point, not a recommendation — always validate against the topic's generic principles before committing.

| Dominant profile signature | Likely binding constraint | Topic section | First-line hypothesis class |
|---|---|---|---|
| `convolution fusion` > 60% | MXU-bound; FLOPs are already well-fused | [Batch-size amortization](#batch-size-lbs-amortization), [Hardware envelope](#hardware-envelope-v5p-vs-v6e-vs-v7x) | lbs-doubling, larger mesh, scale-up |
| Matmul FLOP utilization < 40% | Dimension misalignment (per-chip shape not aligned to MXU tile) | [Dimension alignment + dtype](#dimension-alignment--dtype) | per-chip hidden multiple of 128/256; reshard to a different mesh shape |
| `custom-call` > 15% (CE / softmax) | Logits or softmax kernel cost | [CE / softmax loss](#ce--softmax-loss) | tokamax CE port, vocab-parallel |
| `custom-call` > 15% (splash) | Attention kernel cost | [Splash attention](#splash-attention) | block-size tuning, fused/non-fused bwd selection |
| `custom fusion` dominated by 1 op | Hidden matmul or scatter (often MoE dispatch) | [MoE expert dispatch](#moe-expert-dispatch) | segment_matmul, fused-gate-up, expert-count check |
| `data formatting` > 5% | Layout bridges (often QKV↔attention, MoE reshapes) | [Pallas kernel patterns](#pallas-kernel-patterns-lane-wide), [Splash attention](#splash-attention) | shard_map wrap, layout pin, fused kernel |
| Large `bitcast` / `copy` / `broadcast` at top of HLO Op Stats | Materialized broadcast that couldn't fuse with consumer | [HBM transient memory + I/O](#hbm-transient-memory--io) | refactor fusion boundary; `get_hlo_neighborhood` to isolate |
| `all-reduce` / `all-gather` > 15% | Collective bandwidth / no overlap | [FSDP / collective optimization](#fsdp--collective-optimization), [Sharding strategy](#sharding-strategy) | collective-matmul mode, lbs-doubling, async fusion, mesh axis re-placement |
| TP-over-DCN (across pods) | Collectives across DCN, not ICI | [Sharding strategy](#sharding-strategy) | move TP to ICI; cap TP ≤ 8 within ICI island |
| `loop fusion` > 20% | Elementwise (norms, casts, residuals) | [HBM transient memory + I/O](#hbm-transient-memory--io), [Dimension alignment + dtype](#dimension-alignment--dtype) | dtype tuning (bf16 weights), kernel fusion, RMSNorm consolidation |
| `idle` > 10% (host-device gap) | Input pipeline or host-side bound | [Profile-driven attribution](#profile-driven-attribution) | prefetch, parallel reads (grain / tf.data); `unset LD_PRELOAD` for DLRM-style embeddings |
| `idle` > 10% (collective waits) | Collective bound | [FSDP / collective optimization](#fsdp--collective-optimization) | async collectives, reduce TP, switch DP→FSDP |
| Step time growing over steps | HBM thrash / memory pressure | [VMEM / scratch memory](#vmem--scratch-memory), [HBM transient memory + I/O](#hbm-transient-memory--io) | adamw_bf16 state, AC=full, rematerialization |
| `dynamic-slice` / `update-slice` dominant | KV cache decode (inference-only) | [Profile-driven attribution](#profile-driven-attribution) | larger decode batch, splash attention, StaticCache + jit |
| Many recompiles per step | Dynamic shapes (data-dependent shape paths) | [torch.compile + scan-over-layers](#torchcompile--scan-over-layers) | static shapes, padding, pre-compile all shapes |
| Rematerialization time > 15% in HLO Op Stats | Over-remat | [Activation checkpointing (AC)](#activation-checkpointing-ac) | reduce remat policy if HBM headroom > 20% |
| Looking for a kernel by function (attention, MoE, GLU, SSM, CE, norm, matmul, collective) | n/a — discovery, not diagnosis | [Pallas kernel patterns](#pallas-kernel-patterns-lane-wide) → **Kernel catalog** | walk the cross-repo catalog before writing a new kernel |
| Long-context training (≥16k seq) where dense attention is HBM-bound | seq×seq attention transient | [Splash attention](#splash-attention) | NSA (DeepSeek native sparse, 64k @ FA2-speed); Henry Mko v5e TPU port; or splash with custom mask |
| No profile yet | Unknown | — | **capture a profile first** — never hypothesize against a profile-less frontier |

**Profile attribution principle**: optimizing any bucket < 5% of step is below measurement noise. Use the profile to find top-3 buckets and attack those. See [Profile-driven attribution](#profile-driven-attribution).

---

## TPU hardware envelope (essential without lookup)

### v5p

| | Value |
|---|---|
| bf16 peak | **459 TF/chip** |
| HBM capacity | **95.74 GB/chip** (program-usable ~95 GB) |
| HBM bandwidth | **2.76 TB/s/chip** |
| VMEM cap | **65472 KiB (64 MB) per Tensor Core** — `scoped_vmem_limit_kib` silently clamped to this on libtpu 0.0.41+ |
| TCs per chip | 2 (megacore mode = 1 logical) — `xla_tpu_use_megacore=false` regresses v5p; megacore is mandatory |
| MXU shape | 128×128 bf16 (4 MXUs/TC) |
| ICI | 3D torus, ~100 GB/s per link bidi |
| Critical arithmetic intensity (bf16) | ~180 FLOPs/byte |
| Bytes-per-flop | ~6.0 (HBM-rich; favors low arithmetic-intensity workloads like MoE, sparse) |

**XPK naming caveat**: `v5p-N` counts **TensorCores**, not chips. v5p-8 = 4 chips. v5p-16 = 8 chips. v5p-32 = 16 chips.

### v6e

| | Value |
|---|---|
| bf16 peak | **918 TF/chip** (2× v5p) |
| HBM capacity | **31 GB/chip** (1/3 v5p) |
| HBM bandwidth | **1.64 TB/s/chip** |
| VMEM cap | **96 MB per Tensor Core** (~50% larger than v5p) — unlocks larger Pallas block sizes (e.g., tokamax `v_block=2048` fits) |
| TCs per chip | 1 |
| MXU shape | 256×256 bf16 |
| ICI | 2D torus, lower-radix than v5p |
| Critical arithmetic intensity (bf16) | ~570 FLOPs/byte |
| Bytes-per-flop | ~1.8 (FLOPs-rich; favors high arithmetic-intensity dense workloads) |

**XPK naming convention**: `v6e-N` counts **chips**. v6e-4 = 4 chips. v6e-8 = 8 chips.

### v7x (Ironwood)

| | Value |
|---|---|
| bf16 peak | **2307 TF/chip** (~2.5× v6e, ~5× v5p) |
| FP8 peak | **4614 TF/chip** |
| HBM capacity | **192 GB/chip** (2 × 96 GB across two chiplets — 2× v5p, 6× v6e) |
| HBM bandwidth | **~7.37 TB/s/chip** (~2.7× v5p, ~4.5× v6e) |
| VMEM cap | **~128 MB per Tensor Core** (inferred from v8i's 384 MB = 3× v7x per the deep-dive — v7x docs don't quote VMEM directly; **verify before tuning Pallas block sizes**) |
| TCs per chip | 2 (one per chiplet) |
| Chiplets per chip | 2 (connected via D2D interface ~6× faster than a single ICI link) |
| SparseCores per chip | 4 (2 per chiplet) |
| Chips per VM | 4 (machine type `tpu7x-standard-4t`) |
| MXU shape | 256×256 bf16 |
| ICI | 3D torus, 200 GB/s **per-axis bidirectional** (~2× v5p per link) |
| Max pod | 9216 chips, 3D torus |
| Critical arithmetic intensity (bf16) | ~313 FLOPs/byte (intermediate) |
| Bytes-per-flop | ~3.2 (between v5p's 6.0 and v6e's 1.8 — middle ground) |

**XPK naming convention** (inferred, **unverified**): likely `v7x-N` counts **TensorCores** (i.e., N = 2 × chips, matching v5p / v4). The official doc does not state the XPK convention. The GKE node label is `cloud.google.com/gke-tpu-accelerator=tpu7x` (no `-slice` suffix). See [`concepts/tpu-hardware-generations.md`](concepts/tpu-hardware-generations.md) for the verification plan and the cross-gen table.

**SparseCore offload defaults to `true` on v7x** for `xla_tpu_enable_sparse_core_collective_offload_*` flags — see [libtpu / XLA flags](#libtpu--xla-flags). On generations where SC offload is off by default, this is an explicit toggle.

### v8 (preview — TPU 8t / TPU 8i, GA later in 2026)

Announced 2026-04 at Cloud Next; **no official per-generation docs exist yet** as of 2026-05-21. Numbers below are from the announcement + technical deep-dive blog; treat as preview-quality, verify against `cloud.google.com/tpu/docs/v8t` or `/v8i` once published. **Peak compute is quoted at FP4**; bf16-equivalent is inferred (FP4 throughput / 4) and approximate.

| | TPU 8t (training) | TPU 8i (inference) |
|---|---|---|
| FP4 peak | **12.6 PFLOPs/chip** | **10.1 PFLOPs/chip** |
| bf16-equivalent (inferred ÷4) | ~3150 TF/chip | ~2525 TF/chip |
| HBM capacity | **216 GB HBM3e** | **288 GB HBM3e** |
| HBM bandwidth | 6.53 TB/s | 8.60 TB/s |
| On-chip SRAM (VMEM-equivalent) | not quoted | **384 MB** (3× v7x) |
| TCs per chip | not stated (presume 2) | 2 (explicit) |
| Network topology | 3D torus | **Boardfly** (new, not 3D torus) |
| Max scale | 9600 chips (superpod) | 1152 chips |
| New mechanism | Axion Arm host CPUs | Collectives Acceleration Engine (CAE) — up to 5× on-chip latency reduction |
| Bytes-per-flop (bf16-equiv) | ~2.07 | ~3.41 |
| Price-perf vs v7x | 2.8× better training | 80% better inference |
| GA | later in 2026 | later in 2026 |

**Caveat**: FP4 throughput is not a one-to-one substitute for bf16 work. If your model trains in bf16, the bf16-equivalent column is the relevant one. XPK naming + machine type for v8 are not announced; do not plan iteration ladders on v8 until docs publish.

### Roofline implications (use these before guessing)

- **bytes/flop ratio drives lane selection**: at bf16 with ~1 byte/flop arithmetic intensity (small batches, MoE dispatch, vocab projection), v6e is HBM-bound and v5p is MXU-bound — same code, very different MFU. v7x sits between (~3.2 byte/flop) and is the most balanced of the three for mixed workloads.
- **Critical arithmetic intensity** (the bf16 batch-size cliff where a `[B, D] × [D, F]` matmul becomes compute-bound): v5p ~180, v7x ~313, v6e ~570. Below the threshold the chip waits on HBM regardless of FLOPs available.
- **Collective BW**: ICI all-reduce ~150 GB/s for full mesh on v5p; ~80 GB/s on v6e; v7x has 200 GB/s per-axis bidi (highest per-link of the three generations). Collective fraction of step typically grows on v6e and shrinks on v7x.
- **VMEM = the throughput unlock for newer gens** that v5p doesn't have. Pallas kernels with larger heuristic block sizes (CE, splash) realize wins on v6e (96 MB) / v7x (~128 MB inferred) that hit the wall on v5p (64 MB). v8i's 384 MB will move the ceiling further.
- **HBM headroom**: v6e is the most memory-tight (31 GB); v5p is comfortable (95 GB); v7x is generous (192 GB) and unlocks recipes that OOM on the older gens. Memory-tight recipes (high lbs, AC=selective) work on v5p/v7x that won't fit on v6e.
- **v7x SparseCore defaults**: SC collective offload is on by default on v7x — `xla_tpu_enable_sparse_core_collective_offload_*` flags. Embedding-heavy workloads ride this without explicit opt-in.
- **v8t FP4-first**: training-optimized at FP4 (not bf16). If your code trains in bf16, the bf16-equivalent throughput is what you actually get — see the v8 table caveat.

### Topology cheat sheet

| TPU + topology | Chips | Slice shape | Common use |
|---|---|---|---|
| v5p-8 (XPK) | 4 | 2×2×1 | Smoke / preflight |
| v5p-16 (XPK) | 8 | 2×2×2 | 3B iterate |
| v5p-32 (XPK) | 16 | 4×2×2 | 24B iterate |
| v5p-128 (XPK) | 64 | 4×4×4 | 150B preflight |
| v5p-256 (XPK) | 128 | 4×4×8 | 150B production |
| v6e-4 | 4 | 2×2 | Smoke |
| v6e-8 | 8 | 2×4 | 3B iterate |
| v6e-16 | 16 | 4×4 | 24B / multi-track |
| v6e-32 | 32 | 4×8 | 24B / 150B preflight |
| v6e-256 | 256 | 16×16 | 150B+ production |
| v7x-8 (XPK, **inferred**) | 4 | 2×2×1 | Smoke / preflight |
| v7x-16 (XPK, **inferred**) | 8 | 2×2×2 | 3B iterate |
| v7x-32 (XPK, **inferred**) | 16 | 2×2×4 | 24B iterate |
| v7x-128 (XPK, **inferred**) | 64 | 4×4×4 | 150B preflight (one "cube" = one rack) |
| v7x-256 (XPK, **inferred**) | 128 | 4×4×8 | 150B production |
| v7x-1024 (XPK, **inferred**) | 512 | 8×8×8 | frontier-scale |
| v8t / v8i | — | — | XPK naming + machine type not announced as of 2026-05-21 |

v7x XPK naming is inferred from the 2-TC-per-chip pattern (matches v5p / v4); **verify on next v7x XPK submission** — see [`concepts/tpu-hardware-generations.md`](concepts/tpu-hardware-generations.md) for the verification recipe.

`.env/gke-*.md` is the source of truth per cluster (region, project, kubectl context).

---

## Checkpointing + persistence

**Mechanism**:
- [codebases/orbax.md](codebases/orbax.md) — Google's JAX checkpoint + persistence library. Async save reduces wall-clock by **40% at 300M / 85% at 8B / 97% at 340B**. Multi-host distributed; **reshard-on-load** (load under different sharding than save); multi-tier (HBM → host pinned → GCS, composing with `jax.experimental.layout.Format`). Adopted by Flax / Pax / T5X / MaxText / AXLearn / Keras 3.14+.

**Generic principles**:
- **At 340B / 97% time reduction, async save is structurally cheap** — checkpoint-per-step becomes viable for true fault-tolerance.
- **Reshard-on-load** matters when experiments change FSDP degree mid-run (elastic training).
- **Pinned-host memory kind** via `jax.experimental.layout.Format` is the multi-tier substrate.

**TPU-stack notes**:
- Orbax is JAX-side; **PT/XLA SPMD has separate checkpointing** (`torch.distributed.checkpoint` + `SPMDSavePlanner/SPMDLoadPlanner`). They don't interoperate.

## Canonical reference stacks

These are the shared substrates a hypothesis is most often grounded in. Cite them in proposal rationale rather than re-deriving.

| Stack | Page | Role |
|---|---|---|
| **MaxText** | [codebases/maxtext.md](codebases/maxtext.md) | Reference TPU-tuned training stack (Google). `benchmarks/xla_flags_library.py` is the canonical curated TPU flag library (CF presets, SparseCore offload, host-offload, layout, pipelining). Configs under `MaxText/configs/`. |
| **tokamax** | [codebases/tokamax.md](codebases/tokamax.md) | Pallas-on-Mosaic-TPU kernel suite. Splash attention, linear+softmax CE loss, fused ops. Heuristics encode per-generation block sizes (v5p `v=512`, v6e `v=2048`). |
| **tpu-recipes** | [codebases/tpu-recipes.md](codebases/tpu-recipes.md) | Google's curated tuning recipes per model family. |
| **scaling-book** | [codebases/scaling-book.md](codebases/scaling-book.md) | "How to Scale Your Model" (DeepMind / JAX-ML 2025). Canonical theory: roofline, sharding taxonomy, collective costs, FSDP↔TP↔PP trade-offs. 11 chapters individually ingested as sources. Note: 2025-02-04 — v6e and v7x not covered. |
| **xprof-mcp** | [codebases/xprof-mcp.md](codebases/xprof-mcp.md) | Diagnostic playbook: roofline, per-gen critical intensity, profile-signature → cause decision tree, tool invocation order. See [`sources/2026-xprof-mcp-tpu-optimization.md`](sources/2026-xprof-mcp-tpu-optimization.md) — the highest-density "what to check and why" doc in the wiki. |
| **ultrascale-playbook** | [sources/2025-ultrascale-playbook.md](sources/2025-ultrascale-playbook.md) | Hugging Face multi-thousand-GPU training playbook. FSDP / TP / PP / sequence-parallelism systematically with formulas. GPU-primary but the parallelism logic transfers. |
| **JAX** | [codebases/jax.md](codebases/jax.md) | JAX runtime + compiler (XLA). The default lane for high-performance TPU work. |
| **torchtitan** | [codebases/torchtitan.md](codebases/torchtitan.md) | PyTorch reference training stack used by this autoresearch project's experiments. |
| **pallas-forge** | [codebases/pallas-forge.md](codebases/pallas-forge.md) | Pallas kernel infrastructure (autotuning, autotune-result caching). |
| **ejkernel** | [codebases/ejkernel.md](codebases/ejkernel.md) | Pallas kernel collection. |
| **ringattention** | [codebases/ringattention.md](codebases/ringattention.md) | Ring attention reference implementation. |

**Per-stack canonical flag library**: MaxText's [`benchmarks/xla_flags_library.py`](codebases/maxtext.md) is the single best reference for cross-stack TPU flag bundles (CF, SparseCore offload, host-offload, layout-for-AR-and-RS, pipelining, data-parallel overlap). Re-reading the library before formulating a flag-sweep hypothesis avoids reinventing presets that are already named, version-gated, and refuted-where-applicable.

---

## Optimization topic index

Each topic section: (1) mechanism — concept page citations; (2) generic principles — model-agnostic lessons; (3) generic refuted-pattern principles — the high-level traps. Per-experiment refutations live in the per-model refuted-patterns layer.

The topics are ordered roughly **foundation → diagnosis → kernels/ops → infrastructure → flags** — though the skill should still pick by TLDR table, not by reading top-to-bottom.

### Sharding strategy

The algebra under FSDP, TP, EP, CP, PP — and the choice of mesh axis placement that decides which collectives sit on ICI vs DCN.

**Mechanism — theory**:
- [sources/2025-scaling-book-ch3-sharding.md](sources/2025-scaling-book-ch3-sharding.md) — **the canonical theory**: 4-case sharded-matmul taxonomy + collective-cost formulas.
- [sources/2025-scaling-book-ch5-training.md](sources/2025-scaling-book-ch5-training.md) — Parallelize-for-training: FSDP / TP / PP trade-offs.
- [sources/2025-ultrascale-playbook.md](sources/2025-ultrascale-playbook.md) — HF distributed training playbook (parallelism formulas, sequence parallelism).

**Mechanism — partitioning systems**:
- [sources/2021-gspmd-paper.md](sources/2021-gspmd-paper.md) — **GSPMD**, the foundational annotation+propagation SPMD partitioner backing pjit/jit, T5X, MaxText, Paxml, and PT/XLA SPMD. 50-62% MFU on 2048 TPUv3 cores at trillion-param scale.
- [sources/2024-partir-paper.md](sources/2024-partir-paper.md) — **PartIR**, DeepMind's schedule-based compositional partitioner. Decouples partitioning from model code; performance parity with GSPMD.
- [sources/2026-shardy-overview.md](sources/2026-shardy-overview.md) — **Shardy**, OpenXLA's modern MLIR-based successor combining GSPMD propagation + PartIR composition. Enable with `jax.config.update("jax_use_shardy_partitioner", True)`. Backend currently routes through GSPMD; MLIR-native partitioner planned.
- [sources/2023-jax-jep-14273-shard-map.md](sources/2023-jax-jep-14273-shard-map.md) — `shard_map` JEP, the canonical design doc for manual-mode per-device collectives. Rank-preserving, eager-by-default, composable with `jit`.
- [sources/2026-jax-explicit-sharding.md](sources/2026-jax-explicit-sharding.md) — **explicit sharding** ("sharding-in-types"). Deterministic propagation at the JAX type level; `jax.typeof(x)` queryable inside JIT. Errors on ambiguity rather than silently choosing.
- [sources/2026-pytorch-xla-spmd-user-guide.md](sources/2026-pytorch-xla-spmd-user-guide.md) — **PT/XLA SPMD**, GSPMD-on-PyTorch (lazy tensor + `mark_sharding`). Reference impl: `torchprime`.

**Mechanism — sequence parallelism (long context)**:
- [sources/2023-ring-attention-paper.md](sources/2023-ring-attention-paper.md) — foundational Liu et al. ICLR 2024 Ring Attention; blockwise distributed attention with KV rotation; **device-count× sequence length**.
- [sources/2025-tasp-paper.md](sources/2025-tasp-paper.md) — **Topology-aware Sequence Parallelism**; Hamiltonian decomposition; **3.58× over Ring + Zigzag-Ring**. TPU 3D-torus is exactly the topology this exploits — open hypothesis for TPU port.
- [sources/2026-mtraining-paper.md](sources/2026-mtraining-paper.md) — **Striped Ring + dynamic sparse**; replaces Zigzag (which has >2.4× worker imbalance on RoPE VS pattern); **6× at 512K**; hierarchical NVLink+InfiniBand → maps to TPU ICI+DCN.
- [sources/2024-tokenring-paper.md](sources/2024-tokenring-paper.md) — bidirectional P2P comm variant of Ring Attention.
- [codebases/yunchang-usp.md](codebases/yunchang-usp.md) — **production hybrid Ulysses + Ring** (Ulysses-deg × Ring-deg); heterogeneous-network aware; canonical reference impl.

**Mechanism — concepts**:
- [concepts/tensor-parallelism.md](concepts/tensor-parallelism.md), [concepts/fsdp.md](concepts/fsdp.md), [concepts/expert-parallelism.md](concepts/expert-parallelism.md), [concepts/context-parallelism.md](concepts/context-parallelism.md), [concepts/pipeline-parallelism.md](concepts/pipeline-parallelism.md)
- [concepts/sequence-parallelism.md](concepts/sequence-parallelism.md), [concepts/ring-attention.md](concepts/ring-attention.md)
- [concepts/all-gather.md](concepts/all-gather.md), [concepts/all-reduce.md](concepts/all-reduce.md), [concepts/reduce-scatter.md](concepts/reduce-scatter.md)
- [concepts/multi-shard-sequence-parallel-correction.md](concepts/multi-shard-sequence-parallel-correction.md)
- [concepts/in-graph-xla-collective-lowering.md](concepts/in-graph-xla-collective-lowering.md)
- [concepts/shardy.md](concepts/shardy.md), [concepts/sharding-in-types.md](concepts/sharding-in-types.md), [concepts/jax-experimental-layout.md](concepts/jax-experimental-layout.md) — new partitioner + explicit-sharding + layout-control concept stubs.

**The 4-case sharded matmul taxonomy** (from scaling-book ch3 — internalize this; it grounds every parallelism decision):

| Case | Sharding pattern | Collective required | Total cost |
|---|---|---|---|
| 1 | Neither contracting dim sharded — `A[I, J] × B[J, K]` both replicated | none | local matmul |
| 2 | One contracting dim sharded — `A[I, J_X] × B[J, K]` | **AllGather** along X before matmul | `≈ 2 · sharded_size / (BW · X)` |
| 3 | Both contracting dims sharded matchingly — `A[I, J_X] × B[J_X, K]` | matmul local blocks then **AllReduce** (or **ReduceScatter** if output sharded) | `≈ 2 · output_size / BW` |
| 4 | Conflicting non-contracting shardings — `A[I_X, J] × B[J, K_X]` | invalid — must AllGather one matrix first | — |

**Collective identity**: `AllReduce = AllGather + ReduceScatter`. Same total cost decomposed into two async phases — this is what enables overlap. FSDP replaces DP's AllReduce with `AllGather(params, fwd) + ReduceScatter(grads, bwd)` — same cost, lower memory footprint.

**Megatron-style TP** = Case 2 chained with Case 3: column-sharded first matmul (AllGather-then-matmul), row-sharded second matmul (matmul-then-ReduceScatter).

**Generic principles**:
- **Place high-BW collectives on ICI, low-BW on DCN**: FSDP AllGather / ReduceScatter on ICI (high BW); DP gradient sync on DCN (low BW); pipeline parallelism only across pod boundaries (DCN).
- **TP ≤ 8 within an ICI island**: measured 43% degradation going TP=8 (intra-node) → TP=16 (inter-node). TP across DCN degrades severely.
- **Per-chip dim rule**: when a tensor is sharded across N chips, the **per-chip** size after sharding must satisfy the MXU alignment rule (multiple of 128 on v5p/v5e, 256 on v6e+). A globally-aligned tensor (8512) sharded across 16 chips = 532/chip, neither aligned — **resharding to 8 chips = 1064/chip ≈ 1024 gives ~16% throughput improvement** in a measured case.
- **ICI roofline**: when tensors are sharded, the compute-bound crossover shifts. For TP across N chips on v5e ICI, compute-bound only when the **sharded dim > ~8755**. The batch dim is not what matters for ICI — the sharded dim is.
- **Mosaic kernels must be `shard_map`-wrapped** under GSPMD — auto-partition is `NotImplementedError`. Canonical pattern: `in_specs=(P('fsdp', None), P('fsdp'), P(None, None))` + `psum('fsdp')` + `out_specs=P()` + `check_rep=False`.
- **Sequence parallelism**: a sharded sequence axis avoids per-token replication and amortizes attention activation memory; requires explicit `shard_as` annotations to propagate through scan-over-layers backward.

**Generic refuted-pattern principles**:
- ❌ **TP across DCN** — collectives are too slow. Cap TP at ICI island size (≤ 8 typically).
- ❌ **Two collectives nested on the same mesh axis** — multi-host deadlock. Inner `shard_map`'s `all_gather` on `fsdp` axis collides with the outer FSDP `all_gather` on the same axis. Use a dedicated axis for the inner collective.
- ❌ **Globally-aligned shape with no per-chip check** — if the per-chip dim doesn't divide by 128/256, you get MXU padding regardless of how clean the global shape looks.
- ❌ **Pure DP (replicated weights, no FSDP shard)** — GSPMD replicated-matmul codegen inflates conv-fusion bytes; refutes universally on multi-chip slices.
- ❌ **Pipeline parallelism inside an ICI island** — wastes ICI on low-frequency transfers. PP is for DCN.

### Dimension alignment + dtype

The two cheapest universal optimizations — both visible in the profile as MXU under-utilization or a sea of `f32 → bf16` casts.

**Mechanism**:
- [concepts/dimension-alignment.md](concepts/dimension-alignment.md)
- [concepts/dtype-strategy.md](concepts/dtype-strategy.md)
- [concepts/mxu.md](concepts/mxu.md), [concepts/mxu-utilization.md](concepts/mxu-utilization.md)
- [concepts/int8-quantization.md](concepts/int8-quantization.md), [concepts/fp8-training.md](concepts/fp8-training.md), [concepts/fp4-training.md](concepts/fp4-training.md)
- [sources/2026-xprof-mcp-tpu-optimization.md](sources/2026-xprof-mcp-tpu-optimization.md) — canonical bf16/dim-alignment rules.
- [sources/2023-fp8-lm-paper.md](sources/2023-fp8-lm-paper.md), [sources/2024-coat-paper.md](sources/2024-coat-paper.md) — FP8 training recipes (gradients + states + collectives; +activation/optimizer compression).
- [sources/2025-fp4-training-paper.md](sources/2025-fp4-training-paper.md) — first published FP4 LLM training framework.
- [sources/2024-deepseek-v3-tech-report.md](sources/2024-deepseek-v3-tech-report.md) — production FP8 MoE recipe (671B / 14.8T tok / 2.788M H800-hours).
- [codebases/transformer-engine.md](codebases/transformer-engine.md) — NVIDIA reference impl; `DelayedScaling` recipe, MXFP8, NVFP4.
- [codebases/deepgemm.md](codebases/deepgemm.md) — clean ~300-LOC FP8/FP4 GEMM library (SM90/SM100, MoE grouped GEMM).

**MXU tile sizes** (alignment cliffs):

| Generation | MXU shape |
|---|---|
| v5e, v5p | 128×128 |
| v6e, v7x | 256×256 |

**Dimension alignment rules**:

| Dimension | Minimum | Preferred |
|---|---|---|
| Batch per core | 8 | 128 |
| Total batch | multiple of 64 | multiple of 1024 |
| Hidden / feature dim (global) | multiple of 128 | multiple of 256 (for v6e+) |
| Sharded dim (per chip) | multiple of 128 | multiple of 256 (for v6e+) |

**Training memory budget per parameter** (bf16 mixed-precision + Adam):

| Component | Bytes/param |
|---|---|
| bf16 weights | 2 |
| bf16 gradients | 2 |
| fp32 master weights | 4 |
| Adam m | 4 |
| Adam v | 4 |
| **Total** | **~16** |

**Quantization critical batch** (v5e roofline):

| Precision | Critical batch | Notes |
|---|---|---|
| bf16 | > 240 | Native TPU format |
| int8 weights, bf16 compute | > ~120 | 2× memory bandwidth benefit |
| int8 weights + int8 compute | > ~240 | FLOPs and bytes both halve |

**Generic principles**:
- **bf16 weights are mandatory**: fp32 weights trigger an implicit cast on every matmul (MXU consumes bf16 natively). Converting fp32 weights → bf16 typically gives ~17% device-time improvement. Diagnose via HLO Graph Viewer: inspect matmul weight operands, look for `f32`.
- **Sharded dims are the most-missed alignment check**: the global shape can be aligned (8512) while the per-chip shape (8512 / 16 = 532) is not. Always check the per-chip number.
- **Lower precision shifts the roofline**: int8 weights + bf16 compute halves bandwidth required → halves the critical batch. AQT for weight-only int8 is the recommended starting point.
- **fp32 master weights are non-negotiable for training stability** — they're not the perf cost; the bf16 → fp32 promotion on matmul weights is.

**Generic refuted-pattern principles**:
- ❌ **bf16 master weights for training** — destroys convergence on non-LoRA workloads. The Adam state and master-weights stay fp32 (or stay in COAT-compressed FP8 with BF16 scale); only forward/backward stays bf16.
- ❌ **Naive global-shape alignment without per-chip check** — per-chip = global/N_axis; if the result isn't a 128/256 multiple, MXU pads.
- ❌ **Quantizing weights below bf16 in a training loop** without quality validation — out of scope for this optimization wiki; semantics-changing **unless** the FP8/FP4 framework explicitly preserves training trajectory (FP8-LM, COAT, FP4-training paper all demonstrate near-lossless quality at scale; v7x+v8 generations make these the structural path forward).
- ❌ **bf16-only stack on v7x / v8 hardware** — leaves 50% (v7x FP8 vs bf16) or 75% (v8t FP4 vs bf16-equiv) of compute on the table. No general workaround yet — TPU FP8/FP4 software stack is structurally untested in this wiki.

### Profile-driven attribution

How you read a profile and turn it into a candidate hypothesis. This is the diagnostic foundation everything else builds on.

**Mechanism**:
- [sources/2026-xprof-mcp-tpu-optimization.md](sources/2026-xprof-mcp-tpu-optimization.md) — the canonical "what's slow → first fix" decision tree.
- [sources/2026-xprof-overview-page.md](sources/2026-xprof-overview-page.md)
- [sources/2026-xprof-trace-viewer.md](sources/2026-xprof-trace-viewer.md)
- [sources/2026-xprof-hlo-op-profile.md](sources/2026-xprof-hlo-op-profile.md), [sources/2026-xprof-hlo-op-stats.md](sources/2026-xprof-hlo-op-stats.md)
- [sources/2026-xprof-memory-profile.md](sources/2026-xprof-memory-profile.md), [sources/2026-xprof-memory-viewer.md](sources/2026-xprof-memory-viewer.md)
- [sources/2026-xprof-roofline-model.md](sources/2026-xprof-roofline-model.md)
- [sources/2026-xprof-graph-viewer.md](sources/2026-xprof-graph-viewer.md)
- [sources/2026-xprof-framework-op-stats.md](sources/2026-xprof-framework-op-stats.md)
- [sources/2026-xprof-custom-call-profiling.md](sources/2026-xprof-custom-call-profiling.md)
- [sources/2026-xprof-megascale-stats.md](sources/2026-xprof-megascale-stats.md), [sources/2026-xprof-megascale-viewer.md](sources/2026-xprof-megascale-viewer.md)
- [sources/2026-xprof-perf-counters.md](sources/2026-xprof-perf-counters.md)
- [sources/2026-xprof-capturing-profiles.md](sources/2026-xprof-capturing-profiles.md), [sources/2026-xprof-jax-profiling.md](sources/2026-xprof-jax-profiling.md), [sources/2026-xprof-pytorch-xla-profiling.md](sources/2026-xprof-pytorch-xla-profiling.md)
- [sources/2025-scaling-book-ch9-profiling.md](sources/2025-scaling-book-ch9-profiling.md)
- **PyTorch-side diagnostic complements** (when torchax / PT/XLA experiments need compile-events or runtime-trace analysis):
  - [codebases/tlparse.md](codebases/tlparse.md) — PT2 compile event parser (recompiles, guards, specializations). Canonical for "why is `torch.compile` recompiling?" / "why didn't it compile?"
  - [codebases/holistic-trace-analysis.md](codebases/holistic-trace-analysis.md) — Kineto runtime trace analyzer (temporal breakdown, comm-compute overlap %, idle-time attribution). GPU-side analog of xprof.
  - [sources/2025-ezyang-state-of-torch-compile.md](sources/2025-ezyang-state-of-torch-compile.md) — Edward Yang's mid-2025 status; canonical reference for PT2 maturity + gotchas.
- **JAX-side artifact / model introspection**:
  - [codebases/penzai.md](codebases/penzai.md) — DeepMind's JAX research toolkit; Treescope renders sharded JAX arrays + deeply-nested pytrees; `pz.select` for ablation + activation probing.
- **Inference architecture context**:
  - [sources/2025-vllm-anatomy.md](sources/2025-vllm-anatomy.md) — vLLM internals (PagedAttention, continuous batching, V1 scheduler, chunked prefill).
  - [sources/2025-vllm-tpu-unified-backend.md](sources/2025-vllm-tpu-unified-backend.md) — Oct 2025: vLLM TPU now routes PyTorch + JAX through unified JAX→XLA lowering via Torchax. 3.6× Llama 3.1-8B v6e-1 throughput vs prior gen.
- [concepts/trace-event-categories.md](concepts/trace-event-categories.md), [concepts/decode-profile-signature.md](concepts/decode-profile-signature.md)
- [concepts/mfu.md](concepts/mfu.md), [concepts/step-time.md](concepts/step-time.md)
- [concepts/arithmetic-intensity.md](concepts/arithmetic-intensity.md), [concepts/ici-roofline.md](concepts/ici-roofline.md), [concepts/roofline-model.md](concepts/roofline-model.md)
- [concepts/profile-capture.md](concepts/profile-capture.md), [concepts/hlo-dumping-and-diffing.md](concepts/hlo-dumping-and-diffing.md)

**Canonical XLA op_profile bucket categories**:

| Bucket | Means |
|---|---|
| `convolution fusion` | matmul / MXU work |
| `custom-call` | Pallas kernels (splash, tokamax CE, ragged_dot) |
| `custom fusion` | XLA-fused inner loop bodies; sometimes elementwise + scatter, sometimes hidden matmul work |
| `data formatting` | copies / layout bridges (often QKV↔splash, MoE dispatch reshapes) |
| `loop fusion` | elementwise (norms, casts, residuals) |
| `all-reduce` / `all-gather` / `reduce-scatter` | explicit collectives |
| `idle` | host-barrier / scheduling gaps |

**Investigation tool order** (xprof-mcp canonical sequence — follow this in order, don't skip):

1. `list_runs()` — find the run name
2. `get_overview(run)` — step time, host vs device split, top bottleneck category
3. `get_top_hlo_ops(run)` — hottest ops by time / FLOPs / memory
4. `get_op_profile(run)` — per-program breakdown (fallback when hlo_stats empty, common for inference)
5. `list_hlo_modules(run)` → `get_hlo_module_content(run, module)`
6. `get_hlo_neighborhood(run, instruction_name)` — BFS around a slow op
7. `get_memory_profile(run)` — peak HBM, fragmentation
8. `aggregate_xplane_events(run, plane_regex="/device:TPU:0")` — kernel stats; `event_regex="all-reduce"` for collective stats

**xprof tab → use case** quick map:

| xprof tab / tool | When |
|---|---|
| Overview Page | start here — step time, host-device split, bottleneck category |
| HLO Op Stats | hottest ops table — find dominant `bitcast/copy/broadcast` and big custom-calls |
| HLO Op Profile | per-program breakdown — fallback when Op Stats is empty (inference) |
| Memory Viewer | per-tensor padding %, peak HBM, fragmentation |
| Memory Profile | HBM utilization over step time — diagnoses growing memory |
| Graph Viewer | inspect matmul weight operand dtypes (look for `f32` on weights) — bf16 cast detection |
| Roofline | arithmetic intensity scatter — above-diagonal = compute-bound, far below = HBM-bound |
| Framework Op Stats | per-framework-op (PyTorch / JAX) attribution — bridge to source code |
| Megascale Stats / Viewer | multi-host collective timing, per-host variance |
| Perf Counters | low-level event counters when standard buckets are inconclusive |

**Profile-signature → cause table** (xprof-mcp decision tree):

| Profile signature | Likely cause | First fix |
|---|---|---|
| Device idle > 30%, host-device gap | Input pipeline bound | Prefetch, parallel reads (grain / tf.data) |
| Device idle > 30%, collective waits | All-reduce / all-gather bound | Async collectives, reduce TP, switch DP→FSDP |
| Matmul FLOP utilization < 40% | Dimension misalignment | Per-chip hidden dim multiple of 128 / 256 |
| Large `bitcast` / `copy` / `broadcast` at top of Op Stats | Layout mismatch or materialized broadcast | `get_hlo_neighborhood`, refactor fusion boundary |
| High rematerialization time | Over-remat | Reduce remat policy if HBM headroom > 20% |
| `dynamic-slice` / `update-slice` dominant + duty cycle < 60% | KV cache decode (memory-bandwidth-bound) | Bigger batch, splash attention, paged attention |
| Training OOM | Memory pressure | Selective AC, scan layers, FSDP optimizer-state sharding |

**Generic principles**:
- **As model gets larger, conv-fusion share grows** and non-matmul shares (CE, dispatch, attention) shrink proportionally. A 15% lever at small-model scale can be a 1.4% lever at large-model scale.
- The Pareto-relevant cost is the **top-3 buckets**. Optimize those; ignore the long tail.
- **Always skip the first 1–3 steps** (JIT / XLA compile distorts results); profile after steady state.
- **HLO dump prerequisite**: `XLA_FLAGS=--xla_dump_to=<dir> --xla_dump_hlo_as_text --xla_dump_hlo_as_proto --xla_dump_hlo_pass_re=.*` — needed for `diff_hlo_stages` and HLO-pass-level inspection.

**Generic refuted-pattern principles**:
- ❌ **Skipping profile analysis on routine experiments** — the canonical session failure. Wins below 5% are inside measurement noise.
- ❌ **Optimizing a sub-5% bucket** — wins below measurement noise. Use the profile to find top-3 buckets.
- ❌ **Trusting step-1 measurements** — JIT compile + warm-up distorts. Profile at steady state.

### CE / softmax loss

**Mechanism**:
- [concepts/memory-efficient-cross-entropy.md](concepts/memory-efficient-cross-entropy.md) — online softmax, chunked-XLA, Pallas options
- [concepts/online-softmax-with-logit-sink.md](concepts/online-softmax-with-logit-sink.md)
- [concepts/base2-softmax.md](concepts/base2-softmax.md)
- Source: `raw/code/tokamax/tokamax/_src/ops/linear_softmax_cross_entropy_loss/` — `pallas_mosaic_tpu_kernel.py` (the actual kernel). Heuristics: v5p `(b=1024, h=512, v=512)`; v6e gen 7+ `(b=1024, h=512, v=2048)`.
- [codebases/tokamax.md](codebases/tokamax.md)

**Generic principles**:
- The `[N, V]` logits transient is the dominant cost — bf16 logits at V=128k cost 256 KB/token, V=262k cost 512 KB/token.
- A well-implemented kernel (tokamax) streams logits in VMEM and never materializes the full `[N, V]` tensor in HBM. **Total step HBM traffic, not FLOPs, is the right cost model.**
- Bwd kernel typically needs larger VMEM block than fwd — the VMEM cap may make the v6e heuristic infeasible on v5p (or hung at compile).
- CE share of step is **strongly scale-dependent**: typically 15%+ at small model / small batch, dropping to ~1–2% at large model / large batch as conv-fusion grows.

**Generic refuted-pattern principles**:
- ❌ **Reasoning about CE kernel cost from FLOPs alone** — the canonical failure: chunked-XLA bwd looks dispatch-bound by FLOPs accounting, but it materializes the full `[N, V]` logits in HBM each chunk. Total HBM I/O is 12× tokamax's. Always compute total HBM traffic when comparing kernel implementations.
- ❌ **Pushing kernel `v_block_size` above the hardware VMEM cap** — silent hang or step-2 recompile error. Check [`concepts/vmem.md`](concepts/vmem.md) for the cap before tuning.
- ❌ **Vocab-parallel via `shard_map` on the SAME axis as outer FSDP collectives** — multi-host all_gather inside the shard_map deadlocks against the outer FSDP all-gathers on the same axis. Use a dedicated TP axis.

### MoE expert dispatch

**Mechanism**:
- [concepts/expert-parallelism.md](concepts/expert-parallelism.md)
- [concepts/segment-matmul-pallas-kernel.md](concepts/segment-matmul-pallas-kernel.md)
- [concepts/ragged-dot.md](concepts/ragged-dot.md)
- [concepts/pallas-on-triton-fused-gemm-activation-gemm.md](concepts/pallas-on-triton-fused-gemm-activation-gemm.md)
- [concepts/spmd-safe-fat-collectives.md](concepts/spmd-safe-fat-collectives.md)
- [concepts/deepseekmoe.md](concepts/deepseekmoe.md) — shared experts + fine-grained specialization + auxiliary-loss-free balancing.
- [observations/moe-dispatch-optimization-refuted-at-low-expert-count.md](observations/moe-dispatch-optimization-refuted-at-low-expert-count.md)
- [observations/pallas-mosaic-gspmd-requires-shard-map.md](observations/pallas-mosaic-gspmd-requires-shard-map.md)
- [sources/2024-deepseek-v3-tech-report.md](sources/2024-deepseek-v3-tech-report.md) — production MoE recipe (671B / 37B active, auxiliary-loss-free).
- [codebases/deepep.md](codebases/deepep.md) — production expert-parallel all-to-all (FP8 dispatch + BF16 combine; 90 GB/s @ 12 SMs on H800). GPU-side; TPU analog is SparseCore collective offload.
- [codebases/deepgemm.md](codebases/deepgemm.md) — FP8 grouped GEMM for MoE.
- [codebases/megablocks.md](codebases/megablocks.md) — canonical dropless MoE library (block-sparse GEMM; eliminates `capacity_factor`); MaxText megablox is the TPU port.
- [sources/2022-megablocks-paper.md](sources/2022-megablocks-paper.md) — block-sparse GEMM foundation paper.

**Generic principles**:
- MoE kernel choice (ragged_dot, segment_matmul, Pallas SwiGLU, megablox GMM) is **strongly sensitive to expert count**. A kernel that pays off at E≥16 can regress at E=4 because per-call dispatch overhead × layer-call count dominates.
- Mosaic kernels under GSPMD must be wrapped in `shard_map` — auto-partitioning is not supported (`NotImplementedError`). Canonical pattern in the cited observation.
- Gate+up weight fusion (single `[T, E, H, 2*ffn]` weight + 1 segment_matmul + chunk(2)) is a clean optimization independent of kernel choice — saves a matmul-pair on every layer call.
- MoE all-to-all over DCN is the **classic anti-pattern** — place expert-parallel axis on ICI; if it must cross DCN, batch the dispatch.

**Generic refuted-pattern principles**:
- ❌ **Adopting a "big-E MoE" kernel at low E** — dispatch overhead per Pallas call × ~1k–3k layer-calls/step exceeds the per-matmul saving when E is small.
- ❌ **Auto-partitioning a Mosaic kernel** — always raises `NotImplementedError`. Wrap in `shard_map` with explicit input/output specs.

### Splash attention

**Mechanism**:
- [concepts/splash-attention.md](concepts/splash-attention.md)
- [concepts/attention-block-sizes.md](concepts/attention-block-sizes.md)
- [concepts/custom-splash-masks.md](concepts/custom-splash-masks.md)
- [concepts/block-sparse-offset-masks.md](concepts/block-sparse-offset-masks.md)
- [concepts/flash-attention.md](concepts/flash-attention.md)
- [concepts/ring-attention.md](concepts/ring-attention.md)
- [concepts/native-sparse-attention.md](concepts/native-sparse-attention.md) — DeepSeek NSA; hardware-aligned trainable sparse attention.
- [sources/2026-tokamax-splash-attention.md](sources/2026-tokamax-splash-attention.md)
- [sources/2024-flashattention-3-paper.md](sources/2024-flashattention-3-paper.md) — Hopper-specific FA3 (warp specialization, TMA, WGMMA, FP8). Not TPU-portable; algorithmic ideas (block quantization, incoherent processing) transfer to v7x+ FP8.
- [sources/2025-nsa-paper.md](sources/2025-nsa-paper.md) — Native Sparse Attention (ACL 2025 Best Paper). Three-branch (compressed + selected + sliding) end-to-end trainable; 64k-context at FA2 speed.
- [sources/2025-nsa-tpu-blog.md](sources/2025-nsa-tpu-blog.md) — Henry Mko's TPU v5e port. **Softmax order-invariance + sort-then-process** trick converts NSA's dynamic indexing into Pallas-compatible lexicographical traversal. 2.5× over vectorized JAX.
- [sources/2026-ragged-paged-attention-paper.md](sources/2026-ragged-paged-attention-paper.md) — **the TPU SOTA inference kernel**: 86% MBU decode / 73% MFU prefill on Llama 3 8B / TPU7x. Primary backend in vLLM + SGLang.
- [sources/2025-typhoon-mla-paper.md](sources/2025-typhoon-mla-paper.md) — mixed naive-absorb MLA kernel for shared-prefix decode (RAG, multi-turn, agents); 3-3.24× over pure-absorb.
- **Kernel catalog**: see [Pallas kernel patterns → Kernel catalog → Dense flash/splash row](#pallas-kernel-patterns-lane-wide) for which repos carry which splash variants (tokamax extended, maxtext vendored, maxdiffusion diffusion-tuned, axlearn + logit sink + in-kernel dropout, graphcast banded-sparse for weather models).

**Generic principles**:
- **Standard attention materializes an N×N matrix in HBM** — at N=4096 that's 32 MB per layer per request of pure bandwidth waste. Splash/flash kernels avoid this by tiling and never materializing the full attention matrix in HBM.
- Default to **`sa_block_q=1024, sa_block_kv=1024`** as the cross-model starting point. Cluster validation runs typically find this is universal.
- `sa_block_kv` affects VMEM, not HBM. Halving it frees only tens of MB of HBM (not GB) — don't expect memory relief from block-size changes.
- Fused-bwd kernel has VMEM-saturation risk at multi-host; non-fused bwd is the conservative choice when the profile shows splash < ~5% of step.

**Generic refuted-pattern principles**:
- ❌ **`sa_use_fused_bwd_kernel=True` on multi-host** — VMEM-saturated deadlock at first step on v5p multi-host. Re-check explicitly per (model, hardware) before flipping.
- ❌ **Source-level layout pins around splash** (`with_sharding_constraint`, `jax.experimental.layout.Layout`) — splash's `operand_layout_constraints` is XLA-internal; user-level pins are silently ignored. **Caveat**: filed under GSPMD-mode propagation. Open hypothesis: re-test under [JAX explicit sharding](sources/2026-jax-explicit-sharding.md) (deterministic JAX-level propagation, errors on ambiguity) + the newer [`Format` API](sources/2026-jax-experimental-layout.md) (vs older bare `Layout`).
- ❌ **QKVLayout enum sweeps for HBM-layout effect** — the enum only flips internal `q_ref[...]` access patterns within the kernel; XLA's `LayoutAssignment` doesn't consult it.

### Tensor parallelism (TP)

**Mechanism**:
- [concepts/tensor-parallelism.md](concepts/tensor-parallelism.md)
- [concepts/context-parallelism.md](concepts/context-parallelism.md)
- [concepts/multi-shard-sequence-parallel-correction.md](concepts/multi-shard-sequence-parallel-correction.md)
- [sources/2025-scaling-book-ch3-sharding.md](sources/2025-scaling-book-ch3-sharding.md) — the Megatron-style TP pattern is Case 2 + Case 3 chained.

**Generic principles**:
- TP wins on **wide** per-layer matmuls (large hidden) and **many** experts. TP regresses when per-layer activation collectives can't amortize over enough FLOPs.
- For MoE: TP makes sense at `num_experts ≥ 16` or when per-track matmuls are large. At `E=4` and narrow attention, TP collectives are too thin.
- **TP ≤ 8 within an ICI island** is the load-bearing universal rule. Measured 43% degradation at TP=16 (inter-node) vs TP=8 (intra-node).
- TP axis must be distinct from FSDP axis for the all-reduce-scatter / all-gather pair to be safe (otherwise: nested same-axis collective deadlock).

**Generic refuted-pattern principles**:
- ❌ **TP on a small-E narrow-attention MoE** — collectives dominate, MFU regresses. Re-open only at larger model dimensions.
- ❌ **TP across DCN** — measured 43% degradation; treat as out-of-bounds.

### FSDP / collective optimization

**Mechanism**:
- [concepts/fsdp.md](concepts/fsdp.md)
- [concepts/all-gather.md](concepts/all-gather.md), [concepts/all-reduce.md](concepts/all-reduce.md), [concepts/reduce-scatter.md](concepts/reduce-scatter.md)
- [concepts/async-collectives.md](concepts/async-collectives.md)
- [concepts/collective-bucketing-prefetch.md](concepts/collective-bucketing-prefetch.md)
- [concepts/latency-hiding-scheduler.md](concepts/latency-hiding-scheduler.md)
- [concepts/in-graph-xla-collective-lowering.md](concepts/in-graph-xla-collective-lowering.md)
- [concepts/simple-fsdp-graph-trainer.md](concepts/simple-fsdp-graph-trainer.md)
- [concepts/graph-trainer-fx-passes.md](concepts/graph-trainer-fx-passes.md)
- [sources/2025-scaling-book-ch3-sharding.md](sources/2025-scaling-book-ch3-sharding.md) — FSDP = AllGather(params, fwd) + ReduceScatter(grads, bwd)

**Generic principles**:
- **FSDP cost identity**: same total bytes moved as DP's AllReduce, decomposed into AllGather (fwd) + ReduceScatter (bwd). Lower memory footprint, same network cost.
- **lbs-doubling at FSDP ≥ 8 is the universal cheap win**: amortizes per-step collectives over more tokens. Typical lift ~+0.9 pp MFU per doubling, until you saturate HBM at the ~1.5× current-frontier batch wall.
- Collective-matmul mode V2 (`xla_tpu_*_collective_matmul_mode=none`) is the modern default on libtpu v160+; V1 boolean flags are deprecated.
- Async collective fusion flags are XLA defaults on libtpu v160+ — forcing them ON is a no-op.

**Generic refuted-pattern principles**:
- ❌ **Pure DP / ZeRO-1 (replicated weights, no FSDP shard)** — GSPMD replicated-matmul codegen inflates conv-fusion bytes; refutes universally on multi-chip slices.
- ❌ **V1 collective-matmul bool flags** (`xla_tpu_enable_windowed_einsum_for_*`) — silently rejected as DEPRECATED on libtpu v160+. Use V2 enum.
- ❌ **`--xla_jf_spmd_threshold_for_windowed_einsum_mib`** — silently ignored on libtpu v160+.
- ❌ **Bigger batch without re-checking FSDP divisibility** — `BS % FSDP != 0` is a compile error.
- ❌ **Forcing async-collective fusion flags ON** — they're XLA defaults; flipping them is a no-op on modern libtpu.

### VMEM / scratch memory

**Mechanism**:
- [concepts/vmem.md](concepts/vmem.md), [concepts/vmem-budget.md](concepts/vmem-budget.md), [concepts/vmem-oom-fallthrough.md](concepts/vmem-oom-fallthrough.md)
- [concepts/cmem.md](concepts/cmem.md), [concepts/memory-hierarchy.md](concepts/memory-hierarchy.md)
- [observations/pallas-vmem-budget-independent-of-scoped-vmem-flag.md](observations/pallas-vmem-budget-independent-of-scoped-vmem-flag.md)

**Generic principles**:
- **VMEM caps per gen**: v5p = 64 MB (65472 KiB; silently clamped), v6e = 96 MB, v7x ≈ 128 MB (inferred), v8i = 384 MB.
- `LIBTPU_INIT_ARGS="--xla_tpu_scoped_vmem_limit_kib=81920"` is the cross-lane convention — clamped to hardware cap, no harm.
- The Pallas-kernel-allocated VMEM is **not** always controlled by the scoped_vmem flag; see the linked observation.
- VMEM is the on-chip scratchpad; the hierarchy is Registers → VMEM → HBM → Host RAM. VMEM bandwidth is dramatically higher than HBM — a kernel that keeps tiles in VMEM avoids HBM round-trips.

**Generic refuted-pattern principles**:
- ❌ **VMEM < 65472 KiB on v5p** — measurably regresses (smaller scratchpad = more spills).
- ❌ **Expecting v5p to honor `scoped_vmem_limit_kib > 65472`** — silently clamped; no effect.
- ❌ **Selecting a kernel block size that requires more VMEM than the cap** — silent hang or step-2 recompile error.

### Activation checkpointing (AC)

**Mechanism**:
- [concepts/rematerialization.md](concepts/rematerialization.md)
- [concepts/training-memory-budget.md](concepts/training-memory-budget.md)
- [concepts/host-offload.md](concepts/host-offload.md)
- [concepts/two-level-chunk-recomputation.md](concepts/two-level-chunk-recomputation.md)
- [sources/2025-scaling-book-ch5-training.md](sources/2025-scaling-book-ch5-training.md)

**Trade-offs** (from xprof-mcp):

| Variant | Compute overhead | Memory saved |
|---|---|---|
| Full AC | +30–40% | ~100% of activations |
| Selective AC | +~2.7% | ~70% of activations |
| None | 0% | 0% |

**Generic principles**:
- **AC=selective** is a small-model win — eliminates forward-matmul recompute (~−24% conv-fusion bytes). Trades +40 GiB peak HBM for throughput. Canonical production trade.
- **AC=full** is required at larger model scale — activation residuals dwarf HBM otherwise.
- The break-even depends on **layer count × per-layer activation residual** vs HBM headroom on the target hardware.
- **20% headroom rule**: if HBM headroom > 20%, reduce remat (you're paying compute you don't need to). If headroom < 5%, increase remat or AC.
- JAX knob: `jax.checkpoint_policies.dots_with_no_batch_dims_saveable`. MaxText exposes `remat_policy`. PyTorch/XLA: `torch.distributed.algorithms.checkpoint_wrapper`.

**Generic refuted-pattern principles**:
- ❌ **Assuming AC=selective transfers across the variant matrix** — does not. A 3B/v5p-16 selective win can OOM at 24B/v5p-32 because layer-count is up 4× and residuals dwarf HBM.
- ❌ **AC=offload_dot** — has compile-OOM in practice on at least one variant; verify before committing.

### Batch-size (lbs) amortization

**Generic principles**:
- At FSDP ≥ 8, **lbs-doubling gives ~+0.9 pp MFU** until you saturate either (a) compile-time HBM (typical at 1.5× the current frontier batch) or (b) per-shard activation memory.
- Tokamax CE and similar Pallas kernels are usually neutral at small lbs × small mesh — the lift comes at multi-host with larger batches.
- **Total batch should be a multiple of 1024** for best MXU efficiency (per the dimension-alignment rules).

**Generic refuted-pattern principles**:
- ❌ **Pushing lbs beyond HBM headroom** — `lbs = N+1` step is often a hard cliff (compile-time OOM by GBs).
- ❌ **Batch increase without re-checking FSDP divisibility** — compile error.
- ❌ **Batch increase without re-checking per-chip dim alignment** — total batch can grow while per-chip drops below the 128/256 cliff.

### Pallas kernel patterns (lane-wide)

Two subsections: (A) **Kernel catalog** — where existing kernels live (use this BEFORE writing a new kernel); (B) **Authoring principles** — patterns that apply to any kernel you write.

#### A. Kernel catalog — where Pallas kernels live by function

**START HERE before writing a new kernel.** The [`pallas-kernel-directory`](analyses/2026-04-23-pallas-kernel-directory.md) analysis is a repo-by-repo catalog of **~200 Pallas kernels across ~30 public repositories**, with source paths, stability statements, perf claims, and known callers. Roughly 60% of production-grade TPU kernels are `mosaic_tpu` backend; the rest are Pallas-on-Triton (GPU), Pallas-Mosaic-GPU (SM90/SM100), or Pallas-XLA fallbacks. Every production-grade TPU kernel is either published upstream in `jax-ml/jax` / `openxla/tokamax` or vendored from them with modifications.

**Top-level catalog**: [`analyses/2026-04-23-pallas-kernel-directory.md`](analyses/2026-04-23-pallas-kernel-directory.md) — cross-repo index, grouped by functional category, with subpages by repo group.

**Subpages** (each ~25k of detailed per-kernel rows):
1. [`pallas-kernel-directory/01-upstream-jax-tokamax.md`](analyses/pallas-kernel-directory/01-upstream-jax-tokamax.md) — `jax-ml/jax` + `openxla/tokamax` (~55 kernels, root of the vendoring graph; TPU + all three GPU backends)
2. [`pallas-kernel-directory/02-ai-hypercomputer.md`](analyses/pallas-kernel-directory/02-ai-hypercomputer.md) — MaxText, MaxDiffusion, JetStream (~17 kernels, all `mosaic_tpu`; MaxDiffusion ships the **public Pallas ring attention** integration)
3. [`pallas-kernel-directory/03-inference-engines.md`](analyses/pallas-kernel-directory/03-inference-engines.md) — vllm-project/tpu-inference, sgl-project/sglang-jax, aphrodite-engine (~33 kernels; **tpu-inference is the authoritative author** for the engine tier — MLA v1/v2, fused_moe v1, quantized_matmul blockwise, all_gather_matmul, gdn, sparse_core, batched_rpa; sglang-jax has **~2,000+ tuned block-size entries** as a crown-jewel autotune dictionary)
4. [`pallas-kernel-directory/04-research-labs.md`](analyses/pallas-kernel-directory/04-research-labs.md) — apple/axlearn, recurrentgemma, simply, graphcast, alphafold3 (~18 kernels; **uniquely-novel Mamba/Mamba2/RAttention SSM Pallas in axlearn**, **canonical LRU scan in recurrentgemma**, **fused GLU Pallas in alphafold3 @ v3.0.1**)
5. [`pallas-kernel-directory/05-frameworks-quant.md`](analyses/pallas-kernel-directory/05-frameworks-quant.md) — tunix, qwix, aqt, jaxite, paxml/praxis, pytorch/xla, torchtitan, marin/levanter, pytorch (~18 kernels; quant-focused; marin/levanter autotune harness is the most portable deployment-time tuner)
6. [`pallas-kernel-directory/06-community-research.md`](analyses/pallas-kernel-directory/06-community-research.md) — ejkernel, EasyDeL, jax-flash-attn2, ringattention, flashback, gla-jax, etc. (~50 rows; **ejkernel is the broadest community TPU surface** with 17 TPU kernels)

**Companion meta-analysis**: [`analyses/2026-04-23-pallas-kernel-source-survey.md`](analyses/2026-04-23-pallas-kernel-source-survey.md) — ranks repos as ingest candidates by novelty-vs-tokamax + relevance. Tier-1 list: maxtext, vllm-project/tpu-inference, maxdiffusion, axlearn, sglang-jax, JetStream.

**Quick-pick by function** (canonical TPU source for each kernel family — defer to the catalog for full coverage):

| Kernel family | Canonical TPU source(s) | Notes |
|---|---|---|
| **Dense flash / splash (training)** | `openxla/tokamax` (extended splash with experimental dir), `jax-ml/jax` (legacy `flash_attention`), `axlearn` (splash + logit-sink + in-kernel dropout) | Splash is the default for Gemma/Llama/DeepSeek/Qwen/Mistral/Kimi training |
| **Paged KV / ragged / MLA (inference)** | `vllm-project/tpu-inference` (authoritative), `sgl-project/sglang-jax` (with 2k+ tuned configs), `JetStream` (older — archived 2026-02-01) | MLA v1/v2 (DeepSeek-V2/V3) is novel in tpu-inference; see [concepts/multi-head-latent-attention.md](concepts/multi-head-latent-attention.md) for the architecture |
| **FP8/FP4 GEMM + MoE grouped GEMM (GPU)** | `deepseek-ai/DeepGEMM` (canonical), `NVIDIA/TransformerEngine` (DelayedScaling, NVFP4, MXFP8) | GPU-side reference for FP8/FP4 training kernels; algorithmic ideas transfer to a future TPU FP8 stack |
| **Expert-parallel all-to-all (GPU)** | `deepseek-ai/DeepEP` (production; FP8 dispatch + BF16 combine; 12 SMs for 90 GB/s) | GPU-side reference. TPU analog is SparseCore collective offload via `xla_tpu_enable_sparse_core_collective_offload_*` flags |
| **Ring / context-parallel attention** | `AI-Hypercomputer/maxdiffusion` (public Pallas integration with splash) | Zig-Zag (load-balanced causal) ring attention is **confirmed absent** from all surveyed repos — open gap |
| **Speculative decoding tree** | `sgl-project/sglang-jax` | — |
| **Block-sparse / banded-sparse attention** | `axlearn` (block-sparse offset mask for sliding window), `graphcast` (banded-sparse splash for weather models) | — |
| **Gated-delta / linear attention** | `sgl-project/sglang-jax` (`simple_gla`), `axlearn` (rattention) | — |
| **MoE / grouped matmul** | TPU: `AI-Hypercomputer/maxtext` (megablox gmm/backend/ops), `vllm-project/tpu-inference` (fused_moe v1). GPU: `databricks/megablocks` (canonical dropless MoE; 40% over Tutel, 2.4× over Megatron-LM dense), `deepseek-ai/DeepGEMM` (FP8 grouped GEMM, modern successor) | At low E (≤4) all of these often regress vs masked fallback — see [MoE expert dispatch](#moe-expert-dispatch). MaxText megablox is the TPU port of MegaBlocks' block-sparse abstraction. |
| **MoE support — gather/scatter/routing** | `maxtext` (gather_reduce_sc, sort_activations), `vllm-project/tpu-inference` (sparse_core) | — |
| **Normalization (LayerNorm / RMSNorm)** | `openxla/tokamax`, `axlearn` | — |
| **Fused MLP (GLU / SwiGLU / GEGLU / REGLU)** | TPU: `google-deepmind/alphafold3` @ v3.0.1 (canonical fused GLU), `axlearn`. GPU: `linkedin/Liger-Kernel` (production-grade SwiGLU + GeGLU with in-place replacement) | Fused GLU was a previously-open wiki gap, now closed |
| **Fused training kernels (RMSNorm / RoPE / FusedLinearCE — GPU)** | `linkedin/Liger-Kernel` (production at LinkedIn; HF Transformers one-line patch; 20% throughput / 60% memory vs HF baseline; BSD-2-Clause; arxiv 2410.10989) | GPU analog of TPU tokamax. Worth understanding why GPU shared-memory chunked FusedLinearCE works at scale where TPU VMEM chunked-XLA failed (per v391y refutation). |
| **Paged/ragged/MLA attention + sampling (GPU)** | `flashinfer-ai/flashinfer` (MLSys 2025 Best Paper; powers vLLM + SGLang + TensorRT-LLM + HF TGI + MLC-LLM; JIT customizable attention; sorting-free top-k/p/min-p) | GPU production analog of TPU tpu-inference + sglang-jax. Apache-2.0; SM 7.5+ (Turing through Blackwell). |
| **Matmul (quantized / blockwise)** | `vllm-project/tpu-inference` (quantized_matmul blockwise), `qwix` (quantized_matmul), `aqt` (jax/v2/pallas/) | — |
| **Collective / comms** | `vllm-project/tpu-inference` (`all_gather_matmul`), `JetStream` (collective_matmul) | — |
| **State-space / SSM (Mamba / Mamba2 / SSD)** | `apple/axlearn` (`ssm_kernels/mamba_kernels.py`, `ssd_kernels.py`) | Uniquely-novel; not in tokamax/upstream |
| **LRU / linear-recurrence scan** | `google-deepmind/recurrentgemma` (`jax/pallas.py`) | Canonical Griffin scan |
| **Cross-entropy / loss** | `openxla/tokamax` (`linear_softmax_cross_entropy_loss`) | See [CE / softmax loss](#ce--softmax-loss) topic |
| **PRNG** | `jax-ml/jax` upstream | — |
| **Non-ML (FHE polymul)** | `google/jaxite` (`polymul_kernel.py`) | Niche |

**Tuned block-size dictionaries** (use as autotune priors before launching a fresh search):
- `sgl-project/sglang-jax` — **~2,000+ entries** across `ragged_paged_attention v3`, `simple_gla`, `update_kv_cache`, multimodal flash attention. Crown jewel.
- `vllm-project/tpu-inference` — ~1,200+ entries on `ragged_paged_attention v2/v3`, MLA.
- `tokamax/_src/ops/linear_softmax_cross_entropy_loss/pallas_mosaic_tpu.py` — heuristics: v5p `(b=1024, h=512, v=512)`, v6e gen 7+ `(b=1024, h=512, v=2048)`.
- `marin/levanter` autotune harness — most portable deployment-time tuner found in the survey.

**When to write a fresh kernel vs port an existing one**:
- **Port first.** If the catalog has the kernel family in a Tier-1 source, port + adapt (often a 1-day change). Writing from scratch is multi-day to multi-week.
- **Write only when the catalog confirms the gap** (e.g., Zig-Zag ring attention, the one confirmed-absent kernel in the 2026-04-23 survey). The source-survey's "Direct impact on open wiki state" section identifies these.
- **Re-tune block sizes per generation** even when porting — v5p (VMEM 64 MB) and v6e (96 MB) differ. The block size in the source repo may not fit on your target.

#### B. Authoring principles (for kernels you write or modify)

**Mechanism**:
- [concepts/pallas-kernel.md](concepts/pallas-kernel.md)
- [concepts/mosaic-kernel.md](concepts/mosaic-kernel.md)
- [concepts/manual-mlir-dialect-pallas.md](concepts/manual-mlir-dialect-pallas.md)
- [concepts/custom-trace-annotations.md](concepts/custom-trace-annotations.md)
- [concepts/torch-tpu-pallas-custom-jax-kernel.md](concepts/torch-tpu-pallas-custom-jax-kernel.md)
- [concepts/autotuning.md](concepts/autotuning.md), [concepts/compile-time-aware-autotune-filtering.md](concepts/compile-time-aware-autotune-filtering.md)
- [concepts/in-kernel-dropout.md](concepts/in-kernel-dropout.md)
- [concepts/nvidia-weight-tile-bytes-limit.md](concepts/nvidia-weight-tile-bytes-limit.md)
- [concepts/grouped-program-ids-for-l2.md](concepts/grouped-program-ids-for-l2.md)
- [concepts/pallas-on-triton-fused-gemm-activation-gemm.md](concepts/pallas-on-triton-fused-gemm-activation-gemm.md)
- [concepts/segment-matmul-pallas-kernel.md](concepts/segment-matmul-pallas-kernel.md)
- [codebases/tokamax.md](codebases/tokamax.md), [codebases/pallas-forge.md](codebases/pallas-forge.md), [codebases/ejkernel.md](codebases/ejkernel.md)
- [observations/pallas-mosaic-gspmd-requires-shard-map.md](observations/pallas-mosaic-gspmd-requires-shard-map.md) — **canonical** shard_map pattern

**Generic principles**:
- Mosaic kernels under GSPMD **cannot** be auto-partitioned — always `shard_map`-wrap with explicit `in_specs` / `out_specs` / `check_rep=False` + a `jax.lax.psum` on reductions.
- Pallas kernels have dispatch overhead (~hundreds–thousands of dispatches/step at multi-host). The I/O cost (W rereads from HBM) usually dominates, but at small mesh × small batch dispatch overhead can flip the sign.
- **Autotuning is shape-specific** — re-autotune when shape changes (different batch, seq, hidden). Autotune-result caching (pallas-forge) avoids the recompile.
- **Pre-filter autotune candidates by predicted VMEM budget** — running every config can compile-OOM at scale; filter by `block_q × block_kv × head_dim × bytes_per_elem` before dispatch.
- Custom trace annotations (`jax.named_scope`, `pl.named_scope`) make Pallas kernel timing visible in xprof — without them, kernels show up as anonymous custom-calls.
- **Grouped program IDs improve L2 reuse** on GPU; on TPU the analogue is block-iteration order through `pltpu.emit_pipeline` — both reduce HBM-to-on-chip traffic when adjacent program-IDs share inputs.

**Generic refuted-pattern principles**:
- ❌ **`jax.lax.pmax` inside `shard_map`** — no VJP rule. Even with `stop_gradient(pmax(...))` the VJP transform hits it. Use `all_gather + jax.scipy.special.logsumexp` instead.
- ❌ **`all_gather` inside `shard_map` on the same axis as outer collectives** — multi-host deadlock. Use a dedicated axis.
- ❌ **Adopting a kernel without re-tuning block sizes per generation** — v5p heuristics and v6e heuristics differ (VMEM cap shifts the optimal block).
- ❌ **Writing a new kernel without checking the catalog first** — high probability that the kernel family already exists somewhere in the 200-kernel surface; port + adapt is dramatically cheaper.

### torch.compile + scan-over-layers

Long-form compilation strategies that reduce HLO-module count and compile time.

**Mechanism**:
- [concepts/scan-over-layers.md](concepts/scan-over-layers.md)
- [concepts/torch-tpu-compile-pipeline.md](concepts/torch-tpu-compile-pipeline.md)
- [concepts/torch-tpu-compilation-cache.md](concepts/torch-tpu-compilation-cache.md)
- [concepts/torch-tpu-graph-split-synchronize.md](concepts/torch-tpu-graph-split-synchronize.md)
- [concepts/torch-tpu-precision-modes.md](concepts/torch-tpu-precision-modes.md)
- [concepts/spmd-safe-compile-mode-opt-in.md](concepts/spmd-safe-compile-mode-opt-in.md)
- [concepts/jaxpr-hash-cache-keys.md](concepts/jaxpr-hash-cache-keys.md)
- [concepts/graph-trainer.md](concepts/graph-trainer.md)
- [concepts/torchtitan-tpu-trainer-options.md](concepts/torchtitan-tpu-trainer-options.md)
- [concepts/mark-step-sync.md](concepts/mark-step-sync.md)

**Generic principles**:
- **Scan over identical layers reduces compile time from O(N) to O(1)** — `jax.lax.scan`, torchprime `scan_layers`, torchax `ScannedModule`, Flax `nn.scan`. For a 48-layer transformer, this is dramatic.
- Scan **complicates 2D sharding propagation in the backward pass** — OOM during backward with scan enabled typically needs explicit `shard_as` annotations.
- **Dynamic shapes trigger a fresh compile per shape** — XLA compiles per unique input shape. Variable-length batches or growing KV caches each recompile. Pre-compile + warmup all expected shapes before steady-state.
- **JAX compilation cache** requires `JAX_COMPILATION_CACHE_DIR=gs://...` PLUS `JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=1` AND `JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0` to hit on rerun for non-jit_train_step modules.

**Generic refuted-pattern principles**:
- ❌ **Value-dependent branches** (e.g. `if tensor.sum() > 0: ...`) — break fusion. Shape-dependent on static shapes (`tensor.shape[0] == 1`) is fine.
- ❌ **Compile-time-aware autotune without filtering** — burns wall time on configs that compile-OOM at scale. Pre-filter by predicted VMEM budget.

### HBM transient memory + I/O

**Mechanism**:
- [concepts/hbm.md](concepts/hbm.md), [concepts/hbm-bandwidth.md](concepts/hbm-bandwidth.md)
- [concepts/memory-bound.md](concepts/memory-bound.md), [concepts/memory-bandwidth-utilization.md](concepts/memory-bandwidth-utilization.md)
- [concepts/dma-overhead-heuristic.md](concepts/dma-overhead-heuristic.md)
- [concepts/memory-hierarchy.md](concepts/memory-hierarchy.md)
- [concepts/ridge-point.md](concepts/ridge-point.md)
- [concepts/roofline-model.md](concepts/roofline-model.md)
- [sources/2025-scaling-book-ch1-roofline.md](sources/2025-scaling-book-ch1-roofline.md) — canonical theory.
- [observations/v520-compute-saturated-data-movement-bound.md](observations/v520-compute-saturated-data-movement-bound.md)

**Generic principles**:
- **Compute total step HBM traffic** when comparing kernel implementations. FLOPs alone is incomplete and has cost regressions in practice.
- For a `[N, K] × [K, M]` bf16 matmul: bytes = `2 * (N*K + K*M + N*M)`. If `2*N*K*M / bytes < bytes_per_flop_ratio` of the hardware, the op is HBM-bound.
- The variant matrix is typically at **85–89% HBM utilization** at frontier on v5p — memory-tight. Watch transient peak, not just steady-state.
- **Materialized broadcasts are the silent HBM tax** — broadcasts that can't fuse with their consumer land in HBM as full-size temporaries. Diagnose via Op Stats; fix by restructuring fusion boundary.
- **`TPU_PREMAPPED_BUFFER_SIZE=8589934592`** (8 GB pre-mapped DMA buffer) — raise if hitting `RESOURCE_EXHAUSTED` on DMA.

**Generic refuted-pattern principles**:
- ❌ **Reasoning about kernel cost from FLOPs alone** — the canonical failure mode (v391y-class). Always include I/O bytes in the cost model.
- ❌ **Counting on a broadcast to fuse** when the consumer is a reduction across a different axis — XLA often can't fuse these; the broadcast materializes.

### Hardware envelope (v5p vs v6e vs v7x)

See the [TPU hardware envelope](#tpu-hardware-envelope-essential-without-lookup) section above for the actual numbers.

**Generic principles**:
- A frontier on v5p **may not** be the frontier on v6e — different bytes/flop ratio shifts the ridge point.
- v6e/v7x larger VMEM unlocks Pallas heuristics that don't fit on v5p — re-tune block sizes per generation.
- v6e smaller HBM tightens memory recipes — what fits on v5p/v7x may OOM on v6e at the same lbs.
- v7x's SparseCore-default-on shifts the collective story — embedding-heavy workloads behave differently without explicit opt-in.
- v8t's FP4-first compute means a bf16 workload runs at ~1/4 the headline number.

**Generic refuted-pattern principles**:
- ❌ **Naive recipe clone from v5p to v6e** — almost always misses the heuristic re-tune and the memory headroom shift. Treat each generation as a separate iteration row.
- ❌ **Comparing v8 FP4 headline numbers to v5p/v6e/v7x bf16 numbers** — apples to oranges; use the bf16-equivalent column for bf16 workloads.

### libtpu / XLA flags

**Mechanism**:
- [concepts/libtpu-flags.md](concepts/libtpu-flags.md) — `LIBTPU_INIT_ARGS` namespace, V1→V2 collective-matmul migration, per-gen VMEM caps, probe results.
- [concepts/xla-compiler-flags.md](concepts/xla-compiler-flags.md) — exhaustive XLA flag catalog (TPU + GPU + CPU) with MaxText preset bundles.
- [concepts/xla-dump-flags.md](concepts/xla-dump-flags.md)
- [concepts/xla-fusion.md](concepts/xla-fusion.md)
- [concepts/hlo-dumping-and-diffing.md](concepts/hlo-dumping-and-diffing.md)
- [sources/2022-pathways-paper.md](sources/2022-pathways-paper.md) — the orchestration system whose error-propagation design motivates `xla_tpu_use_enhanced_launch_barrier=true`. This project sets `false` (single-pod runs don't need Pathways' error propagation; the barrier deadlocks the FSDP all-gather on multi-host).
- MaxText canonical: `raw/code/maxtext/benchmarks/xla_flags_library.py`.

**Universal multi-host stack (libtpu v160+)**:
```
LIBTPU_INIT_ARGS=
  --xla_tpu_scoped_vmem_limit_kib=81920          # capped to hw cap on v5p
  --xla_tpu_use_enhanced_launch_barrier=false    # mandatory on multi-host
  --xla_tpu_reduce_scatter_collective_matmul_mode=none  # V2 enum, beats default
  --xla_tpu_all_gather_collective_matmul_mode=none
```

**MaxText named bundles** (use these by name when transferring recipes):
- `CF_FOR_ALL_GATHER` / `CF_FOR_ALL_REDUCE` / `CF_FOR_ALL_REDUCE_AND_ALL_GATHER` — continuation fusion presets
- `ENABLE_SPARSECORE_OFFLOADING_*` (RS / AG / AR variants + base) — SC collective offload
- `LAYOUT_FOR_ALL_REDUCE_SCATTER` / `REDUCE_SCATTER_FUSION` — layout pin for RS
- `HOST_OFFLOAD_FLAGS` — host-offload friendly scheduler features
- `DISABLE_COLLECTIVE_MATMUL` — V2 enum mode=none on both AG and RS
- `DISABLE_BUNDLE_AWARE_COST_MODEL` — fixed a 3× backward regression in MoE (<internal-bug>)
- `ENHANCED_LAUNCH_BARRIER` — Pathways error propagation; **inverse** of this project's default (we set `false`)

**Generic principles**:
- Verify libtpu version before transferring a flag stack from another model — newer libtpu can move a flag to a different namespace or deprecate it silently.
- An XLA flag a release ago may be a no-op today.
- **Prefer named MaxText bundles over ad-hoc flag soups** — they're version-tracked and represent the canonical curated stack.

**Generic refuted-pattern principles**:
- ❌ **V1 collective-matmul bool flags** (`xla_tpu_enable_windowed_einsum_for_*`) — DEPRECATED on libtpu v160+. Silently rejected.
- ❌ **`--xla_jf_spmd_threshold_for_windowed_einsum_mib`** — silently ignored on libtpu v160+.
- ❌ **`xla_tpu_use_megacore=false` on v5p** — regresses. Megacore is mandatory.
- ❌ **`xla_tpu_enable_aggressive_loop_fusion_layout_opt=true`** — harmful on at least one MoE workload (v354 series); verify per-model before flipping.

---

## How to maintain this file

- **Don't add per-experiment data here** — that goes in the model page and the per-model refuted-patterns layer.
- **Do add a generic principle here** when an experiment teaches a lesson that crosses model families (e.g. "FLOPs-only cost analysis is incomplete" is a generalizable principle; "v391y refuted on 3B/v5p-16" is not).
- **Do add a new topic section** when a new optimization category (e.g. a new hardware feature, a new kernel ecosystem) becomes load-bearing across multiple models.
- **Do update the TLDR table** when a new profile bucket → topic mapping is observed.
- **Don't duplicate concept page content** — link to it; this file is the index, not the depth.
- **Promote shared findings from per-model wikis to here** — when a per-model autoresearch repo discovers a concept, observation, or kernel pattern that generalizes, the principle (not the experiment ID) belongs in the relevant topic.

## See also

- [SCHEMA.md](../SCHEMA.md) — wiki structure, page types
- [wiki/index.md](index.md) — full catalog of pages (this file is the optimization-specific index; that file is the everything index)
- [wiki/experiments/program.md](experiments/program.md) — the procedural loop spec
- [.claude/skills/formulate-hypothesis/SKILL.md](../.claude/skills/formulate-hypothesis/SKILL.md) — the consumer of this file
- [.claude/skills/edit-model-code/SKILL.md](../.claude/skills/edit-model-code/SKILL.md) — invoked at code-edit time; references this file's hardware envelope for VMEM/HBM constraints

## Sources

The shared-substrate sources this index is grounded in (in addition to per-topic citations):

- `raw/code/maxtext/benchmarks/xla_flags_library.py` — MaxText canonical flag library
- `raw/code/scaling-book/` — DeepMind/JAX-ML "How to Scale Your Model" (11 chapters)
- `raw/code/tokamax/` — Pallas-on-Mosaic kernel suite (CE, splash, fused ops)
- `raw/code/xprof-mcp/docs/TPU_OPTIMIZATION.md` — xprof-mcp diagnostic playbook
- `raw/code/jax/`, `raw/code/torchtitan/` — runtime substrates

**Regenerating this file**: see [`model-optimization-index-regenerate-prompt.md`](model-optimization-index-regenerate-prompt.md) for the self-contained prompt that rebuilds this page from current wiki state. Heuristic: regenerate end-to-end when 10+ new sources have landed since last regen.
