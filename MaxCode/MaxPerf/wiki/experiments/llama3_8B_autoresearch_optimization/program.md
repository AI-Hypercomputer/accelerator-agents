<!-- disableFinding(LINK_RELATIVE_G3DOC) -->
<!-- disableFinding(LINE_OVER_80) -->
<!-- disableFinding(LIST_NO_LINE) -->
<!-- disableFinding(SNIPPET_EM_DASH) -->
<!-- disableFinding(WHITESPACE_LINES) -->
<!-- disableFinding(SPACES) -->
<!-- disableFinding(HTML_OPEN) -->
<!-- disableFinding(HTML_BROKEN) -->

# performance-autoresearch — Llama 3 8B on TPU via torchax

Variant of Karpathy's [autoresearch](https://github.com/karpathy/autoresearch) adapted for **performance optimization** on a specific stack: Google's **Llama 3 8B** (`meta-llama/Meta-Llama-3-8B`), running under **torchax (PyTorch-on-JAX)** on a **TPU v6e-4** host, fine-tuning on `wikitext-2-raw-v1` packed at a fixed seq_len. An AI agent modifies the trainer + sharding + compiler / runtime flags to maximize throughput.

**The model's output distribution is invariant.** The agent optimizes *how* the same computation runs — sharding, attention kernel, remat policy, XLA flags, layer implementation, training config — never the model architecture itself.

## Fixed bindings

Single-environment program (no `program-<env>.md` split yet). These are the facts the loop must respect.

| Binding | Value |
|---|---|
| Hardware | TPU v6 Lite, **4 chips**, ~31.25 GiB HBM per chip, peak bf16 FLOP rate ~946 TFLOP/s per chip (device_information from xprof_mcp), peak HBM BW 1525.5 GB/s per chip. Mesh options: 1D (`fsdp=4`) or 2D (`dp=2,fsdp=2`) etc. HBM ridge point 578 FLOPs/byte. |
| Conda env | `llama3_py313`; Python 3.13.13. Python binary: `/home/$USER/miniconda3/envs/llama3_py313/bin/python`. |
| Trainer | `wiki/experiments/llama3_autoresearch_optimization/torchax/train.py`. `python -m train` from that folder (sys.path dance for `model/` and `data.py`). |
| Baseline command | `python -m train --steps 20 --batch_size 1 --seq_len 1024 --strategy fsdp --profile_dir $PROFILE_DIR --profile_steps 10 11 12` (run from `wiki/experiments/llama3_autoresearch_optimization/torchax/`). `$PROFILE_DIR` = repo-root-anchored `raw/profiles/<YYYY-MM-DD>-<slug>/`. |
| Compiler / runtime env | `LIBTPU_INIT_ARGS`: per experiment. `XLA_FLAGS`: `--xla_dump_to=$HLO_DIR --xla_dump_hlo_as_text` recommended, other flags per experiment. **`--xla_tpu_*` flags go in `LIBTPU_INIT_ARGS`, not `XLA_FLAGS`** — XLA_FLAGS errors with `Unknown flags` otherwise. |
| libtpu version | **0.0.40** (confirmed latest across PyPI, libtpu-lts-releases, libtpu-nightly-releases). No newer version available as of 2026-04-25. When a flag is rejected as unknown, `strings $CONDA_PREFIX/lib/python3.13/site-packages/libtpu/libtpu.so | grep xla_tpu_<token>` is the source of truth. |
| Profile directory | `raw/profiles/<YYYY-MM-DD>-<slug>/` under the wiki repo root. **Gitignored.** |
| HLO dump directory | `raw/profiles/<YYYY-MM-DD>-<slug>/hlo/` (colocated). |
| xprof_mcp server | `xprof --logdir=gs://tpu-pytorch-alekseyv-us-central2/autoresearch --port=8791` (GCS-backed). To make a captured profile queryable in the **xprof browser UI**: upload the session dir to `gs://tpu-pytorch-alekseyv-us-central2/autoresearch/<run-name>/` (with the `plugins/profile/...` subtree inside), then `mcp__xprof__list_runs` reports it and it appears in the dropdown at `http://localhost:8791/`. Historical local path `/mnt/disks/persist/torch-tpu/dump_folder/profiles/plugins/profile/<name>/` still works for older symlinked runs but is not the current server logdir. |
| xprof browser URL | **Per-run direct URL**: `http://localhost:8791/?run=<run-name>` (TensorBoard-style — opens the interactive trace viewer for that run). Every experiment page **must** include this URL under `## Profile` alongside the on-disk path so the reviewer can click straight through to the visual inspector. Convention for `<run-name>`: the last path component of the profile directory, e.g. `2026-04-25-llama3-exp25-splash-block1024`. |
| Stack dimension | Two parallel execution stacks exist: **torchax** (HuggingFace PyTorch wrapped via torchax — the original path, exp 1–33 + the 2026-04-25 baseline) and **jax** (from-scratch Flax NNX port — exp 34+). Writeups live under `torchax/experiments/` and `jax/experiments/` respectively; each folder has its own `OBSERVATIONS.md` and `RESULTS.tsv`. Experiment numbers continue globally across both stacks (commit history stays linear); the folder path + slug disambiguates (e.g. `torchax/experiments/2026-04-25-exp25-splash-block1024-accepted.md` vs `jax/experiments/2026-04-25-exp34-jax-baseline-accepted.md`). Include the stack word in the slug when the experiment could run on either stack (e.g. `exp35-jax-splash-attention`). |
| Experiment page slug | `<YYYY-MM-DD>-exp<NN>-<short-slug>-<verdict-suffix>.md` in `<stack>/experiments/`, where `<verdict-suffix>` ∈ { `accepted`, `rejected`, `potential` } per the rule below. The original baseline `2026-04-25-baseline.md` has no verdict suffix (it is the reference, not a hypothesis test). |
| Experiment **branch** | Each experiment is a git branch named `v6e8-llama3-8b-torchax-<YYYYMMDD>-exp<NN>-<short-slug>` (or `…-baseline` for the reference). The branch holds the exact `train.py` + `model/` + `data.py` state used for the run — the workload pulls a Docker image whose contents mirror that branch tip, so the branch is the reproducibility surface. Even env-only experiments get their own branch (the experiment page + submit script are committed to it). Convention: branch from the most recent `-accepted` ancestor; document the parent branch in the experiment page frontmatter as `branched_from:`. |
| Experiment verdict suffix | **Rule**: once an experiment's verdict is final, the page filename MUST end with one of three suffixes before `.md`, reflecting the `RESULTS.tsv` status. `-accepted` for `keep` / `supported` (hypothesis confirmed and useful). `-rejected` for `discard` / `refuted` / `crash` / `invalid` (hypothesis disconfirmed or unusable). `-potential` for `parked` / `inconclusive` (hypothesis not settled — revisit when conditions change). The suffix is part of the filename at file-creation time; rename on status change. This makes directory listings of `wiki/experiments/<program>/` self-summarizing. |
| Timeout per experiment | **15 minutes** wall-clock. Kill and mark `crash` otherwise. |

## Model constraints (architecture — off-limits)

These are fixed by the model identity and cannot be changed — changing them means you're training a different model.

| Field | Value |
|---|---|
| n_layers | 32 |
| n_heads | 32 |
| n_kv_heads | 8 (GQA 4:1) |
| head_dim | 128 |
| hidden_dim | 4096 |
| intermediate (FFN) | 14336 |
| vocab_size | 128256 |
| max_position | 8192 |
| rope_theta | 500000.0 |
| tie_word_embeddings | false |

## The contract

The final trained model must be equivalent to a baseline-trained model of the same architecture — same output distribution (bitwise up to floating-point rounding for a fixed seed/config). Everything about *how* that computation is expressed, scheduled, sharded, stored, or compiled is tunable.

## What you CAN do

Anything preserving the output-distribution contract:

- Modify any file in `wiki/experiments/llama3_autoresearch_optimization/torchax/` (`train.py`, `model/sharding.py`, `data.py`). Also the text-only-forward monkey-patch in `train.py` if the change preserves Llama 3 semantics.
- **Training config**: `global_batch_size`, per-chip batch, `seq_len`, grad accumulation, `--strategy` (`fsdp` / `tp`), `--fsdp` / `--dp` / `--tp` mesh shape, `--dtype`, `--profile_*`, `--learning_rate`, `--warmup_steps`.
- **Environment / compiler**: any XLA flag or `LIBTPU_INIT_ARGS` value. Persistent compile cache (`JAX_COMPILATION_CACHE_DIR`).
- **Sharding**: mesh shape, axis names, `PartitionSpec` choices, FSDP/TP mix. (`model/sharding.py` owns the rules.)
- **Attention kernel**: default (XLA through HF) → `tokamax.dot_product_attention` (splash) / `jax.experimental.pallas.ops.tpu.splash_attention` / custom Pallas — as long as numerically equivalent. GQA is native in splash — don't `jnp.repeat` K/V.
- **Remat policy**: `full` / selective / none, `jax.checkpoint_policies.save_and_offload_only_these_names`.
- **Optimizer implementation**: custom `scale_by_adam` with bf16 mu+nu, host-offload of opt state, AdamW variants.
- **Precision mix**: bf16 compute with fp32 master optional; fp8 acceptable if loss sanity passes.
- **Module structure**: fused QKV vs separate projections, different RoPE, scan-over-layers (`torchax.train.ScannedModule`), different residual fusion — provided math is preserved.
- **JIT boundaries**: where `jit` is placed, `donate_argnums`, `out_shardings`, `in_shardings`, `static_argnums`.
- **Data pipeline**: prefetch, packing, async host→device.
- **Pallas kernels** for any op with numerical equivalence.

## What you CANNOT do

- Change the **model architecture** per the table above. These define the parameter count and trained-model identity.
- Change `--model_id` away from `meta-llama/Meta-Llama-3-8B` (the `-it` instruction-tuned variant is a different model).
- Quantize trained weights below bf16 (int8/int4 weight quantization is off-limits for this loop; activation int8 is also off-limits).
- Disable or approximate attention (linear attention, sliding-window on a full-attention layer are off). Swapping to a numerically-equivalent kernel is fine. Note Llama 3 **already** has SW-512 on non-global layers — those layers must keep their local-mask semantics.
- Skip / approximate the optimizer step, norm layers, or residual connections. Skip `final_logit_softcapping=30.0`.
- Modify anything outside `wiki/experiments/llama3_autoresearch_optimization/torchax/`. The wiki source pages, the torchax submodule, tokamax, HF transformers, etc., are strictly read-only from this program's perspective (they are the reference material).
- Cheat the metric: skip or warp step-timing, fake the profile, log crashes as `discard`.

**If a torchax / tokamax / transformers source change seems necessary**: stop the loop and ask the human. Explain what you'd change, why, and what experiment it unblocks. Do not proceed until authorized.

## Hardware / parallelism constraints

- **Mesh divisibility**: every per-chip-shard axis must divide evenly into the mesh axis size. On `fsdp=4`, `global_batch_size ≥ 4` and must be a multiple of 4. Shrinking batch below the mesh size is a pre-trace `IndivisibleError`. Pre-check before proposing shape changes.
- **`num_kv_heads=8` does not divide `tp>=4`**. K/V projections under TP with tp>=4 must be replicated (documented in `model/sharding.py`). Known memory-leaving-on-the-table corner.
- **HBM budget**: the baseline at seq=1024 batch=1 already hits **95% HBM**. Any experiment growing state (bigger batch, longer seq, less remat, extra saved activations) OOMs immediately. **Memory-ceiling rule: win memory before throughput.** Structural order is *memory win → throughput win*, not the reverse.
- **Attention N² materialization**: the default (XLA-through-HF) path materializes the `[B, n_heads, S, S]` score tensor. At batch=4, seq=2048 this OOMs on v6e-4. Splash attention avoids this.

## Pallas kernel landscape

Two questions to ask on every profile before proposing an experiment:

1. **Which existing Pallas / Mosaic-TPU kernel could replace this op?** Default on this stack is `HF PyTorch → torchax (aten lowering) → JAX primitives → XLA → TPU MXU/VPU`. Pallas kernels substitute a **custom-call** path that keeps tiled intermediates in VMEM and typically wins on memory-bound ops.
2. **Which ops would benefit from a Pallas kernel that doesn't yet exist?** Build-candidates surface when (a) the op shows up in `loop fusion` with high HBM bytes_accessed and low flops_tf, (b) no upstream Pallas kernel covers it, and (c) XLA's existing fusion can't eliminate the HBM round-trip.

### Decision rule for "Pallas vs. XLA"

- **Compute-bound above the HBM ridge point** (OI > 578 FLOPs/byte on v6e): Pallas rarely wins. XLA's MXU lowering is hard to beat for dense GEMM. Our `convolution fusion` shows 1376 FLOPs/byte, 27 % HBM BW — firmly compute-bound, don't touch.
- **Memory-bound below the ridge point**: Pallas candidate. Keep tiles in VMEM, fuse consecutive HBM round-trips. Our `loop fusion` shows ~3 FLOPs/byte at 62 % HBM BW — strongly memory-bound, big target.
- **Fusion of matmuls sharing an input** (heuristic 1 from sibling program.md): fusing matmuls that share the same input helps **iff output shapes are similar**. Llama 3's `gate_proj` + `up_proj` both map `hidden=2560 → intermediate=10240` — symmetric output, good fuse. QKV under GQA are asymmetric (Q is 4× K/V) — bad fuse.
- **Always** profile first. Proposing a Pallas kernel without a matched profile signal is speculation.

### Existing Pallas / Mosaic-TPU kernels to TRY

| Op in Llama 3 | Existing kernel | Current path on this stack | Expected win | Wiring |
|---|---|---|---|---|
| `Llama4Attention.forward` (all 32 layers) | [`jax.experimental.pallas.ops.tpu.splash_attention`](../../concepts/splash-attention.md) or [`tokamax.dot_product_attention(implementation='pallas_mosaic_tpu')`](../../sources/2026-tokamax-splash-attention.md) | HF SDPA → torchax aten lowering → XLA. Materializes full `[B, n_heads, S, S]` score tensor → N² HBM traffic + OOM at batch=4 seq=2048. | **15–40 % on attention**. Native GQA (8:2). Handles Llama 3's hybrid local-SW-512 + global attention via `splash_attention_mask_builders`. Unlocks longer seq at bigger batch. | Monkey-patch `Llama4Attention.forward` in a local file; bridge torch ↔ JAX via `torchax.interop.call_jax(splash_fn, q, k, v, mask)`. |
| Cross-entropy + LM head output | [`tokamax.linear_softmax_cross_entropy_loss`](../../sources/2026-tokamax-supported-ops.md) (TPU Mosaic) | Hand-rolled `log_softmax` + `gather` in `train.py`'s `forward_loss`, upcast to fp32. Materializes `[B, S, 262144]` fp32 logits ≈ 4 GiB at batch=2 seq=1024. | ~4 GiB of HBM freed; smaller compute impact. Unlocks batch=4 from current 92 % HBM (exp 4) or comparable. | Replace the hand-rolled CE with a `tokamax.linear_softmax_cross_entropy_loss(hidden, lm_head_weight, labels)` call from the JAX side. Needs torchax interop. |
| `jax.experimental.pallas.ops.tpu.flash_attention` | Older flash-attention TPU kernel, pre-splash. | n/a — splash supersedes. | Don't use; splash covers everything flash does plus GQA + sparse masks. | — |
| `tokamax.ragged_dot` | Grouped matmul for MoE. | n/a — Llama 3 is dense, not MoE. | Skip. | — |

### Pallas kernels to BUILD (gaps in the ecosystem)

Driven by: profile signal + existing tokamax Wave 1 finding that **TPU `gated_linear_unit` and `layer_norm` have NO TPU-specific Pallas kernel — they silently fall back to XLA**. Every one of these is a kernel we would write and drop into `torchax/kernels/` (or upstream to tokamax) if the profile justified the effort.

| Op | Why it's a build-target | Where in profile | Sketch | Expected |
|---|---|---|---|---|
| **TPU Mosaic RMSNorm** | Llama 3 has 5 RMSNorm calls per layer × 32 layers = **210 RMSNorm calls per forward**. No TPU Pallas kernel in tokamax; XLA runs them as fused loops. Memory-bound. | `loop fusion` (multiply_add_fusion + fusion.95 + multiply_add_fusion.4 are the top three at ~140 ms aggregate — matches the RMSNorm-pattern count). Each runs at 87–93 % HBM BW (near peak). | Mosaic-TPU block kernel: accumulate sum-of-squares in VMEM across a block of the hidden dim, invsqrt, scale by learnable weight, write once. Fuse with input residual-add when applicable. | **3–8 % on step time**. Tight HBM-BW-limited kernel, near-peak achievable given current 87 % BW usage — the win is avoiding redundant HBM passes (sum-squares read + normalize read + scale read = 3 passes → 1 pass). |
| **TPU Mosaic SwiGLU (GLU + activation)** | `gate_proj` + `up_proj` share input `x`. Currently 2 separate matmuls + elementwise `act(gate) * up` + `down_proj` matmul. No TPU Pallas kernel. | `convolution fusion` covers the 3 matmuls; the elementwise `act*up` lives in `loop fusion`. Per-layer MLP time ≈ 15 ms × 42 = 630 ms aggregate. | Fused Mosaic-TPU kernel: tile `x`, compute both matmuls sharing the HBM read of `x`, apply activation × mul inline, write a single intermediate to HBM for `down_proj`. | **5–15 % on MLP time**. One shared HBM read of `x` instead of two; one intermediate tensor instead of three. Matches the "fuse-matmuls-sharing-input-when-outputs-are-symmetric" heuristic — gate and up are symmetric. |
| **Fused residual-add + RMSNorm** | Transformer pattern `x = x + attn(norm(x))` does `residual-add` then next layer's `RMSNorm` reads the same tensor — two HBM round-trips on consecutive memory-bound ops. | `loop fusion` — sub-ops `add_convert_fusion` (42 ms) and `subtract_subtract_fusion` (35 ms) both at 89–93 % HBM BW. | Extend the RMSNorm Pallas kernel above to accept a residual-input: `out = rmsnorm(x + residual, scale)`. Fuses both ops into one HBM pass. | **2–5 % on step**. Pure HBM-round-trip elimination. Low risk; wins compound with the RMSNorm kernel. |
| **Fused final logit softcap + log-softmax + NLL** | Llama 3 applies `logits = 30 * tanh(logits / 30)` *after* `lm_head` — materializes `[B, S, V=262144]` bf16 (~0.5 GiB/sample), *then* the CE kernel does another pass. | Not directly visible at the op level (inside the convolution-fusion + loop-fusion pipe), but straightforward to estimate from the `[B, S, V]` tensor sizes. | Extend the tokamax CE kernel (or build a variant): take `hidden`, `lm_head.T`, and `labels` → apply `30*tanh(logits/30)` inline → log-softmax → NLL. Never materialize the `[B, S, V]` tensor. | **0.5 GiB × batch saved**; small wall-clock gain from elimination of one extra pass. Mostly a memory-unlock. |
| **Fused Q/K RMSNorm + RoPE** | Llama 3 applies `q_norm` and `k_norm` right after Q/K projection and before RoPE. Three small ops in a row on the same tensor. | Hidden inside the attention fusion; visible as extra `loop fusion` per-layer sub-ops. | Pallas kernel: `out = rope(norm(x, scale), cos, sin)` in a single pass over the `[B, S, H, D]` tensor. | **1–2 % on step**. Small but cheap; stack-able with splash kernel. |

### When to propose vs. defer a new kernel

- If the build cost is > 2 experiment cycles (~30 min) **and** the expected win < 5 %, **defer** and file it as a follow-up.
- If a profile clearly points at a specific memory-bound hotspot > 10 % of step time with no existing Pallas coverage, **build**.
- Always check the [tokamax codebase page](../../codebases/tokamax.md) for hidden kernels (e.g. `ring_attention_kernel` — exists but not wired to the public API) before writing from scratch.

## Known correctness bug

**NaN loss at seq ≥ 2048** with the current scaffold. Clean at seq ∈ {512, 1024}. `final_logit_softcapping` already applied in the text-only forward patch; root cause unlocalized (bf16 attention overflow / hybrid SW+global mask edge case candidates). Experiments at seq ≥ 2048 cannot yield `supported` until this is fixed. Until then, **run at seq=1024 unless the experiment targets the NaN fix itself**.

**Loss sanity check** — required after changes that can affect numerics (attention kernel swap, precision change, custom Pallas, optimizer internals, init/seed change). Rerun for ≥ 20 steps and verify the loss trajectory matches the baseline within bf16-reorder noise at the same steps. If it diverges, the change is a cheat or a bug — mark `invalid` and revert.

**Simplicity criterion**:

- TPS win from **deleting** code → clear keep.
- ~1% TPS win with 200 lines of fragile new code → probably discard; log the idea as a follow-up.
- No-op-or-faster plus simpler → always keep.

## Metric

**Primary: TPS (tokens per second)** = `global_batch_size × seq_len / median_step_time_s`. Higher is better. Invariant under batch/seq tuning.

**Secondary: MFU** — sanity check. Use detailed matmul-FLOP accounting (not `6PT` with `P=8B` — Llama 3's PLE tables add params without matmul FLOPs; the right `P` is ~4B non-PLE matmul params; see [2026-04-25-baseline.md](2026-04-25-baseline.md) for the breakdown). Baseline MFU was ~26% at seq=2048, ~23% at seq=1024. If TPS rises but MFU falls, flag it in observations.

**Diagnostic: step_time** — human-readable signal. Not a decision metric.

**Measurement protocol**: median step_time over **steps 6–15** if steps 1–5 include warm-up and steps 1 is a recompile (which it currently is, ~150 s — see exp2/"refuted" for the attempted fix). At `--steps 20`, steady-state is steps 2+ because step 1 is the second compile, and practical reporting uses steps 6–15. Compile time is recorded separately; does not enter the metric.

## Output format — per run

Extracted from the trainer's summary block + xprof_mcp `get_overview` + `get_memory_profile`:

```
tps:             <global_batch × seq_len / median step_time>
mfu_percent:     <detailed matmul FLOPs / (peak × wall time × n_chips)>
step_time_ms:    <median over steps 6-15; fall back to 2-<N-1> for short runs>
peak_hbm_gib:    <from get_memory_profile>
compile_seconds: <step 0 wall time — diagnostic only>
config:          <key flags / code change vs baseline, one-line>
```

## Ledger — `<stack>/experiments/RESULTS.tsv`

One ledger per stack (`torchax/experiments/RESULTS.tsv` and `jax/experiments/RESULTS.tsv`). Append one row per run (tab-separated, gitignored):

```
exp_id    date    tps    mfu_percent    step_time_ms    peak_hbm_gib    config    status    description
```

- `exp_id` — `exp<NN>` (or `baseline`). Numbering continues globally across both stacks — use the folder path to disambiguate which stack the row belongs to.
- Others as per template. `status` ∈ `{keep, discard, crash, parked, inconclusive}`, and the experiment-page filename's verdict suffix (`-accepted` / `-rejected` / `-potential`) must match.

## Observations log — `<stack>/experiments/OBSERVATIONS.md`

One section per experiment. **The experiment page** (`<YYYY-MM-DD>-exp<NN>-<slug>-<verdict-suffix>.md` — see "Experiment verdict suffix" rule in the Fixed bindings table) remains the canonical per-experiment file per `SCHEMA.md`; `OBSERVATIONS.md` is a **skim-and-reason** aggregation log for the human reviewer to quickly thread through the session's arc.

Template per block (append-only, commit alongside code):

```markdown
### exp<NN> — <commit>  <one-line summary>

**Config**:
- Command diff from prior keep: <flag diffs + code change summary>
- Profile path: `raw/profiles/<YYYY-MM-DD>-<slug>/` (on-disk, gitignored; link as relative markdown for IDE click-through)
- **Profile browser URL** (mandatory if run executed): `http://localhost:8791/?run=<YYYY-MM-DD>-<slug>`
- HLO dump path: `raw/profiles/<YYYY-MM-DD>-<slug>/hlo/`
- Experiment page: `<YYYY-MM-DD>-exp<NN>-<slug>-<verdict-suffix>.md` (accepted/rejected/potential — see Fixed bindings)

**Hypothesis**:
<2–4 sentences; what you expect and why. Cite profile signal / wiki page / prior follow-up.>

**Changes made**:
- <file:function — what was swapped — equivalence argument>
- <config flag change — old → new — why>

**Expected outcome**:
<TPS / MFU / HBM prediction + mechanism>

**Actual outcome**:
- TPS: X → Y (Δ Z%)
- MFU: A → B
- Step time: P → Q ms
- Peak HBM: M → N GiB
- Compile time: C → D s

**Profile signals**:
- Bottleneck (from `get_overview`): compute | memory | host | collective
- Top ops change (from `get_top_hlo_ops`): which ops moved, how bytes_accessed / FLOPs shifted
- HLO patterns: fusion formed/broke, collective placement, custom-call presence
- HLO-pass diff (if relevant, from `diff_hlo_stages`): what the compiler did differently

**Analysis**:
<If matched expected — why the mechanism worked. If not — what really happened, confounds, what it reveals about the system. If crash — root cause, and whether it invalidates the hypothesis or just the implementation. If numerics-affecting — report sanity loss vs baseline at a common step.>

**Decision**: keep | discard | crash | invalid

**Follow-ups** (ideas this surfaced):
- <idea 1 — what, why plausible>
- <idea 2 — ...>
```

## The experiment loop

**Branch model (MANDATORY — updated 2026-04-25 after 16-experiment retrospective)**: each experiment lives on **its own branch**, forked from the current session-trunk tip (or another kept branch — see "hierarchical forking" below). The model is inherited from the core autoresearch doc and applies to this program.

### Why branches matter

Each experiment branch is a **persistent rollback point**. At any future moment you can `git checkout perfautoresearch/v6e4-<date>-exp<NN>-<slug>` to recover the exact code state that experiment ran on, read its `OBSERVATIONS.md` block, and start a new experiment from that state to explore an alternative direction. In-place editing on a single branch destroys that — the rejected experiments' code is gone the moment the next edit lands.

The branch structure is therefore a **tree of hypotheses**, not a linear history:
- The **trunk** (`perfautoresearch/v6e4-<date>`) is the main line of compounding wins — the sequence of *kept* experiments that make up the current best config.
- **Side branches** are experiments that were run but not (yet) merged — discarded, crashed, or still-under-evaluation.
- **Hierarchical forking**: a new experiment can fork from **any** kept branch in the tree, not only the current trunk tip. If exp 12 gave us splash+bf16-CE and we want to go back to exp 6's simpler baseline to try a totally different direction (say, tokamax CE instead of splash), we fork directly from `...-exp06-...` — exp 12's unrelated code isn't in the way. Use this when an earlier kept state is a cleaner ancestor for the new idea than the current trunk is.

### Naming and commit rules

- **Session trunk**: `perfautoresearch/v6e4-<YYYYMMDD>` (e.g. `perfautoresearch/v6e4-20260423`). Created once per session, branched from `main`. Holds only **kept** experiments.
- **Per-experiment branch**: `perfautoresearch/v6e4-<YYYYMMDD>-exp<NN>-<short-slug>`. Forked from the session-trunk tip (or a chosen prior kept branch) at the start of each experiment. Every code edit + the experiment page + the `OBSERVATIONS.md` block + the `RESULTS.tsv` row for this experiment commit **to this branch**, not to trunk.
- **Keep**: `git checkout <session-trunk>` + `git merge --no-ff <exp-branch>`. The `--no-ff` preserves a merge commit so the trunk history shows which changes were kept as discrete units. Next experiment forks from this new trunk tip — **unless** a hierarchical fork from an older state is the better ancestor for the next hypothesis.
- **Discard / crash**: `git checkout <session-trunk>`. Leave the experiment branch alone — unmerged but present, fully recoverable. If the `OBSERVATIONS.md` update on the discarded branch has useful follow-ups, cherry-pick just that file onto the trunk (`git checkout <branch> -- path/to/OBSERVATIONS.md`, commit on trunk). Do **not** cherry-pick the experiment's code change onto trunk — that's what "discard" means.
- **Hierarchical branching example**: exp 18 might say "fork from exp 5's selective-remat branch, add tokamax CE, skip splash." The branch name reflects it: `...-20260423-exp18-tokamax-ce-from-exp05`. The descriptive suffix is free-form; include the parent exp number when forking from anywhere other than the current trunk tip.

### Retrospective

The first 16 experiments in this program **violated this model** — code was overwritten in-place on `main` without per-experiment branches. Exps 1–16's exact intermediate code states are not recoverable from git; only the experiment pages describe what the code change was. The violation is preserved as commit `71a45ae exp1-16: compound best-config reached` (the consolidation commit). From exp 17 onward, this discipline is enforced.

No `git reset --hard` in the happy path. Failed attempts are parked as unmerged branches for later review; the trunk's git log is a clean record of only the *kept* changes.

LOOP FOREVER:

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read papers referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

1. **State check.** Read the tail of `OBSERVATIONS.md` for recent context and the last `status: keep` row of `RESULTS.tsv`. The trainer source on `main` is at that kept state.
2. **Generate a hypothesis**, priority order:
   a. **Profile-driven** — highest-signal gap in the most recent profile: top slow op, low HBM BW utilization on bandwidth-bound op, collective wait, `async-collective-done` stall, missing fusion. Query `get_overview`, `get_top_hlo_ops`, `get_op_profile`, `get_memory_profile`.
   b. **Follow-up** — an unexplored idea from a previous observation's "follow-ups".
   c. **Wiki-driven** — consult current accumulated wiki knowledge base and if some information is missing consider checking internet to check on potential bottlenecks observed for the current model and for ideas to try.
   d. **Heuristics table below** — accumulated rules of thumb.

   **Within these, prefer structural changes (shape, sharding, code) over flag-only changes.** Exp 1 (2026-04-25, async-collective flag bundle) was `refuted` with a 25% regression because the compiler had already found a good schedule for the small workload; giving it more freedom to reorder broke compute-fusion memory locality. On already-decent configs, XLA/LIBTPU flag-only experiments tend to be null or noise unless they follow a structural change that unlocks them.

   **Memory-ceiling rule** (applies here — 95% HBM on baseline): memory-saving change must come before any state-growing change.

3. **Implement** the hypothesis. One experiment = one `<YYYY-MM-DD>-exp<NN>-<slug>-<verdict-suffix>.md` page + code changes to `torchax/` + a `RESULTS.tsv` row + an `OBSERVATIONS.md` block.
4. **Run** the baseline command with the hypothesis-adjusted config / env:
   ```bash
   cd wiki/experiments/llama3_autoresearch_optimization/torchax
   LIBTPU_INIT_ARGS="..." XLA_FLAGS="..." \
     python -m train --profile_dir $PROFILE_DIR --profile_steps 10 11 12 \
       --steps 20 --batch_size <B> --seq_len <S> [other flags] \
       2>&1 | tee /tmp/llama3_<exp>.log
   ```
5. **Classify the outcome**. A `CompileTimeHbmOom` or `RESOURCE_EXHAUSTED` log line by itself is not necessarily fatal — XLA may retry silently. Wait for a step-log line (success) OR a `RuntimeProgramAllocationFailure` followed by no further progress (crash). After timeout with no step summary → crash. Dumb bug → fix on main and rerun; fundamentally broken → crash row, skip to step 8.
6. **Immediately symlink the profile** into the live xprof logdir with a descriptive name (`llama3_exp<NN>_<slug>_<YYYYMMDD>`) so xprof_mcp can still access it later. Record the symlink name in the experiment page and the observation block.
7. **Extract metrics** via xprof_mcp:
   - `list_runs` → confirm the symlinked run is visible
   - `get_overview` → step time, device duty cycle, MFU when computed
   - `get_top_hlo_ops(limit=15)` → op time / FLOPs / bytes_accessed
   - `get_op_profile` → nested breakdown, HBM BW utilization per op
   - `get_memory_profile` → peak HBM, stack vs heap, fragmentation
   - Optional for HLO-level changes: `list_hlo_modules`, `get_hlo_module_content`, `diff_hlo_stages` (needs HLO dumps enabled).
8. **Write up** the experiment page (full `experiment` template per `SCHEMA.md`) + append to `OBSERVATIONS.md` + append to `RESULTS.tsv`. The **HLO-op diff table** (baseline vs this run, top ops, time / bytes deltas) is the single most diagnostic piece of evidence.
9. **Decision**:
   - **keep** — TPS improved beyond noise, no tracked-metric regression, loss trajectory within bf16-reorder noise. Commit to main.
   - **discard** — regression, no change, or correctness preserved but not worth the code complexity. Revert code changes on main; keep the experiment page + observation block (they are durable reasoning).
   - **crash** — OOM, compiler bug, infra failure. Revert code; file the experiment page with `status: crash` and the root-cause analysis.
   - **invalid** — the change modified model semantics (loss sanity fails) or otherwise violated the contract. Revert, mark the page `verdict: invalid`, do NOT report the measured number as a win.
10. **Loop.**

## Improving the approach itself

If during experimentation a generalizable improvement surfaces (a better measurement protocol, a richer profile-reading heuristic, a new anti-cheat case, a new reference page to consult), commit the improvement **to this `program.md`** directly on main (methodology updates aren't experiments) and add a `## approach evolution` entry to `OBSERVATIONS.md`:

```markdown
### approach update — <date>
<what changed in program.md, why, and the experiment that revealed it>
```

## Heuristics accumulated

Rules of thumb, not invariants.

- **Flag-only experiments on a small workload often regress** (exp 1 lesson). The stock XLA scheduler has usually found a good-enough schedule; giving it more freedom to fuse/reorder can break compute-side memory locality it had by default. Flag experiments become useful right after a structural change; prefer structural first.
- **`--xla_tpu_*` flags go in `LIBTPU_INIT_ARGS`, not `XLA_FLAGS`.** XLA_FLAGS errors with `Unknown flags` otherwise. (Exp 1 gotcha.)
- **Flag-catalog drift vs installed runtime**: a named flag in a doc may not be in the installed libtpu. Dump `strings libtpu.so | grep xla_tpu_<token>` as source of truth. (Exp 1 gotcha — `_overlap_compute_collective_comms` does not exist; real name is `_tc`.)
- **Tied-weight dedup in torchax `JittableModule` complicates `out_shardings`** (exp 2 lesson). Building a sharding pytree from `weights.items()` or `jax.tree.map(.sharding, ...)` for tied weights (`lm_head.weight` ↔ `embed_tokens.weight` in Llama 3) yields a collapsed single-device sharding for the duplicate; `jit` rejects the mismatch. Skip the explicit pin, or route around the dedup.
- **Measurement**: use median step-time over steps 6–15 with `--steps 20`. Steps 0 and 1 are both compiles (~150 s each; the step-1 recompile is open). Short runs (< 10 steady-state steps) undercount slow outliers.
- **Bytes-accessed is the diagnostic metric** for reorder-driven regressions. Wall-clock alone was ambiguous in exp 1; the HLO-op bytes_accessed blowup (2.5× on convolution fusion) was decisive evidence for the mechanism.
- **Splash attention supports GQA natively** (from sibling-wiki heuristic 10 and `2026-tokamax-splash-attention.md`). With `num_kv_heads=8` and `num_heads=8`, don't pre-split K/V with `jnp.repeat` — it wastes ~4× K/V memory and HBM bandwidth.
- **HBM ratchet**: once at 95%+ HBM, every memory-saving change unlocks a potential throughput change (bigger batch, longer seq, less remat). Track peak HBM per experiment; when it drops, immediately queue the now-tryable state-growing change as a follow-up.
- **HF checkpoint is multimodal-only**. `Llama4ForCausalLM.from_pretrained(meta-llama/Meta-Llama-3-8B)` silently re-inits all weights (key-name mismatch). Load `Llama4ForConditionalGeneration`; monkey-patch `.forward` to skip the multimodal orchestrator (boolean `input_ids[mask] = pad` is NonConcreteBooleanIndexError under jit). Apply `final_logit_softcapping=30.0` in the patched forward. (Baseline lesson.)
- **Compile time dominates short runs**. Step 0 compile ~150 s, step 1 recompile ~150 s. `scan-over-layers` (via `torchax.train.ScannedModule`) should collapse 42-layer unroll to one scan-body compile. Planned follow-up.

## Reference material at a glance

- Bottleneck diagnosis: [xprof — overview page](../../sources/2026-xprof-overview-page.md), [xprof — roofline model](../../sources/2026-xprof-roofline-model.md), [xprof — HLO op stats](../../sources/2026-xprof-hlo-op-stats.md), [xprof — HLO op profile](../../sources/2026-xprof-hlo-op-profile.md), [xprof — memory profile](../../sources/2026-xprof-memory-profile.md), [xprof — trace viewer](../../sources/2026-xprof-trace-viewer.md).
- Crown jewel: [xprof-mcp — TPU optimization guide](../../sources/2026-xprof-mcp-tpu-optimization.md) — roofline, dimension alignment, dtype, fusion, remat, KV cache, decision trees, XLA flag catalog.
- Attention: [flash-attention](../../concepts/flash-attention.md), [splash-attention](../../concepts/splash-attention.md), [attention-block-sizes](../../concepts/attention-block-sizes.md), [tokamax — splash attention](../../sources/2026-tokamax-splash-attention.md), [tokamax — autotuning](../../sources/2026-tokamax-autotuning.md).
- Sharding / parallelism: [fsdp](../../concepts/fsdp.md), [tensor-parallelism](../../concepts/tensor-parallelism.md), [sharding](../../concepts/sharding.md), [collective-communication](../../concepts/collective-communication.md), [reduce-scatter](../../concepts/reduce-scatter.md), [all-gather](../../concepts/all-gather.md), [ici](../../concepts/ici.md).
- Remat / compile: [rematerialization](../../concepts/rematerialization.md), [scan-over-layers](../../concepts/scan-over-layers.md), [xla-fusion](../../concepts/xla-fusion.md), [hlo-dumping-and-diffing](../../concepts/hlo-dumping-and-diffing.md), [xla-flags](../../concepts/xla-flags.md), [async-collectives](../../concepts/async-collectives.md), [latency-hiding-scheduler](../../concepts/latency-hiding-scheduler.md), [custom-call](../../concepts/custom-call.md), [pallas-kernel](../../concepts/pallas-kernel.md).
- Kernels library: [tokamax codebase](../../codebases/tokamax.md), [tokamax — supported ops](../../sources/2026-tokamax-supported-ops.md), [tokamax — basic usage](../../sources/2026-tokamax-basic-usage.md).
- Runtime: [torchax codebase](../../codebases/torchax.md) — `JittableModule`, `interop.jax_view`, `jax_shard_map`, `ScannedModule`.
- Prior art: [jax-huggingface codebase](../../codebases/jax-huggingface.md) + its four source pages (the `2026-jax-huggingface-part-{1,2,3,4}.md` series).
- MCP tools: [xprof-mcp codebase](../../codebases/xprof-mcp.md) — 18 tools.
- Methodology root: [autoresearch codebase](../../codebases/autoresearch.md).
