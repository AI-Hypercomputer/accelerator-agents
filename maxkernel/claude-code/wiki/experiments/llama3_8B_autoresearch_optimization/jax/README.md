# Native-JAX (Flax NNX) Llama 3 8B trainer

Companion folder to [`../torchax/`](../torchax/README.md) — a **pure-JAX** port of the same Llama 3 8B training pipeline. No torchax, no PyTorch at run time. Uses Flax NNX for module construction, `optax.adamw` for the optimizer, and the same splash / tokamax / scan-over-layers kernel stack the torchax sibling validated.

The trainer mirrors the torchax sibling **flag-for-flag** so the two stacks can be A/B'd directly. All design decisions about kernel choice, remat policy, batch/seq sweet-spot, and CE impl are inherited from the torchax experiment series — see [`../program.md`](../program.md) and the [torchax experiments index](../torchax/experiments/README.md).

## Final status (2026-04-27)

🏆 **Frontier**: ~7,700 tok/s/chip / 43.3 % MFU on v6e-8 at `bs=4 seq=8192 fsdp=8` (mean of 3 reruns; peak run 7,768/43.6 %). vs MaxText `tpu-recipes-v0.1.4` reference (7,069 / 44.6 %): **+8.9 % per-chip throughput**. The reported 1.0 pp MFU gap is FLOP-counter normalization — under MaxText's accounting we measure 49.0 % MFU on the same throughput, **+4.4 pp above** their reported number.

**Cumulative climb 2026-04-26**: torchax exp 20 baseline 4,591 → JAX exp 28b 7,768 = **+69.2 % per-chip**.

**The frontier stack** (image `precast-1`):

| Layer | Choice | Source |
|-------|--------|--------|
| Compute | bf16 with fp32 master weights (AMP) + adamw mu/nu fp32 | torchax exp 20 |
| Sharding | FSDP=8, TP=1, scan-over-layers (`nothing_saveable` remat) | torchax exp 40 |
| Attention | `tokamax._src.ops.experimental.tpu.splash_attention` with `use_base2_exp=1`, `fuse_reciprocal=1`, `max_logit_const=30`; `bq=2048 bkv=1024` fwd, `bq_dkv=2048 bkv_dkv=2048` bwd, `fused_bwd=True` | torchax exp 72a + JAX kernel-tune |
| Cross-entropy | `tokamax.linear_softmax_cross_entropy_loss` impl `chunked_xla` with autotune | torchax exp 62b |
| XLA flags | Full MaxText flag stack: `HOST_OFFLOAD_FLAGS` (12 scheduler flags) + `DISABLE_COLLECTIVE_MATMUL` + `LAYOUT_FOR_ALL_REDUCE_SCATTER` + `DATA_PARALLEL_OVERLAP` + `CF_FOR_ALL_GATHER` recipe flags | MaxText `xla_flags_library.py` |
| SparseCore offload | All three FSDP collectives offloaded (`xla_tpu_enable_sparse_core_collective_offload_{all_reduce,reduce_scatter,all_gather}=true`) | JAX exp 27/28b |
| VMEM limit | `--xla_tpu_scoped_vmem_limit_kib=98304` | MaxText DENSE_VMEM_LIMIT |
| Density | `bs=4 seq=8192` | exp 28b density sweep |

**Headline profile breakdown (exp 28b)**: matmul 60.1 % / MXU util 65.8 %, splash custom-call 25.5 %, loop fusion 9.2 %, async-done 1.4 %, all-reduce 0.3 %, **TC idle 0.014 % (saturated)**.

**Loss-validation** (100 steps × 3 stack configurations at identical RNG seed): max |Δ| = **0.0003** between full optimized stack and minimal-flags reference, median Δ = 0.0000 — bf16 precision floor. **Optimization stack delivers +19.9 % per-chip throughput at literally identical loss curve.** Details: [exp 65/66/67](experiments/2026-04-27-jax-exp65-67-loss-validation-100steps.md).

### What was tried — 2026-04-26/27 ablation summary (40 experiments)

| Wave | Experiments | Outcome |
|------|-------------|---------|
| Frontier discovery | exp 1e-13 (port), exp 12-18 (MaxText XLA flags + bkv match) | 6,529 → 7,471/chip |
| 🏆 Frontier advance | **exp 27/28b (SparseCore offload of RS+AG)** | 7,471 → **7,768/chip** |
| Density/kernel/VMEM sweep | exp 29-49 (VMEM 64k/80k/128k, bkv 512/2048, bq 4096, scan unroll, bs=3/5/6, etc.) | All refuted (within ±1 % noise or worse) |
| XLA flag flips | exp 50-60 (collective matmul ENABLE, bundle-aware off, async permute, megacore-fusion-ag, overlap-compute-collective off, latency-hiding rerun=0, loop-invariant chain off, …) | All refuted |
| Code change | exp 54/55: pre-cast bf16 weights once per train_step | -0.5 to -1.1 % (XLA already fuses cast in matmul prologue) |
| Named-remat policies | exp 35/36/37/38: `save_qkv_proj`, `qkv_proj_offloaded` at bs=4/5/6 | All refuted (compile OOM or PCIe latency > savings) |
| Loss validation | exp 65/66/67/68/69 (100 steps × 3 configs + 2 LR-matched MaxText comparison) | Bit-equivalent (max \|Δ\|=0.0003) |
| HLO inspection | hypotheses #2 + #3 refutation via `XLA_FLAGS=--xla_dump_to=...` | XLA already emits both fusions as `kind=kOutput` Mosaic kernels |

**Bf16-MXU regime is empirically saturated.** No surviving knob, flag, or kernel-replacement hypothesis produces a measurable win on top of exp 28b at this hardware/dtype/shape.

### Open hypotheses (1 remaining)

- [int8 weight-only quantization (AQT/qwix)](../../../hypotheses/llama3-jax-int8-weight-quantization.md) — expected +15-30 % step time. **Only viable per-chip throughput lever after the bf16-MXU regime saturated.** Would shift the 65.8 % bf16-MXU utilization onto the 2× int8-MXU path; weights drop fp32→int8 (4× smaller HBM footprint), activations stay bf16 in stage 1.

### Refuted hypotheses (validated cheaply via HLO before kernel-write)

- ~~Pallas RMSNorm + matmul-prologue fusion~~ — XLA already inlines RMSNorm into each matmul's `kind=kOutput` Mosaic kernel. See [hypothesis page](../../../hypotheses/llama3-jax-rmsnorm-matmul-prologue-fusion.md) for HLO refs (`fused_computation.47` for Q-proj inlines `fused_computation.25`'s RMSNorm body).
- ~~Pallas SwiGLU + down_proj fusion~~ — XLA already fuses silu+mult+down_proj+residual into a single `kind=kOutput` kernel (`fused_computation.40` containing `fused_computation.8`'s silu*u body, `convolution.111`, and the residual add). See [hypothesis page](../../../hypotheses/llama3-jax-pallas-swiglu-downproj-fusion.md). HLO evidence preserved at [`raw/profiles/2026-04-27-jax-hlodump-exp28b/`](../../../../raw/profiles/2026-04-27-jax-hlodump-exp28b/).

## Layout

```
jax/
  README.md                this file
  train.py                 native-JAX trainer (fire.Fire)
  data.py                  wikitext-2-raw-v1 packer + fake_dataloader
  splash_attn.py           tokamax-splash dispatch (verbatim from ../torchax/)
  Dockerfile               jax-ai-image base + transformers + flax + tokamax
  model/
    __init__.py            re-exports
    modeling_llama3.py     Flax NNX port of LlamaForCausalLM(+Scan)
    sharding.py            FSDP / TP plans + mesh helpers; SCAN_SHARDING_PLAN
    weight_loader.py       HF safetensors -> NNX param tree (sharded device_put)
  experiments/
    README.md              experiment index for this stack
```

## Architectural decisions

- **Param naming matches HF dot-for-dot.** `model.embed_tokens.weight`, `model.layers.{i}.self_attn.{q,k,v,o}_proj.weight`, etc. so the weight loader is a 1:1 mapping with no transpose layer.
- **Linear weight shape is `(out, in)`** (matches `torch.nn.Linear`). Forward computes `x @ weight.T`.
- **RoPE precomputed once** at `LlamaRotaryEmbedding.__init__` (single layer-type — much simpler than Gemma 4's hybrid sliding/full).
- **Attention dispatch via env var** `JAX_ATTENTION_IMPL=splash|xla`. The trainer sets this based on `--use_splash` and registers the mesh with `set_splash_mesh(mesh)` so the kernel's `shard_map` has a concrete `Mesh` (Mosaic custom calls cannot be auto-partitioned).
- **Scan-over-layers** is implemented by stacking each per-layer Linear/RMSNorm weight along a new leading dim (size `num_hidden_layers`). The forward calls `jax.lax.scan` over a functional decoder body that indexes the stacked params layer-by-layer. Sharding plan in `model/sharding.py` (`SCAN_SHARDING_PLAN`) prepends a `None` to every per-layer PartitionSpec so the stack dim stays unsharded.
- **Tokamax CE goes through `shard_map`** with `psum` across `fsdp` (matches the torchax wrapper exactly). `chunked_xla` requires fp32 inputs (validated by torchax exp 62b vs invalid exp 66 — bf16 destroys precision in the lse accumulator).
- **MFU formula** is the MaxText train-step TFLOPs (×3 for fwd + 2× bwd). Identical lines to torchax `train.py` lines 596-632.

## Running

The trainer expects to be invoked with `python -m train` from this folder so `model/` and `data.py` resolve as siblings on `sys.path`.

```bash
WIKI_ROOT="/mnt/disks/persist/torch-tpu/tpu_performance_autoresearch_wiki"
cd "$WIKI_ROOT/wiki/experiments/llama3_8B_autoresearch_optimization/jax"

# canonical: bs=3 seq=8192 fp32-master + bf16-compute, scan + tokamax-CE
python -m train \
    --use_real_data=True \
    --use_splash=True \
    --use_scan=True \
    --use_tokamax_ce=True \
    --tokamax_ce_impl=chunked_xla \
    --tokamax_ce_autotune=True \
    --weights_dtype=fp32 \
    --compute_dtype=bf16 \
    --master_dtype=fp32 \
    --batch_size=3 \
    --seqlen=8192 \
    --train_steps=15
```

Add `--profile_dir=$PROFILE_DIR --profile_step=5` to capture an xprof trace at step 5.

## Flags

Same set as the torchax sibling. Key flags only:

| Flag | Default | Notes |
|---|---|---|
| `--model_id` | `meta-llama/Meta-Llama-3-8B` | HF repo. |
| `--batch_size` | `4` | Per-chip. Global = `batch_size × fsdp`. Sweet-spot at this shape: `bs=3 seq=8192` (torchax exp set). |
| `--seqlen` | `1024` | Smoke-test default; production runs use 8192. |
| `--train_steps` | `15` | Step 0 compiles; steps 0-1 dropped from MFU. |
| `--tp_parallelism` | `1` | 1 = pure FSDP across all chips; >1 = 2D mesh. |
| `--weights_dtype` | `bf16` | `bf16` or `fp32` (master). |
| `--compute_dtype` | `match` | `match` inherits `weights_dtype`; `bf16` with `weights_dtype=fp32` → AMP. |
| `--master_dtype` | `match` | `fp32` forces optimizer mu/nu to fp32 even when weights are bf16. |
| `--use_splash` | `False` | True = splash kernel via env-gated dispatch. Trainer registers the mesh. |
| `--use_scan` | `False` | True = `LlamaForCausalLMScan` (32 layers stacked into one scan body). |
| `--scan_remat_policy` | `nothing_saveable` | `jax.checkpoint_policies` attribute name OR a MaxText named-policy: `save_qkv_proj`, `save_out_proj`, `save_dot_except_mlp`, `qkv_proj_offloaded`, `minimal_offloaded` (resolved by `_resolve_scan_policy` in `train.py`). |
| `--use_tokamax_ce` | `False` | True = `tokamax.linear_softmax_cross_entropy_loss` (skip lm_head materialization). |
| `--tokamax_ce_impl` | `mosaic_tpu` | `chunked_xla` is the validated faster path (exp 62b). |
| `--tokamax_ce_autotune` | `False` | True = autotune CE block sizes. |
| `--profile_dir` / `--profile_step` | unset | xprof capture at the given step. |
| `--learning_rate` | `1e-5` | optax.adamw peak LR; constant schedule (no warmup/cosine in this trainer). |
| `--use_real_data` | `True` | True = wikitext-2-raw-v1; False = `data.fake_dataloader` (fresh-random tokens, model can't memorize — used for 100-step loss-validation runs). |

Env-var knobs (also passable via the `xpk` `--env=...` flag in `/tmp/llama3_run/xpk/exp_jax_maxtext_flags.sh`):

| Env | Default | Notes |
|---|---|---|
| `JAX_ATTENTION_IMPL` | `splash` | `splash` or `xla` (SDPA fallback). |
| `USE_TOKAMAX_SPLASH` | `1` | 1 = tokamax-shipped splash (with the perf knobs below); 0 = jax-experimental splash (reference impl, no perf knobs). |
| `TOKAMAX_USE_BASE2_EXP` | `1` | Rewrite `exp(x)` as `2^(x*log2_e)` in tokamax-splash softmax. |
| `TOKAMAX_FUSE_RECIPROCAL` | `1` | Fuse the softmax reciprocal into the inner matmul. |
| `TOKAMAX_MAX_LOGIT_CONST` | `30` | Fixed max-logit estimate (avoids the per-block reduction). |
| `SPLASH_BQ` / `SPLASH_BKV` / `SPLASH_BKV_COMPUTE` | `2048` / `1024` / `1024` | Forward splash block sizes. |
| `SPLASH_BQ_DKV` / `SPLASH_BKV_DKV` / `SPLASH_BKV_DKV_COMPUTE` | `2048` / `2048` / `2048` | Backward splash block sizes. |
| `SPLASH_FUSED_BWD` | `1` | Use the fused-bwd splash kernel. |
| `JAX_SCAN_UNROLL` | `1` | scan unroll factor (refuted: 2 was -2.8 % at bs=4). |
| `JAX_PRECAST_BF16_WEIGHTS` | `0` | Pre-cast all weights to compute_dtype outside the scan loop (refuted: -0.5 to -1.1 % — XLA already fuses cast). |
| `XLA_FLAGS` | unset | extra XLA flags for HLO dump etc. (the LIBTPU stack is set separately by the launcher). |

## See also

- [`../program.md`](../program.md) — shared experiment protocol.
- [`../torchax/`](../torchax/) — sibling stack (primary, validated).
- [`experiments/`](experiments/) — JAX-stack experiments index.
- [`experiments/2026-04-26-jax-exp27-28-sparsecore-rs-ag-offload-frontier.md`](experiments/2026-04-26-jax-exp27-28-sparsecore-rs-ag-offload-frontier.md) — the 7,768/chip frontier writeup with profile breakdown, density sweep, and exp 29-60 ablation table.
- [`experiments/2026-04-27-jax-exp65-67-loss-validation-100steps.md`](experiments/2026-04-27-jax-exp65-67-loss-validation-100steps.md) — 100-step loss-curve validation; max |Δ| = 0.0003 across 3 stack configs.
- [`../../../hypotheses/`](../../../hypotheses/) — open and refuted hypothesis pages (1 open: int8/AQT; 2 refuted via HLO inspection: Pallas RMSNorm+matmul, Pallas SwiGLU+down_proj).
