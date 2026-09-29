---
title: "Qwen3 8B — jax"
type: model
architecture: qwen3-cx
lane: jax
sizes: ["8B"]
hardwares: ["v6e-8"]
tags: [qwen3-cx, jax]
created: 2026-06-02
updated: 2026-06-12
---

# Qwen3 8B — jax

Qwen3 8B (`Qwen/Qwen3-8B`) as a native-JAX (Flax NNX) port on **TPU v6e-8**.
The lane is live: the from-scratch Flax NNX trainer has a baseline and a
Splash+tokamax/remat frontier run for cross-lane comparison against
[torchax](qwen3-cx-torchax.md).

## Target metrics

- Primary: MaxText-style MFU at seq_len=8192 (v6e bf16 peak ≈ 918 TFLOPS/chip).
  As of 2026-06-12, Qwen3 native-JAX/torchax reporting matches the MaxText
  recipe image by counting attention as the full non-causal QK/AV matmul term;
  execution remains causal.
- Secondary: tokens/sec (and tok/s/chip), step time, peak HBM.

## How to run

```bash
cd wiki/experiments/qwen3_cx_autoresearch_optimization/jax
python -u train.py --model_id=Qwen/Qwen3-8B --use_real_data=False \
  --batch_size=1 --seqlen=2048 --tp_parallelism=1 \
  --train_steps=20 --weights_dtype=bf16
```

## Variant matrix

| Size | Hardware | Status | Baseline (step / TPS / MFU) | Current best (step / TPS / MFU) | Open hyps | Frontier exp |
|------|----------|--------|-----------------------------|----------------------------------|-----------|--------------|
| 8B | v6e-8 | live | 512 ms / 31,955 TPS / 20.5% pre-normalization MFU @ seq2048 bs8 | 4.36 s / 60,346 TPS confirmed, 60,351 TPS provisional peer, 7,543-7,544 tok/s/chip / 43.3% old MFU, about 49.2% MaxText-style MFU @ seq8192 bs32 no-scan+remat+tokamax-Splash+maxlogit30+MaxTextCE+shard-acts except no MLP-intermediate `_sac`+rerun3+vmem100352+customcall-vmem-adjust+selective-resources+collective-matmul-none+layeragg-false+all-experimental-scheduler-false+SparseCore offload for RS/AR+aggregator with AG on async collective fusion+MLP up-proj first+RS latency multiplier 3+AR latency multiplier 2; confirmed HLO/profile; peak HBM 30.93/31.25 GiB with 0.318 GiB free | 2 | [2026-06-05 v273 MLP up-proj first + RS3+AR2 + selective resources rerun](../experiments/qwen3_cx_autoresearch_optimization/jax/experiments/2026-06-05-v273-no-scan-sparsecore-rs-ar-rs3-ar2-mlp-up-proj-first-selective-resources-rerun.md) |

*Baseline captured at seq 2048 (global batch 8). **Cross-lane: jax 20.5% MFU /
3,994 tok/s/chip beats torchax 19.2% / 3,724 (+7.3% tok/s/chip, +1.3 pp)** at the
identical shape — the native-JAX path has lower collective/dispatch overhead
(profile: 21.7% collective vs torchax 31.3%). See the
[jax baseline](../experiments/qwen3_cx_autoresearch_optimization/jax/experiments/2026-06-02-qwen3-jax-v6e8-baseline.md)
and [torchax baseline](../experiments/qwen3_cx_autoresearch_optimization/torchax/experiments/2026-06-02-qwen3-torchax-v6e8-baseline.md).*

Latest checked runtime isolate: [v261](../experiments/qwen3_cx_autoresearch_optimization/jax/experiments/2026-06-05-v261-no-scan-sparsecore-rs-ar-rs3-ar2-mlp-up-proj-first-no-offload-tracing.md)
disabled active SparseCore RS/AR offload tracing on the v260 frontier. It kept
the optimized HLO byte-identical and reduced profile artifacts about **6.4x**,
but throughput and XProf step time did not improve, so it is not carried as a
performance lever. [v262](../experiments/qwen3_cx_autoresearch_optimization/jax/experiments/2026-06-05-v262-no-scan-sparsecore-rs-ar-rs3-mlp-up-proj-first-ar-default.md)
then removed only AR latency multiplier `2`; it selected a distinct HLO but
only tied throughput and worsened XProf step time to **4391.2 ms**, so AR2
remains carried. [v263](../experiments/qwen3_cx_autoresearch_optimization/jax/experiments/2026-06-05-v263-no-scan-sparsecore-rs-ar-rs4-ar2-mlp-up-proj-first.md)
then raised only RS latency multiplier from `3` to `4` and produced a
provisional high-water mark (**59,947 tok/s**, **4387.5 ms** XProf).
[v264](../experiments/qwen3_cx_autoresearch_optimization/jax/experiments/2026-06-05-v264-no-scan-sparsecore-rs-ar-rs4-ar2-mlp-up-proj-first-rerun.md)
exactly reproduced the v263 HLO and improved the best full-window summary to
**59,954 tok/s / 43.0% MFU** with **4387.7 ms / 67.9% MXU** in XProf, so RS4
+ AR2 is now the confirmed frontier.
[v265](../experiments/qwen3_cx_autoresearch_optimization/jax/experiments/2026-06-05-v265-no-scan-sparsecore-rs-ar-rs5-ar2-mlp-up-proj-first.md)
then raised only RS latency multiplier to `5`; it selected a distinct HLO and
tied throughput/profile at **59,955 tok/s / 4387.6 ms** while raising peak HBM
to **31.0 GiB**, so do not carry RS5.
[v266](../experiments/qwen3_cx_autoresearch_optimization/jax/experiments/2026-06-05-v266-no-scan-sparsecore-rs-ar-rs4-mlp-up-proj-first-ar-default.md)
then kept RS4 but returned AR latency to default. It matched v264 aggregate HLO
counts but regressed to **59,942 tok/s / 4388.5 ms**, so keep AR2. The immediate
RS/AR scalar-latency neighborhood is now bracketed around RS4+AR2.
[v267](../experiments/qwen3_cx_autoresearch_optimization/jax/experiments/2026-06-05-v267-no-scan-sparsecore-rs-ar-rs4-ar2-mlp-up-proj-first-layer-scheduler.md)
then enabled the latency-hiding layer scheduler on top of the SparseCore
aggregator. It failed compile at **66.54 GiB / 31.25 GiB** HBM, so keep the
layer scheduler disabled.
[v268](../experiments/qwen3_cx_autoresearch_optimization/jax/experiments/2026-06-05-v268-no-scan-sparsecore-rs-ar-rs4-ar2-mlp-up-proj-first-selective-resources.md)
then forced selective resources on after v264's flagfile showed selective
resources were disabled. It selected a new HLO
(`a92469ed51218d7814d863f5e78b678967b81a836d9a558a1d9c966cc4a206e0`) and
raised worker1 throughput to **60,336 tok/s / 43.3% MFU** with **4359.7 ms /
68.2% MXU** in XProf. Treat this as a provisional high-water mark only:
peak HBM is **31.16 GiB / 31.25 GiB**, so an exact rerun is required before
replacing v264 as the confirmed frontier.
[v269](../experiments/qwen3_cx_autoresearch_optimization/jax/experiments/2026-06-05-v269-no-scan-sparsecore-rs-ar-rs4-ar2-mlp-up-proj-first-selective-resources-rerun.md)
exactly reran v268 and reproduced the HLO byte-for-byte with **60,338 tok/s /
43.3% MFU** on worker1, **60,318 tok/s / 43.2% MFU** on worker0, and **4359.6
ms / 68.2% MXU** in XProf. Carry selective resources as part of the confirmed
frontier. The caveat is memory pressure: XProf reports **31.16 GiB / 31.25
GiB** peak HBM with only **0.0896 GiB** free, so future probes should avoid
mechanisms that increase live memory and can test small scoped-VMEM downshifts.
[v270](../experiments/qwen3_cx_autoresearch_optimization/jax/experiments/2026-06-05-v270-no-scan-sparsecore-rs-ar-rs4-ar2-mlp-up-proj-first-selective-resources-vmem99840.md)
then lowered only scoped VMEM from `100352` to `99840` KiB. It recovered only
**0.0274 GiB** of extra free HBM (**0.117 GiB** total free) while regressing to
**60,116 tok/s / 43.1% MFU** and **4374.8 ms / 67.9% MXU** in XProf, with a
changed HLO. Keep scoped VMEM `100352` on the confirmed v269 frontier.
[v271](../experiments/qwen3_cx_autoresearch_optimization/jax/experiments/2026-06-05-v271-no-scan-sparsecore-rs-ar-rs5-ar2-mlp-up-proj-first-selective-resources.md)
then raised only RS latency multiplier from `4` to `5` under selective
resources. It selected a distinct HLO
(`a88aac126db1ab1d3d04db6f74b572b2f51091637084b82d470246003e9a7d8a`) but failed
before step 0 while loading `jit_train_step`: XLA attempted an additional
**2.12 GiB** allocation with only **995.89 MiB** free. No XProf was produced.
Close RS5 under selective resources and keep RS4; the next bracket probe is
the low side (`RS3+AR2` with selective resources).
[v272](../experiments/qwen3_cx_autoresearch_optimization/jax/experiments/2026-06-05-v272-no-scan-sparsecore-rs-ar-rs3-ar2-mlp-up-proj-first-selective-resources.md)
then tested that low side. It completed cleanly with **60,351 tok/s / 43.3%
MFU** on worker1 and **60,318 tok/s / 43.2% MFU** on worker0, with **4358.2 ms
/ 68.1% MXU** in XProf. The HLO changed
(`81e3c3c58b3fbfa12c0b203bdc9d0df428913fad3462ea5552e36cd07dcd5a96`) and
peak HBM dropped to **30.93 GiB / 31.25 GiB**, leaving **0.3181 GiB** free.
Treat as a provisional memory-headroom tie, not yet a confirmed frontier
replacement; exact rerun before carrying RS3+AR2 over RS4+AR2.
[v273](../experiments/qwen3_cx_autoresearch_optimization/jax/experiments/2026-06-05-v273-no-scan-sparsecore-rs-ar-rs3-ar2-mlp-up-proj-first-selective-resources-rerun.md)
exactly reran v272 and reproduced the HLO byte-for-byte with **60,346 tok/s /
43.3% MFU** on worker0, **60,322 tok/s / 43.3% MFU** on worker1, and **4358.1
ms / 68.2% MXU** in XProf. Carry RS3+AR2 selective resources as the confirmed
safer frontier: the speed gain over v269 is noise-level, but free HBM improves
from **0.0896 GiB** to **0.3181 GiB** while preserving the same MFU band.
[v274](../experiments/qwen3_cx_autoresearch_optimization/jax/experiments/2026-06-05-v274-no-scan-sparsecore-rs-ar-rs3-mlp-up-proj-first-selective-resources-ar-default.md)
then removed only AR latency multiplier `2` on the confirmed RS3 selective
base. It selected a distinct same-size HLO
(`361b10fd3f580e2829ae2a660ad81c4cf38bc62bfbcf0da033214495b2a5c44e`) and kept
the same **0.3181 GiB** free-HBM profile, but regressed to **60,334 tok/s /
4359.4 ms**. Keep AR2.
[v275](../experiments/qwen3_cx_autoresearch_optimization/jax/experiments/2026-06-05-v275-no-scan-sparsecore-rs-ar-rs3-ar2-mlp-up-proj-first-selective-resources-vmem100864.md)
then raised only scoped VMEM from `100352` to `100864` KiB on the v273 base.
It selected a distinct HLO
(`6e82cf40ff59806506583187b0d32c3242ae906e8aa6a6f49a2e725aa2c747a6`) with the
same main collective/matmul/custom-call counts and unchanged **0.3181 GiB**
free HBM, but regressed to **60,311 tok/s / 43.2% MFU** and **4360.2 ms**.
Keep scoped VMEM `100352`.
[v276](../experiments/qwen3_cx_autoresearch_optimization/jax/experiments/2026-06-05-v276-no-scan-sparsecore-rs-ar-rs3-ar2-mlp-up-proj-first-selective-resources-shmem95.md)
then lowered only scheduler shared-memory limit from `100` to `95`. The HLO
was byte-identical to v273
(`81e3c3c58b3fbfa12c0b203bdc9d0df428913fad3462ea5552e36cd07dcd5a96`) and HBM
was unchanged, but throughput/profile did not improve (**60,344 tok/s** best
worker average, **4358.5 ms**). Keep shared-memory limit `100`.
[v277](../experiments/qwen3_cx_autoresearch_optimization/jax/experiments/2026-06-05-v277-no-scan-sparsecore-rs-ar-rs3-ar2-mlp-up-proj-first-selective-resources-shmem90.md)
then lowered shared-memory limit further to `90`. It kept the same optimized
train-step text as v273 but changed the memory profile substantially:
**29.03 GiB / 31.25 GiB** peak with **2.2202 GiB** free. Runtime regressed to
**60,294 tok/s / 43.2% MFU** and **4361.9 ms**, so do not carry the lower-HBM
schedule.
[v278](../experiments/qwen3_cx_autoresearch_optimization/jax/experiments/2026-06-05-v278-no-scan-sparsecore-rs-ar-rs3-ar2-mlp-up-proj-first-selective-resources-bs5-shmem90.md)
then tried to spend v277's HBM headroom on per-chip batch 5/global batch 40.
It failed during `jit_train_step` compile with **33.10 GiB / 31.25 GiB** HBM
used, exceeding capacity by **1.86 GiB**. The largest temp remained the
full-vocabulary logits allocation `bf16[5,8192,151936]` at **11.59 GiB**.
The emitted optimized HLO has SHA256
`ab88cab233d7fabd595e00652e22e66f18f26c37a849847de761e41db7e85d9a`
(17,671,203 bytes / 106,145 lines, `tokamax` count 2). This closes direct
batch scaling for the current MaxText-CE path; a streaming/fused lm-head+CE
mechanism is needed before retrying batch 5.
[v279](../experiments/qwen3_cx_autoresearch_optimization/jax/experiments/2026-06-05-v279-no-scan-sparsecore-rs-ar-rs3-ar2-mlp-up-proj-first-selective-resources-tokamax-ce-bs5.md)
then tested the existing Tokamax Mosaic TPU streaming CE path on top of the
v273 RS3+AR2 selective-resources stack with per-chip batch 5/global batch 40.
It completed cleanly and fit comfortably (**29.50 GiB / 31.25 GiB** peak HBM,
**1.7418 GiB** free), but regressed to **56,318 tok/s / 40.4% MFU** and
**5829.3 ms / 64.4% MXU** in XProf. The HLO confirms Tokamax CE custom calls
(`f205e693bc93ac8db8e11c71199fbe2b562d62cdebe6d3c6410cf9924ebe77bb`,
19,922,224 bytes / 119,222 lines). Do not carry Tokamax CE: existing
streaming CE avoids the logits OOM but is too slow to challenge the v273
MaxText-CE batch-4 frontier.

Current long-sequence frontier update: [v205](../experiments/qwen3_cx_autoresearch_optimization/jax/experiments/2026-06-05-v205-no-scan-sparsecore-all-collectives.md)
adds SparseCore offload for all-gather, reduce-scatter, and all-reduce to the
v183 no-scan stack. [v206](../experiments/qwen3_cx_autoresearch_optimization/jax/experiments/2026-06-05-v206-no-scan-sparsecore-all-collectives-rerun.md)
exactly reran it and confirmed **59,351 tok/s / 42.6% MFU** at seq8192, global
batch 32. [v208](../experiments/qwen3_cx_autoresearch_optimization/jax/experiments/2026-06-05-v208-no-scan-sparsecore-aggregator-rerun.md)
enabled the SparseCore collective aggregator and marginally raised the
best-worker average to **59,372 tok/s / 42.6% MFU** with byte-identical HLO.
[v210](../experiments/qwen3_cx_autoresearch_optimization/jax/experiments/2026-06-05-v210-no-scan-sparsecore-aggregator-rerun3.md)
reproduced the band at **59,370 tok/s / 42.6% MFU**, so the aggregator flag is
now carried as a marginal confirmed runtime scheduling win.
[v211](../experiments/qwen3_cx_autoresearch_optimization/jax/experiments/2026-06-05-v211-no-scan-sparsecore-concurrent-offload.md)
tested concurrent SparseCore offloading on top of v210. It changed the HLO,
raised compiled memory slightly to 27.68 GiB, and regressed to
**59,268 tok/s / 42.5% MFU**, so do not carry
`--xla_tpu_enable_concurrent_sparse_core_offloading=true`.
[v218](../experiments/qwen3_cx_autoresearch_optimization/jax/experiments/2026-06-05-v218-no-scan-sparsecore-ag-rs-only.md)
then isolated all-reduce offload by keeping all-gather/reduce-scatter on
SparseCore but returning all-reduce to async collective fusion. It completed
cleanly and reduced the visible XProf async-done bucket, but still regressed to
**59,359 tok/s / 42.6% MFU** with a changed HLO, so keep all-reduce SparseCore
offload enabled.
[v219](../experiments/qwen3_cx_autoresearch_optimization/jax/experiments/2026-06-05-v219-no-scan-sparsecore-ag-ar-only.md)
then isolated reduce-scatter offload by keeping all-gather/all-reduce on
SparseCore but returning reduce-scatter to async collective fusion. It regressed
hard to **58,269 tok/s / 41.8% MFU** with lower MXU, so keep reduce-scatter
SparseCore offload enabled too.
[v220](../experiments/qwen3_cx_autoresearch_optimization/jax/experiments/2026-06-05-v220-no-scan-sparsecore-rs-ar-only.md)
then isolated all-gather offload by keeping reduce-scatter/all-reduce on
SparseCore but returning all-gather to async collective fusion. It produced a
provisional new best of **59,441 tok/s / 7,430 tok/s/chip / 42.7% MFU** with
XProf step time **4428.4 ms** and MXU **67.5%**, versus v210's indexed
4433.3 ms / 67.4%. Because the margin over v208/v210 is small, require an
exact rerun before carrying this as confirmed.
[v221](../experiments/qwen3_cx_autoresearch_optimization/jax/experiments/2026-06-05-v221-no-scan-sparsecore-rs-ar-only-rerun.md)
exactly reran v220 and confirmed the same operating band: worker0 reached
**59,435 tok/s** over steps 2-19 and **59,478 tok/s** over steps 15-19; worker1
reached **59,406 tok/s** over steps 2-19 and **59,478 tok/s** over steps 15-19,
with a slow traced step depressing the full-window average. XProf remained
matched at **4429.0 ms** step time, **67.4% MXU**, and **30.93 GiB** peak HBM,
and the optimized train-step HLO was byte-identical to v220. Carry the
v220/v221 collective split as confirmed; best single measured run remains
v220's **59,441 tok/s / 42.7% MFU**.
[v222](../experiments/qwen3_cx_autoresearch_optimization/jax/experiments/2026-06-05-v222-no-scan-sparsecore-rs-ar-no-aggregator.md)
then disabled only the SparseCore collective aggregator on top of v220/v221. It
kept byte-identical HLO but regressed to **59,246 tok/s / 42.5% MFU** with
XProf step time **4439.1 ms** and MXU **66.8%**. Keep the aggregator enabled.
[v223](../experiments/qwen3_cx_autoresearch_optimization/jax/experiments/2026-06-05-v223-no-scan-sparsecore-rs-ar-nd-reduce-scatter.md)
then added only the TPU recipe ND reduce-scatter offload flag and failed during
compilation with `Check failed: span_counts_for_colors_[i] == span_count` in
`VerifyReduceScatter`. Do not carry ND reduce-scatter offload for this shape.
[v224](../experiments/qwen3_cx_autoresearch_optimization/jax/experiments/2026-06-05-v224-no-scan-sparsecore-rs-ar-rs-v2.md)
then added only `--xla_tpu_enable_sparse_core_reduce_scatter_v2=true`. It ran
cleanly with byte-identical HLO but regressed to **59,204 tok/s / 42.5% MFU**
and **4442.6 ms / 66.8-66.9% MXU** in XProf. Do not carry reduce-scatter v2.
[v225](../experiments/qwen3_cx_autoresearch_optimization/jax/experiments/2026-06-05-v225-no-scan-sparsecore-rs-ar-rs-latency3.md)
then added only
`--xla_tpu_sparse_core_reduce_scatter_latency_multiplier=3`. It changed the
optimized train-step HLO to
`50dae614cf9c7f4cb48f50ed3bee40310a972e141d58476da21b951462d66b88`, lowered
profiled step time to **4402.9 ms**, and raised best-worker full-window
throughput to **59,732 tok/s / 42.9% MFU**. Treat v225 as a provisional new
best; require exact rerun before carrying it over the confirmed v220/v221
frontier.
[v226](../experiments/qwen3_cx_autoresearch_optimization/jax/experiments/2026-06-05-v226-no-scan-sparsecore-rs-ar-rs-latency3-rerun.md)
exactly reran v225. It reproduced the exact HLO hash and kept `async-done` low
(**577.3 ms / 0.5%**), but full-window throughput was weaker at **59,519 tok/s
/ 42.7% MFU** with **4422.3 ms / 67.3% MXU** in XProf. Carry RS-latency3 as a
marginal supported flag, but separate the **59,732 tok/s / 42.9% MFU** best
observed value from the conservative confirmed rerun value.
[v227](../experiments/qwen3_cx_autoresearch_optimization/jax/experiments/2026-06-05-v227-no-scan-sparsecore-rs-ar-rs-latency2.md)
then lowered the RS latency multiplier to `2`. It selected another HLO hash
(`247229cca7ccc50e7d695af1efe8f798dbc119cabe31b220617e3c7f7d3f2dd1`) and
reached **59,661 tok/s / 42.9% MFU** with **4406.7 ms / 67.4% MXU**. This is
stronger than the latency3 rerun and close to v225's high-water mark; require
an exact latency2 rerun before replacing the conservative confirmed frontier.
[v228](../experiments/qwen3_cx_autoresearch_optimization/jax/experiments/2026-06-05-v228-no-scan-sparsecore-rs-ar-rs-latency2-rerun.md)
confirmed v227 with byte-identical HLO and a slightly stronger best-worker
full-window average of **59,681 tok/s / 42.9% MFU**; late/no-trace windows were
about **59,71k / 59,69k tok/s**. Carry RS latency multiplier `2` as the
confirmed native-JAX frontier. Best observed single full-window remains v225's
**59,732 tok/s**, but it was not reproduced by the latency3 rerun.
[v229](../experiments/qwen3_cx_autoresearch_optimization/jax/experiments/2026-06-05-v229-no-scan-sparsecore-rs-ar-rs-latency2-ar-latency2.md)
then added only `--xla_tpu_sparse_core_all_reduce_latency_multiplier=2` on top
of confirmed RS latency2. It selected a new HLO hash
(`8906838f769ee38abcec2c0498a68eace061c517c362b19e7efa7e4cc5854278`) and
raised best-worker full-window throughput to **59,738 tok/s / 42.9% MFU** with
**4402.4 ms / 67.5% MXU**. Treat this as the new provisional best and rerun
before carrying AR latency2.
[v230](../experiments/qwen3_cx_autoresearch_optimization/jax/experiments/2026-06-05-v230-no-scan-sparsecore-rs-ar-rs-latency2-ar-latency2-rerun.md)
confirmed v229 with byte-identical HLO and improved the best-worker full-window
average to **59,749 tok/s / 42.9% MFU** with **4400.7 ms / 67.5% MXU**. Carry
RS latency multiplier `2` plus all-reduce latency multiplier `2` as the
confirmed native-JAX frontier.
[v231](../experiments/qwen3_cx_autoresearch_optimization/jax/experiments/2026-06-05-v231-no-scan-sparsecore-rs-ar-rs-latency2-ar-latency3.md)
raised only the all-reduce latency multiplier to `3`. It selected a much
smaller HLO (`3c4126a41ca2ac948d1e2ae3aced9ba66b47fae10d95381d1d2a8d2d99f20743`)
and regressed to **57,586 tok/s / 41.3% MFU**, with XProf step time **4565.7
ms**, MXU **66.0%**, direct `all-gather` **2.9%**, and `async-done` **2.6%**.
Do not carry AR latency `3`.
[v232](../experiments/qwen3_cx_autoresearch_optimization/jax/experiments/2026-06-05-v232-no-scan-sparsecore-rs-ar-rs2-ar2-max-async-ag1.md)
then added only `--xla_max_concurrent_async_all_gathers=1` on top of v230. It
kept the exact same optimized HLO and memory profile but landed slightly below
the frontier at **59,730 tok/s / 42.9% MFU** with XProf step time
**4402.4 ms**, so do not carry the all-gather concurrency cap.
[v233](../experiments/qwen3_cx_autoresearch_optimization/jax/experiments/2026-06-05-v233-no-scan-sparsecore-rs-ar-rs2-ar2-max-async-ag2.md)
tested the less restrictive cap `--xla_max_concurrent_async_all_gathers=2`. It
changed the HLO (`94c2737b...`, 17.78 MB), exposed direct `all-gather` at
**4.7%**, dropped MXU to **65.7%**, and regressed to
**58,311 tok/s / 41.8% MFU**. Do not carry async all-gather concurrency caps.
[v234](../experiments/qwen3_cx_autoresearch_optimization/jax/experiments/2026-06-05-v234-no-scan-sparsecore-rs-ar-rs1-ar2.md)
then kept AR2 but lowered only RS latency multiplier to `1`. It selected a
distinct HLO (`02d404f3...`, 19.96 MB) and landed at
**59,643 tok/s / 42.8% MFU**, with XProf step time **4408.3 ms** and MXU
**67.3%**. Keep v230's RS2+AR2 setting.
[v235](../experiments/qwen3_cx_autoresearch_optimization/jax/experiments/2026-06-05-v235-no-scan-sparsecore-rs-ar-rs3-ar2.md)
then kept AR2 but raised only RS latency multiplier to `3`. It selected a
distinct HLO (`d4d46797...`, 19.91 MB) and landed at
**59,747 tok/s / 42.9% MFU**, with XProf step time **4400.9 ms** and MXU
**67.4%**. This is below v230 on the primary full-window average but matched
late/no-trace windows, so rerun before closing RS3+AR2.
[v236](../experiments/qwen3_cx_autoresearch_optimization/jax/experiments/2026-06-05-v236-no-scan-sparsecore-rs-ar-rs3-ar2-rerun.md)
exactly reran RS3+AR2 and reproduced the same HLO
(`d4d46797...`, 19.91 MB), with worker0 reaching **59,750 tok/s / 42.9% MFU**
and XProf **4401.1 ms / 67.0% MXU / 30.93 GiB**. This is one tok/s above v230
but slightly worse in MXU, so treat RS3+AR2 as a same-band confirmed schedule
rather than a material new MFU step.
[v237](../experiments/qwen3_cx_autoresearch_optimization/jax/experiments/2026-06-05-v237-no-scan-sparsecore-rs-ar-rs4-ar2.md)
raised only the RS latency multiplier to `4`. It changed HLO
(`6277ba88...`, 19.90 MB), raised peak HBM to **31.16 GiB**, worsened profile
step time to **4404.0 ms**, and regressed to **59,725 tok/s / 42.8% MFU**.
Close the high-side RS latency bracket.
[v238](../experiments/qwen3_cx_autoresearch_optimization/jax/experiments/2026-06-05-v238-no-scan-sparsecore-rs-ar-rs3-ar2-no-offload-tracing.md)
disabled active SparseCore RS/AR offload tracing on RS3+AR2. It kept the exact
v236 HLO and much smaller profile artifacts but only tied at
**59,749 tok/s / 42.9% MFU**, so tracing is a diagnostics-size knob rather than
a performance lever.
[v239](../experiments/qwen3_cx_autoresearch_optimization/jax/experiments/2026-06-05-v239-no-scan-sparsecore-rs-ar-rs3-ar2-vmem98304.md)
then lowered scoped VMEM from 100352 KiB to 98304 KiB on the post-SparseCore
RS3+AR2 stack. It selected a distinct smaller HLO (`6ca611a...`, 19.84 MB) but
regressed to **59,507 tok/s / 42.7% MFU** with XProf step time **4418.9 ms**.
Keep scoped VMEM 100352 KiB for the current frontier.
[v240](../experiments/qwen3_cx_autoresearch_optimization/jax/experiments/2026-06-05-v240-no-scan-sparsecore-rs-ar-rs3-ar2-vmem100864.md)
then tested the high side at 100864 KiB. It changed HLO again
(`774cf325...`, 19.91 MB) and produced a good late window, but the full-window
average stayed below frontier at **59,714 tok/s / 42.8% MFU** with XProf
**4404.1 ms / 67.4% MXU / 30.93 GiB**. Close the immediate RS3+AR2 VMEM
bracket around 100352 KiB.
[v241](../experiments/qwen3_cx_autoresearch_optimization/jax/experiments/2026-06-05-v241-no-scan-sparsecore-rs-ar-rs2-ar2-vmem100864.md)
then moved the high-side VMEM retune to the material RS2+AR2 v230 frontier. It
also changed HLO (`1932dd46...`, 19.93 MB) but regressed to **59,702 tok/s /
42.8% MFU** with **4404.0 ms** profile step time. Immediate post-SparseCore
VMEM retuning is closed for both frontier HLO families; keep 100352 KiB.
[v242](../experiments/qwen3_cx_autoresearch_optimization/jax/experiments/2026-06-05-v242-no-scan-sparsecore-rs-ar-rs2-ar2-rerun4.md)
then changed only `--xla_latency_hiding_scheduler_rerun=3` to `4` on the
material v230 RS2+AR2 frontier. It reproduced v230's optimized train-step HLO
byte-for-byte (`8906838f...`, 19.93 MB) and matched the profile band at
**4400.4 ms / 66.9-67.9% MXU / 30.93 GiB**, but the best-worker full-window
average was **59,740 tok/s / 42.9% MFU**, below the v230/v236 primary frontier.
Keep rerun `3` unless a later structural change opens a new interaction.
[v243](../experiments/qwen3_cx_autoresearch_optimization/jax/experiments/2026-06-05-v243-no-scan-sparsecore-rs-ar-rs2-ar2-shmem95.md)
then lowered only the scheduler shared-memory cap from 100 to 95 on the same
RS2+AR2 frontier. It again reproduced v230's optimized train-step HLO
byte-for-byte, but regressed to **59,342 tok/s / 42.6% MFU** with XProf
**4432.6 ms / 66.9% MXU / 30.93 GiB**. Do not carry shmem95; this reinforces
that the remaining gap is unlikely to be solved by scalar scheduler-memory
flags alone.
[v244](../experiments/qwen3_cx_autoresearch_optimization/jax/experiments/2026-06-05-v244-no-scan-sparsecore-rs-ar-rs2-ar2-tokamax-ce.md)
then retested Tokamax streamed CE on the post-SparseCore RS2+AR2 frontier. It
improved the old no-scan Tokamax CE band by about 1.2k tok/s and reduced peak
HBM to **27.02 GiB**, but throughput was still only **55,493 tok/s / 39.8% MFU**
with XProf **4745.4 ms / 63.3% MXU**. The Tokamax CE backward custom-call alone
took **10,181.2 ms / 8.9%** of profiled train-step time, so do not carry this
CE path without a targeted kernel/config fix.
[v245](../experiments/qwen3_cx_autoresearch_optimization/jax/experiments/2026-06-05-v245-no-scan-sparsecore-rs-ar-rs2-ar2-tokamax-ce-bs5.md)
used v244's HBM headroom for per-chip batch 5/global batch 40. It fit at
**29.16 GiB** and improved to **55,885 tok/s / 40.0% MFU**, but XProf still
showed **~5.88 s** steps and **63.4-64.2% MXU**. Close Tokamax CE batch
scaling; the remaining Tokamax path issue is the CE backward custom-call cost,
not batch size.
[v246](../experiments/qwen3_cx_autoresearch_optimization/jax/experiments/2026-06-05-v246-no-scan-sparsecore-rs-ar-rs2-ar2-no-mem-pressure-tracking.md)
then disabled TPU scheduler memory-pressure tracking on the material v230
RS2+AR2 frontier. It reproduced the exact v230 HLO (`8906838f...`, 19.93 MB)
and profile band (**4401.5 ms / 67.5% MXU / 30.93 GiB**), but throughput
remained below frontier at **59,732 tok/s / 42.9% MFU**. Keep memory-pressure
tracking enabled and close scalar scheduler-memory flags on this HLO family.
[v247](../experiments/qwen3_cx_autoresearch_optimization/jax/experiments/2026-06-05-v247-no-scan-sparsecore-rs-ar-rs2-ar2-no-offload-tracing.md)
then disabled active SparseCore RS/AR offload tracing on the material RS2+AR2
HLO. It again reproduced `8906838f...` and the same profile shape
(**4402.2 ms / 67.5% MXU / 30.93 GiB**) while reaching only **59,748 tok/s /
42.9% MFU** on the primary full-window average. Treat tracing as a
diagnostics-size knob; it is not a frontier lever.
[v248](../experiments/qwen3_cx_autoresearch_optimization/jax/experiments/2026-06-05-v248-no-scan-sparsecore-rs-ar-rs2-ar2-levanter-ce.md)
then replaced MaxText CE with a vendored Levanter Pallas TPU fused CE path. It
compiled the intended `linear_softmax_cross_entropy_loss_{fwd,bwd}_pallas`
surface and completed with normal loss, but regressed to **56,088 tok/s /
40.2% MFU**, **4702.9 ms / 63.5% MXU**, and **29.37 GiB** peak HBM. Levanter CE
forward itself was only **1.0%** of device time, but the generated fusion
surface was slower and memory remained too high. Do not carry Levanter CE at
`b=1024,h=256,v=512`.
[v249](../experiments/qwen3_cx_autoresearch_optimization/jax/experiments/2026-06-05-v249-no-scan-sparsecore-rs-ar-rs2-ar2-levanter-ce-default-blocks.md)
then retuned only the Levanter CE block sizes to the fallback/default
`b=1024,h=512,v=1024`. It recovered to **58,427 tok/s / 41.9% MFU** with
**4499.4 ms / 67.2% MXU / 29.38 GiB**, proving v248's smaller block shape was
poor, but it still missed the v230/v236 frontier. One bounded Levanter
autotune-set probe remains (`h=512,v=768`); otherwise close this CE branch.
[v250](../experiments/qwen3_cx_autoresearch_optimization/jax/experiments/2026-06-05-v250-no-scan-sparsecore-rs-ar-rs2-ar2-levanter-ce-h512-v768.md)
tested that bounded probe. It regressed to **57,792 tok/s / 41.4% MFU** with
**4553.0 ms / 66.2% MXU / 29.38 GiB** despite a lower forward CE custom-call
time, because generated fusion work increased. Close Levanter Pallas CE
block-size tuning; v249 is the best Levanter point but not a frontier.
[v251](../experiments/qwen3_cx_autoresearch_optimization/jax/experiments/2026-06-05-v251-no-scan-sparsecore-rs-ar-rs2-ar2-levanter-xla-ce.md)
then switched the same Levanter CE API from `implementation="pallas_tpu"` to
`implementation="xla"`. It compiled and ran cleanly, but regressed further to
**54,073 tok/s / 38.8% MFU** with **4861.4 ms / 61.8% MXU / 29.23 GiB**. The
compiled memory fell to **26.00 GiB**, so the failure is not HBM capacity; XProf
shows higher custom-call, loop/data-formatting, and custom-fusion work. Close
Levanter CE as the next replacement path for MaxText CE.
[v252](../experiments/qwen3_cx_autoresearch_optimization/jax/experiments/2026-06-05-v252-no-scan-sparsecore-rs-ar-rs2-ar2-fused-mlp-gate-up.md)
then tried a source-level MLP change: keep `gate_proj`/`up_proj` params but
concatenate their weights and issue one larger input projection before splitting
SwiGLU gate/up. It compiled but failed at program load with only **1.54 GiB**
free for a **1.82 GiB** allocation; compiled memory rose to **28.57 GiB**.
[v253](../experiments/qwen3_cx_autoresearch_optimization/jax/experiments/2026-06-05-v253-no-scan-sparsecore-rs-ar-rs2-ar2-fused-mlp-gate-up-vmem98304.md)
lowered scoped VMEM to **98304** as a bounded headroom probe, but failed the same
way (**1.81 GiB** requested, **1.54 GiB** free; compiled memory **28.56 GiB**).
Close fused gate/up projection: the concatenated-weight path is structurally too
memory-heavy at this frontier.
[v254](../experiments/qwen3_cx_autoresearch_optimization/jax/experiments/2026-06-05-v254-no-scan-sparsecore-rs-ar-rs2-ar2-qknorm-after-transpose.md)
then moved Q/K RMSNorm after the q/k transpose so QK-norm and RoPE operate in
the same `(B,H,T,D)` layout, but the launch omitted the frontier-required
Tokamax Splash env and produced HLO with `tokamax` count 0. Its **56,438 tok/s /
40.5% MFU** result is invalid for frontier comparison; rerun as
[v257](../experiments/qwen3_cx_autoresearch_optimization/jax/experiments/2026-06-05-v257-no-scan-sparsecore-rs-ar-rs2-ar2-qknorm-after-transpose-tokamax.md)
with the correct env reached only **59,328 tok/s / 42.5% MFU** with
**4430.7 ms / 67.1% MXU / 30.95 GiB** peak HBM. Keep the original QK-norm
placement.
[v255](../experiments/qwen3_cx_autoresearch_optimization/jax/experiments/2026-06-05-v255-no-scan-sparsecore-rs-ar-rs2-ar2-explicit-swiglu.md)
then rewrote MLP SiLU as explicit `g * sigmoid(g)`, but the launch omitted the
frontier-required Tokamax Splash env and produced HLO with `tokamax` count 0, so
it is invalid. [v256](../experiments/qwen3_cx_autoresearch_optimization/jax/experiments/2026-06-05-v256-no-scan-sparsecore-rs-ar-rs2-ar2-explicit-swiglu-tokamax.md)
reran the same image with `USE_TOKAMAX_SPLASH=1` and
`TOKAMAX_MAX_LOGIT_CONST=30.0`. It completed cleanly with valid loss and
Tokamax HLO markers but only reached **59,328 tok/s / 42.5% MFU** with
**4431.8 ms / 67.1% MXU / 30.94 GiB** peak HBM. Keep the original
`jax.nn.silu(g)` expression.
[v258](../experiments/qwen3_cx_autoresearch_optimization/jax/experiments/2026-06-05-v258-no-scan-sparsecore-rs-ar-rs2-ar2-mlp-up-proj-first.md)
then tested a narrower MLP scheduling probe by computing `up_proj` before
`gate_proj` while keeping the same params, `checkpoint_name(..., "mlpwi")`, and
`jax.nn.silu(g) * u` math. It completed cleanly with Tokamax markers and a
distinct train-step HLO
(`e413f9db56a30764ebd38a27dc3587ae3889261d16b56473026515e2c91f758a`), reaching
worker0 **59,904 tok/s / 43.0% MFU** and worker1 **59,875 tok/s / 42.9% MFU**.
XProf step time improved to **4391.1 ms** with **67.6% MXU / 30.94 GiB** peak
HBM. Treat this as a same-band supported scheduling signal and the best
profiled step-time point so far, but the conservative confirmed throughput
frontier remains the v230/v236 band until another structural probe turns this
small signal into a larger primary-metric gain.
[v259](../experiments/qwen3_cx_autoresearch_optimization/jax/experiments/2026-06-05-v259-no-scan-sparsecore-rs-ar-rs3-ar2-mlp-up-proj-first.md)
then composed the v258 source ordering with v236's RS3+AR2 latency setting. It
completed cleanly with a distinct train-step HLO
(`57b4add6fe5afe1d02e0423086570ab999f02a2f6185bf935a208f7928d4b26a`) and
reached worker1 **59,926 tok/s / 7,491 tok/s/chip / 43.0% MFU** with best
observed step **60,028 tok/s**. XProf improved again to **4390.3 ms / 67.7%
MXU / 30.94 GiB**. Treat this as the new provisional best same-band point and
run an exact rerun before declaring it confirmed.
[v260](../experiments/qwen3_cx_autoresearch_optimization/jax/experiments/2026-06-05-v260-no-scan-sparsecore-rs-ar-rs3-ar2-mlp-up-proj-first-rerun.md)
exactly reran v259 and reproduced the optimized HLO byte-for-byte. Worker1
landed at **59,903 tok/s / 43.0% MFU** with best observed step **60,030 tok/s**;
worker0 was depressed by one slow post-profile step but otherwise matched the
same schedule. XProf improved again to **4389.7 ms / 67.8% MXU / 30.94 GiB**.
Carry MLP up-proj-first + RS3+AR2 as the current confirmed native-JAX frontier:
best full-window summary remains v259's **59,926 tok/s**, with v260 confirming
the HLO/profile schedule and rounded **43.0% MFU** band.
The MaxText reference
comparison is captured in
[2026-06-05 Qwen3 JAX vs MaxText v6e-8 gap analysis](../analyses/2026-06-05-qwen3-jax-maxtext-gap-analysis.md):
native JAX already exceeds MaxText v006 in tok/s/chip, while MaxText reports a
higher MFU due shape/accounting differences.
[v212](../experiments/qwen3_cx_autoresearch_optimization/jax/experiments/2026-06-05-v212-no-scan-sparsecore-aggregator-bs3-comparison.md)
then reran the current native-JAX stack at MaxText's exact bs3/chip shape and
reported **58,970 tok/s / 7,371 tok/s/chip / 42.3% MFU**, beating MaxText v006's
**55,064 tok/s / 6,883 tok/s/chip / 44.92% MFU**. This confirms the remaining
MaxText/JAX MFU gap is not a token-throughput gap.
[v213](../experiments/qwen3_cx_autoresearch_optimization/jax/experiments/2026-06-05-v213-no-scan-tiled-ce-bs4.md)
tested a native-JAX vocab-tiled CE prototype. It passed toy CPU equivalence but
failed train-step compile at **36.99G / 31.25G HBM** because autodiff retained
many per-chunk `bf16/f32/pred[32768,4096]` tensors. Do not carry naive tiled CE;
[v214](../experiments/qwen3_cx_autoresearch_optimization/jax/experiments/2026-06-05-v214-no-scan-remat-tiled-ce-bs4.md)
then rematerialized the chunk max and sum/picked-logit functions. It reduced
the compile OOM to **35.78G / 31.25G HBM** but still retained many chunk-shaped
values, so chunk remat at size 4096 is also invalid.
[v215](../experiments/qwen3_cx_autoresearch_optimization/jax/experiments/2026-06-05-v215-no-scan-remat-tiled-ce-chunk2048-bs4.md)
tested chunk size 2048 and worsened to **54.36G / 31.25G HBM**. Do not continue
the remat-only chunk-size sweep.
[v216](../experiments/qwen3_cx_autoresearch_optimization/jax/experiments/2026-06-05-v216-no-scan-custom-vjp-tiled-ce-bs4.md)
tested an explicit custom VJP. It removed chunk predicates but worsened to
**38.12G / 31.25G HBM** due many `f32[32768,4096]` chunk-dot temps. Future
CE/logit work needs a faithful streaming/Pallas kernel or a different
structural approach; the cheap
[v217 scan custom-VJP](../experiments/qwen3_cx_autoresearch_optimization/jax/experiments/2026-06-05-v217-scan-custom-vjp-tiled-ce-local-invalid.md)
candidate failed local CPU equivalence and was not launched.

## Cross-variant open hypotheses

Ranked after the confirmed [2026-06-04 v118](../experiments/qwen3_cx_autoresearch_optimization/jax/experiments/2026-06-04-v118-shard-acts-s8k-bs4-tokamax-splash-layeragg-collective-matmul-none.md) / [v123](../experiments/qwen3_cx_autoresearch_optimization/jax/experiments/2026-06-04-v123-shard-acts-s8k-bs4-tokamax-splash-layeragg-collective-matmul-none-rerun.md) frontier (40.5% MFU / 64.0-64.1% MXU; peak HBM 26.91 GiB):

Recent loop state: v136/v137 showed host-transfer overlap 48 was only a
noise-level tie; v138/v139 closed max host send/recv 128/64 as no-gain ties;
v140 closed scheduler rerun4 as a no-gain tie. v141 tested scheduler
shared-memory limit 95 as a no-op/no-gain tie; v142 tested the stronger 90 cap,
changed the train-step scheduling hash but kept 26.88 GiB compiled memory and
still reached only 56,452 tok/s best-worker average. Do not carry overlap48,
sendrecv128, sendrecv64, rerun4, shared-memory 95, or shared-memory 90. v143
rechecked lower scoped VMEM 73728 in the newer collective-matmul-none family;
it changed HLO, raised compiled memory slightly, lowered MXU to 63.3%, and
regressed to 55,810 tok/s. v144 tested the upper-side scoped VMEM point 86016,
changed HLO to a 787,753-byte schedule, improved XProf step time to 1992.6 ms,
and reached worker0 56,609 tok/s average with 56,728 best observed. v145 reran
the same setting, hit the same train-step program, and stayed above the old
frontier at worker0 56,603 tok/s. Carry scoped VMEM 86016 as the current
confirmed frontier. v146 tested the upper bracket 90112 KiB, selected a fresh
787,246-byte HLO with 26.87 GiB compiled memory, improved XProf step average to
1990.9 ms, and reached worker0 56,635 tok/s average with 56,764 best observed.
v147 exactly reran 90112, hit the same train-step program, and improved to
worker0 56,644 tok/s average with 56,785 best observed. Carry scoped VMEM 90112
as the current confirmed frontier. v148 tested 94208 KiB, selected a fresh
779,591-byte HLO with 26.87 GiB compiled memory, improved XProf step average to
1988.6 ms, and reached worker0 56,773 tok/s average with 56,904 best observed.
Treat 94208 as provisional supported; the next probe is an exact v149 rerun
before carrying it. v149 exactly reran 94208, hit the same train-step program,
and reached worker0 56,768 tok/s average with 56,856 best observed. Carry
scoped VMEM 94208 as the current frontier. The next probe is 98304 KiB to
continue the upper bracket. v150 tested 98304 KiB, selected a fresh 787,060-byte
HLO with 26.86 GiB compiled memory, improved XProf step average to 1959.9 ms,
and reached worker0 57,578 tok/s average with 57,685 best observed. v151
exactly reran 98304, hit the same train-step program, and reached worker0
57,572 tok/s average with a new 57,707 best observed steady step. Carry scoped
VMEM 98304 as the current frontier. The next probe is 102400 KiB to continue
the upper bracket. v152 tested 102400 KiB, selected a fresh 782,971-byte HLO
with train-step program `12059987095245408810`, but regressed to worker0
57,410 tok/s average and XProf step average 1966.8 ms. Do not carry 102400;
v153 tested the midpoint 100352 KiB, selected a fresh 783,949-byte HLO with
train-step program `10728347668315821976`, improved XProf step average to
1954.6 ms, and reached worker0 57,757 tok/s average with 57,888 best observed.
v154 exactly reran 100352, hit the persistent compile cache for the same
program family, and reached worker0 57,754 tok/s average with normal loss.
Carry scoped VMEM 100352 as the confirmed frontier. v155 tested 101376 KiB,
emitted a byte-identical optimized HLO to v153, but worker0 regressed to 57,588
tok/s; do not continue the high-side VMEM bracket without a new mechanism.
v156 retested smaller forward Splash KV tiles (`SPLASH_BKV=512`) on the current
batch-4 frontier; it changed HLO but regressed to 56,541 tok/s, so keep default
forward KV tiling. v157 retested smaller DKV KV tiles
(`SPLASH_BKV_DKV=1024`) on the same frontier; it regressed harder to 55,258
tok/s / 39.6% MFU and expanded the DKV custom-call scratch, so keep default DKV
tiling too. v158 checked whether explicitly restoring the layer-scheduler plus
sparse-core collective-aggregator pair changes the vmem100352 schedule family;
it tied at 57,755 tok/s but emitted a byte-identical HLO to v153, so it is a
no-gain no-op. v159 tested the lower-side scoped-VMEM midpoint at 99328 KiB;
it changed HLO but regressed to 57,519 tok/s / 41.2% MFU. v160 tested the
high-side midpoint at 100864 KiB; it completed at 57,762 tok/s best-worker
average but emitted a byte-identical optimized HLO to v153/v158, kept 26.86 GiB
compiled memory, and did not beat v153's best observed step. Treat v160 as
same-HLO noise and keep 100352 KiB as the local VMEM peak. A full retrospective
is filed at [2026-06-04 Qwen3 JAX retrospective](../analyses/2026-06-04-qwen3-cx-jax-retrospective.md);
the next probe should pivot to a structural CE/lm-head, collective-layout, or
kernel hypothesis rather than another scalar VMEM/Splash flag. v161 ran that
pivot by changing only `lm_head.weight` sharding to vocab-axis FSDP
(`("fsdp", None)`) on the confirmed v154 stack. It reached 57,765 tok/s
best-worker average with a 57,908 best step, changed HLO to 785,947 bytes
(`3648b17f7b1fdb754e9b9330400997ed2e06c3f25543cb549bb90559a8b7a348`), and
reduced compiled memory to 26.76 GiB, but v162 exact-rerun refuted the carry:
the same image/shape/flags completed cleanly with normal loss yet fell to
56,494 tok/s best-worker average / 40.5% MFU, with no new HLO dump under the
v162 prefix, consistent with replaying the v161 cached program. Do not carry
lm-head vocab-axis FSDP; keep v153/v154 as the confirmed frontier and pivot to a
different structural CE/lm-head, collective-layout, or kernel hypothesis. v163
then tested a narrower MaxText CE logits boundary constraint
(`P("fsdp", None, None)`) on the confirmed base sharding. It emitted a distinct
784,080-byte HLO
(`9f7762782921bf93af7dddc93c2c93d04207b6f73ab0ec2ceb205709d93c903f`) but
regressed to 56,408 tok/s best-worker average / 40.4% MFU. Do not carry the
full-logits boundary constraint. v164 moved the constraint earlier to the final
hidden state before `lm_head`; it also emitted a distinct 784,126-byte HLO
(`59fb6287422baec57b38443c7e58063e531ba1d98205de3aca1017ce5c37aa01`) and
regressed to 56,380 tok/s best-worker average / 40.4% MFU. Do not carry the
final-hidden constraint either.
v165/v166 tested a different sharding lever by replicating only RMSNorm params.
That reduced train-step HLO size to 778,215 bytes and all-gather occurrences
from 211 to 177, but repeated throughput stayed below the frontier at
57,730/57,735 tok/s best-worker average. Do not carry replicated RMSNorm
params.
v169/v170 later established a new frontier by removing only the MLP
intermediate `_sac`, reaching 57,935-57,947 tok/s repeated average and 58,051
best observed at 41.5% MFU. v171-v175 closed local VMEM and adjacent activation
boundary variants; each changed little or regressed, so v169's gain is isolated
to the large MLP intermediate boundary. v176 retested replicated RMSNorm params
on top of the v169 schedule. It reduced all-gather text count from 140 to 116
and HLO size to 778,035 bytes, but reached only 57,718/57,702 tok/s at 41.4%
MFU, so the RMSNorm replication lever remains refuted even in the no-MLP-sac
schedule family.
v177 similarly retested fully replicated `lm_head.weight` on top of v169. It
emitted a distinct 780,640-byte HLO with all-gather text count 136, but
throughput regressed to 56,449/56,423 tok/s at 40.5% MFU. The CE/lm-head
parameter-sharding interactions remain closed unless a genuinely new fused or
streamed CE mechanism is introduced.
v178 removed the unused one-hot-target cotangent from the MaxText CE custom VJP.
CPU equivalence passed, but no fresh HLO was emitted and throughput stayed at
57,770/57,749 tok/s, indicating the original target-cotangent path was already
dead-code-eliminated from the compiled train step.
v179 tried disabling remat on top of v169, but compile failed hard: XLA reported
340.87 GiB HBM required, dominated by scan-stacked MLP/attention activations.
v180 then tried the narrower host-offload remat policy on top of v169. It
preserved CPU equivalence and completed cleanly, but regressed to
49,761/49,743 tok/s at 35.7% MFU while growing optimized HLO to 1,303,770 bytes
and memory-report bytes to 54.71 GiB. The remat/offload branch is closed for
this batch-4 seq8192 schedule unless a new activation-memory mechanism is
added.
v181 retested Tokamax streamed CE on the current v169 schedule family. It
lowered the memory report to 23.15 GiB and improved substantially over old v031,
but still reached only 53,280/53,265 tok/s at 38.2% MFU. The simple Tokamax CE
swap remains closed; future CE work needs a faster streamed/fused
implementation, not this Pallas Mosaic CE path.
v182 disabled scan on top of v169 and, unlike old v042, the unrolled graph fit.
It produced a very large 19,043,435-byte HLO and raised memory to 27.66 GiB, but
steady-state throughput improved to 58,307/58,330 tok/s with a 58,443 best step
at 41.8% MFU. v183 exactly reran the same runtime schedule; stdout was lost
after pod cleanup, but profile traces confirmed steady device spans at
58,288-58,291 tok/s with 58,302 best per-device span. Carry no-scan as the
confirmed new frontier, while accounting for the much larger HLO and heavier
compile/warmup cost. v184 retuned scoped VMEM upward to 100864 KiB on the
confirmed no-scan frontier. It emitted a distinct 19,043,156-byte HLO
(`8997b4c6...`) and completed cleanly, but only reached 58,311 tok/s worker0
average with a 58,421 best step, so it is a changed-HLO tie/refutation. Keep
100352 KiB for the current no-scan frontier. v185 tested the lower side at
99328 KiB; it also emitted a distinct HLO (`6e904d24...`, 19,043,751 bytes) but
regressed to 57,815 tok/s / 41.5% MFU. The immediate no-scan VMEM retune around
100352 is closed. v186 tested scheduler shared-memory limit 95 on no-scan; it
emitted the exact v182/v183 HLO and regressed to 58,104 tok/s / 41.7% MFU, so
that same-HLO scheduler cap is also closed. v187 raised host-transfer overlap
to 48; it also emitted the exact frontier HLO and only tied at 58,324 tok/s
with a 58,440 best step, so host-transfer overlap 48 is closed too. v188 tried
smaller forward Splash KV tiles with `SPLASH_BKV=512`, but the launch was
invalid because `SPLASH_BKV_COMPUTE` remained 1024; Tokamax rejected the
configuration before train-step compilation. v188 is not a performance result;
the intended test must set both `SPLASH_BKV=512` and `SPLASH_BKV_COMPUTE=512`.
v189 ran that corrected pair. It completed cleanly with normal loss, emitted a
fresh 19,042,499-byte HLO (`668dfe34...`) at the same 27.66 GiB memory, but
regressed to 56,837 tok/s / 40.8% MFU and profile-derived 4.616-4.630 s
train-step spans. Smaller forward Splash KV tiles are closed for the no-scan
frontier. v190 then tested the corresponding backward DKV tile pair
`SPLASH_BKV_DKV=1024 SPLASH_BKV_DKV_COMPUTE=1024`; it emitted another fresh
HLO (`362c1236...`, 19,050,551 bytes) but regressed harder to 55,489 tok/s /
39.8% MFU with profile-derived 4.730-4.739 s spans. Smaller Splash KV tiles are
closed on both forward and DKV paths for the no-scan frontier. v191 tested
Tokamax Splash's experimental scheduler in the no-scan family. It was valid and
emitted a fresh HLO (`4a924f03...`, 19,043,039 bytes), but reached only
57,516/57,495 tok/s at 41.2% MFU with profile spans around 4.563-4.573 s. Do
not carry `TOKAMAX_USE_EXP_SCHED=1`. v192 removed all activation sharding
constraints under no-scan. It was valid and closer, reaching 58,088/58,067
tok/s at 41.6% MFU, but profile spans of 4.518-4.527 s and a 58,244 best step
remain below the confirmed no-scan frontier. The changed HLO (`a05d8bbc...`,
19,042,581 bytes) falls into the same high-count family as v189/v191, so do
not carry wholesale `--shard_acts=False`. v193 then tested the individual
attention-output `_sac` removal under no-scan by reusing the v172 image. It
matched v192's near-tie band at 58,087/58,075 tok/s with profile spans around
4.518-4.525 s and a high-count HLO (`7d751697...`, 19,043,256 bytes), so do not
carry that individual boundary removal either. v194 similarly tested the
pre-attention norm-output `_sac` removal under no-scan by reusing the v173
image. It also landed in the same near-tie band at 58,086/58,064 tok/s with
profile spans around 4.518-4.528 s and high-count HLO `ad96ef69...`; do not
carry that boundary removal. v195 tested the pre-MLP norm-output `_sac` removal
under no-scan by reusing the v174 image and reran CPU equivalence
successfully. It was much slower at 56,654/56,628 tok/s with profile spans
around 4.650-4.663 s and high-count HLO `aa8c6604...`, so do not carry that
boundary removal either. v196 tested the decoder layer input `_sac` removal
under no-scan by reusing the v175 image. It was a near-tie at 58,099/58,083
tok/s, but still below the confirmed no-scan frontier with profile spans around
4.517-4.525 s and high-count HLO `f94aebd8...`. The no-scan single-boundary
activation-constraint sweep is closed; carry only the v169 MLP-intermediate
`_sac` removal. v197 pivoted to the retrospective's collective-matmul
dependency by restoring both all-gather and reduce-scatter collective-matmul
modes to `auto` under no-scan; it failed before step 0 with compile HBM OOM
at 35.56 GiB used / 31.25 GiB capacity, so the `none` pair remains required
for the current target shape. Next isolate the two modes one at a time.
v198 isolated all-gather collective-matmul `auto` with reduce-scatter still
`none`; it also failed before step 0, using 32.67 GiB of 31.25 GiB, and emitted
a 28,098,167-byte HLO (`52ec4c05...`) with many more collective-permute/copy
occurrences. Keep all-gather collective-matmul mode `none`; isolate
reduce-scatter `auto` next. v199 isolated reduce-scatter collective-matmul
`auto` with all-gather still `none`; it failed even harder, using 35.87 GiB of
31.25 GiB with a 26,420,592-byte HLO (`66a1b7df...`) and 34,627 copy
occurrences. The no-scan collective-matmul mode sweep is closed: keep both
modes `none`.

1. Attention/custom-call reduction — v052 improves the target frontier by adding `TOKAMAX_MAX_LOGIT_CONST=30.0` to the v045 tokamax Splash path, v056 adds a marginal scheduler-rerun throughput gain, v068 moves the VMEM setting from 98304 to 81920, v072/v073 add `--xla_tpu_enable_tpu_custom_call_scoped_vmem_adjustments=true`, and v074/v075 add selective resources for a repeatable 51k tok/s / 36.6% band. v076 shows latency-hiding scheduler resource sharing is a tie/slight regression; v077 shows async collective fusion with Mosaic custom calls reclassifies custom-call work into custom fusion but regresses wall time; v078/v079 show multi-compute overlap is non-harmful but not a durable gain; v080 shows dependent-collectives layer scheduling is also an in-band non-improvement; v081 shows forcing collective-compute fusion is harmful; v082 shows fusing multiple async collectives is a tie; v083 shows kloop async collective fusion crashes libtpu; v084 shows start/done-only async collective fusion regresses throughput to 50,656 tok/s; v085 shows staged collective-compute pipelining crashes libtpu's `CollectiveComputeDecomposer`; v086 shows forced ILP latency-hiding scheduling is valid but only ties; v087 shows forced BRKGA LHS is also an in-band tie; v088 shows forcing layer scheduler alone is invalid without sparse-core collective aggregation; v089/v090 pair layer scheduler with sparse-core collective aggregation and repeat above the old 51,021 band, with v090 reaching 51,033 tok/s best worker; v091 shows adding BRKGA to that pair regresses to the old band; v092 shows adding ILP to that pair also regresses to 51,016 tok/s; v093 shows resource serializing reduces HBM to 28.03 GiB but slows throughput to 50,977 tok/s; v094 shows multi-compute overlap on the layeragg pair is valid but only reaches 51,022 tok/s; v095 shows dependent-collectives scheduling on the layeragg pair is also valid but only reaches 51,018 tok/s; v096 shows sparse-core aggregator alone is valid but only reaches 51,011 tok/s, so the v089/v090 gain requires the explicit pair; v101 shows disabling base async collective fusion regresses to 50,361 tok/s while raising all-gather time; v113 shows disabling only the all-gather fusion subtype reproduces that lower-HBM/all-gather-cost tradeoff and regresses to 50,355 tok/s, so all-gather fusion is required; v114 shows disabling only the reduce-scatter fusion subtype is valid and keeps the normal train-step hash/profile, but only reaches 51,008 tok/s and does not improve the frontier; v115 shows disabling only async collective fusion multiple-steps changes the train-step hash/HLO shape, but only reaches 51,023 tok/s and does not improve the frontier, closing the ACF subtype sweep. v102 shows disabling scheduler memory-pressure tracking is only an in-band tie/slight regression at 51,019 tok/s; v103 shows disabling the broad all-experimental scheduler bundle is also an in-band tie/slight regression at 51,004 tok/s; v104 shows lowering host-transfer overlap limit from 24 to 12 is valid but only reaches 51,007 tok/s with the same train-step HLO/profile shape; v108 shows raising the same limit to 48 is also only an in-band tie at 51,014 tok/s; v105 shows lowering max concurrent host send/recv from 100 to 64 is also valid but only reaches 51,006 tok/s with the same train-step HLO/profile shape; v109 shows raising the same limit to 128 is also only an in-band tie at 51,008 tok/s with the normal frontier HLO/profile; v106 shows lowering scheduler shared-memory limit from 100 to 90 creates a real lower-HBM schedule but slows to 51,000 tok/s; v107 shows a milder limit of 95 returns to the normal HLO/profile and still only reaches 51,012 tok/s; v110 shows raising latency-hiding scheduler reruns from 3 to 4 on the layeragg frontier is valid but only reaches 51,009 tok/s with the normal HLO/profile; v111 shows raising scoped VMEM to 86016 KiB on the layeragg frontier selects a real different train-step hash but regresses to 50,003 tok/s / 35.9% MFU; v112 shows `TOKAMAX_DQ_REDUCTION_STEPS=3` also selects a real changed train-step HLO and lowers DKV custom-call time, but regresses to 50,853 tok/s because convolution fusion grows; v116 adds the MaxText `DISABLE_COLLECTIVE_MATMUL` pair (`all_gather_collective_matmul_mode=none`, `reduce_scatter_collective_matmul_mode=none`) and jumps to 56,009 tok/s avg / 40.2% MFU with MXU 63.1%, train-step time 84.6 s, peak HBM 22.21 GiB, and HLO size 795,784 bytes; v117 confirms the same batch-3 frontier; v118 raises per-chip batch to 4/global 32 and reaches 56,445 tok/s / 40.5% MFU with peak HBM 26.91 GiB, but scaling is sublinear; v119-v122 show batch 5/global 40 remains a near-fit HBM OOM even after shared-memory and scoped-VMEM scalar tweaks; v123 confirms the batch-4 frontier; v124 omits the explicit layer-scheduler plus sparse-core aggregator flags but falls back to `auto` and reproduces the exact same train-step HLO/profile; v125 disables selective resources and selects a real changed HLO, but regresses to 56,387 tok/s / 40.4% MFU; v126 forces the layer-scheduler plus sparse-core aggregator pair explicitly `false` and still reproduces the exact v123 HLO/profile; v127 disables the broad all-experimental scheduler bundle on that simplified stack and again reproduces the exact v123 HLO/profile, reaching an in-band 56,453 tok/s average but lower best-observed step than v123; v128 disables TPU scheduler memory-pressure tracking and still emits the exact same HLO/profile but drops back to 56,448 tok/s avg, so do not carry it; v129 disables custom-call scoped VMEM adjustments and again preserves the exact HLO/profile but only reaches 56,449 tok/s best-worker average; v130 disables AG backward pipelining and also preserves the exact HLO/profile but reaches only 56,449 tok/s best-worker average; v131 disables base async collective fusion, changes the final train-step program, raises HBM to 27.10 GiB, lowers MXU to 63.2%, and regresses to 55,482 tok/s / 39.8% MFU, so base ACF remains required after collective-matmul-none; v132 disables only the all-gather ACF subtype, selects the same 632,630-byte HLO/hash as v131, exposes a 4.0% all-gather bucket, and regresses to 55,479 tok/s / 39.8% MFU, so all-gather ACF remains required too; v133 disables only the reduce-scatter ACF subtype, keeps the exact v123/v127 HLO/hash and profile, and reaches an in-band 56,440 tok/s / 40.5% MFU without improving the frontier; v134 disables only async collective fusion multiple-steps, changes the train-step program to `14153491495991100093` with a 641,277-byte HLO, and regresses to 56,294 tok/s / 40.4% MFU, so multiple-step ACF remains required and the post-cmnone ACF sweep is closed; v135 lowers host-transfer overlap limit to 12, keeps the exact v123/v127 HLO/profile, and reaches only 56,436 tok/s / 40.5% MFU without improving the frontier; v136 raises host-transfer overlap limit to 48, keeps the exact frontier HLO/profile, and reaches a noise-level 56,456 tok/s best-worker average but only 56,589 best step, requiring a rerun before carry. v097-v100 exhaust qseq2 runtime correctness probes: dense static masking, no-base2, no-fused-reciprocal, and combined conservative softmax all remain NaN/invalid despite 39%+ nominal MFU. The next probe is an overlap48 rerun.
2. CE/logit materialization — v026 shows batch 4/global 32 OOMs on two 9.27 GiB `bf16[4,8192,151936]` tensors. v030/v031 prove tokamax CE removes the memory wall, but it is too slow; v032 proves MaxText CE plus remat offload also fits batch 4, but data formatting/offload overhead leaves it at 45,953 tok/s / 32.9% MFU. v033/v034 show TP2 at v041's global-batch-24 target shape is invalid both with scan and without scan; v042 shows disabling scan under TP1 also OOMs on a 6.96 GiB full-vocab logit temp plus unrolled MLP/Splash temps. v043/v044 show integer-label custom VJP over full logits is also invalid because it promotes/materializes a 13.91 GiB f32 lm-head-dot temp. v161/v162 show lm-head vocab-axis FSDP sharding is not a durable throughput gain; v163 shows an explicit FSDP layout constraint on the full logits is a real but slower HLO; v164 shows the same for the final hidden state before `lm_head`. v200 shows the trainer's plain integer-label `log_softmax + gather` CE is invalid on the current no-scan frontier because it materializes two 18.55 GiB f32 full-vocab temps and OOMs at 48.39 GiB HBM. v201 shows Tokamax CE under no-scan lowers compiled memory to 22.50 GiB, but only reaches about 54.3k tok/s, so it is not a batch-4 throughput carry. v202 shows that memory relief is enough to fit batch 5/global batch 40 at 25.62 GiB compiled memory, but throughput remains only about 54.9k tok/s, so Tokamax CE batch scaling is also closed. v203 shows direct batch-5 scaling of the faster MaxText CE no-scan frontier is a near-fit but still invalid, OOMing at 32.10 GiB with an 11.59 GiB `bf16[5,8192,151936]` lm-head/logit temp. v213 shows naive Python/XLA vocab-tiled CE is also invalid: it removes full-vocab logits but autodiff saves per-chunk logits and predicates, OOMing at 36.99G HBM with many `bf16/f32/pred[32768,4096]` values. v214 rematerializes chunk computations and reduces the OOM to 35.78G, but still keeps enough chunk-shaped state to miss HBM by 4.54G. v215 lowers remat chunk size to 2048 and worsens to 54.36G HBM, closing the remat-only chunk-size sweep. v216 replaces autodiff with an explicit custom VJP and removes predicates, but worsens to 38.12G because the unrolled backward has 52 `f32[32768,4096]` temps. Further CE work needs a genuinely streamed/fused CE implementation or a deeper sharding/layout rewrite, not tokamax for batch-4 throughput, tokamax batch scaling, direct MaxText CE batch scaling, host offload batch scaling, naive TP2, no-scan unrolling, full-logit label custom VJP, plain integer CE, naive autodiffed tiled CE, remat-only tiled CE, chunk-size-only tiled CE, unrolled custom-VJP tiled CE, lm-head vocab-axis FSDP, full-logits boundary constraint, or final-hidden boundary constraint.

## Variant-specific open hypotheses

(none yet)

## Retired hypotheses

(none yet)

## Knobs translation matrix

| Technique | 8B/v6e-8 | Notes |
|-----------|----------|-------|
| Native JAX / Flax NNX baseline | 512 ms / 31,955 TPS / 20.5% MFU | fsdp=8,tp=1, seq2048, global batch 8 |
| Splash + TP4 + per-chip batch 8 | 831 ms / 39,438 TPS / 25.4% MFU | fsdp=2,tp=4, seq2048, global batch 16; peak HBM 27.75 GiB |
| Splash + tokamax CE + remat + fused RoPE | 1.370 s / 47,846 TPS / 30.8% MFU | fsdp=4,tp=2, seq2048, global batch 32; peak HBM 31.23 GiB |
| Splash + tokamax CE + remat + fused RoPE, batch relief | 1.056 s / 46,526 TPS / 29.9% MFU | fsdp=4,tp=2, seq2048, global batch 24; peak HBM 28.32 GiB |
| Splash + tokamax CE + remat + fused RoPE, seq4096 | 788 ms / 41,592 TPS / 27.8% MFU | fsdp=4,tp=2, seq4096, global batch 8; peak HBM 25.90 GiB |
| Splash + tokamax CE + remat + fused RoPE, seq8192 | 882 ms / 37,148 TPS / 26.6% MFU | fsdp=4,tp=2, seq8192, global batch 4; peak HBM 26.57 GiB |
| Scan + remat + Splash + MaxText CE + activation sharding, seq8192 | 4.07 s / 48,312 TPS / 34.6% MFU | fsdp=8,tp=1, seq8192, global batch 24; peak HBM 28.33 GiB |
| Scan + remat + tokamax Splash + MaxText CE + activation sharding, seq8192 | 4.01 s / 49,069 TPS / 35.2% MFU | fsdp=8,tp=1, seq8192, global batch 24; peak HBM 28.33 GiB |
| + `TOKAMAX_MAX_LOGIT_CONST=30.0` | 3.90 s / 50,376 TPS / 36.1% MFU | v052 frontier; valid loss, peak HBM 28.33 GiB |
| + scheduler rerun3 | 3.90 s / 50,383 TPS / 36.1% MFU | v056 throughput frontier; marginal gain, same profile, peak HBM 28.33 GiB |
| + VMEM 81920 | 3.87 s / 50,790 TPS / 36.4% MFU | v068 throughput frontier; valid loss, MXU 56.5%, peak HBM 28.32 GiB |
| + custom-call scoped VMEM adjust | 3.87 s / 50,793 TPS / 36.4% MFU | v072/v073 frontier; valid loss, MXU 56.4-56.6%, peak HBM 28.32 GiB; tiny but repeatable gain |
| + selective resources | 3.85 s / 51,021 TPS / 36.6% MFU | v074/v075 prior frontier; valid loss, MXU 56.7-56.8%, peak HBM 28.32 GiB |
| + resource sharing | 3.86 s / 51,017 TPS / 36.6% MFU | v076 refuted/tie; real HLO with resource sharing enabled, but no gain over v074/v075 |
| + Mosaic async collective fusion | 3.86 s / 50,976 TPS / 36.5% MFU | v077 refuted; custom-call drops to 18.2% but custom fusion rises to 6.3%, wall time regresses |
| + layer multi-compute overlap | 3.85 s / 51,025 TPS best observed, 51,021 rerun / 36.6% MFU | v078/v079 supported tie; non-harmful, but no durable improvement over v074/v075 |
| + dependent-collectives layer scheduler | 3.86 s / 51,017 TPS / 36.6% MFU | v080 refuted/tie; valid HLO, no profile or throughput win |
| + collective-compute fusion | 4.62 s / 42,555 TPS / 30.5% MFU | v081 refuted hard; convolution fusion rises to 55.8%, MXU falls to 47.4% |
| + fuse multiple async collectives | 3.85 s / 51,021 TPS / 36.6% MFU | v082 supported tie; profile unchanged, no durable gain |
| + fuse kloop async collectives | invalid | v083 compile-time libtpu crash in `async_all_gather_emitter.cc`; exit 139 |
| + start/done-only async collective fusion | 3.88 s / 50,656 TPS / 36.3% MFU | v084 refuted; real HLO variant, lower MXU and no custom-call win |
| + staged collective-compute pipelining | invalid | v085 compile-time libtpu SIGSEGV in `CollectiveComputeDecomposer`; exit 139 |
| + ILP latency-hiding scheduler | 3.85 s / 51,016 TPS / 36.6% MFU | v086 supported tie; profile in-band, no durable improvement |
| + BRKGA latency-hiding scheduler | 3.85 s / 51,023 TPS best worker / 36.6% MFU | v087 supported tie; profile in-band, below v078 best-observed 51,025 |
| + latency-hiding layer scheduler | invalid | v088 startup failure; requires sparse-core collective aggregator |
| + layer scheduler + sparse-core collective aggregator | 3.85 s / 51,033 TPS best observed, 51,026+ repeated / 36.6% MFU | v089/v090 supported; valid pair, profile in-band, new scheduler frontier |
| + layeragg + BRKGA latency-hiding scheduler | 3.86 s / 51,020 TPS / 36.6% MFU | v091 refuted/tie; valid but below v090 |
| + layeragg + ILP latency-hiding scheduler | 3.86 s / 51,016 TPS / 36.6% MFU | v092 refuted/tie; valid but below v090 |
| + layeragg + resource serializing | 3.86 s / 50,977 TPS / 36.6% MFU | v093 refuted/tie; peak HBM drops to 28.03 GiB but wall time worsens |
| + layeragg + multi-compute overlap | 3.85 s / 51,022 TPS / 36.6% MFU | v094 refuted/tie; valid but below v090 |
| + layeragg + dependent collectives scheduling | 3.85 s / 51,018 TPS / 36.6% MFU | v095 refuted/tie; valid but below v090 |
| + sparse-core aggregator only | 3.86 s / 51,011 TPS / 36.6% MFU | v096 refuted/tie; valid, below v074/v075 and v089/v090, pair is required |
| + layeragg without base async collective fusion | 3.91 s / 50,361 TPS / 36.1% MFU | v101 refuted; valid and lower HBM, but all-gather rises to 2.3% and wall time regresses |
| + layeragg without all-gather async collective fusion subtype | 3.91 s / 50,355 TPS / 36.1% MFU | v113 refuted; valid and lower HBM, but all-gather rises to 2.3%, MXU drops to 56.1%, and wall time regresses |
| + layeragg without reduce-scatter async collective fusion subtype | 3.86 s / 51,008 TPS / 36.6% MFU | v114 refuted/tie; valid, normal train-step hash/profile, below repeated frontier |
| + layeragg without async collective fusion multiple-steps | 3.85 s / 51,023 TPS / 36.6% MFU | v115 refuted/tie; valid and changed train-step hash/HLO shape, but below v090 best-observed and no profile win |
| + layeragg without scheduler memory-pressure tracking | 3.85 s / 51,019 TPS / 36.6% MFU | v102 refuted/tie; valid, same HLO size/profile as v089/v090, below repeated frontier |
| + layeragg without all-experimental scheduler bundle | 3.86 s / 51,004 TPS / 36.6% MFU | v103 refuted/tie; valid, same HLO size/profile as v089/v090/v102, below repeated frontier |
| + layeragg host-transfer overlap limit 12 | 3.86 s / 51,007 TPS / 36.6% MFU | v104 refuted/tie; valid, same HLO size/profile as v089/v090/v102/v103, below repeated frontier |
| + layeragg host-transfer overlap limit 48 | 3.86 s / 51,014 TPS / 36.6% MFU | v108 refuted/tie; valid, same HLO size/profile as v089/v090/v102-v105/v107, below repeated frontier |
| + layeragg max concurrent host send/recv 64 | 3.86 s / 51,006 TPS / 36.6% MFU | v105 refuted/tie; valid, same HLO size/profile as v089/v090/v102-v104, below repeated frontier |
| + layeragg max concurrent host send/recv 128 | 3.86 s / 51,008 TPS / 36.6% MFU | v109 refuted/tie; valid, normal HLO/profile, below repeated frontier |
| + layeragg scheduler shared-memory limit 90 | 3.86 s / 51,000 TPS / 36.6% MFU | v106 refuted/tie; real lower-HBM schedule, peak 28.15 GiB, but loop/custom-call time worsens and throughput falls |
| + layeragg scheduler shared-memory limit 95 | 3.86 s / 51,012 TPS / 36.6% MFU | v107 refuted/tie; returns to normal HLO/profile, peak 28.32 GiB, below repeated frontier |
| + layeragg scheduler rerun4 | 3.86 s / 51,009 TPS / 36.6% MFU | v110 refuted/tie; valid, normal HLO/profile, below repeated frontier |
| + layeragg scoped VMEM 86016 | 3.93 s / 50,003 TPS / 35.9% MFU | v111 refuted; real HLO/hash change, but MXU drops to 55.7% and loop fusion grows |
| + layeragg Tokamax DQ reduction steps 3 | 3.87 s / 50,853 TPS / 36.5% MFU | v112 refuted; real HLO/hash change and lower DKV custom-call time, but convolution fusion grows and wall time regresses |
| + layeragg collective-matmul modes none, batch 3 | 3.51 s / 56,017 TPS repeated avg, 56,172 best / 40.2% MFU | v116/v117 supported; `DISABLE_COLLECTIVE_MATMUL` pair shrinks HLO to 795,784 bytes, drops peak HBM to 22.21 GiB, lowers loop fusion to 10.1%, raises MXU to 62.9-63.1%, and reruns with the same train-step program hash |
| + layeragg collective-matmul modes none, batch 4 | 4.64 s / 56,445 TPS avg, 56,557 best / 40.5% MFU | v118 supported; batch 4/global 32 fits, peak HBM 26.91 GiB, MXU 64.1%, but throughput scaling is only about +0.8% over batch 3 |
| + collective-matmul modes none, batch 4, no explicit layeragg pair | 4.64 s / 56,447 TPS avg, 56,576 best / 40.5% MFU | v124 supported tie; explicit layer-scheduler+sparse-core-aggregator flags omitted but final HLO/profile are identical to v118/v123 |
| + layeragg collective-matmul modes none, batch 4, no selective resources | 4.65 s / 56,387 TPS avg, 56,526 best / 40.4% MFU | v125 refuted; `selective_resources=false` changes train-step hash/HLO but slightly regresses with unchanged HBM |
| + collective-matmul modes none, batch 4, layeragg explicitly false | 4.64 s / 56,448 TPS avg, 56,579 best / 40.5% MFU | v126 supported tie; explicit layer-scheduler+sparse-core-aggregator false still preserves exact v123 HLO/profile |
| + collective-matmul modes none, batch 4, layeragg false, no all-experimental scheduler bundle | 4.65 s / 56,453 TPS avg, 56,579 best / 40.5% MFU | v127 supported tie; `all_experimental_scheduler_features=false` is applied but final HLO/profile remain byte-identical to v123/v126 |
| + collective-matmul modes none, batch 4, layeragg false, no all-experimental scheduler bundle, no TPU memory-pressure tracking | 4.65 s / 56,448 TPS avg, 56,597 best / 40.5% MFU | v128 refuted/tie; `xla_tpu_enable_scheduler_memory_pressure_tracking=false` is applied but final HLO/profile remain byte-identical and no throughput gain appears |
| + collective-matmul modes none, batch 4, layeragg false, no all-experimental scheduler bundle, no custom-call VMEM adjust | 4.65 s / 56,449 TPS best-worker avg, 56,587 best / 40.5% MFU | v129 refuted/tie; `xla_tpu_enable_tpu_custom_call_scoped_vmem_adjustments=false` is applied but final HLO/profile remain byte-identical and no throughput gain appears |
| + collective-matmul modes none, batch 4, layeragg false, no all-experimental scheduler bundle, no AG backward pipelining | 4.65 s / 56,449 TPS best-worker avg, 56,571 best / 40.5% MFU | v130 refuted/tie; `xla_tpu_enable_ag_backward_pipelining=false` is applied but final HLO/profile remain byte-identical and no throughput gain appears |
| + collective-matmul modes none, batch 4, layeragg false, no all-experimental scheduler bundle, no base ACF | 4.73 s / 55,482 TPS best-worker avg, 55,533 best / 39.8% MFU | v131 refuted; `xla_tpu_enable_async_collective_fusion=false` changes HLO/hash, raises HBM to 27.10 GiB, lowers MXU to 63.2%, and regresses |
| + collective-matmul modes none, batch 4, layeragg false, no all-experimental scheduler bundle, no all-gather ACF subtype | 4.73 s / 55,479 TPS best-worker avg, 55,547 best / 39.8% MFU | v132 refuted; `xla_tpu_enable_async_collective_fusion_fuse_all_gather=false` selects the same no-ACF-style HLO/hash as v131, exposes all-gather at 4.0%, and regresses |
| + collective-matmul modes none, batch 4, layeragg false, no all-experimental scheduler bundle, no reduce-scatter ACF subtype | 4.65 s / 56,440 TPS best-worker avg, 56,561 best / 40.5% MFU | v133 supported/tie; `xla_tpu_enable_async_collective_fusion_fuse_reduce_scatter=false` keeps the exact v123/v127 HLO/profile, but does not improve the frontier |
| + collective-matmul modes none, batch 4, layeragg false, no all-experimental scheduler bundle, no ACF multiple-steps | 4.66 s / 56,294 TPS best-worker avg, 56,425 best / 40.4% MFU | v134 refuted; `xla_tpu_enable_async_collective_fusion_multiple_steps=false` changes HLO/hash and regresses, so keep multiple-step ACF enabled |
| + collective-matmul modes none, batch 4, layeragg false, no all-experimental scheduler bundle, host-transfer overlap limit 12 | 4.65 s / 56,436 TPS best-worker avg, 56,558 best / 40.5% MFU | v135 refuted/tie; `xla_tpu_host_transfer_overlap_limit=12` keeps the exact v123/v127 HLO/profile but does not improve the frontier |
| + collective-matmul modes none, batch 4, layeragg false, no all-experimental scheduler bundle, host-transfer overlap limit 48 | 4.65 s / 56,456 TPS best-worker avg, 56,589 best / 40.5% MFU | v136 provisional/tie; exact v123/v127 HLO/profile, +3 tok/s avg over v127 was noise-level |
| + collective-matmul modes none, batch 4, layeragg false, no all-experimental scheduler bundle, host-transfer overlap limit 48 rerun | 4.65 s / 56,456 TPS best-worker avg, 56,566 best / 40.5% MFU | v137 refuted/tie; same train-step program, worker1 dropped to 56,383 TPS avg and best steps stayed below v123, so restore overlap 24 |
| + collective-matmul modes none, batch 4, layeragg false, no all-experimental scheduler bundle, max concurrent host send/recv 128 | 4.65 s / 56,447 TPS best-worker avg, 56,550 best / 40.5% MFU | v138 refuted/tie; exact v123/v127 HLO/profile, but no throughput gain over the 100 baseline |
| + collective-matmul modes none, batch 4, layeragg false, no all-experimental scheduler bundle, max concurrent host send/recv 64 | 4.65 s / 56,439 TPS best-worker avg, 56,557 best / 40.5% MFU | v139 refuted/tie; exact v123/v127 HLO/profile, but lower throughput than the 100 baseline |
| + collective-matmul modes none, batch 4, layeragg false, no all-experimental scheduler bundle, scheduler rerun4 | 4.65 s / 56,445 TPS best-worker avg, 56,549 best / 40.5% MFU | v140 refuted/tie; exact v123/v127 HLO/profile, but no throughput gain over rerun3 |
| + collective-matmul modes none, batch 4, layeragg false, no all-experimental scheduler bundle, scheduler shared-memory limit 95 | 4.65 s / 56,455 TPS best-worker avg, 56,567 best / 40.5% MFU | v141 refuted/tie; exact v123/v127 HLO/profile and 26.88 GiB compiled memory, but no throughput gain over the frontier band |
| + collective-matmul modes none, batch 4, layeragg false, no all-experimental scheduler bundle, scheduler shared-memory limit 90 | 4.65 s / 56,452 TPS best-worker avg, 56,556 best / 40.5% MFU | v142 refuted/tie; real train-step scheduling hash change, but same 787,978-byte HLO size, same 26.88 GiB memory, and no throughput gain |
| + collective-matmul modes none, batch 4, layeragg false, no all-experimental scheduler bundle, scoped VMEM 73728 | 4.70 s / 55,810 TPS best-worker avg, 55,909 best / 40.0% MFU | v143 refuted; HLO grows to 792,825 bytes, compiled memory rises to 26.89 GiB, MXU drops to 63.3%, and throughput regresses |
| + collective-matmul modes none, batch 4, layeragg false, no all-experimental scheduler bundle, scoped VMEM 86016 | 4.63 s / 56,609 TPS best-worker avg, 56,728 best / 40.6% MFU | v144 provisional supported; HLO changes to 787,753 bytes, compiled memory stays 26.88 GiB, XProf step avg improves to 1992.6 ms, rerun before carry |
| + collective-matmul modes none, batch 4, layeragg false, no all-experimental scheduler bundle, scoped VMEM 86016 rerun | 4.63 s / 56,603 TPS best-worker avg, 56,708 best / 40.6% MFU | v145 supported; same train-step program as v144, no fresh HLO due compile cache, confirms 86016 as the current frontier |
| + collective-matmul modes none, batch 4, layeragg false, no all-experimental scheduler bundle, scoped VMEM 90112 | 4.62 s / 56,635 TPS best-worker avg, 56,764 best / 40.6% MFU | v146 provisional supported; fresh train-step program `13912291835542571918`, HLO 787,246 bytes, compiled memory 26.87 GiB, XProf step avg 1990.9 ms; rerun before carry |
| + collective-matmul modes none, batch 4, layeragg false, no all-experimental scheduler bundle, scoped VMEM 90112 rerun | 4.62 s / 56,644 TPS best-worker avg, 56,785 best / 40.6% MFU | v147 supported; same train-step program as v146, no fresh HLO due compile cache, confirms 90112 as the current frontier |
| + collective-matmul modes none, batch 4, layeragg false, no all-experimental scheduler bundle, scoped VMEM 94208 | 4.62 s / 56,773 TPS best-worker avg, 56,904 best / 40.7% MFU | v148 provisional supported; fresh train-step program `15797032912653574046`, HLO 779,591 bytes, compiled memory 26.87 GiB, XProf step avg 1988.6 ms; rerun before carry |
| + collective-matmul modes none, batch 4, layeragg false, no all-experimental scheduler bundle, scoped VMEM 94208 rerun | 4.62 s / 56,768 TPS best-worker avg, 56,856 best / 40.7% MFU | v149 supported; same train-step program as v148, no fresh HLO due compile cache, confirms 94208 as the current frontier |
| + collective-matmul modes none, batch 4, layeragg false, no all-experimental scheduler bundle, scoped VMEM 98304 | 4.55 s / 57,578 TPS best-worker avg, 57,685 best / 41.3% MFU | v150 provisional supported; fresh train-step program `6628478088121256461`, HLO 787,060 bytes, compiled memory 26.86 GiB, XProf step avg 1959.9 ms; rerun before carry |
| + collective-matmul modes none, batch 4, layeragg false, no all-experimental scheduler bundle, scoped VMEM 98304 rerun | 4.55 s / 57,572 TPS best-worker avg, 57,707 best / 41.3% MFU | v151 supported; same train-step program as v150, no fresh HLO due compile cache, confirms 98304 as the current frontier |
| + collective-matmul modes none, batch 4, layeragg false, no all-experimental scheduler bundle, scoped VMEM 102400 | 4.57 s / 57,410 TPS best-worker avg, 57,516 best / 41.2% MFU | v152 refuted; fresh train-step program `12059987095245408810`, HLO 782,971 bytes, compiled memory 26.86 GiB, XProf step avg 1966.8 ms; valid but slower than 98304 |
| + collective-matmul modes none, batch 4, layeragg false, no all-experimental scheduler bundle, scoped VMEM 100352 | 4.54 s / 57,757 TPS best-worker avg, 57,888 best / 41.4% MFU | v153 provisional supported; fresh train-step program `10728347668315821976`, HLO 783,949 bytes, compiled memory 26.86 GiB, XProf step avg 1954.6 ms; rerun before carry |
| + collective-matmul modes none, batch 4, layeragg false, no all-experimental scheduler bundle, scoped VMEM 100352 rerun | 4.54 s / 57,754 TPS best-worker avg, 57,862 best / 41.4% MFU | v154 supported; same program family as v153 via persistent compile cache, confirms 100352 as the current frontier |
| + collective-matmul modes none, batch 4, layeragg false, no all-experimental scheduler bundle, scoped VMEM 101376 | 4.55 s / 57,588 TPS worker0 avg, 57,740 best / 41.3% MFU | v155 refuted/no-gain; flag applied and fresh HLO emitted, but optimized HLO is byte-identical to v153 and throughput regresses |
| + collective-matmul modes none, batch 4, layeragg false, no all-experimental scheduler bundle, scoped VMEM 100352, Splash BKV512 | 4.64 s / 56,541 TPS best-worker avg, 56,643 best / 40.5% MFU | v156 refuted; `SPLASH_BKV=512` changes HLO to 783,923 bytes but regresses throughput, matching the older bs3 direction |
| + collective-matmul modes none, batch 4, layeragg false, no all-experimental scheduler bundle, scoped VMEM 100352, Splash BKV_DKV1024 | 4.75 s / 55,258 TPS best-worker avg, ~55,380 best / 39.6% MFU | v157 refuted; `SPLASH_BKV_DKV=1024` changes HLO to 783,982 bytes and expands DKV custom-call scratch, so keep default DKV tiling |
| + collective-matmul modes none, batch 4, layeragg pair true, no all-experimental scheduler bundle, scoped VMEM 100352 | 4.54 s / 57,755 TPS best-worker avg, ~57,870 best / 41.4% MFU | v158 refuted/tie; true/true layer-scheduler pair is accepted but optimized HLO is byte-identical to v153, so no new gain |
| + collective-matmul modes none, batch 4, layeragg false, no all-experimental scheduler bundle, scoped VMEM 99328 | 4.56 s / 57,519 TPS best-worker avg, ~57,625 best / 41.2% MFU | v159 refuted; lower midpoint changes HLO to 784,575 bytes but regresses, so 100352 remains the local VMEM peak |
| + collective-matmul modes none, batch 4, layeragg false, no all-experimental scheduler bundle, scoped VMEM 100864 | 4.54 s / 57,762 TPS best-worker avg, 57,854 best / 41.4% MFU | v160 refuted/tie; optimized HLO is byte-identical to v153/v158 (`3734fae0...`), best step is below v153, and the tiny avg uptick is same-HLO noise |
| + lm_head vocab-axis FSDP sharding, collective-matmul modes none, batch 4, scoped VMEM 100352 | 4.54-4.64 s / 57,765 TPS first run, 56,494 TPS rerun / 41.4% then 40.5% MFU | v161/v162 refuted; first run emitted distinct 785,947-byte HLO and 26.76 GiB compiled memory, but exact rerun fell below frontier, so do not carry |
| + MaxText CE logits constrained to `P("fsdp", None, None)` | 4.65 s / 56,408 TPS best-worker avg, 56,742 best / 40.4% MFU | v163 refuted; distinct 784,080-byte HLO (`9f776278...`) but slower than v153/v154, so do not carry full-logits boundary constraint |
| + final hidden state constrained with `_sac` before `lm_head` | 4.65 s / 56,380 TPS best-worker avg, 56,726 best / 40.4% MFU | v164 refuted; distinct 784,126-byte HLO (`59fb6287...`) but slower than v153/v154, so do not carry final-hidden boundary constraint |
| + replicated RMSNorm params | 4.54 s / 57,730-57,735 TPS best-worker avg, 57,889 best / 41.4% MFU | v165/v166 refuted/tie; train-step HLO shrinks to 778,215 bytes and all-gather count drops 211→177, but repeated throughput remains below v153/v154 |
| + replicated RMSNorm params, scoped VMEM 100864 | 4.54 s / 57,724 TPS best-worker avg, 57,854 best / 41.4% MFU | v167 refuted; HLO is byte-identical to v165 (`f928230a...`) and throughput is lower than v165/v166 and v153/v154 |
| + fully replicated `lm_head.weight` | 4.64 s / 56,464 TPS best-worker avg, 56,875 best / 40.5% MFU | v168 refuted; distinct 780,820-byte HLO (`49953ae3...`) reduces all-gather text but explodes copy/fusion count and regresses wall time |
| + remove MLP intermediate `_sac` only | 4.52 s / 57,947 TPS first run, 57,935 TPS rerun; 58,051 best / 41.5% MFU | v169/v170 supported; distinct 783,769-byte HLO (`443b5f34...`), same 26.86 GiB memory, new confirmed frontier |
| + remove MLP intermediate `_sac` only, scoped VMEM 100864 | 4.52 s / 57,941 TPS best-worker avg, 58,050 best / 41.5% MFU | v171 refuted/tie; byte-identical HLO to v169 (`443b5f34...`) and does not beat the confirmed v169/v170 band |
| + remove MLP intermediate `_sac` and attention-output `_sac` | 4.54 s / 57,770 TPS best-worker avg, 57,922 best / 41.4% MFU | v172 refuted; CPU equivalence passes and HLO changes to 783,590 bytes (`fdcf6528...`), but throughput regresses, so keep the attention-output boundary constraint |
| + remove MLP intermediate `_sac` and pre-attention norm-output `_sac` | 4.54 s / 57,766 TPS best-worker avg, 57,884 best / 41.4% MFU | v173 refuted; CPU equivalence passes and HLO changes to 783,587 bytes (`06b6bbab...`), but throughput regresses, so keep the pre-attention norm-output constraint |
| + remove MLP intermediate `_sac` and pre-MLP norm-output `_sac` | 4.54 s / 57,759 TPS best-worker avg, 57,883 best / 41.4% MFU | v174 refuted; CPU equivalence passes and HLO changes to 783,590 bytes (`14ca3242...`), but throughput regresses, so keep the pre-MLP norm-output constraint |
| + remove MLP intermediate `_sac` and decoder layer input `_sac` | 4.54 s / 57,751 TPS best-worker avg, 57,866 best / 41.4% MFU | v175 refuted; CPU equivalence passes and HLO changes to 783,587 bytes (`76fc341d...`), but throughput regresses, so v169's gain is isolated to the large MLP intermediate |
| + remove MLP intermediate `_sac` and replicate RMSNorm params | 4.54 s / 57,718 TPS best-worker avg, 57,868 best / 41.4% MFU | v176 refuted; CPU equivalence passes and HLO changes to 778,035 bytes (`d6839ab4...`), all-gather text count drops to 116, but wall time regresses below v169/v170 |
| + remove MLP intermediate `_sac` and fully replicate `lm_head.weight` | 4.65 s / 56,449 TPS best-worker avg, 56,839 best / 40.5% MFU | v177 refuted; CPU equivalence passes and HLO changes to 780,640 bytes (`ac73f41c...`), but full lm-head replication remains a large wall-time regression |
| + remove MLP intermediate `_sac` and drop unused MaxText CE target cotangent | 4.54 s / 57,770 TPS best-worker avg, 57,941 best / 41.4% MFU | v178 refuted/no-op; CPU equivalence passes, no fresh HLO emitted, and throughput stays below v169/v170 |
| + remove MLP intermediate `_sac` and disable remat | invalid | v179 compile-time HBM OOM; XLA reports 340.87 GiB required, dominated by `bf16[36,4,8192,12288]` and f32 scan-stacked activations |
| + remove MLP intermediate `_sac` and enable remat host offload | 5.27 s / 49,761 TPS best-worker avg, 49,888 best / 35.7% MFU | v180 refuted; CPU equivalence passes and the run completes, but HLO grows to 1,303,770 bytes (`262e50f1...`), memory report rises to 54.71 GiB, and offload overhead dominates |
| + remove MLP intermediate `_sac` and switch MaxText CE to Tokamax CE | 4.92 s / 53,280 TPS best-worker avg, 53,515 best / 38.2% MFU | v181 refuted; HLO changes to 791,173 bytes (`caca667e...`) and memory drops to 23.15 GiB, but the Tokamax CE custom-call path remains much slower than MaxText CE |
| + remove MLP intermediate `_sac` and disable scan | 4.49 s / 58,288-58,330 TPS confirmed, 58,443 best / 41.8% MFU | v182/v183 supported; unrolled HLO grows to 19,043,435 bytes (`1ba7015c...`) and memory rises to 27.66 GiB, but steady-state throughput repeats above v169/v170 |
| + remove MLP intermediate `_sac`, disable scan, scoped VMEM 100864 | 4.49 s / 58,311 TPS worker0 avg, 58,421 best / 41.8% MFU | v184 refuted/tie; HLO changes to 19,043,156 bytes (`8997b4c6...`) with flat 27.65 GiB memory, but throughput stays inside the v182/v183 band and below v182's best step |
| + remove MLP intermediate `_sac`, disable scan, scoped VMEM 99328 | 4.54 s / 57,815 TPS worker0 avg, 58,004 best / 41.5% MFU | v185 refuted; HLO changes to 19,043,751 bytes (`6e904d24...`) with flat 27.66 GiB memory, but lower VMEM clearly regresses |
| + remove MLP intermediate `_sac`, disable scan, scheduler shared-memory limit 95 | 4.51 s / 58,104 TPS worker0 avg, 58,210 best / 41.7% MFU | v186 refuted/tie; optimized HLO is byte-identical to v182/v183 (`1ba7015c...`), so the cap does not improve the frontier |
| + remove MLP intermediate `_sac`, disable scan, host-transfer overlap limit 48 | 4.49 s / 58,324 TPS worker0 avg, 58,440 best / 41.8% MFU | v187 refuted/tie; optimized HLO is byte-identical to v182/v183 (`1ba7015c...`) and best step remains below v182's 58,443 |
| + qseq2 dense static mask | invalid | v097 reaches 54,832 TPS / 39.3% nominal MFU but NaNs from step 1; dense mask changes HLO custom-call operands but does not fix qseq2 correctness |
| + qseq2 no base2 exp | invalid | v098 reaches 55,574 TPS / 39.8% nominal MFU but NaNs from step 1; metadata confirms `use_base2_exp=false`, so base2 exp is not the isolated correctness bug |
| + qseq2 no fused reciprocal | invalid | v099 reaches 54,909 TPS / 39.4% nominal MFU but NaNs from step 0; metadata confirms `fuse_reciprocal=false`, so fused reciprocal alone is not the isolated correctness bug |
| + qseq2 conservative softmax | invalid | v100 reaches 54,766 TPS / 39.3% nominal MFU but NaNs from step 0; metadata confirms `use_base2_exp=false` and `fuse_reciprocal=false`, retiring qseq2 runtime numeric-control probes |

## Iteration ladder

1. Port Qwen3 8B to native JAX (Flax NNX), matching the torchax baseline number. Done.
2. Apply Splash + TP/batch scaling to establish the first native-JAX frontier. Done: v012.
3. Remove the CE/log-softmax HBM wall and recover batch 32. Done: v015.
4. Create HBM margin, then test longer sequence lengths.

## See also

- [Qwen3 8B — torchax](qwen3-cx-torchax.md) — primary lane (live frontier).
- Program: [`../experiments/qwen3_cx_autoresearch_optimization/program.md`](../experiments/qwen3_cx_autoresearch_optimization/program.md)
- Lane README: [`../experiments/qwen3_cx_autoresearch_optimization/jax/README.md`](../experiments/qwen3_cx_autoresearch_optimization/jax/README.md)

## Sources

- `wiki/experiments/qwen3_cx_autoresearch_optimization/jax/` (scaffold).
