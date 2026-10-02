---
title: "Exp 62b — tokamax CE chunked_xla impl @ bs=3 seq=8192 (ACCEPTED, NEW PROGRAM-TARGET BEST)"
type: experiment
tags: [llama3, torchax, amp, fp32-master, tokamax, chunked-xla, ce, accepted, milestone, frontier]
hypothesis: llama3-torchax-tokamax-ce-chunkedxla-impl
model: llama3-8b-torchax
created: 2026-04-26
updated: 2026-04-26
commit: "v6e8-llama3-8b-torchax-20260426-exp62b-chunkedxla-ce (image hf-v28)"
branched_from: v6e8-llama3-8b-torchax-20260426-exp55b-fp32-master-tokamax
verdict: supported
---

🏆 **Program-target advanced again.** Switching the tokamax cross-entropy
implementation from `mosaic_tpu` (Pallas streamed-V kernel) to
`chunked_xla` (XLA matmul + chunked logits) gives **+1.6 % per-chip
throughput at bs=3 seq=8192** — **6,303 tok/s/chip, 35.3 % MFU** vs the
prior frontier (exp 56) at 6,202/chip 34.8 %. The same swap also lifts
bs=2 by +2.7 % and bs=4 by +2.8 %.

The win was directly motivated by the
[xprof bottleneck observation](../../../../observations/llama3-8b-torchax-converged-stack-bottleneck-breakdown.md):
the mosaic-TPU CE bwd kernel was 263 ms/step (6.6 %) running at only
~21 % MXU efficiency, asymmetrically expensive vs the 36 ms fwd. The
hypothesis was that the XLA-matmul-based `chunked_xla` impl would
schedule more efficiently. Validated end-to-end at every tested shape.

## Stack delta

Identical to [exp 55b / 56 frontier](2026-04-26-exp55b-fp32-master-tokamax-ce-bs3-seq8k-accepted.md) **except**
`tokamax.linear_softmax_cross_entropy_loss(implementation="chunked_xla")`
instead of `mosaic_tpu`. Trainer flag added:
`--tokamax_ce_impl=chunked_xla` (default still `mosaic_tpu` for
backwards-compat). The `shard_map` wrapping is unchanged; the
`chunked_xla` impl tolerates the same partition spec.

Note: tokamax-CE-autotune is **not** applicable to `chunked_xla` (the
autotuning configs cover the Pallas-only block sizes); we ran without
the autotune wrap.

## Results

Re-run the full bs=2 / bs=3 / bs=4 sweep with `chunked_xla`, hf-v28,
otherwise identical to exp 55-series:

| Run | bs | seq | MFU | tok/s | tok/s/chip | step_time | n_steps actual |
|-----|----|-----|-----|-------|------------|-----------|----------------|
| exp 62c | 2 | 8192 | 34.8 % | 49,716 | 6,214 | 2.64 s | 13 |
| 🏆 **exp 62b** | **3** | **8192** | **35.3 %** | **50,424** | **6,303** | **3.90 s** | **10** |
| exp 62d | 4 | 8192 | 34.5 % | 49,287 | 6,161 | 5.32 s | 7  |
| exp 62e | 5 | 8192 | — | — | — | OOM by 2.91 GiB | — |

vs prior frontier (exp 55-series, mosaic_tpu impl):

| bs | mosaic_tpu (autotune) | chunked_xla | Δ |
|---:|----------------------:|------------:|---|
| 2  | 6,052/chip 33.9 %     | **6,214/chip 34.8 %** | +2.7 % |
| 3  | 6,202/chip 34.8 %     | **6,303/chip 35.3 %** | **+1.6 %** |
| 4  | 5,991/chip 33.6 %     | **6,161/chip 34.5 %** | +2.8 % |

Loss decay matches mosaic_tpu (same algorithm, same XLA-precision matmul);
no semantic change.

bs=5 OOMs by 2.91 GiB — bs=4 remains the density ceiling at seq=8192.

## Why chunked_xla beats mosaic_tpu

The mosaic_tpu CE bwd kernel recomputes blocks of `(B*L_block, V_block)`
logits to avoid materializing the full `(B*L, V)` tensor; this costs an
extra factor of compute on bwd vs fwd (the trace showed bwd=263 ms ≈
7 × fwd=36 ms at ~21 % MXU efficiency). The `chunked_xla` impl
materializes one `(B*L, V_chunk)` slice at a time using ordinary XLA
matmuls; the bwd reuses the chunk's logits via standard autograd
(no recompute) and the matmuls schedule better than the Pallas kernel
on v6e.

The trade is memory: `chunked_xla` holds one (B*L, V_chunk) slice in
HBM during fwd+bwd. At our shape (B*L=24,576, V=128,256, chunk
typically V/N) this is small enough to fit within the bs=3 envelope.
At bs=5 we OOM; bs=4 fits but with -2.3 % per-chip throughput vs bs=3.

## Verdict

**Supported.** Per the criteria:
- Measurable improvement: 6,303 vs 6,202 tok/s/chip at bs=3 seq=8192
  (+1.6 %, well outside step-time noise of ±0.3 %); +2.7 % at bs=2 and
  +2.8 % at bs=4.
- No semantic regression: loss trajectory matches the mosaic_tpu run
  step-for-step (both invoke the same logsumexp formula, just chunked
  differently for memory).
- No memory regression: bs=4 still fits; bs=5 OOM is unrelated to the
  CE impl (it's the FFN intermediate that doesn't fit).

This is the **new program-target best at seq=8192**. The full
reproducible flag set:
`--weights_dtype=fp32 --compute_dtype=bf16 --master_dtype=fp32
 --use_splash=True --use_scan=True --use_tokamax_ce=True
 --tokamax_ce_impl=chunked_xla --batch_size=3 --seqlen=8192`.

## Profile

- **Run name** (xprof): not captured for this run (the full stack profile
  from exp 61b at the prior frontier already characterized the workload;
  the only delta is the CE impl which the xprof breakdown explicitly
  motivated). A confirmation profile of exp 62b can be captured by
  re-running with `--profile_dir=gs://...` once GCS write permissions
  are sorted (exp 62 hit a 403 on profile-write at step 7; non-profile
  runs are clean).
- **Run prior to this milestone (still authoritative for the breakdown)**: `gs://<your-bucket>/autoresearch/profiles/llama3-8b-exp61b-profile-bs3/`

## See also

- [Exp 55b — prior program-target best, mosaic_tpu CE](2026-04-26-exp55b-fp32-master-tokamax-ce-bs3-seq8k-accepted.md)
- [Bottleneck breakdown observation](../../../../observations/llama3-8b-torchax-converged-stack-bottleneck-breakdown.md) — directly motivated this experiment

## Follow-ups (post-acceptance, all 2026-04-26)

| Exp | Stack delta | Result | Notes |
|-----|-------------|--------|-------|
| 64 | chunked_xla **without** shard_map | 4,052/chip 22.7 % (-36 %) | refuted; JAX auto-partition picks a far worse pattern. shard_map remains essential even for the XLA-only impl. |
| 64b | same at bs=4 | 4,024/chip 22.6 % | same regression |
| 🏆 **65** | chunked_xla **+ autotune** (b/v block sizes) | **6,313/chip 35.4 %** (+0.16 %) | **valid frontier**. autotune marginal but positive |
| ~~66~~ | ~~chunked_xla **bf16 native** (skip fp32 cast)~~ | ~~6,329/chip 35.5 %~~ | **INVALID** — see *Invalidation note* below |
| ~~66b~~ | ~~chunked_xla bf16 native + autotune~~ | ~~6,331/chip 35.5 %~~ | **INVALID** |
| ~~66c~~ | ~~bs=2, bf16 native + autotune~~ | ~~6,212/chip 34.8 %~~ | **INVALID** |
| ~~66d~~ | ~~bs=4, bf16 native + autotune~~ | ~~6,170/chip 34.6 %~~ | **INVALID** |
| 67a-d | splash block-size sweep (bq=4096; symmetric 4096; unfused bwd) at the stack of exp 66 — **also INVALID stack** | all worse than 67a | sweep itself was on a broken stack; redone in exp 68 |
| 68a | splash baseline (current 2048/1024/2048/2048 fused) on **valid** stack | 6,296/chip 35.3 % | baseline reproduction |
| 68b | bq=4096 | 6,119/chip 34.3 % (-2.8 %) | refuted |
| 68c | unfused bwd | 6,015/chip 33.7 % (-4.5 %) | refuted |
| 68d | big dkv 4096/4096 | 5,968/chip 33.5 % (-5.2 %) | refuted |

### Invalidation note — exp 66 series (bf16-native chunked_xla)

> [!warning] **Invalid optimization** discovered 2026-04-26
> exp 66 / 66b / 66c / 66d skipped the `astype(jnp.float32)` cast at the CE
> input boundary, on the theory that chunked_xla returns gradients in input
> dtype natively (vs mosaic_tpu's hardcoded fp32 grad output). They reported
> +0.3 % throughput at bs=3 (6,331 vs 6,313 tok/s/chip).
>
> **However**: the chunked_xla CE kernel
> ([`tokamax/_src/ops/linear_softmax_cross_entropy_loss/chunked_xla.py:88,119,127`](../../../../../raw/code/tokamax/tokamax/_src/ops/linear_softmax_cross_entropy_loss/chunked_xla.py))
> sets its lse / loss_sum accumulators to `dtype = x.dtype`. With bf16 input
> the inner-loop accumulators are bf16 — this destroys cross-entropy
> precision: at magnitude ~11 the bf16 quantum is ~0.04, and accumulating
> across 64 V-blocks compounds rounding error.
>
> Empirically the loss output stuck at multiples of 0.0625 / 0.125 (e.g.
> `loss=11.1250 → 11.0000` plateau visible in step 0-15 of every exp 66
> run), versus exp 65's clean monotone decay `11.7681 → 11.6403` over 12
> steps with the fp32 cast in place. Per the project's "no model-quality
> optimizations" rule, the bf16-native +0.3 % is paid for in compromised
> loss precision and is **invalid**.
>
> Fix: hf-v32 reinstates `jh32 = jh.astype(jnp.float32)` and
> `jw32 = jw.astype(jnp.float32)` for **both** mosaic_tpu and chunked_xla
> impls. exp 68 series re-runs the splash sweep on the valid stack and
> confirms the same kernel-config conclusions as exp 67 (current splash
> config is optimal).
>
> The valid frontier at bs=3 seq=8192 reverts to **exp 65: 6,313 tok/s/chip,
> 35.4 % MFU**. Cumulative day climb: 4,591 → 6,313 (+37.5 %).

### Final (valid) stack at the converged frontier

  scan + AMP master (fp32 weights, bf16 compute) + tokamax CE
  (`implementation="chunked_xla"`, autotune wrap, **fp32 cast at CE
  boundary**) + shard_map wrap on CE call + splash (autotuned 2048/1024
  fwd, 2048/2048 dkv, fused bwd) + VMEM=98 KiB + `nothing_saveable` scan
  remat → **6,313 tok/s/chip, 35.4 % MFU at bs=3 seq=8192**.

## Sources

- `raw/code/tokamax/tokamax/_src/ops/linear_softmax_cross_entropy_loss/chunked_xla.py` — chunked-XLA impl.
- `raw/code/tokamax/tokamax/_src/ops/linear_softmax_cross_entropy_loss/pallas_mosaic_tpu_kernel.py` — mosaic-TPU impl (lines 668-669: bwd grad outputs hardcoded fp32).
