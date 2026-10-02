## [2026-06-05] loop-iteration | v001 maxtext baseline

**Op**: loop-iteration
**Experiment**: [2026-06-05-qwen3-maxtext-v001-baseline.md](file:///mnt/disks/persist/torch-tpu/tpu_performance_autoresearch_wiki_ag/wiki/experiments/qwen3_ag_autoresearch_optimization/maxtext/experiments/2026-06-05-qwen3-maxtext-v001-baseline.md)
**Verdict**: open
**Notes**: Obtaining MaxText baseline for Qwen3-8B.

## [2026-06-05] loop-iteration | v002 maxtext baseline bs=2

| Date | ID | Hypothesis | Step Time | MFU | Verdict |
|---|---|---|---|---|---|
| 2026-06-05 | [v001](experiments/2026-06-05-qwen3-maxtext-v001-baseline.md) | Initial baseline run on MaxText, bs=1 | 1.39s | 22.4% | `bs=1` baseline collected. |
| 2026-06-05 | [v002](experiments/2026-06-05-qwen3-maxtext-v002-baseline-bs2.md) | Try `bs=2` to see if MFU improves | 3.70s | 14.8% | **FAILED**. MFU degraded due to severe memory pressure / remat. |

**Op**: loop-iteration
**Experiment**: [2026-06-05-qwen3-maxtext-v002-baseline-bs2.md](file:///mnt/disks/persist/torch-tpu/tpu_performance_autoresearch_wiki_ag/wiki/experiments/qwen3_ag_autoresearch_optimization/maxtext/experiments/2026-06-05-qwen3-maxtext-v002-baseline-bs2.md)
**Verdict**: closed
**Notes**: Running MaxText baseline for Qwen3-8B at bs=2. Results are sub-optimal due to memory constraints.
## [2026-06-05] loop-iteration | v003 maxtext baseline seq=2048 bs=16

**Op**: loop-iteration
**Experiment**: [2026-06-05-qwen3-maxtext-v003-baseline-seq2048.md](file:///mnt/disks/persist/torch-tpu/tpu_performance_autoresearch_wiki_ag/wiki/experiments/qwen3_ag_autoresearch_optimization/maxtext/experiments/2026-06-05-qwen3-maxtext-v003-baseline-seq2048.md)
**Verdict**: open
**Notes**: Running maxtext with matching seq=2048 and bs=16 to compare against JAX lane's 33.0% MFU.
| 2026-06-05 | [v003](experiments/2026-06-05-qwen3-maxtext-v003-baseline-seq2048.md) | seq=2048 bs=16 baseline | N/A | N/A | **FAILED**. OOM. MaxText requires 41.76G HBM for bs=16 seq=2048, which exceeds the 31.25G v6e-8 limit. (Note: torchtitan JAX lane successfully fits bs=16 seq=2048). |
## [2026-06-05] loop-iteration | v006 maxtext baseline seq=8192 bs=2

| Date | ID | Hypothesis | Step Time | MFU | Verdict |
|---|---|---|---|---|---|
## [2026-06-05] loop-iteration | v006 maxtext baseline seq=8192 bs=2

**Op**: loop-iteration
**Experiment**: [2026-06-05-qwen3-maxtext-v006-baseline-seq8192-bs2.md](file:///mnt/disks/persist/torch-tpu/tpu_performance_autoresearch_wiki_ag/wiki/experiments/qwen3_ag_autoresearch_optimization/maxtext/experiments/2026-06-05-qwen3-maxtext-v006-baseline-seq8192-bs2.md)
**Verdict**: closed
**Notes**: Baseline achieved with `per_device_batch_size=2` and `sa_block_*=1024`. Tokens/s/device: 5355. Step Time: 3.06s. MFU: ~34.9%.

