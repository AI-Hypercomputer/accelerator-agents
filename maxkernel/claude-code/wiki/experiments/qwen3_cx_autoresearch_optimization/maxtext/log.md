## [2026-06-12] follow-up | MaxText bs3 base-b improves to ~6,998 tok/s/chip, ~45.7% MFU

Workload `alekseyv-qwen3-cc5-maxtext-v001-base-b` ran another bs3 MaxText
variant under the `qwen3_cc5` GCS root:
`gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-12-qwen3-maxtext-v001-base/mt-v001-base-b/`.
It completed with `EXIT_CODE=0`. Differences from v007 include
`use_iota_embed=False`, no HLO upload, and `upload_all_profiler_results=False`;
XLA memory report dropped device temp from **13.1 GiB** to **10.8 GiB**.

Pre-profile steady steps 1-9 reached **417.7-419.2 TFLOP/s/device** and
**6,973-6,998 tok/s/device**. The profile overhead landed on step 14;
post-profile steps 15-19 settled at **416.7-417.7 TFLOP/s/device** and
**6,956-6,973 tok/s/device**. This raises the observed MaxText reference from
the earlier **44.9% MFU / 6,883 tok/s/chip** to roughly **45.4-45.7% MFU /
6,956-6,998 tok/s/chip**, depending on whether pre-profile or post-profile
steady steps are used. Profile path:
`tensorboard/plugins/profile/2026_06_12_06_18_19/`.

## [2026-06-12] rerun + profile | MaxText Qwen3-8B bs3 recipe reproduced (6,831-6,845 tok/s/chip, 44.6-44.8% MFU)

Reran the efficient bs3 recipe as `alekseyv-qwen3-maxtext-v007-bs3r` to collect
a fresh profile and HLO. The run completed with `EXIT_CODE=0`. Output root:
`gs://<your-bucket>/autoresearch/qwen3_maxtext/2026-06-12-qwen3-8b-v6e8-recipe-v007-bs3-rerun/alekseyv-qwen3-maxtext-v007-bs3r/`.
Profile landed at `tensorboard/plugins/profile/2026_06_12_06_07_07/`.

XProf summary: **3611.7 ms**, **62.2% MXU**, **0.3% idle**. Op profile is
train-step dominated: convolution fusion **49.6%** and Splash custom-calls
**32.6%**. Steady post-profile steps 15-19 reached **409.2-410.0
TFLOP/s/device**, **6,831-6,845 tok/s/device**, **44.6-44.8% MFU**. The
optimized HLO is byte-identical to v006: SHA256
`598fe616e4031e66c7a0709c0e0dfd04bec2566041590378c0a4fb1ea931cf84`
(1,498,186 bytes / 8,013 lines).

MFU audit: MaxText's recipe image counts attention as the full non-causal
QK/AV matmul term for reporting. The native Qwen3 JAX/torchax trainers were
counting triangular causal attention, underreporting MFU by **13.76%** at this
shape. The native reporting code now uses the MaxText-style attention counter
and labels the printout `MaxText-style`.

## [2026-06-05] baseline | MaxText Qwen3-8B v6e-8: supported (5,896 tok/s/chip, 38.5% MFU)

First Qwen3-8B MaxText lane baseline. Uses native MaxText `qwen3-8b` support
from the runner image, synthetic data, seq=8192, fsdp=8, conservative
`per_device_batch_size=1`, xplane profiling after step 10, and HLO dump enabled.
Valid run `alekseyv-qwen3-maxtext-v003` completed 20 steps and proved artifact
capture. Median steady-state over steps 3-9 and 16-19: 1.3893 s/step,
353.22 TFLOP/s/device, 5,896.46 tok/s/device, 47,171.65 aggregate tok/s,
38.48% MFU. Final loss 0.2334. Profile landed at
`gs://<your-bucket>/autoresearch/qwen3_maxtext/2026-06-05-qwen3-8b-v6e8-baseline-v003/alekseyv-qwen3-maxtext-v003/tensorboard/plugins/profile/2026_06_05_01_13_31/`
with two xplane protos and two trace JSONs. HLO landed under the matching
`xla_dump/` directory for both hosts; optimized train-step HLO is 1,507,976
bytes with SHA256 `3ea3f8e823438e2c7617af3e5a2881083f1a7a8ee30786e6f0fa3520d8b6d365`.
Peak HBM from memory report is 29.07 GiB, so batch scaling has little headroom.

Note: `v002` was an invalid config-validation attempt because `tokenizer_type`
was redundantly overridden; `v003` removed the duplicate key.

## [2026-06-05] recipe inspection + bs2/bs3 probes | MaxText Qwen3-8B v6e-8: supported (6,883 tok/s/chip, 44.9% MFU)

Inspected local MaxText Trillium recipes. The likely source of the suspected
~45% MFU recipe is `llama3_1_8b_8192` in
`/mnt/disks/persist/maxtext-main/benchmarks/maxtext_trillium_model_configs.py`,
not a Qwen3-specific recipe. It uses seq8192, `per_device_batch_size=4`,
custom remat/offload, 2048 Splash tiles, scoped VMEM 98304,
reduce-scatter layout, data-parallel overlap, async all-gather continuation
fusion, SparseCore all-reduce offload, and host-offload scheduler flags.

`v004` attempted the direct Qwen3 adaptation `per_device_batch_size=2` plus
`num_vocab_tiling=16`, but the runner image rejected the key:
`ValueError: Key num_vocab_tiling was passed at the command line but isn't in
config.` A diagnostic grep inside `/app/MaxText` confirmed that the image has
Qwen3 support but no vocab-tiling implementation. The local MaxText mirror has
vocab tiling, but it requires newer dependencies than this image provides.

`v005` therefore tested the first legacy-compatible scaling step:
baseline v003 stack with `per_device_batch_size=2` and no vocab tiling. It
completed all 20 steps with xprof and HLO capture, reaching **42.74% MFU** on
worker 1.

`v006` then tested `per_device_batch_size=3`, matching the local
`llama3_1_8b_8192_no_collective_matmul` batch-density recipe. It completed all
20 steps with xprof and HLO capture. Worker 1 steady-state median over steps
3-9 and 16-19: 3.571 s/step, 412.32 TFLOP/s/device, 6,883.00 tok/s/device,
55,064 aggregate tok/s, **44.92% MFU**. This effectively reproduces the
suspected ~45% MaxText recipe level for Qwen3-8B on v6e-8. Profile landed at
`gs://<your-bucket>/autoresearch/qwen3_maxtext/2026-06-05-qwen3-8b-v6e8-recipe-v006-bs3/alekseyv-qwen3-maxtext-v006-bs3/tensorboard/plugins/profile/2026_06_05_02_39_21/`.
HLO landed under the matching `xla_dump/` directory; optimized train-step HLO
SHA256 is `598fe616e4031e66c7a0709c0e0dfd04bec2566041590378c0a4fb1ea931cf84`.
For comparison, v005 profile landed at
`gs://<your-bucket>/autoresearch/qwen3_maxtext/2026-06-05-qwen3-8b-v6e8-recipe-v005-bs2/alekseyv-qwen3-maxtext-v005-bs2/tensorboard/plugins/profile/2026_06_05_02_30_49/`.
