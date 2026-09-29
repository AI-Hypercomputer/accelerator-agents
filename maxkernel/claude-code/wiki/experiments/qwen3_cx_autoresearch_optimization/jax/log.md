## [2026-06-12] loop-iteration | cc5 v081 launched: 2k current-stack control on v6e-demo-hjajoo

Launching workload `alekseyv-qwen3-jax-v081-2kctl` on cluster
`v6e-demo-hjajoo` (`tpu-prod-env-one-vm/us-east5`) to establish the requested
2k-context native-JAX baseline. Capacity check before launch found two Ready
2x4 `tpu-v6e-slice` nodes in nodepool `v6e-demo-hjajoo-np-0`, Kueue
`multislice-queue` with zero pending/admitted workloads, and no active Qwen3
workloads.

The run reuses the carried explicit-SiLU shmem90 stack from image
`qwen3-8b-jax:v067-explicit-silu`, changing the shape to
`--seqlen=2048 --batch_size=4`. Runtime flags keep no-scan remat, activation
sharding, Tokamax Splash max-logit 30, MaxText CE, scoped VMEM **100352 KiB**,
latency-hiding scheduler rerun **3**, SparseCore reduce-scatter/all-reduce
offload, SparseCore collective aggregator, RS latency multiplier **3**, AR
latency multiplier **2**, selective resources, and shared-memory limit **90**.

Run/cache paths:
`gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-12-qwen3-jax-v081-2k-bs4-current-control`
and
`gs://<your-bucket>/autoresearch/qwen3_cc5/jax_lane_cache_v081_2k_bs4ctl`.

## [2026-06-12] loop-iteration | cc5 v081 2k current-stack control: supported (65,505 tok/s, 42.1% script MFU)

Workload `alekseyv-qwen3-jax-v081-2kctl` completed cleanly on
`v6e-demo-hjajoo` with both TPU hosts on nodepool `v6e-demo-hjajoo-np-0`.
Worker summaries were:

- `gke-tpu-77a6681c-wwzz`: **65,448 tok/s**, **8,181 tok/s/chip**,
  **42.1%** script MFU, `EXIT_CODE=0`.
- `gke-tpu-77a6681c-6221`: **65,505 tok/s**, **8,188 tok/s/chip**,
  **42.1%** script MFU, `EXIT_CODE=0`.

XProf run
`2026-06-12-qwen3-jax-v081-2k-bs4-current-control/2026_06_12_22_27_48`
reported **1008.1 ms** step time, **0.1 ms** stddev, **56.5% MXU**, about
**1.0%** op-profile idle, and peak HBM **18.95 GiB / 31.25 GiB** with
**12.296 GiB** free. The train-step profile remains matmul-heavy but with more
short-context overhead than the 8k frontier: `convolution fusion` was
**7525.467 ms / 62.0%**, `custom-call` **1503.348 ms / 12.4%**, `loop fusion`
**1486.835 ms / 12.3%**, `custom fusion` **804.665 ms / 6.6%**, and
`data formatting` **449.972 ms / 3.7%**.

Train-step HLO modules `module_0264` and `module_0267` were byte-identical:
SHA256 `c6b001141eb5b66b941054abd357bc8683e8e95bece3fb3b6d74b494348ff786`,
**19,968,324 bytes / 121,278 lines**, compile memory **16.00 GiB**, and
preallocated temp **10.25 GiB**. Counts: `all-gather=4595`,
`all-reduce=154`, `reduce-scatter=1780`, `async-start=802`,
`async-done=802`, `convolution=2100`, `custom-call=2514`, `copy=10614`,
`fusion=22515`, `dot_general=7035`, `splash=1778`, `tokamax=2`,
`jvp_jit_silu__=0`, `exponential=183`, `dynamic-slice=150`,
`dynamic-update-slice=216`.

Verdict: supported as the 2k current-stack control, not yet optimized. The next
2k-specific probe should increase batch size before touching scheduler flags:
v081 leaves **12.296 GiB** runtime HBM free and shows only **56.5% MXU**, so
`--batch_size=6` / global batch **48** is the conservative next experiment.

## [2026-06-12] loop-iteration | cc5 v082 launched: 2k batch6 scale

Launching workload `alekseyv-qwen3-jax-v082-2kbs6` on `v6e-demo-hjajoo` to
test the first 2k-specific optimization after v081. The only intended shape
change is `--batch_size=6` at `--seqlen=2048`, increasing global batch from
**32** to **48**. Runtime stack remains v081's explicit-SiLU shmem90 stack:
no-scan remat, activation sharding, Tokamax Splash max-logit 30, MaxText CE,
scoped VMEM **100352 KiB**, latency-hiding scheduler rerun **3**, SparseCore
reduce-scatter/all-reduce offload, SparseCore collective aggregator, RS latency
multiplier **3**, AR latency multiplier **2**, selective resources, and
shared-memory limit **90**.

Support requires clean finite loss plus throughput above v081's **65,505
tok/s** and ideally higher MXU than **56.5%**. Compile OOM, profile step time
that cancels the larger token batch, or a larger custom-call/loop-fusion share
refutes direct 2k batch scaling.

Run/cache paths:
`gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-12-qwen3-jax-v082-2k-bs6-batch-scale`
and
`gs://<your-bucket>/autoresearch/qwen3_cc5/jax_lane_cache_v082_2k_bs6`.

## [2026-06-12] loop-iteration | cc5 v082 2k batch6 scale: supported (67,987 tok/s, 43.7% script MFU)

Workload `alekseyv-qwen3-jax-v082-2kbs6` completed cleanly on
`v6e-demo-hjajoo` with normal finite loss, improving the v081 2k current-stack
control from **65,505 tok/s** to **67,987 tok/s** best-worker average. Worker0
reported **67,987 tok/s**, **8,498 tok/s/chip**, **43.7% script MFU**; worker1
reported **67,980 tok/s**, **8,498 tok/s/chip**, **43.7% script MFU**. XProf
run
`2026-06-12-qwen3-jax-v082-2k-bs6-batch-scale/2026_06_12_22_43_08`
shows **1456.5 ms** average step time, **0.3 ms** standard deviation,
**61.6% MXU**, and **0.4%** op-profile idle. Peak runtime HBM was
**21.28 GiB / 31.25 GiB**, leaving **9.9688 GiB** free.

Top HLO buckets by time were convolution fusion **22,487.957 ms / 64.4%**,
custom-call **4,523.430 ms / 13.0%**, loop fusion **4,195.456 ms / 12.0%**,
custom fusion **1,589.972 ms / 4.6%**, and data formatting
**1,354.823 ms / 3.9%**. Fresh train-step HLO
`module_0267.jit_train_step.cl_854318611.after_optimizations.txt` was
**19,958,402 bytes**, **120,868 lines**, SHA256
`d32ddad1763f328ce81a016e475f5abc412fbeacba26d865ad31b93448b10f9d`, with
compiled memory total **18.14 GiB** and preallocated temp **12.39 GiB**. Verdict:
supported; continue direct 2k batch scaling to per-chip **batch_size=8** before
touching scheduler/kernel flags.

## [2026-06-12] loop-iteration | cc5 v083 launched: 2k batch8 scale

Launching workload `alekseyv-qwen3-jax-v083-2kbs8` on `v6e-demo-hjajoo`.
This is a one-axis continuation from v082: only `--batch_size` changes from
**6** to **8** at sequence length **2048**, increasing global batch from **48**
to **64**. Runtime stack remains the v081/v082 explicit-SiLU shmem90 stack:
Tokamax Splash, MaxText CE, no-scan remat, activation sharding, scoped VMEM
**100352 KiB**, selective-resource latency-hiding rerun **3**, SparseCore
reduce-scatter/all-reduce offload, SparseCore collective aggregator, and
collective-matmul modes disabled.

Support requires clean finite loss plus throughput above v082's **67,987
tok/s** and ideally MXU above **61.6%**. OOM or a step-time increase that
overwhelms the token batch refutes further direct batch scaling.

Run dir:
`gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-12-qwen3-jax-v083-2k-bs8-batch-scale`

Compile cache:
`gs://<your-bucket>/autoresearch/qwen3_cc5/jax_lane_cache_v083_2k_bs8`.

## [2026-06-12] loop-iteration | cc5 v083 2k batch8 scale: supported/tie (68,184 tok/s, 43.8% script MFU)

Workload `alekseyv-qwen3-jax-v083-2kbs8` completed cleanly on
`v6e-demo-hjajoo` with normal finite loss, reaching worker0 **68,184 tok/s** /
**8,523 tok/s/chip** / **43.8% script MFU** and worker1 **68,182 tok/s** /
**8,523 tok/s/chip** / **43.8% script MFU**. This is only **+0.3%** over v082's
**67,987 tok/s**. Profile-window steps were noisy and included **67,321-68,028
tok/s**, so the apparent gain is marginal.

XProf run
`2026-06-12-qwen3-jax-v083-2k-bs8-batch-scale/2026_06_12_22_58_59`
shows **1945.0 ms** average step time, **15.2 ms** standard deviation,
**63.2% MXU**, and **0.3%** op-profile idle. Peak runtime HBM was
**22.54 GiB / 31.25 GiB**, leaving **8.7082 GiB** free. Top HLO buckets by time
were convolution fusion **30,476.399 ms / 65.5%**, custom-call
**6,051.219 ms / 13.0%**, loop fusion **5,595.362 ms / 12.0%**, custom fusion
**1,835.653 ms / 3.9%**, and data formatting **1,794.640 ms / 3.9%**.

Fresh train-step HLO
`module_0267.jit_train_step.cl_854318611.after_optimizations.txt` was
**19,739,913 bytes**, **118,334 lines**, SHA256
`c7523344a6b91d9bed5f7fea0d9b968120459e8be511db1b727b78226b7f202a`, with
compiled memory total **19.51 GiB** and preallocated temp **13.75 GiB**. Verdict:
supported/tie; it is the best script MFU so far by a tiny margin, but not enough
to declare batch scaling healthy. Run one larger batch jump before pivoting.

## [2026-06-12] loop-iteration | cc5 v084 launched: 2k batch12 saturation check

Launching workload `alekseyv-qwen3-jax-v084-2kbs12` on `v6e-demo-hjajoo`.
This is a one-axis continuation from v083: only `--batch_size` changes from
**8** to **12** at sequence length **2048**, increasing global batch from **64**
to **96**. Runtime stack remains the v081-v083 explicit-SiLU shmem90 stack:
Tokamax Splash, MaxText CE, no-scan remat, activation sharding, scoped VMEM
**100352 KiB**, selective-resource latency-hiding rerun **3**, SparseCore
reduce-scatter/all-reduce offload, SparseCore collective aggregator, and
collective-matmul modes disabled.

Support requires clean finite loss plus throughput meaningfully above v083's
**68,184 tok/s**. A near-flat result around **68k tok/s**, OOM, or higher
non-matmul overhead refutes further direct 2k batch scaling and should pivot the
lane to scheduler/kernel-shape changes.

Run dir:
`gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-12-qwen3-jax-v084-2k-bs12-batch-scale`

Compile cache:
`gs://<your-bucket>/autoresearch/qwen3_cc5/jax_lane_cache_v084_2k_bs12`.

## [2026-06-12] loop-iteration | cc5 v084 2k batch12 scale: supported (69,547 tok/s, 44.7% script MFU)

Workload `alekseyv-qwen3-jax-v084-2kbs12` completed cleanly on
`v6e-demo-hjajoo` with normal finite loss, reaching worker0 **69,517 tok/s** /
**8,690 tok/s/chip** / **44.7% script MFU** and worker1 **69,547 tok/s** /
**8,693 tok/s/chip** / **44.7% script MFU**. This is a real improvement over
v083's **68,184 tok/s**. XProf run
`2026-06-12-qwen3-jax-v084-2k-bs12-batch-scale/2026_06_12_23_12_45`
shows **2844.7 ms** average step time, **3.0 ms** standard deviation,
**65.5% MXU**, and **0.3%** op-profile idle. Peak runtime HBM was
**26.22 GiB / 31.25 GiB**, leaving **5.0216 GiB** free.

Top HLO buckets by time were convolution fusion **45,384.405 ms / 66.5%**,
custom-call **9,100.467 ms / 13.3%**, loop fusion **8,366.298 ms / 12.3%**,
data formatting **2,762.872 ms / 4.0%**, and custom fusion
**1,709.908 ms / 2.5%**. Fresh train-step HLO
`module_0267.jit_train_step.cl_854318611.after_optimizations.txt` was
**19,758,199 bytes**, **118,066 lines**, SHA256
`3402ba6fae40990de244b278f7fa74e6e39a937d5b8a537be73d51e51a0cf0c9`, with
compiled memory total **23.04 GiB** and preallocated temp **17.29 GiB**.
Verdict: supported and current 2k frontier; batch scaling is still improving,
but memory headroom is now only about **5 GiB**.

## [2026-06-12] loop-iteration | cc5 v085 launched: 2k batch16 near-ceiling probe

Launching workload `alekseyv-qwen3-jax-v085-2kbs16` on `v6e-demo-hjajoo`.
This is a one-axis continuation from v084: only `--batch_size` changes from
**12** to **16** at sequence length **2048**, increasing global batch from
**96** to **128**. Runtime stack remains the v081-v084 explicit-SiLU shmem90
stack: Tokamax Splash, MaxText CE, no-scan remat, activation sharding, scoped
VMEM **100352 KiB**, selective-resource latency-hiding rerun **3**, SparseCore
reduce-scatter/all-reduce offload, SparseCore collective aggregator, and
collective-matmul modes disabled.

Support requires clean finite loss plus throughput above v084's **69,547
tok/s**. OOM, severe variance, or near-flat/lower throughput closes direct 2k
batch scaling and should pivot the lane to scheduler/kernel-shape changes.

Run dir:
`gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-12-qwen3-jax-v085-2k-bs16-batch-scale`

Compile cache:
`gs://<your-bucket>/autoresearch/qwen3_cc5/jax_lane_cache_v085_2k_bs16`.

## [2026-06-12] loop-iteration | cc5 v085 2k batch16 scale: supported (70,184 tok/s, 45.1% script MFU)

Workload `alekseyv-qwen3-jax-v085-2kbs16` completed cleanly on
`v6e-demo-hjajoo` with normal finite loss, reaching worker0 **70,184 tok/s** /
**8,773 tok/s/chip** / **45.1% script MFU** and worker1 **70,173 tok/s** /
**8,772 tok/s/chip** / **45.1% script MFU**. This is the current best 2k result.
XProf run
`2026-06-12-qwen3-jax-v085-2k-bs16-batch-scale/2026_06_12_23_26_32`
shows **3765.2 ms** average step time, **0.5 ms** standard deviation,
**66.8% MXU**, and **0.2%** op-profile idle. Peak runtime HBM was
**29.25 GiB / 31.25 GiB**, leaving only **1.9936 GiB** free.

Top HLO buckets by time were convolution fusion **60,183.943 ms / 66.7%**,
custom-call **12,220.286 ms / 13.5%**, loop fusion **11,255.317 ms / 12.5%**,
data formatting **3,598.883 ms / 4.0%**, and custom fusion
**1,912.212 ms / 2.1%**. Fresh train-step HLO
`module_0267.jit_train_step.cl_854318611.after_optimizations.txt` was
**19,785,942 bytes**, **118,485 lines**, SHA256
`6aa440aba434018b4f22849e8064082c66b383914de899ee6186c180937043f2`, with
compiled memory total **27.89 GiB** and preallocated temp **22.14 GiB**.
Verdict: supported and current frontier; raw batch scaling is now close to the
HBM ceiling.

## [2026-06-12] loop-iteration | cc5 v086 launched: 2k batch16 scoped-VMEM 98304

Launching workload `alekseyv-qwen3-jax-v086-2kbs16vmem` on
`v6e-demo-hjajoo`. This keeps v085's per-chip **batch_size=16**, global batch
**128**, and sequence length **2048**, changing only
`--xla_tpu_scoped_vmem_limit_kib` from **100352** to **98304**. Runtime stack
otherwise remains Tokamax Splash, MaxText CE, no-scan remat, activation
sharding, selective-resource latency-hiding rerun **3**, SparseCore
reduce-scatter/all-reduce offload, SparseCore collective aggregator, and
collective-matmul modes disabled.

Support requires matching or beating v085's **70,184 tok/s** with lower memory
pressure or lower variance. If throughput drops with no memory benefit, keep
the v085 **100352 KiB** schedule and pivot away from local VMEM.

Run dir:
`gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-12-qwen3-jax-v086-2k-bs16-vmem98304`

Compile cache:
`gs://<your-bucket>/autoresearch/qwen3_cc5/jax_lane_cache_v086_2k_bs16_vmem98304`.

## [2026-06-12] loop-iteration | cc5 v086 2k batch16 VMEM 98304: supported (70,564 tok/s, 45.4% script MFU)

Workload `alekseyv-qwen3-jax-v086-2kbs16vmem` completed cleanly on
`v6e-demo-hjajoo` with finite loss and `EXIT_CODE=0` on both workers. Worker0
reported **70,562 tok/s**, **8,820 tok/s/chip**, **45.4% script MFU**; worker1
reported **70,564 tok/s**, **8,820 tok/s/chip**, **45.4% script MFU**. This is
a small but valid improvement over v085's **70,184 tok/s** / **45.1%** at the
same batch16 and 2k context shape.

XProf run
`2026-06-12-qwen3-jax-v086-2k-bs16-vmem98304/2026_06_12_23_40_02` shows
**3743.6 ms** average step time, **2.5 ms** standard deviation, **67.4% MXU**,
and **0.2%** op-profile idle. Peak runtime HBM remained **29.25 GiB / 31.25
GiB**, with **1.9934 GiB** free, so the lower scoped-VMEM setting improved
schedule quality but did not create memory headroom. Top HLO buckets by time
were convolution fusion **60,205.877 ms / 67.1%**, custom-call
**12,148.863 ms / 13.5%**, loop fusion **10,919.887 ms / 12.2%**, data
formatting **3,537.343 ms / 3.9%**, and custom fusion **1,818.442 ms / 2.0%**.

Fresh train-step HLO
`module_0267.jit_train_step.cl_854318611.after_optimizations.txt` was
**19,710,086 bytes / 117,857 lines**, SHA256
`2b5d503bed2876282591e25458d1515c3d63f1a1ad371f4c6c9cb0d9a020218c`, with
compiled memory **27.89 GiB** and preallocated temp **22.13 GiB**. Verdict:
carry **98304 KiB** as the current 2k batch16 frontier, but do not continue
blind local VMEM sweeps because memory usage did not improve. Next probe should
target scheduler/codegen quality at the same shape, or rerun v086 if a
confirmation run is needed before a riskier change.

## [2026-06-12] loop-iteration | cc5 v087 launched: 2k batch16 scheduler rerun4

Launching workload `alekseyv-qwen3-jax-v087-2kbs16r4` on `v6e-demo-hjajoo`.
This keeps the v086 2k frontier fixed: per-chip **batch_size=16**, global batch
**128**, sequence length **2048**, scoped VMEM **98304 KiB**, Tokamax Splash,
MaxText CE, no-scan remat, activation sharding, SparseCore reduce-scatter and
all-reduce offload, SparseCore collective aggregator, and shared-memory limit
**90**. The only intended runtime flag change is:

```bash
--xla_latency_hiding_scheduler_rerun=4
```

Support requires beating v086's **70,564 tok/s** / **45.4% script MFU**, or
matching throughput with lower XProf step-time variance. If rerun4 slows down
or gives identical performance, keep v086's rerun3 frontier and pivot to a
different scheduler/codegen axis.

Run dir:
`gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-12-qwen3-jax-v087-2k-bs16-rerun4`

Compile cache:
`gs://<your-bucket>/autoresearch/qwen3_cc5/jax_lane_cache_v087_2k_bs16_rerun4`.

## [2026-06-12] loop-iteration | cc5 v087 2k batch16 scheduler rerun4: refuted/tie (70,554 tok/s, 45.4% script MFU)

Workload `alekseyv-qwen3-jax-v087-2kbs16r4` completed cleanly on
`v6e-demo-hjajoo` with finite loss and `EXIT_CODE=0` on both workers. Worker0
reported **70,554 tok/s**, **8,819 tok/s/chip**, **45.4% script MFU**; worker1
reported **70,541 tok/s**, **8,818 tok/s/chip**, **45.4% script MFU**. This is
slightly below v086's **70,564 tok/s** at the same 2k batch16 shape.

XProf run
`2026-06-12-qwen3-jax-v087-2k-bs16-rerun4/2026_06_12_23_55_46` shows
**3746.5 ms** average step time, **3.3 ms** standard deviation, **67.2% MXU**,
and **0.2%** op-profile idle. Peak runtime HBM was unchanged at **29.25 GiB /
31.25 GiB**, with **1.9934 GiB** free. Top HLO buckets by time were convolution
fusion **60,182.900 ms / 67.0%**, custom-call **12,145.678 ms / 13.5%**, loop
fusion **10,946.289 ms / 12.2%**, data formatting **3,538.512 ms / 3.9%**, and
custom fusion **1,967.148 ms / 2.2%**.

Fresh train-step HLO was byte-identical to v086: **19,710,086 bytes / 117,857
lines**, SHA256
`2b5d503bed2876282591e25458d1515c3d63f1a1ad371f4c6c9cb0d9a020218c`, compiled
memory **27.89 GiB**, preallocated temp **22.13 GiB**. Verdict: do not carry
`--xla_latency_hiding_scheduler_rerun=4`; keep v086's rerun3 / VMEM98304 as the
current 2k batch16 frontier.

## [2026-06-12] loop-iteration | cc5 v088 launched: 2k batch16 VMEM 97280

Launching workload `alekseyv-qwen3-jax-v088-2kbs16vm97` on
`v6e-demo-hjajoo`. This keeps the v086 2k frontier fixed: per-chip
**batch_size=16**, global batch **128**, sequence length **2048**, scheduler
rerun **3**, Tokamax Splash, MaxText CE, no-scan remat, activation sharding,
SparseCore reduce-scatter/all-reduce offload, SparseCore collective aggregator,
and shared-memory limit **90**. The only intended runtime flag change is:

```bash
--xla_tpu_scoped_vmem_limit_kib=97280
```

Support requires beating v086's **70,564 tok/s** / **45.4% script MFU**, or
improving XProf step time below **3743.6 ms**. If lower VMEM slows down or
keeps memory/profile unchanged, keep **98304 KiB** as the lower-side VMEM
frontier and pivot away from local VMEM.

Run dir:
`gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-12-qwen3-jax-v088-2k-bs16-vmem97280`

Compile cache:
`gs://<your-bucket>/autoresearch/qwen3_cc5/jax_lane_cache_v088_2k_bs16_vmem97280`.

## [2026-06-13] loop-iteration | cc5 v088 2k batch16 VMEM 97280: refuted (70,549 tok/s, 45.4% script MFU)

Workload `alekseyv-qwen3-jax-v088-2kbs16vm97` completed cleanly on
`v6e-demo-hjajoo` with finite loss and `EXIT_CODE=0` on both workers. Worker0
reported **70,546 tok/s**, **8,818 tok/s/chip**, **45.4% script MFU**; worker1
reported **70,549 tok/s**, **8,819 tok/s/chip**, **45.4% script MFU**. This is
below v086's **70,564 tok/s** at the same 2k batch16 shape.

XProf run
`2026-06-12-qwen3-jax-v088-2k-bs16-vmem97280/2026_06_13_00_09_42` shows
**3748.9 ms** average step time, **3.5 ms** standard deviation, **67.4% MXU**,
and peak runtime HBM **29.26 GiB / 31.25 GiB**, with **1.9883 GiB** free. Top
HLO buckets by time were convolution fusion **60,274.105 ms / 67.1%**,
custom-call **12,131.671 ms / 13.5%**, loop fusion **10,938.990 ms / 12.2%**,
data formatting **3,540.592 ms / 3.9%**, and custom fusion **1,884.894 ms /
2.1%**.

Fresh train-step HLO was distinct from v086: **19,719,421 bytes / 117,767
lines**, SHA256
`5ec20392283b09491d01e672599a13e450a608f3e30dbd73bfb74b6404d530d1`, compiled
memory **27.89 GiB**. Verdict: do not carry **97280 KiB**; keep v086's
**98304 KiB** as the current 2k batch16 VMEM frontier and pivot away from local
VMEM bracketing.

## [2026-06-13] loop-iteration | cc5 v089 launched: 2k batch18 on VMEM98304 frontier

Launching workload `alekseyv-qwen3-jax-v089-2kbs18` on `v6e-demo-hjajoo`.
This keeps the v086 runtime frontier fixed: sequence length **2048**, scoped
VMEM **98304 KiB**, scheduler rerun **3**, Tokamax Splash, MaxText CE, no-scan
remat, activation sharding, SparseCore reduce-scatter/all-reduce offload,
SparseCore collective aggregator, and shared-memory limit **90**. The only
intended shape change is:

```bash
--batch_size=18
```

Rationale: v086 still had about **1.99 GiB** runtime HBM free at batch16, while
v087/v088 scheduler/VMEM probes did not improve throughput. Batch18 is higher
OOM risk, but it is the direct remaining path to higher 2k MFU if the HBM
margin is sufficient.

Support requires beating v086's **70,564 tok/s** / **45.4% script MFU** without
severe XProf variance. OOM or flat/slower throughput closes raw batch scaling
above batch16 for this 2k frontier.

Run dir:
`gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-13-qwen3-jax-v089-2k-bs18-vmem98304`

Compile cache:
`gs://<your-bucket>/autoresearch/qwen3_cc5/jax_lane_cache_v089_2k_bs18_vmem98304`.

## [2026-06-13] loop-iteration | cc5 v089 2k batch18 VMEM98304: invalid/refuted (program-load HBM exhaustion)

Workload `alekseyv-qwen3-jax-v089-2kbs18` reached train-step compile but failed
before step 0 on both workers with `EXIT_CODE=1`. The runtime error was:

```text
RESOURCE_EXHAUSTED: Error loading program 'jit_train_step': Attempting to reserve 26.25G at the bottom of memory. That was not possible. There are 25.52G free, 0B reserved, and 25.52G reservable.
```

No XProf profile window was produced. Fresh optimized train-step HLO was
**18,234,233 bytes / 110,093 lines**, SHA256
`9e5ee3a2862cc0ce435db1cbd5cba9ab323c16a23b8abead397d1a68aad3bc60`, with HLO
memory report total **30.84 GiB** and preallocated temp **25.08 GiB**. Verdict:
batch18 is invalid at the v086 VMEM98304 frontier. Since the load failure missed
by roughly **0.73 GiB**, the next direct batch-scaling probe should test
**batch17**.

## [2026-06-13] loop-iteration | cc5 v090 launched: 2k batch17 on VMEM98304 frontier

Launching workload `alekseyv-qwen3-jax-v090-2kbs17` on `v6e-demo-hjajoo`.
This keeps the v086 runtime frontier fixed and changes only:

```bash
--batch_size=17
```

Rationale: v089 batch18 compiled but failed program load by about **0.73 GiB**
of bottom-of-memory reservation. Batch17 is the narrow fit check between the
supported batch16 frontier and the invalid batch18 upper bound.

Support requires beating v086's **70,564 tok/s** / **45.4% script MFU**. OOM or
flat/slower throughput closes direct batch scaling above batch16.

Run dir:
`gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-13-qwen3-jax-v090-2k-bs17-vmem98304`

Compile cache:
`gs://<your-bucket>/autoresearch/qwen3_cc5/jax_lane_cache_v090_2k_bs17_vmem98304`.

## [2026-06-13] loop-iteration | cc5 v090 2k batch17 VMEM98304: supported tie / provisional frontier (70,570 tok/s, 45.4% script MFU)

Workload `alekseyv-qwen3-jax-v090-2kbs17` completed cleanly on
`v6e-demo-hjajoo` with finite loss and `EXIT_CODE=0` on both workers. Worker0
reported **70,568 tok/s**, **8,821 tok/s/chip**, **45.4% script MFU**; worker1
reported **70,570 tok/s**, **8,821 tok/s/chip**, **45.4% script MFU**. This is
only **6 tok/s** above v086's **70,564 tok/s** batch16 result, so it is a
best-by-script result but not a durable frontier until rerun.

XProf run
`2026-06-13-qwen3-jax-v090-2k-bs17-vmem98304/2026_06_13_01_44_57` shows
**3978.0 ms** average step time, **2.4 ms** standard deviation, **67.5% MXU**,
and peak runtime HBM **30.14 GiB / 31.25 GiB**, with only **1.1031 GiB** free.
Top HLO buckets by time were convolution fusion **63,923.412 ms / 67.0%**,
custom-call **12,930.844 ms / 13.6%**, loop fusion **11,695.853 ms / 12.3%**,
data formatting **3,741.159 ms / 3.9%**, and custom fusion **1,928.880 ms /
2.0%**.

Fresh optimized train-step HLO was **19,715,571 bytes / 117,956 lines**, SHA256
`46f4d60dd591bf2ee06acbb8dfaeb1fc91bcd20f8e684fe35035079358d51acd`, with HLO
memory report total **28.78 GiB** and preallocated temp **23.03 GiB**. Verdict:
carry v090 only as provisional best-by-script; rerun batch17 exactly before
banking it, and do not probe batch18 further without a separate memory-reduction
mechanism.

## [2026-06-13] loop-iteration | cc5 v091 launched: exact 2k batch17 VMEM98304 rerun

Launching workload `alekseyv-qwen3-jax-v091-2kbs17r` on `v6e-demo-hjajoo`.
This is an exact v090 rerun except for workload identity, fresh compile cache,
and fresh profile/HLO output directory. The model/runtime shape remains:
sequence length **2048**, `--batch_size=17`, scoped VMEM **98304 KiB**,
scheduler rerun **3**, Tokamax Splash, MaxText CE, no-scan remat, activation
sharding, SparseCore reduce-scatter/all-reduce offload, SparseCore collective
aggregator, and shared-memory limit **90**.

Rationale: v090 beat v086 by only **6 tok/s** while using much tighter HBM
headroom. The loop needs an exact repeat before treating batch17 as a true
frontier.

Support requires another clean finite-loss run in the v090 band. A result near
v086/v087/v088 makes batch17 a noise-band tie rather than a durable improvement.

Run dir:
`gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-13-qwen3-jax-v091-2k-bs17-vmem98304-rerun`

Compile cache:
`gs://<your-bucket>/autoresearch/qwen3_cc5/jax_lane_cache_v091_2k_bs17_vmem98304_rerun`.

## [2026-06-13] loop-iteration | cc5 v091 2k batch17 VMEM98304 rerun: refuted as confirmation (70,557 tok/s, 45.4% script MFU)

Workload `alekseyv-qwen3-jax-v091-2kbs17r` exactly reran v090 with fresh cache
and profile/HLO output. It completed cleanly on `v6e-demo-hjajoo` with finite
loss and `EXIT_CODE=0` on both workers. Worker0 reported **70,550 tok/s**,
**8,819 tok/s/chip**, **45.4% script MFU**; worker1 reported **70,557 tok/s**,
**8,820 tok/s/chip**, **45.4% script MFU**. This does not confirm v090's
**70,570 tok/s** high-water mark and is slightly below v086's stable
**70,564 tok/s** batch16 frontier.

XProf run
`2026-06-13-qwen3-jax-v091-2k-bs17-vmem98304-rerun/2026_06_13_01_59_26` shows
**3979.0 ms** average step time, **1.6 ms** standard deviation, **67.5% MXU**,
and the same profile shape as v090. Top HLO buckets by time were convolution
fusion **63,975.526 ms / 67.1%**, custom-call **12,932.782 ms / 13.6%**, loop
fusion **11,696.227 ms / 12.3%**, data formatting **3,737.899 ms / 3.9%**, and
custom fusion **1,937.624 ms / 2.0%**.

Fresh optimized train-step HLO was byte-identical to v090: **19,715,571 bytes /
117,956 lines**, SHA256
`46f4d60dd591bf2ee06acbb8dfaeb1fc91bcd20f8e684fe35035079358d51acd`, with HLO
memory report total **28.78 GiB** and preallocated temp **23.03 GiB**. Verdict:
batch17 is a noise-band tie/slight regression, not a durable frontier. Keep v086
batch16 VMEM98304 as the supported 2k frontier and return to one-axis
scheduler/codegen probes at that shape.

## [2026-06-13] loop-iteration | cc5 v092 launched: 2k batch16 VMEM98304 shmem100

Launching workload `alekseyv-qwen3-jax-v092-2kbs16shm100` on
`v6e-demo-hjajoo`. This returns to the supported v086 shape and changes only:

```bash
--xla_tpu_scheduler_percent_shared_memory_limit=100
```

Rationale: v090/v091 close direct batch scaling above batch16. The profile is
still dominated by convolution fusion/codegen buckets, so the next 2k-specific
probe should test whether the scheduler benefits from full shared-memory budget
at batch16. The longer-context shmem100 branch was too tight and not durable,
but the 2k batch16 frontier has about **1.99 GiB** runtime HBM free and has not
tested this exact combination with VMEM98304.

Support requires beating v086's **70,564 tok/s** / **45.4% script MFU** or XProf
step time below **3743.6 ms**, without unsafe HBM headroom.

Run dir:
`gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-13-qwen3-jax-v092-2k-bs16-vmem98304-shmem100`

Compile cache:
`gs://<your-bucket>/autoresearch/qwen3_cc5/jax_lane_cache_v092_2k_bs16_vmem98304_shmem100`.

## [2026-06-13] loop-iteration | cc5 v092 2k batch16 VMEM98304 shmem100: supported provisional (70,606 tok/s, 45.4% script MFU)

Workload `alekseyv-qwen3-jax-v092-2kbs16shm100` completed cleanly on
`v6e-demo-hjajoo` with finite loss and `EXIT_CODE=0` on both workers. Worker0
reported **70,606 tok/s**, **8,826 tok/s/chip**, **45.4% script MFU**; worker1
reported **70,591 tok/s**, **8,824 tok/s/chip**, **45.4% script MFU**. This is
the new best observed script throughput for the 2k lane.

XProf run
`2026-06-13-qwen3-jax-v092-2k-bs16-vmem98304-shmem100/2026_06_13_02_13_35`
shows **3743.2 ms** average step time, **3.1 ms** standard deviation, and
**67.4% MXU**, versus v086's **3743.6 ms** and **67.4% MXU**. Top HLO buckets by
time were convolution fusion **60,181.248 ms / 67.1%**, custom-call
**12,133.557 ms / 13.5%**, loop fusion **10,918.561 ms / 12.2%**, data
formatting **3,538.545 ms / 3.9%**, and custom fusion **1,842.162 ms / 2.1%**.

Fresh optimized train-step HLO was byte-identical to v086/v087: **19,710,086
bytes / 117,857 lines**, SHA256
`2b5d503bed2876282591e25458d1515c3d63f1a1ad371f4c6c9cb0d9a020218c`, with HLO
memory report total **27.89 GiB** and preallocated temp **22.13 GiB**. Verdict:
carry v092 only provisionally because the HLO is unchanged; launch an exact
rerun before banking shmem100.

## [2026-06-13] loop-iteration | cc5 v093 launched: exact 2k batch16 VMEM98304 shmem100 rerun

Launching workload `alekseyv-qwen3-jax-v093-2kbs16shm100r` on
`v6e-demo-hjajoo`. This is an exact v092 rerun except for workload identity,
fresh compile cache, and fresh profile/HLO output directory.

Rationale: v092 set a new high-water mark, but its optimized HLO is byte-
identical to v086, so the lane needs a confirmation run before treating
`--xla_tpu_scheduler_percent_shared_memory_limit=100` as a real improvement.

Support requires another clean finite-loss run near **70,606 tok/s** or a
matching XProf step time near **3743.2 ms**. A result in the v086/v087 band
refutes shmem100 as a durable 2k improvement.

Run dir:
`gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-13-qwen3-jax-v093-2k-bs16-vmem98304-shmem100-rerun`

Compile cache:
`gs://<your-bucket>/autoresearch/qwen3_cc5/jax_lane_cache_v093_2k_bs16_vmem98304_shmem100_rerun`.

## [2026-06-13] loop-iteration | cc5 v093 2k batch16 VMEM98304 shmem100 rerun: supported (70,603 tok/s, 45.4% script MFU)

Workload `alekseyv-qwen3-jax-v093-2kbs16shm100r` exactly reran v092 with fresh
cache and profile/HLO output. It completed cleanly on `v6e-demo-hjajoo` with
finite loss and `EXIT_CODE=0` on both workers. Worker0 reported **70,599
tok/s**, **8,825 tok/s/chip**, **45.4% script MFU**; worker1 reported
**70,603 tok/s**, **8,825 tok/s/chip**, **45.4% script MFU**. This confirms the
v092/v093 shmem100 batch16 band and keeps v092's **70,606 tok/s** as the best
observed result.

XProf run
`2026-06-13-qwen3-jax-v093-2k-bs16-vmem98304-shmem100-rerun/2026_06_13_02_27_06`
shows **3741.4 ms** average step time, **3.8 ms** standard deviation, and
**67.5% MXU**, better than v086's **3743.6 ms**. Top HLO buckets by time were
convolution fusion **60,206.935 ms / 67.2%**, custom-call **12,133.267 ms /
13.5%**, loop fusion **10,917.732 ms / 12.2%**, data formatting **3,538.837 ms
/ 3.9%**, and custom fusion **1,789.932 ms / 2.0%**.

Fresh optimized train-step HLO remained byte-identical to v086/v092:
**19,710,086 bytes / 117,857 lines**, SHA256
`2b5d503bed2876282591e25458d1515c3d63f1a1ad371f4c6c9cb0d9a020218c`, with HLO
memory report total **27.89 GiB** and preallocated temp **22.13 GiB**. Verdict:
carry shmem100 as the current supported 2k batch16 runtime frontier, while
noting that it did not alter optimized HLO.

## [2026-06-13] loop-iteration | cc5 v094 launched: 2k batch17 VMEM98304 shmem100

Launching workload `alekseyv-qwen3-jax-v094-2kbs17shm100` on
`v6e-demo-hjajoo`. This keeps the supported v093 shmem100 runtime frontier and
changes only:

```bash
--batch_size=17
```

Rationale: batch17 with shmem90 was a noise-band tie/slight regression, but
v092/v093 confirmed a small shmem100 runtime improvement at batch16. This tests
whether the same runtime branch makes the adjacent batch17 shape worthwhile.

Support requires beating v092's **70,606 tok/s** high-water mark or clearly
improving the v090/v091 batch17 profile. OOM, tight-memory instability, or a
result in the prior **70.55k** batch17 band refutes further batch17 retries.

Run dir:
`gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-13-qwen3-jax-v094-2k-bs17-vmem98304-shmem100`

Compile cache:
`gs://<your-bucket>/autoresearch/qwen3_cc5/jax_lane_cache_v094_2k_bs17_vmem98304_shmem100`.

## [2026-06-13] loop-iteration | cc5 v094 2k batch17 VMEM98304 shmem100: invalid/OOM before step 0

Workload `alekseyv-qwen3-jax-v094-2kbs17shm100` failed during the first
`jit_train_step` program load on both workers. The runtime error was:

```text
RESOURCE_EXHAUSTED: Error loading program 'jit_train_step': Attempting to
allocate 2.10G. That was not possible. There are 1.02G free.
```

No valid throughput summary or XProf profile was produced. This closes direct
batch scaling above batch16 on the v092/v093 shmem100 branch: batch18 failed at
program load, batch17 at shmem90 was a noise-band tie/slight regression, and
batch17 at shmem100 leaves too little HBM for runtime allocation. Keep v092's
**70,606 tok/s / 45.4% script MFU** as the 2k high-water mark and return to
2k-specific codegen/layout probes at batch16.

## [2026-06-13] loop-iteration | cc5 v095 launched: 2k batch16 shmem100 Splash BKV2048

Launching workload `alekseyv-qwen3-jax-v095-2kbs16sp2048` on
`v6e-demo-hjajoo`. This reuses the confirmed v093 batch16 shmem100 frontier and
changes only the forward Splash KV tile:

```bash
SPLASH_BKV=2048
SPLASH_BKV_COMPUTE=2048
```

Rationale: the v092/v093 profile still spends about **13.5%** in
Splash/custom-call and the current logs show `bq=2048 bkv=1024` at seqlen
**2048**. Matching BKV to the full sequence length may reduce short-context
forward attention tiling overhead. Support requires beating v092's **70,606
tok/s** or showing a same-band run with lower custom-call/data-formatting time
and no HBM regression.

Run dir:
`gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-13-qwen3-jax-v095-2k-bs16-vmem98304-shmem100-splash2048`

Compile cache:
`gs://<your-bucket>/autoresearch/qwen3_cc5/jax_lane_cache_v095_2k_bs16_vmem98304_shmem100_splash2048`.

## [2026-06-13] loop-iteration | cc5 v095 2k batch16 shmem100 Splash BKV2048: supported provisional (70,976 tok/s, 45.6% script MFU)

Workload `alekseyv-qwen3-jax-v095-2kbs16sp2048` completed cleanly on
`v6e-demo-hjajoo` with `EXIT_CODE=0` on both workers. It changed only
`SPLASH_BKV=2048 SPLASH_BKV_COMPUTE=2048` from the v092/v093 batch16 shmem100
frontier. The logs confirmed:

```text
bq=2048 bkv=2048 bq_dkv=2048 bkv_dkv=2048 fused_bwd=True
```

Worker0 reported **70,976 tok/s / 8,872 tok/s/chip / 45.6% script MFU**.
Worker1 reported **70,970 tok/s / 8,871 tok/s/chip / 45.6% script MFU**. Loss
ended finite at **12.0467**.

XProf run
`2026-06-13-qwen3-jax-v095-2k-bs16-vmem98304-shmem100-splash2048/2026_06_13_02_54_18`
shows **3720.2 ms** average step time, **67.9% MXU**, and **0.2% idle**. Top
buckets: convolution fusion **60,158.769 ms / 67.5%**, custom-call
**11,695.131 ms / 13.1%**, loop fusion **10,908.781 ms / 12.2%**, data
formatting **3,539.631 ms / 4.0%**, custom fusion **1,793.108 ms / 2.0%**.
Compared with v093, custom-call drops from **12,133.267 ms** to **11,695.131
ms**, and profile step time improves from **3741.4 ms** to **3720.2 ms**.

Memory is the risk: XProf peak HBM is **31.14 GiB / 31.25 GiB**, with only
**0.1018 GiB** free. The optimized train-step HLO changed to **19,375,297 bytes
/ 115,424 lines**, SHA256
`a8b0c4b3833fb22bb72597d9e68cdf28fd70e18ba0a695e247b2ebf56ed69bd2`, with HLO
memory total **27.88 GiB** and preallocated temp **22.12 GiB**. Text counts vs
the v086/v092/v093 family: `copy` **20415 -> 17231**, `fusion` **30778 ->
30274**, `splash` **2428 -> 1880**, `tokamax` **2 -> 1**; `custom-call`,
`convolution`, and collectives remain same-count.

Verdict: supported provisional new 2k high-water mark. Carry BKV2048 only if an
exact rerun confirms the gain despite the very tight HBM margin. Next: v096
fresh-cache/profile exact rerun.

## [2026-06-13] loop-iteration | cc5 v096 launched: exact 2k batch16 shmem100 Splash BKV2048 rerun

Launching workload `alekseyv-qwen3-jax-v096-2ksp2048r` on `v6e-demo-hjajoo`.
This is an exact v095 rerun except for workload identity, compile cache, and
profile/HLO output path.

Rationale: v095 moved the observed 2k high-water mark to **70,976 tok/s /
45.6% script MFU** and improved profile step time to **3720.2 ms**, but it left
only **0.1018 GiB** free HBM. A clean rerun near the same band is required
before carrying `SPLASH_BKV=2048 SPLASH_BKV_COMPUTE=2048` as the durable 2k
frontier.

Run dir:
`gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-13-qwen3-jax-v096-2k-bs16-vmem98304-shmem100-splash2048-rerun`

Compile cache:
`gs://<your-bucket>/autoresearch/qwen3_cc5/jax_lane_cache_v096_2k_bs16_vmem98304_shmem100_splash2048_rerun`.

## [2026-06-13] loop-iteration | cc5 v096 exact Splash BKV2048 rerun: supported (70,959 tok/s, 45.6% script MFU)

Workload `alekseyv-qwen3-jax-v096-2ksp2048r` completed cleanly on
`v6e-demo-hjajoo` with normal finite loss ending at **12.0467**. Worker1
reported **70,959 tok/s**, **8,870 tok/s/chip**, and **45.6% script MFU**;
worker0 reported **70,957 tok/s**, **8,870 tok/s/chip**, and **45.6% script
MFU**.

XProf run
`2026-06-13-qwen3-jax-v096-2k-bs16-vmem98304-shmem100-splash2048-rerun/2026_06_13_03_10_13`
reported **3723.7 ms** average step time and **67.8% MXU**. Buckets remain in
the v095 shape: convolution fusion **60,179.211 ms / 67.4%**, custom-call
**11,696.061 ms / 13.1%**, loop fusion **10,928.655 ms / 12.2%**, data
formatting **3,543.761 ms / 4.0%**, and custom fusion **1,830.332 ms / 2.1%**.
Memory is still tight at **31.14 GiB / 31.25 GiB** peak HBM, with only
**0.1018 GiB** free.

The optimized train-step HLO is byte-identical to v095: **19,375,297 bytes /
115,424 lines**, SHA256
`a8b0c4b3833fb22bb72597d9e68cdf28fd70e18ba0a695e247b2ebf56ed69bd2`, HLO memory
total **27.88 GiB**, preallocated temp **22.12 GiB**.

Verdict: supported / confirmed 2k frontier. Carry
`SPLASH_BKV=2048 SPLASH_BKV_COMPUTE=2048` as the current best 2k stack.

## [2026-06-13] loop-iteration | cc5 v097 launched: 2k batch16 Splash BKV2048 shmem95 robustness probe

Launching workload `alekseyv-qwen3-jax-v097-2ksp2048sh95` on
`v6e-demo-hjajoo`. This changes only
`--xla_tpu_scheduler_percent_shared_memory_limit=100` to **95** from the
confirmed v095/v096 frontier. Shape, image, compile-cache isolation, profile
window, scoped VMEM **98304 KiB**, `SPLASH_BKV=2048`, and
`SPLASH_BKV_COMPUTE=2048` remain fixed.

Rationale: v095/v096 confirm the BKV2048 throughput gain, but runtime HBM leaves
only **0.1018 GiB** free. A slightly lower scheduler shared-memory cap may
produce a more robust schedule without giving back the BKV2048 throughput gain.
Support requires clean finite loss plus throughput near the confirmed
**70.95k tok/s** band, ideally with more HBM margin or no profile regression.
OOM, a return to the v092/v093 **70.60k tok/s** band, or a worse profile refutes
carrying shmem95.

Run dir:
`gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-13-qwen3-jax-v097-2k-bs16-vmem98304-splash2048-shmem95`

Compile cache:
`gs://<your-bucket>/autoresearch/qwen3_cc5/jax_lane_cache_v097_2k_bs16_vmem98304_splash2048_shmem95`.

## [2026-06-13] loop-iteration | cc5 v097 2k batch16 Splash BKV2048 shmem95: refuted/tie (70,965 tok/s, 45.6% script MFU)

Workload `alekseyv-qwen3-jax-v097-2ksp2048sh95` completed cleanly on
`v6e-demo-hjajoo` with normal finite loss ending at **12.0467**. Worker1
reported **70,965 tok/s**, **8,871 tok/s/chip**, and **45.6% script MFU**;
worker0 reported **70,958 tok/s**, **8,870 tok/s/chip**, and **45.6% script
MFU**.

XProf run
`2026-06-13-qwen3-jax-v097-2k-bs16-vmem98304-splash2048-shmem95/2026_06_13_03_26_34`
reported **3722.8 ms** average step time, **67.9% MXU**, and **0.2% idle**.
Memory did not improve versus v095/v096: peak HBM remains **31.14 GiB /
31.25 GiB**, with only **0.1018 GiB** free.

The optimized train-step HLO is byte-identical to v095/v096:
**19,375,297 bytes / 115,424 lines**, SHA256
`a8b0c4b3833fb22bb72597d9e68cdf28fd70e18ba0a695e247b2ebf56ed69bd2`, HLO memory
total **27.88 GiB**, preallocated temp **22.12 GiB**.

Verdict: refuted/tie. shmem95 is valid and remains in the BKV2048 performance
band, but it does not beat v095's **70,976 tok/s** high-water mark and gives no
HBM relief. Do not carry shmem95; keep shmem100 on the frontier.

## [2026-06-13] loop-iteration | cc5 v098 launched: 2k batch16 Splash BKV2048 VMEM100352 bracket

Launching workload `alekseyv-qwen3-jax-v098-2ksp2048vm100` on
`v6e-demo-hjajoo`. This changes only scoped VMEM from **98304 KiB** to
**100352 KiB** on the confirmed v095/v096 BKV2048 shmem100 frontier.

Rationale: the pre-BKV2048 batch16 VMEM sweep favored **98304 KiB**, but
BKV2048 changed the train-step HLO and raised runtime HBM from the v086/v092
~29.25 GiB band to **31.14 GiB**. A single high-side VMEM retest checks whether
the new attention tiling shifted the scheduler optimum. Support requires clean
finite loss plus throughput above the confirmed **70.95k tok/s** band or a
profile/memory improvement without throughput loss. OOM, a lower-profile
throughput band, or unchanged HLO/profile refutes carrying VMEM100352.

Run dir:
`gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-13-qwen3-jax-v098-2k-bs16-vmem100352-splash2048-shmem100`

Compile cache:
`gs://<your-bucket>/autoresearch/qwen3_cc5/jax_lane_cache_v098_2k_bs16_vmem100352_splash2048_shmem100`.

## [2026-06-13] loop-iteration | cc5 v098 2k batch16 Splash BKV2048 VMEM100352: refuted (70,561 tok/s, 45.4% script MFU)

Workload `alekseyv-qwen3-jax-v098-2ksp2048vm100` completed cleanly on
`v6e-demo-hjajoo` with normal finite loss ending at **12.0464**. Worker0
reported **70,561 tok/s**, **8,820 tok/s/chip**, and **45.4% script MFU**;
worker1 reported **70,552 tok/s**, **8,819 tok/s/chip**, and **45.4% script
MFU**.

XProf run
`2026-06-13-qwen3-jax-v098-2k-bs16-vmem100352-splash2048-shmem100/2026_06_13_03_41_20`
reported **3745.7 ms** average step time and **67.3% MXU**, worse than the
confirmed v095/v096 frontier. Runtime memory also worsened: peak HBM
**31.17 GiB / 31.25 GiB**, only **0.0775 GiB** free, **2.52%** fragmentation.

Fresh optimized train-step HLO changed from the v095/v096/v097 family:
**19,451,153 bytes / 116,052 lines**, SHA256
`ed70db95bbdffbae13742257ec59811d04e4feb7fe8a5f2493f2a02cb020d9de`, HLO memory
total **27.88 GiB**, preallocated temp **22.13 GiB**. Notable count changes:
`copy` **17231 -> 19182**, `all-gather` **7072 -> 6999**, `fusion`
**30274 -> 30259**; `custom-call`, `convolution`, `splash`, and `tokamax`
counts are unchanged.

Verdict: refuted. Do not carry VMEM100352; it is slower and tighter on HBM.

## [2026-06-13] loop-iteration | cc5 v099 launched: 2k batch16 Splash BKV2048 VMEM97280 low-side bracket

Launching workload `alekseyv-qwen3-jax-v099-2ksp2048vm97` on
`v6e-demo-hjajoo`. This changes only scoped VMEM from **98304 KiB** to
**97280 KiB** on the confirmed v095/v096 BKV2048 shmem100 frontier.

Rationale: v098 showed the high-side VMEM100352 schedule is distinct but slower
and more memory constrained. A low-side VMEM97280 retest checks whether the
BKV2048 stack can recover HBM headroom or a faster schedule below the current
98304 KiB point. Support requires clean finite loss plus throughput above the
confirmed **70.95k tok/s** band or materially better memory/profile with no
throughput loss. OOM, lower throughput, or no memory benefit refutes carrying
VMEM97280 and closes the immediate VMEM bracket.

Run dir:
`gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-13-qwen3-jax-v099-2k-bs16-vmem97280-splash2048-shmem100`

Compile cache:
`gs://<your-bucket>/autoresearch/qwen3_cc5/jax_lane_cache_v099_2k_bs16_vmem97280_splash2048_shmem100`.

## [2026-06-13] loop-iteration | cc5 v099 2k batch16 Splash BKV2048 VMEM97280: refuted/memory-only tie (70,899 tok/s, 45.6% script MFU)

Workload `alekseyv-qwen3-jax-v099-2ksp2048vm97` completed cleanly on
`v6e-demo-hjajoo` with normal finite loss ending at **12.0469**. Worker1
reported **70,899 tok/s**, **8,862 tok/s/chip**, and **45.6% script MFU**;
worker0 reported **70,889 tok/s**, **8,861 tok/s/chip**, and **45.6% script
MFU**.

XProf run
`2026-06-13-qwen3-jax-v099-2k-bs16-vmem97280-splash2048-shmem100/2026_06_13_03_55_34`
reported **3727.5 ms** average step time and **67.7% MXU**. Runtime memory
improved only slightly: peak HBM **31.13 GiB / 31.25 GiB**, **0.1139 GiB** free,
**1.71%** fragmentation.

Fresh optimized train-step HLO changed from the v095/v096/v097 family:
**19,384,632 bytes / 115,334 lines**, SHA256
`144caf5989b6207f46d0d69929f6adc8b030da0c96b98e066d92098a9dfca227`, HLO memory
total **27.88 GiB**, preallocated temp **22.12 GiB**. Counts:
`all-gather=7071`, `all-reduce=165`, `reduce-scatter=2774`,
`async-start=391`, `async-done=391`, `convolution=3071`,
`custom-call=3846`, `copy=17356`, `fusion=30295`, `dot_general=7337`,
`splash=1880`, `tokamax=1`.

Verdict: refuted / memory-only tie. VMEM97280 gives a tiny HBM margin gain but
regresses throughput and step time. Do not carry it for performance. The
immediate BKV2048 VMEM bracket is closed; keep **98304 KiB**.

## [2026-06-13] loop-iteration | cc5 v100 launched: 2k batch16 Splash BKV2048 DQ reduction steps 3

Launching workload `alekseyv-qwen3-jax-v100-2ksp2048dq3` on
`v6e-demo-hjajoo`. This returns to the confirmed v095/v096 frontier
(VMEM **98304 KiB**, shmem100, BKV2048) and changes only
`TOKAMAX_DQ_REDUCTION_STEPS=3`.

Rationale: VMEM and shmem scalar probes did not beat the frontier. The profile
still spends about **13.1%** in custom-call/Splash, and prior long-context
DQ-reduction probes changed the train-step HLO and reduced DKV custom-call time
but regressed convolution time. At 2k context with full-sequence BKV2048, the
same mechanism may have a different balance. Support requires clean finite loss
plus throughput above the confirmed **70.95k tok/s** band or a profile win in
custom-call/attention without wall-time loss. A changed HLO with lower
throughput or worse convolution refutes carrying DQ reduction steps.

Run dir:
`gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-13-qwen3-jax-v100-2k-bs16-vmem98304-splash2048-dqsteps3`

Compile cache:
`gs://<your-bucket>/autoresearch/qwen3_cc5/jax_lane_cache_v100_2k_bs16_vmem98304_splash2048_dqsteps3`.

## [2026-06-13] loop-iteration | cc5 v100 2k batch16 Splash BKV2048 DQ reduction steps 3: failed compile OOM

Workload `alekseyv-qwen3-jax-v100-2ksp2048dq3` failed before measured
throughput with an XLA TPU compile-time HBM OOM. The runtime error reported
**32.62 GiB** used against **31.25 GiB** capacity, exceeding HBM by
**1.38 GiB**; total HBM usage was at least **32.88 GiB** with program
**26.90 GiB** and arguments **5.72 GiB**.

The optimized train-step HLO was still dumped and copied to
`/tmp/qwen3-v100-hlo/`. HLO SHA is
`9b4f56aa346c8eff4ba2826cbd834efbfeb3db1fe51d3a8541a6916b0cda0c0d`
(**19,559,653 bytes**, **116,098 lines**). The HLO memory report shows
**32.56 GiB** total bytes and **26.80 GiB** preallocated temp, far above the
v095/v096 frontier. Counts also shifted upward versus v096/v097:
`custom-call=3924`, `convolution=3138`, `async-start=397`, `async-done=397`.

Additional diagnostic: the runtime Splash config line continued to show
`bq_dq=None bkv_dq=None`, so `TOKAMAX_DQ_REDUCTION_STEPS=3` did not express as
the explicit DQ tile knobs. Do not carry this reduction-steps knob on the 2k
BKV2048 frontier. If DQ tuning is retried, use the explicit DQ tile env knobs
or pair it with a memory-reducing mechanism.

## [2026-06-13] loop-iteration | cc5 v101 launched: 2k batch16 Splash BKV2048 shmem90

Launching workload `alekseyv-qwen3-jax-v101-2ksp2048sh90` on
`v6e-demo-hjajoo`. This keeps the confirmed v095/v096 BKV2048 frontier fixed
(batch16, seqlen2048, scoped VMEM **98304 KiB**, Tokamax Splash, MaxText CE,
SparseCore RS/AR offload plus aggregator) and changes only
`--xla_tpu_scheduler_percent_shared_memory_limit=100` to **90**.

Rationale: v097 showed shmem95 was byte-identical to the frontier and tied
throughput, while v100 showed DQ-reduction HLO growth immediately OOMs. A
stronger shared-memory cap is a low-risk scheduler probe that may change async
collective/Splash staging or improve HBM headroom. Support requires throughput
above the confirmed **70.95k tok/s** band, or same-band throughput with clear
runtime memory improvement.

Run dir:
`gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-13-qwen3-jax-v101-2k-bs16-vmem98304-splash2048-shmem90`

Compile cache:
`gs://<your-bucket>/autoresearch/qwen3_cc5/jax_lane_cache_v101_2k_bs16_vmem98304_splash2048_shmem90`.

## [2026-06-13] loop-iteration | cc5 v101 2k batch16 Splash BKV2048 shmem90: refuted for speed, useful headroom branch

Workload `alekseyv-qwen3-jax-v101-2ksp2048sh90` completed cleanly on
`v6e-demo-hjajoo`. Worker0 reported **70,918 tok/s** and worker1
**70,913 tok/s**, both **45.6% script MFU** with finite loss ending at
**12.0467**. This is below the v095/v096 best band (**70,959-70,976 tok/s**),
so shmem90 is not a speed replacement.

XProf run
`2026-06-13-qwen3-jax-v101-2k-bs16-vmem98304-splash2048-shmem90/2026_06_13_04_23_40`
shows **3723.2 ms**, **67.7% MXU**, and **0.2% idle**. Top buckets are
convolution fusion **60,203.978 ms / 67.5%**, custom-call
**11,697.219 ms / 13.1%**, and loop fusion **10,928.690 ms / 12.2%**.

The key positive signal is runtime HBM: **29.24 GiB / 31.25 GiB** with
**2.0020 GiB free**, versus roughly **0.10 GiB free** on the v095/v096/v097
BKV2048 shmem100/95 branch. The train-step HLO is byte-identical to the speed
frontier: SHA
`a8b0c4b3833fb22bb72597d9e68cdf28fd70e18ba0a695e247b2ebf56ed69bd2`,
**19,375,297 bytes / 115,424 lines**, HLO memory **27.88 GiB** with
**22.12 GiB** preallocated temp. Counts:
`all-gather=7072`, `all-reduce=165`, `reduce-scatter=2774`,
`custom-call=3846`, `convolution=3071`, `copy=17231`.

Verdict: do not carry shmem90 for best batch16 throughput, but use it as a
headroom branch. Next: v102 tests whether that headroom lets **batch17 +
BKV2048** fit and beat the batch16 speed frontier.

## [2026-06-13] loop-iteration | cc5 v102 launched: 2k batch17 Splash BKV2048 shmem90

Launching workload `alekseyv-qwen3-jax-v102-2kbs17sp2048sh90` on
`v6e-demo-hjajoo`. This reuses the v101 BKV2048 shmem90 headroom branch and
changes only `--batch_size=16` to `--batch_size=17`.

Rationale: v101 does not beat batch16 speed, but it leaves **2.0020 GiB** free
at runtime, much more than the **~0.10 GiB** free v095/v096 speed branch.
Earlier direct batch17 probes without BKV2048 were only a noise-band tie, and
batch17 on shmem100 failed allocation. This run tests whether BKV2048 plus the
shmem90 headroom is enough for batch17 to fit and lift tokens/sec above the
confirmed **70.95k tok/s** band.

Run dir:
`gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-13-qwen3-jax-v102-2k-bs17-vmem98304-splash2048-shmem90`

Compile cache:
`gs://<your-bucket>/autoresearch/qwen3_cc5/jax_lane_cache_v102_2k_bs17_vmem98304_splash2048_shmem90`.

## [2026-06-13] loop-iteration | cc5 v102 2k batch17 Splash BKV2048 shmem90: refuted (70,882 tok/s, 45.6% script MFU)

Workload `alekseyv-qwen3-jax-v102-2kbs17sp2048sh90` completed cleanly on
`v6e-demo-hjajoo`. Worker0 reported **70,875 tok/s** and worker1 **70,882
tok/s**, both **45.6% script MFU**, with finite loss ending at **12.0464**.
This fits, but it is below both the v101 shmem90 batch16 headroom branch and
the v095/v096 speed frontier.

XProf run
`2026-06-13-qwen3-jax-v102-2k-bs17-vmem98304-splash2048-shmem90/2026_06_13_04_38_38`
shows **3963.8 ms**, **67.8% MXU**, and **0.2% idle**. Top buckets are
convolution fusion **64,000.513 ms / 67.4%**, custom-call
**12,442.653 ms / 13.1%**, and loop fusion **11,702.587 ms / 12.3%**.
Runtime HBM was **30.13 GiB / 31.25 GiB** with **1.1117 GiB free**.

The train-step HLO is a distinct batch17 program:
`53bb713300fc39dad07b7869e6cfed54010b2a7bcfd9db57e16b037f9e393610`,
**19,380,811 bytes / 115,523 lines**, HLO memory **28.77 GiB** with
**23.02 GiB** preallocated temp. Counts:
`all-gather=7069`, `all-reduce=165`, `reduce-scatter=2774`,
`custom-call=3842`, `convolution=3071`, `copy=17669`.

Verdict: refuted. shmem90 creates enough headroom to fit batch17+BKV2048, but
step time rises too much and tokens/sec regresses. Close batch17+BKV2048. Next:
probe an intermediate shared-memory cap between shmem95 (speed tie, no headroom
relief) and shmem90 (headroom win, speed loss).

## [2026-06-13] loop-iteration | cc5 v103 launched: 2k batch16 Splash BKV2048 shmem92

Launching workload `alekseyv-qwen3-jax-v103-2ksp2048sh92` on
`v6e-demo-hjajoo`. This returns to batch16 BKV2048 and changes only
`--xla_tpu_scheduler_percent_shared_memory_limit=100` to **92**.

Rationale: shmem95 tied the speed frontier but did not improve HBM; shmem90
created **2.0020 GiB** free HBM but regressed throughput by about **58-63
tok/s**. shmem92 tests whether the threshold can provide useful headroom while
keeping throughput near or above the confirmed **70.95k tok/s** band.

Run dir:
`gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-13-qwen3-jax-v103-2k-bs16-vmem98304-splash2048-shmem92`

Compile cache:
`gs://<your-bucket>/autoresearch/qwen3_cc5/jax_lane_cache_v103_2k_bs16_vmem98304_splash2048_shmem92`.

## [2026-06-13] loop-iteration | cc5 v103 2k batch16 Splash BKV2048 shmem92: refuted for speed, useful headroom bracket

Workload `alekseyv-qwen3-jax-v103-2ksp2048sh92` completed cleanly on
`v6e-demo-hjajoo` with `EXIT_CODE=0` on both workers. It reached worker0
**70,928 tok/s** / **8,866/chip** / **45.6%** script MFU and worker1
**70,929 tok/s** / **8,866/chip** / **45.6%**, with normal loss ending at
**12.0467**. This is below the confirmed v095/v096 frontier
(**70,959-70,976 tok/s**), so shmem92 is not a speed improvement.

XProf run
`2026-06-13-qwen3-jax-v103-2k-bs16-vmem98304-splash2048-shmem92/2026_06_13_04_53_02`
shows **3722.8 ms** average step time, **67.7%** MXU, and **0.2%** idle.
Op profile remains the same shape as the shmem90 branch: convolution fusion
**60,201.382 ms / 67.5%**, custom-call **11,697.155 ms / 13.1%**, loop fusion
**10,932.819 ms / 12.3%**, Splash fwd residuals **5,905.514 ms / 6.6%**, and
Splash DKV **5,789.964 ms / 6.5%**. Memory profile is the useful part:
**29.24 / 31.25 GiB** peak HBM with **2.0020 GiB** free, stack reservation
**23.2933 GiB**, heap **5.9508 GiB**. The optimized train-step HLO is
byte-identical to v101:
`a8b0c4b3833fb22bb72597d9e68cdf28fd70e18ba0a695e247b2ebf56ed69bd2`,
**19,375,297 bytes / 115,424 lines**, compiled memory **27.88 GiB**, prealloc
temp **22.12 GiB**, counts `all-gather=7072`, `all-reduce=165`,
`reduce-scatter=2774`, `async-start=391`, `async-done=391`,
`convolution=3071`, `custom-call=3846`, `copy=17231`, `fusion=30274`,
`dot_general=7301`, `tokamax=1`.

Verdict: refuted for speed, but it brackets the shared-memory threshold.
Next: v104 tests shmem93, the last midpoint between HBM-tight shmem95 and the
headroom-but-slower shmem92/shmem90 branch.

## [2026-06-13] loop-iteration | cc5 v104 launched: 2k batch16 Splash BKV2048 shmem93

Launching workload `alekseyv-qwen3-jax-v104-2ksp2048sh93` on
`v6e-demo-hjajoo`. This reuses v095/v096 exactly except
`--xla_tpu_scheduler_percent_shared_memory_limit=93`, keeping batch16,
seqlen2048, VMEM98304, BKV2048/BKV_COMPUTE2048, no-scan remat, activation
sharding, Tokamax Splash, MaxText CE, SparseCore RS/AR offload + aggregator,
and collective-matmul modes none.

Run dir:
`gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-13-qwen3-jax-v104-2k-bs16-vmem98304-splash2048-shmem93`

Compile cache:
`gs://<your-bucket>/autoresearch/qwen3_cc5/jax_lane_cache_v104_2k_bs16_vmem98304_splash2048_shmem93`.

## [2026-06-13] loop-iteration | cc5 v104 2k batch16 Splash BKV2048 shmem93: refuted, tight-HBM side of threshold

Workload `alekseyv-qwen3-jax-v104-2ksp2048sh93` completed cleanly on
`v6e-demo-hjajoo` with `EXIT_CODE=0` on both workers. It reached worker0
**70,946 tok/s** / **8,868/chip** / **45.6%** script MFU and worker1
**70,949 tok/s** / **8,869/chip** / **45.6%**, with normal loss ending at
**12.0467**. This is slightly faster than shmem92 but still below the confirmed
v095/v096 frontier (**70,959-70,976 tok/s**).

XProf run
`2026-06-13-qwen3-jax-v104-2k-bs16-vmem98304-splash2048-shmem93/2026_06_13_05_08_43`
shows **3725.4 ms** average step time, **67.8%** MXU, and **0.2%** idle. Op
profile remains frontier-shaped: convolution fusion **60,203.395 ms / 67.4%**,
custom-call **11,695.673 ms / 13.1%**, loop fusion **10,925.321 ms / 12.2%**,
Splash fwd residuals **5,905.679 ms / 6.6%**, and Splash DKV **5,788.313 ms /
6.5%**. Memory profile is tight again: **31.14 / 31.25 GiB** peak HBM, only
**0.1018 GiB** free, stack reservation **23.2933 GiB**, heap **7.8510 GiB**.
The optimized train-step HLO is byte-identical to v095/v096/v097/v101/v103:
`a8b0c4b3833fb22bb72597d9e68cdf28fd70e18ba0a695e247b2ebf56ed69bd2`,
**19,375,297 bytes / 115,424 lines**, compiled memory **27.88 GiB**, prealloc
temp **22.12 GiB**, counts `all-gather=7072`, `all-reduce=165`,
`reduce-scatter=2774`, `async-start=391`, `async-done=391`,
`convolution=3071`, `custom-call=3846`, `copy=17231`, `fusion=30274`,
`dot_general=7301`, `tokamax=1`.

Verdict: shmem93 is refuted. The runtime HBM transition sits between shmem92
and shmem93: shmem92/90 give about **2.0 GiB** free but slower runtime, while
shmem93/95/100 are tight-HBM and faster. Pivot away from shmem bracketing.

## [2026-06-13] loop-iteration | cc5 v105 launched: 2k batch16 Splash BKV2048 all scheduler features true

Launching workload `alekseyv-qwen3-jax-v105-2ksp2048allsched` on
`v6e-demo-hjajoo`. This reuses the confirmed v095/v096 BKV2048 frontier and
changes only `--xla_tpu_enable_all_experimental_scheduler_features=false` to
`true`. The intent is to test whether the broad scheduler bundle helps the
current attention/custom-call schedule, where older qseq runs suggested `true`
could be marginally better but later simplified stacks tolerated `false`.

Run dir:
`gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-13-qwen3-jax-v105-2k-bs16-vmem98304-splash2048-allsched`

Compile cache:
`gs://<your-bucket>/autoresearch/qwen3_cc5/jax_lane_cache_v105_2k_bs16_vmem98304_splash2048_allsched`.

## [2026-06-13] loop-iteration | cc5 v105 2k batch16 Splash BKV2048 all scheduler features true: refuted (70,943 tok/s, 45.6% script MFU)

Workload `alekseyv-qwen3-jax-v105-2ksp2048allsched` completed cleanly on
`v6e-demo-hjajoo` with normal finite loss, reaching worker0 **70,932 tok/s**
and worker1 **70,943 tok/s**, both **45.6%** script MFU. This does not beat
the confirmed v095/v096 frontier (**70,959-70,976 tok/s**).

XProf run
`2026-06-13-qwen3-jax-v105-2k-bs16-vmem98304-splash2048-allsched/2026_06_13_05_24_05`
reported **3725.5 ms** average step time, **67.8%** MXU, and **0.2%** idle.
The op-profile shape remained in-family: convolution fusion
**60,171.262 ms / 67.4%**, custom-call **11,695.170 ms / 13.1%**, loop fusion
**10,931.714 ms / 12.2%**, forward residuals **5,905.176 ms**, and DKV total
**5,788.314 ms**. Peak HBM was still tight at **31.14 GiB / 31.25 GiB** with
only **0.1018 GiB** free.

Fresh optimized train-step HLO under `/tmp/qwen3-v105-hlo/` was byte-identical
to v095/v096/v103/v104: SHA256
`a8b0c4b3833fb22bb72597d9e68cdf28fd70e18ba0a695e247b2ebf56ed69bd2`,
**19,375,297 bytes / 115,424 lines**, compiled memory **27.88 GiB**, and
preallocated temp **22.12 GiB**. Counts remained `all-gather=7072`,
`all-reduce=165`, `reduce-scatter=2774`, `async-start=391`,
`async-done=391`, `convolution=3071`, `custom-call=3846`, `copy=17231`,
`fusion=30274`, `dot_general=7301`, `tokamax=1`.

Verdict: refuted for speed. Keep
`--xla_tpu_enable_all_experimental_scheduler_features=false` on the BKV2048
frontier and pivot to narrower collective-fusion probes.

## [2026-06-13] loop-iteration | cc5 v106 launched: 2k batch16 Splash BKV2048 no reduce-scatter ACF

Launching workload `alekseyv-qwen3-jax-v106-2ksp2048noacfrs` on
`v6e-demo-hjajoo`. This reuses v095/v096 exactly and changes only
`--xla_tpu_enable_async_collective_fusion_fuse_reduce_scatter=true` to
`false`, while leaving base ACF, all-gather ACF, multi-step ACF, shmem100,
VMEM **98304 KiB**, SparseCore reduce-scatter/all-reduce offload, and
`SPLASH_BKV=2048` / `SPLASH_BKV_COMPUTE=2048` unchanged.

Support requires beating the v095/v096 **70,959-70,976 tok/s** band, or at
least matching it with lower XProf step time or less custom-call/loop-fusion
time. A same-HLO in-band runtime refutes this reduce-scatter ACF ablation.

Run dir:
`gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-13-qwen3-jax-v106-2k-bs16-vmem98304-splash2048-noacfrs`

Compile cache:
`gs://<your-bucket>/autoresearch/qwen3_cc5/jax_lane_cache_v106_2k_bs16_vmem98304_splash2048_noacfrs`.

## [2026-06-13] loop-iteration | cc5 v106 2k batch16 Splash BKV2048 no reduce-scatter ACF: refuted (70,945 tok/s, 45.6% script MFU)

Workload `alekseyv-qwen3-jax-v106-2ksp2048noacfrs` completed cleanly on
`v6e-demo-hjajoo` with normal finite loss, reaching worker0 **70,945 tok/s** /
**8,868 tok/s/chip** / **45.6%** script MFU and worker1 **70,941 tok/s** /
**8,868 tok/s/chip** / **45.6%** script MFU. This is below the confirmed
v095/v096 frontier (**70,959-70,976 tok/s**).

XProf run
`2026-06-13-qwen3-jax-v106-2k-bs16-vmem98304-splash2048-noacfrs/2026_06_13_05_41_03`
reported **3726.5 ms** average step time, **67.8%** MXU, and **0.2%** idle.
Memory remained tight at **31.14 GiB / 31.25 GiB**, **0.1018 GiB** free,
with **23.2933 GiB** stack and **7.8510 GiB** heap. Op profile stayed
in-family: convolution fusion **60,202.129 ms / 67.4%** and data formatting
**3,541.861 ms / 4.0%** in the top-op extract, with the other major buckets
matching the prior BKV2048 profile shape.

Fresh optimized train-step HLO under `/tmp/qwen3-v106-hlo/` had SHA256
`a44dcef073d4f52793299c1a64717bbcf09fce4d2b8a3d73b21248f137cc6226`,
**19,375,306 bytes / 115,424 lines**, compiled memory **27.88 GiB**, and
preallocated temp **22.12 GiB**. Counts were unchanged from the frontier:
`all-gather=7072`, `all-reduce=165`, `reduce-scatter=2774`,
`async-start=391`, `async-done=391`, `convolution=3071`,
`custom-call=3846`, `copy=17231`, `fusion=30274`, `dot_general=7301`,
`tokamax=1`. Diff versus v105 shows instruction renumbering rather than a
meaningful structural change.

Verdict: refuted. Keep reduce-scatter ACF enabled on the BKV2048 frontier.

## [2026-06-13] loop-iteration | cc5 v107 launched: 2k batch16 Splash BKV2048 all-reduce ACF

Launching workload `alekseyv-qwen3-jax-v107-2ksp2048acf-ar` on
`v6e-demo-hjajoo`. This reuses v095/v096 exactly and changes only
`--xla_tpu_enable_async_collective_fusion_fuse_all_reduce=false` to `true`.
All-gather ACF, reduce-scatter ACF, multi-step ACF, shmem100, VMEM
**98304 KiB**, SparseCore reduce-scatter/all-reduce offload, and
`SPLASH_BKV=2048` / `SPLASH_BKV_COMPUTE=2048` remain unchanged.

Support requires beating the v095/v096 **70,959-70,976 tok/s** band, or at
least matching it with lower XProf step time. Same-HLO in-band runtime means
all-reduce ACF is inert on this frontier.

Run dir:
`gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-13-qwen3-jax-v107-2k-bs16-vmem98304-splash2048-acf-ar`

Compile cache:
`gs://<your-bucket>/autoresearch/qwen3_cc5/jax_lane_cache_v107_2k_bs16_vmem98304_splash2048_acf_ar`.

## [2026-06-13] loop-iteration | cc5 v107 2k batch16 Splash BKV2048 all-reduce ACF: refuted/inert (70,945 tok/s, 45.6% script MFU)

Workload `alekseyv-qwen3-jax-v107-2ksp2048acf-ar` completed cleanly on
`v6e-demo-hjajoo` with normal finite loss, reaching worker0 **70,943 tok/s** /
**8,868 tok/s/chip** / **45.6%** script MFU and worker1 **70,945 tok/s** /
**8,868 tok/s/chip** / **45.6%** script MFU. This does not beat the confirmed
v095/v096 frontier (**70,959-70,976 tok/s**).

XProf run
`2026-06-13-qwen3-jax-v107-2k-bs16-vmem98304-splash2048-acf-ar/2026_06_13_05_56_10`
reported **3725.1 ms** average step time, **67.8%** MXU, and **0.2%** idle.
Memory stayed unchanged at **31.14 GiB / 31.25 GiB** peak with only
**0.1018 GiB** free. Op profile remained matched to the BKV2048 frontier:
convolution fusion **60,190.155 ms / 67.4%**, custom-call
**11,695.236 ms / 13.1%**, loop fusion **10,921.098 ms / 12.2%**, forward
residuals **5,905.228 ms**, and DKV total **5,788.328 ms**.

Fresh optimized train-step HLO under `/tmp/qwen3-v107-hlo/` was byte-identical
to v095/v096/v105: SHA256
`a8b0c4b3833fb22bb72597d9e68cdf28fd70e18ba0a695e247b2ebf56ed69bd2`,
**19,375,297 bytes / 115,424 lines**, compiled memory **27.88 GiB**, and
preallocated temp **22.12 GiB**. Counts stayed `all-gather=7072`,
`all-reduce=165`, `reduce-scatter=2774`, `async-start=391`,
`async-done=391`, `convolution=3071`, `custom-call=3846`, `copy=17231`,
`fusion=30274`, `dot_general=7301`, `tokamax=1`.

Verdict: refuted/inert. Keep all-reduce ACF disabled on the BKV2048 frontier.

## [2026-06-13] loop-iteration | cc5 v108 launched: 2k batch16 Splash BKV2048 async all-gather cap1

Launching workload `alekseyv-qwen3-jax-v108-2ksp2048ag1` on
`v6e-demo-hjajoo`. This reuses v095/v096 exactly and adds only
`--xla_max_concurrent_async_all_gathers=1`. The prior long-seq cap1 runs were
mostly same-HLO ties while cap2 exposed direct all-gather and regressed; the
current check is whether cap1 helps the tight-HBM 2k BKV2048 schedule.

Support requires beating the v095/v096 **70,959-70,976 tok/s** band, or at
least matching it with lower XProf step time or lower all-gather/convolution
fusion tail. Same-HLO in-band runtime refutes the cap.

Run dir:
`gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-13-qwen3-jax-v108-2k-bs16-vmem98304-splash2048-ag1`

Compile cache:
`gs://<your-bucket>/autoresearch/qwen3_cc5/jax_lane_cache_v108_2k_bs16_vmem98304_splash2048_ag1`.

## [2026-06-13] loop-iteration | cc5 v108 2k batch16 Splash BKV2048 async all-gather cap1: refuted/default (70,939 tok/s, 45.6% script MFU)

Workload `alekseyv-qwen3-jax-v108-2ksp2048ag1` completed cleanly on
`v6e-demo-hjajoo` with normal finite loss, reaching worker0 **70,935 tok/s** /
**8,867 tok/s/chip** / **45.6%** script MFU and worker1 **70,939 tok/s** /
**8,867 tok/s/chip** / **45.6%** script MFU. This is below the confirmed
v095/v096 frontier (**70,959-70,976 tok/s**).

XProf run
`2026-06-13-qwen3-jax-v108-2k-bs16-vmem98304-splash2048-ag1/2026_06_13_06_11_14`
reported **3726.6 ms** average step time, **67.8%** MXU, and **0.2%** idle.
Memory was unchanged at **31.14 GiB / 31.25 GiB** peak with only
**0.1018 GiB** free. Op profile remained matched to the BKV2048 frontier:
convolution fusion **60,194.825 ms / 67.4%**, custom-call
**11,695.335 ms / 13.1%**, loop fusion **10,922.114 ms / 12.2%**, forward
residuals **5,905.324 ms**, and DKV total **5,788.331 ms**.

Fresh optimized train-step HLO under `/tmp/qwen3-v108-hlo/` was byte-identical
to v095/v096/v105/v107: SHA256
`a8b0c4b3833fb22bb72597d9e68cdf28fd70e18ba0a695e247b2ebf56ed69bd2`,
**19,375,297 bytes / 115,424 lines**, compiled memory **27.88 GiB**, and
preallocated temp **22.12 GiB**. Counts stayed `all-gather=7072`,
`all-reduce=165`, `reduce-scatter=2774`, `async-start=391`,
`async-done=391`, `convolution=3071`, `custom-call=3846`, `copy=17231`,
`fusion=30274`, `dot_general=7301`, `tokamax=1`.

Flagfile inspection showed v095 already resolved
`--xla_max_concurrent_async_all_gathers=1`, so v108 made an existing default
explicit. Verdict: refuted/default. Do not carry any change.

## [2026-06-13] loop-iteration | cc5 v109 launched: 2k batch16 Splash BKV2048 async reduce-scatter cap1

Launching workload `alekseyv-qwen3-jax-v109-2ksp2048rs1` on
`v6e-demo-hjajoo`. This reuses v095/v096 exactly and adds only
`--xla_max_concurrent_async_reduce_scatters=1`. The current flagfile leaves
this axis at `auto`, unlike all-gather concurrency which already resolves to
1. This tests whether constraining reduce-scatter async concurrency helps the
SparseCore reduce-scatter offload schedule or is a no-op.

Support requires beating the v095/v096 **70,959-70,976 tok/s** band, or at
least matching it with lower XProf step time. Same-HLO in-band runtime refutes
the cap.

Run dir:
`gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-13-qwen3-jax-v109-2k-bs16-vmem98304-splash2048-rs1`

Compile cache:
`gs://<your-bucket>/autoresearch/qwen3_cc5/jax_lane_cache_v109_2k_bs16_vmem98304_splash2048_rs1`.

## [2026-06-12] stop | /stop-experiment session end

**Op**: stop
**Pages created**: none
**Pages updated**: `wiki/experiments/qwen3_cx_autoresearch_optimization/jax/log.md`
**Notes**: Clean shutdown via /stop-experiment. Loop markers removed from
`.claude/.loop_active.json` and `.codex/.loop_active.json`; no further loop
iterations should be scheduled.
**Reaped orphan workloads**: none. `kubectl` found no non-terminal
`alekseyv-qwen3*` JobSets; retained completed/failed JobSets were left in place.
**Outstanding lint items**: shutdown-critical checks passed for live work
(`status: in_progress` stubs: 0; active Qwen3 workloads: 0; per-lane bad
log-operation routing: 0). Structural backlog remains: recent v075-v080
experiment pages need canonical `## Profile` / `## HLO Dump` / phase-3 audit
headings, repo-wide warnings count is 3, and global `wiki/log.md` still has 3
historical `loop-iteration` entries from 2026-06-05.
**Session metrics**: latest stopped tail covered v079 and v080 after the MaxText
comparison branch: v079 explicit-SiLU shmem95 was refuted at 59,326 tok/s /
4434.2 ms; v080 sparse-label MaxText CE was invalid/refuted due to compile OOM
at 39.19 GiB required HBM. Frontier remains the current shmem90 explicit-SiLU
family at about 59,326-59,330 tok/s, with historical v273 still the old
high-water mark at 60,346 tok/s but not reproduced on the current cluster.

## [2026-06-12] loop-iteration | cc5 v080 sparse-label MaxText CE: invalid/refuted (compile OOM; 39.19 GiB HBM requirement)

Workload `alekseyv-qwen3-cc5-jax-v080-sparsece` failed during first train-step
compile before any throughput metrics. XLA reported a permanent TPU compile OOM:
used **39.19 GiB** of **31.25 GiB** HBM, exceeding capacity by **7.95 GiB**.
The program required **39.19 GiB**, with **39.11 GiB** HLO temp; total HBM usage
including reservation was at least **39.45 GiB**.

The OOM top allocations show the sparse-label CE path made the full-vocab logits
live as both **18.55 GiB** `f32[4,8192,151936]` and **9.27 GiB**
`bf16[4,8192,151936]` temporaries. The compile memory report for the failed HLO
listed **44.86 GiB** total bytes and a **39.11 GiB** preallocated temp.

Fresh failed train-step HLO in `/tmp/qwen3-v080-hlo` has SHA256
`b53b3502a8c775f87c39060a6da2a0b53392bff8dca0ff5108d1a0901d3e2d5a`,
**18,953,768 bytes / 114,408 lines**. Counts changed materially:
`all-gather=4134`, `all-reduce=158`, `reduce-scatter=1766`,
`async-start=414`, `async-done=414`, `convolution=2070`,
`custom-call=2216`, `copy=10427`, `fusion=19939`, `dot_general=6673`,
`scatter=2044`, `gather=4168`, `dynamic-update-slice=216`, `splash=2210`,
`tokamax=2`, `jvp_jit_silu__=0`, `exponential=182`.

Verdict: invalid/refuted. The local sparse-label CE VJP was mathematically
equivalent, but the full-shape HLO became scatter/gather-heavy and worsened
logits live range. Do not carry this implementation; a viable CE direction must
avoid full `[B,T,V]` logits, not merely avoid one-hot target construction.

## [2026-06-12] loop-iteration | cc5 v080 launched: sparse-label MaxText CE

Launching workload `alekseyv-qwen3-cc5-jax-v080-sparsece` to test a
source-level CE shape change after closing the shared-memory bracket. The
experiment image `qwen3-8b-jax:v080-sparse-maxtext-ce` is a thin layer over
`qwen3-8b-jax:v067-explicit-silu` that replaces only `train.py`: the
`use_maxtext_ce` path now calls a sparse-label custom VJP instead of building a
full one-hot `[B,T,V]` target tensor before MaxText/T5X CE.

Local CPU equivalence against the original one-hot CE passed inside the image
context: identical loss **3.2754132747650146**, max loss diff **0.0**, max
logits-gradient diff **7.45e-09**, and `allclose(atol=1e-6, rtol=1e-6)=true`.

Runtime stack remains the carried explicit-SiLU shmem90 configuration: np0
placement, no-scan remat, activation sharding, Tokamax Splash max-logit 30,
MaxText CE flag enabled, batch size **4** per chip / global batch **32**,
all-gather and reduce-scatter ACF enabled, all-reduce ACF disabled, SparseCore
reduce-scatter/all-reduce offload, SparseCore collective aggregator, RS latency
multiplier **3**, AR latency multiplier **2**, selective resources, scoped VMEM
**100352**, shared-memory limit **90**, and latency-hiding scheduler rerun
**3**.

Support requires clean finite loss plus lower HLO memory/size or lower XProf
time without dropping below the durable **~59.33k tok/s / 4433.8 ms** frontier.

Run/cache paths:
`gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-12-qwen3-jax-v080-v067-sparse-maxtext-ce`
and
`gs://<your-bucket>/autoresearch/qwen3_cc5/jax_lane_cache_v080_sparse_maxtext_ce`.

## [2026-06-12] loop-iteration | cc5 v079 explicit-SiLU shmem95 midpoint: refuted (59,326 host-best, 4434.2 ms; same shmem100 HLO)

Workload `alekseyv-qwen3-cc5-jax-v079-shm95` completed cleanly on np0 with
finite loss and no compile OOM, but did not improve the frontier. Worker
summaries: `gke-tpu-0d0013c1-l5jf` reported **59,308 tok/s / 7,414
tok/s/chip / 42.5% old MFU** (about **48.35% MaxText-style MFU**), while
`gke-tpu-0d0013c1-9rkl` reported **59,326 tok/s / 7,416 tok/s/chip / 42.5%
old MFU** (about **48.37% MaxText-style MFU**).

XProf
`2026-06-12-qwen3-jax-v079-v067-shmem95/2026_06_12_17_08_35`
reported **4434.2 ms** average step time, **1.3 ms** stddev, **67.0% MXU**,
and **0.2%** idle. Peak HBM matched the shmem100 family at
**30.93 GiB / 31.25 GiB**, leaving only **0.3181 GiB** free.

Fresh train-step HLO in `/tmp/qwen3-v079-hlo` was byte-identical to v077/v078:
SHA256 `4e271536a745449917e002dcb8f554b509bb0d7cff82da955d0ad7005ca97ebc`,
**19,895,806 bytes / 119,353 lines**, with aggregate counts
`all-gather=4628`, `all-reduce=158`, `reduce-scatter=1766`,
`async-start=394`, `async-done=394`, `convolution=2106`,
`custom-call=2256`, `copy=9918`, `fusion=22711`, `dot_general=7122`,
`splash=2211`, `tokamax=2`, `jvp_jit_silu__=0`, `exponential=183`.

Verdict: refuted. shmem95 does not produce a useful midpoint; it lands in the
same tight shmem100 HLO/memory family and is not faster. Close this
shared-memory bracket and pivot back to source/HLO directions from the
retrospective.

## [2026-06-12] loop-iteration | cc5 v079 launched: explicit-SiLU shmem95 midpoint

Launching workload `alekseyv-qwen3-cc5-jax-v079-shm95` as the midpoint after
v077/v078 showed that shmem100 is valid but not reliably faster and leaves only
**0.3181 GiB** HBM free. This run keeps image
`qwen3-8b-jax:v067-explicit-silu`, np0 placement, no-scan remat, activation
sharding, Tokamax Splash max-logit 30, MaxText CE, batch size **4** per chip /
global batch **32**, all-gather and reduce-scatter ACF enabled, all-reduce ACF
disabled, SparseCore reduce-scatter/all-reduce offload, SparseCore collective
aggregator, RS latency multiplier **3**, AR latency multiplier **2**, selective
resources, scoped VMEM **100352**, and latency-hiding scheduler rerun **3**.
The only tested change is
`--xla_tpu_scheduler_percent_shared_memory_limit=95`.

Support requires beating the durable shmem90 band (**~59.33k tok/s** or
**4433.8 ms**) without shmem100's tight HBM margin. A same-family tie or slower
profile closes this shared-memory bracket and should pivot back to code/HLO
directions from the retrospective.

Run/cache paths:
`gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-12-qwen3-jax-v079-v067-shmem95`
and
`gs://<your-bucket>/autoresearch/qwen3_cc5/jax_lane_cache_v079_explicit_silu_shmem95`.

## [2026-06-12] loop-iteration | cc5 v078 exact v077 shmem100 rerun: refuted as confirmation (59,329 host-best, 4434.7 ms; same HLO)

Workload `alekseyv-qwen3-cc5-jax-v078-shm100r` completed cleanly on np0 with
finite loss and no compile OOM, but did not confirm v077's small gain. Worker
summaries: `gke-tpu-0d0013c1-l5jf` reported **59,306 tok/s / 7,413
tok/s/chip / 42.5% old MFU** (about **48.36% MaxText-style MFU**), while
`gke-tpu-0d0013c1-9rkl` reported **59,329 tok/s / 7,416 tok/s/chip / 42.5%
old MFU**.

XProf
`2026-06-12-qwen3-jax-v078-v077-shmem100-rerun/2026_06_12_16_55_37`
reported **4434.7 ms** average step time, **1.4 ms** stddev, **66.9% MXU**,
**0.3%** idle, and the same tight peak HBM as v077: **30.93 GiB / 31.25 GiB**
with only **0.3181 GiB** free. Top buckets were same-family but slower:
convolution fusion **56,857.906 ms / 53.4%**, custom-call **29,423.697 ms /
27.6%**, and loop fusion **10,719.645 ms / 10.1%**.

Fresh train-step HLO in `/tmp/qwen3-v078-hlo` was byte-identical to v077:
SHA256 `4e271536a745449917e002dcb8f554b509bb0d7cff82da955d0ad7005ca97ebc`,
**19,895,806 bytes / 119,353 lines**, same aggregate counts, and still distinct
from the shmem90 v068/v071/v073/v075 family.

Verdict: refuted as confirmation. The shmem100 HLO is valid but too tight and
not reliably faster; do not carry shared-memory limit 100. Next useful bracket
is an explicit-SiLU midpoint such as shmem95 to test whether partial scheduler
memory restores any of v077's profile gain without the full HBM squeeze.

## [2026-06-12] loop-iteration | cc5 v078 launched: exact v077 explicit-SiLU shmem100 rerun

Launching workload `alekseyv-qwen3-cc5-jax-v078-shm100r` as the required
confirmation run for v077. The manifest is an exact rerun of v077 except for
workload name, fresh compile cache, and run/HLO output path. It keeps image
`qwen3-8b-jax:v067-explicit-silu`, np0 placement, no-scan remat, activation
sharding, Tokamax Splash max-logit 30, MaxText CE, batch size **4** per chip /
global batch **32**, all-gather and reduce-scatter ACF enabled, all-reduce ACF
disabled, SparseCore reduce-scatter/all-reduce offload, SparseCore collective
aggregator, RS latency multiplier **3**, AR latency multiplier **2**, selective
resources, scoped VMEM **100352**, shared-memory limit **100**, and
latency-hiding scheduler rerun **3**.

Support requires reproducing v077's **59.34k tok/s / 4432.6 ms** band without
HBM OOM. A return to the shmem90 **59.33k / 4433.8 ms** band or worse treats
v077 as noise/tie.

Run/cache paths:
`gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-12-qwen3-jax-v078-v077-shmem100-rerun`
and
`gs://<your-bucket>/autoresearch/qwen3_cc5/jax_lane_cache_v078_explicit_silu_shmem100_rerun`.

## [2026-06-12] loop-iteration | cc5 v077 explicit-SiLU shmem100: provisional support (59,340 host-best, 4432.6 ms; tight HBM)

Workload `alekseyv-qwen3-cc5-jax-v077-shm100` completed cleanly on np0 with
finite loss and no compile OOM. Worker summaries:
`gke-tpu-0d0013c1-l5jf` reported **59,320 tok/s / 7,415 tok/s/chip / 42.5%
old MFU** (about **48.37% MaxText-style MFU**), while
`gke-tpu-0d0013c1-9rkl` reported **59,340 tok/s / 7,418 tok/s/chip / 42.5%
old MFU**.

XProf
`2026-06-12-qwen3-jax-v077-v067-shmem100/2026_06_12_16_44_04`
reported **4432.6 ms** average step time, **0.1 ms** stddev, **66.8% MXU**,
**0.2%** idle, and peak HBM **30.93 GiB / 31.25 GiB** with only
**0.3181 GiB** free. Top buckets improved slightly over the shmem90 band:
convolution fusion **56,850.644 ms / 53.4%**, custom-call **29,423.942 ms /
27.7%**, and loop fusion **10,725.841 ms / 10.1%**.

Fresh train-step HLO in `/tmp/qwen3-v077-hlo` changed slightly from the
shmem90 v068/v071/v073/v075 family: SHA256
`4e271536a745449917e002dcb8f554b509bb0d7cff82da955d0ad7005ca97ebc`,
**19,895,806 bytes / 119,353 lines**, with counts `all-gather=4628`,
`all-reduce=158`, `reduce-scatter=1766`, `async-start=394`,
`async-done=394`, `convolution=2106`, `custom-call=2256`, `copy=9918`,
`fusion=22711`, `dot_general=7122`, `splash=2211`, `tokamax=2`,
`jvp_jit_silu__=0`.

Verdict: provisional support. Explicit-SiLU plus shmem100 is the first
post-v068 runtime interaction to produce both a small host-throughput gain and
a lower profile time, but it restores the tight **0.318 GiB** HBM margin. Run an
exact fresh-cache rerun before carrying.

## [2026-06-12] loop-iteration | cc5 v077 launched: current explicit-SiLU stack with shared-memory limit 100

Launching workload `alekseyv-qwen3-cc5-jax-v077-shm100` as a current
explicit-SiLU scheduler-memory interaction probe. v076 showed ACF
multiple-steps false creates a smaller but slower HLO, so the carried runtime
returns to the normal v067/v068 stack. This run changes only:

```text
--xla_tpu_scheduler_percent_shared_memory_limit=100
```

The run keeps image `qwen3-8b-jax:v067-explicit-silu`, np0 placement, fresh
compile cache, no-scan remat, activation sharding, Tokamax Splash max-logit 30,
MaxText CE, batch size **4** per chip / global batch **32**, all-gather and
reduce-scatter ACF enabled, all-reduce ACF disabled, SparseCore
reduce-scatter/all-reduce offload, SparseCore collective aggregator, RS latency
multiplier **3**, AR latency multiplier **2**, selective resources, scoped VMEM
**100352**, and latency-hiding scheduler rerun **3**.

This exact combination was not found in the current log: shmem100 was the
historical v273 setting, while the explicit-SiLU v067-v076 family was run at
shmem90. Support requires beating the current durable **59.33k tok/s /
4433.8 ms** band or preserving same-band throughput with a lower XProf step and
no HBM regression.

Run/cache paths:
`gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-12-qwen3-jax-v077-v067-shmem100`
and
`gs://<your-bucket>/autoresearch/qwen3_cc5/jax_lane_cache_v077_explicit_silu_shmem100`.

## [2026-06-12] loop-iteration | cc5 v076 no ACF multistep on current explicit-SiLU HLO: refuted (59,208 host-best, 4442.5 ms; smaller slower HLO)

Workload `alekseyv-qwen3-cc5-jax-v076-nomstep` completed cleanly on np0 with
finite loss, but regressed. Worker summaries:
`gke-tpu-0d0013c1-l5jf` reported **59,190 tok/s / 7,399 tok/s/chip / 42.4%
old MFU** (about **48.26% MaxText-style MFU**), while
`gke-tpu-0d0013c1-9rkl` reported **59,208 tok/s / 7,401 tok/s/chip / 42.5%
old MFU**.

XProf
`2026-06-12-qwen3-jax-v076-v067-no-acf-multistep/2026_06_12_16_31_03`
reported **4442.5 ms** average step time, **1.2 ms** stddev, **66.8% MXU**,
**0.3%** idle, and peak HBM **28.96 GiB / 31.25 GiB** with **2.2829 GiB**
free. Top buckets showed the regression clearly: convolution fusion increased
to **60,745.317 ms / 57.0%**, custom-call was **29,461.243 ms / 27.6%**, and
loop fusion was **10,853.464 ms / 10.2%**.

Fresh train-step HLO in `/tmp/qwen3-v076-hlo` changed materially from the
v068/v071/v073/v075 family: SHA256
`9fd4c99f6f83890d3cf14e9cfb83aec092c0f4d8bf55b0637307053b2c15e97d`,
**13,793,666 bytes / 79,880 lines**, with counts `all-gather=1489`,
`all-reduce=158`, `reduce-scatter=1766`, `async-start=253`,
`async-done=253`, `convolution=2136`, `custom-call=870`, `copy=7100`,
`fusion=10394`, `dot_general=5451`, `splash=2211`, `tokamax=2`,
`jvp_jit_silu__=0`.

Verdict: refuted. Disabling ACF multiple-steps creates a smaller and slightly
lower-memory HLO, but it worsens the convolution-fusion schedule and profile.
Keep ACF multiple-steps enabled.

## [2026-06-12] loop-iteration | cc5 v076 launched: current explicit-SiLU stack without ACF multistep

Launching workload `alekseyv-qwen3-cc5-jax-v076-nomstep` as the remaining
current-stack async collective fusion scheduler isolate. v075 showed disabling
only reduce-scatter ACF is accepted but inert, producing a byte-identical HLO
and a slower **4436.1 ms** profile. This run restores reduce-scatter ACF and
changes only:

```text
--xla_tpu_enable_async_collective_fusion_multiple_steps=false
```

The run keeps image `qwen3-8b-jax:v067-explicit-silu`, np0 placement, fresh
compile cache, no-scan remat, activation sharding, Tokamax Splash max-logit 30,
MaxText CE, batch size **4** per chip / global batch **32**, all-gather and
reduce-scatter on async collective fusion, all-reduce ACF disabled, SparseCore
reduce-scatter/all-reduce offload, SparseCore collective aggregator, RS latency
multiplier **3**, AR latency multiplier **2**, selective resources, scoped VMEM
**100352**, shared-memory limit **90**, and latency-hiding scheduler rerun
**3**.

Run/cache paths:
`gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-12-qwen3-jax-v076-v067-no-acf-multistep`
and
`gs://<your-bucket>/autoresearch/qwen3_cc5/jax_lane_cache_v076_no_acf_multistep`.

Support requires beating the current durable **59.33k tok/s / 4433.8 ms** band
or preserving same-band throughput with lower XProf step time and unchanged HBM.

## [2026-06-12] loop-iteration | cc5 v075 no reduce-scatter ACF on current explicit-SiLU HLO: refuted (59,299 host-best, 4436.1 ms; same HLO)

Workload `alekseyv-qwen3-cc5-jax-v075-norsacf` completed cleanly on np0 with
finite loss, but landed below the current RS3+AR2 frontier band. Worker
summaries: `gke-tpu-0d0013c1-9rkl` reported **59,299 tok/s / 7,412
tok/s/chip / 42.5% old MFU** (about **48.35% MaxText-style MFU**), while
`gke-tpu-0d0013c1-l5jf` reported **59,278 tok/s / 7,410 tok/s/chip / 42.5%
old MFU**.

XProf
`2026-06-12-qwen3-jax-v075-v067-no-rs-acf/2026_06_12_16_18_44`
reported **4436.1 ms** average step time, **1.2 ms** stddev, **66.7% MXU**,
**0.3%** idle, and unchanged peak HBM **29.03 GiB / 31.25 GiB** with
**2.2202 GiB** free.

Fresh train-step HLO in `/tmp/qwen3-v075-hlo` was byte-identical to the
v068/v071/v073 current family: SHA256
`84df522a5e6dfdd158288d856af82a5b24506c30db8c0fb29dfb46c952378a38`,
**19,895,815 bytes / 119,353 lines**, with counts `all-gather=4628`,
`all-reduce=158`, `reduce-scatter=1766`, `async-start=394`,
`async-done=394`, `convolution=2106`, `custom-call=2256`, `copy=9918`,
`fusion=22711`, `dot_general=7122`, `splash=2211`, `tokamax=2`,
`jvp_jit_silu__=0`.

Verdict: refuted. Disabling reduce-scatter async collective fusion is inert
under SparseCore reduce-scatter offload and slightly worsens the measured
profile. Keep the default reduce-scatter ACF setting.

## [2026-06-12] loop-iteration | cc5 v075 launched: current explicit-SiLU stack without reduce-scatter ACF

Launching workload `alekseyv-qwen3-cc5-jax-v075-norsacf` as a current-stack
async collective fusion subtype isolate. v074 showed RS4+AR2 changes the
current HLO but regresses throughput/profile and raises HBM, so the carried
runtime returns to RS3+AR2. This run keeps image
`qwen3-8b-jax:v067-explicit-silu`, np0 placement, fresh compile cache,
no-scan remat, activation sharding, Tokamax Splash max-logit 30, MaxText CE,
batch size **4** per chip / global batch **32**, all-gather on async collective
fusion, SparseCore reduce-scatter/all-reduce offload, SparseCore collective
aggregator, RS latency multiplier **3**, AR latency multiplier **2**, selective
resources, scoped VMEM **100352**, shared-memory limit **90**, and
latency-hiding scheduler rerun **3**.

The only intended runtime change is disabling the reduce-scatter ACF subtype:

```text
--xla_tpu_enable_async_collective_fusion_fuse_reduce_scatter=false
```

Run/cache paths:
`gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-12-qwen3-jax-v075-v067-no-rs-acf`
and
`gs://<your-bucket>/autoresearch/qwen3_cc5/jax_lane_cache_v075_no_rs_acf`.

Support requires beating the current durable **59.33k tok/s / 4433.8 ms** band
or preserving same-band throughput with lower XProf step time and unchanged HBM.

## [2026-06-12] loop-iteration | cc5 v074 RS4+AR2 on current explicit-SiLU HLO: refuted (59,299 host-best, 4434.9 ms; changed HLO, higher HBM)

Workload `alekseyv-qwen3-cc5-jax-v074-rs4ar2` completed cleanly on np0 with
finite loss, but regressed below the current RS3+AR2 band. Worker summaries:
`gke-tpu-0d0013c1-9rkl` reported **59,299 tok/s / 7,412 tok/s/chip / 42.5%
old MFU**, while `gke-tpu-0d0013c1-l5jf` reported **59,270 tok/s / 7,409
tok/s/chip / 42.5% old MFU**.

XProf
`2026-06-12-qwen3-jax-v074-v067-rs4-ar2/2026_06_12_16_07_49`
reported **4434.9 ms** average step time, **0.1 ms** stddev, **66.8% MXU**,
**0.3%** idle, and increased peak HBM **29.26 GiB / 31.25 GiB** with
**1.9876 GiB** free. Top buckets worsened slightly: convolution fusion
**56,961.658 ms / 53.5%**, custom-call **29,436.022 ms / 27.6%**, and loop
fusion **10,728.823 ms / 10.1%**.

Fresh train-step HLO in `/tmp/qwen3-v074-hlo` changed from v068/v071:
SHA256 `f54ac89e5ef732ad1613d04af6314a9eecd98c26b77eaac5955bc50a94e51582`,
**19,881,844 bytes / 119,268 lines**, with counts `all-gather=4622`,
`all-reduce=158`, `reduce-scatter=1766`, `async-start=391`,
`async-done=391`, `convolution=2108`, `custom-call=2254`, `copy=9923`,
`fusion=22677`, `dot_general=7116`, `splash=2211`, `tokamax=2`,
`jvp_jit_silu__=0`.

Verdict: refuted. RS4+AR2 selects a distinct schedule on the current lean HLO,
but it raises HBM and slows the profile. Keep RS latency multiplier `3`.

## [2026-06-12] loop-iteration | cc5 v074 launched: current explicit-SiLU stack with RS4+AR2

Launching workload `alekseyv-qwen3-cc5-jax-v074-rs4ar2` as a current-stack
reduce-scatter latency-model isolate. v073 showed all-reduce ACF is an
identical-HLO profile tie (**59,322 tok/s**, **4432.1 ms**) and not a carried
change. This run returns to the normal explicit-SiLU stack and changes only the
reduce-scatter SparseCore latency multiplier from **3** to **4**. The older
v032 RS4 probe was on the larger v030/v273 HLO with shared-memory limit 100;
this run closes the same high-side bracket on the current lean shmem90 HLO.

The run keeps image `qwen3-8b-jax:v067-explicit-silu`, np0 placement, fresh
compile cache, no-scan remat, activation sharding, Tokamax Splash max-logit 30,
MaxText CE, batch size **4** per chip / global batch **32**, all-gather on
async collective fusion, SparseCore reduce-scatter/all-reduce offload,
SparseCore collective aggregator, AR latency multiplier **2**, selective
resources, scoped VMEM **100352**, shared-memory limit **90**, and
latency-hiding scheduler rerun **3**.

Run/cache paths:
`gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-12-qwen3-jax-v074-v067-rs4-ar2`
and
`gs://<your-bucket>/autoresearch/qwen3_cc5/jax_lane_cache_v074_rs4_ar2`.

Support requires beating the current durable **59.33k tok/s / 4433.8 ms** band
or preserving same-band throughput with lower XProf step time and unchanged HBM.

## [2026-06-12] loop-iteration | cc5 v073 all-reduce ACF on explicit-SiLU HLO: refuted/tie (59,322 host-best, 4432.1 ms; same HLO)

Workload `alekseyv-qwen3-cc5-jax-v073-aracf` completed cleanly on np0 with
finite loss. Worker summaries were below the current high side:
`gke-tpu-0d0013c1-9rkl` reported **59,322 tok/s / 7,415 tok/s/chip / 42.5%
old MFU**, while `gke-tpu-0d0013c1-l5jf` reported **59,294 tok/s / 7,412
tok/s/chip / 42.5% old MFU**.

XProf
`2026-06-12-qwen3-jax-v073-v067-allreduce-acf/2026_06_12_15_56_32`
reported **4432.1 ms** average step time, **0.1 ms** stddev, **66.8% MXU**,
**0.3%** idle, and unchanged peak HBM **29.03 GiB / 31.25 GiB** with
**2.2202 GiB** free. Top buckets stayed same-family: convolution fusion
**56,932.672 ms / 53.5%**, custom-call **29,433.682 ms / 27.6%**, and loop
fusion **10,722.739 ms / 10.1%**.

Fresh train-step HLO in `/tmp/qwen3-v073-hlo` was byte-identical to v068/v071:
SHA256 `84df522a5e6dfdd158288d856af82a5b24506c30db8c0fb29dfb46c952378a38`,
**19,895,815 bytes / 119,353 lines**, with counts `all-gather=4628`,
`all-reduce=158`, `reduce-scatter=1766`, `async-start=394`,
`async-done=394`, `convolution=2106`, `custom-call=2256`, `copy=9918`,
`fusion=22711`, `dot_general=7122`, `splash=2211`, `tokamax=2`,
`jvp_jit_silu__=0`.

Verdict: refuted/tie. All-reduce ACF is accepted but inert under SparseCore
all-reduce offload. The slightly lower XProf window is not backed by host
throughput or an HLO/profile-shape change, so do not carry it.

## [2026-06-12] loop-iteration | cc5 v073 launched: current explicit-SiLU stack with all-reduce ACF enabled

Launching workload `alekseyv-qwen3-cc5-jax-v073-aracf` as a current-stack
all-reduce async collective fusion isolate. v072 showed scheduler resource
sharing is an identical-HLO regression (**59,306 tok/s**, **4437.3 ms**), so
the carried runtime returns to the explicit-SiLU RS3+AR2 stack with resource
sharing disabled/default. This run keeps the same image
`qwen3-8b-jax:v067-explicit-silu`, np0 placement, fresh compile cache, no-scan
remat, activation sharding, Tokamax Splash max-logit 30, MaxText CE, batch size
**4** per chip / global batch **32**, all-gather on async collective fusion,
SparseCore reduce-scatter/all-reduce offload, SparseCore collective aggregator,
RS latency multiplier **3**, AR latency multiplier **2**, selective resources,
scoped VMEM **100352**, shared-memory limit **90**, and latency-hiding
scheduler rerun **3**.

The only intended runtime change is enabling all-reduce async collective fusion:

```text
--xla_tpu_enable_async_collective_fusion_fuse_all_reduce=true
```

Run/cache paths:
`gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-12-qwen3-jax-v073-v067-allreduce-acf`
and
`gs://<your-bucket>/autoresearch/qwen3_cc5/jax_lane_cache_v073_allreduce_acf`.

Support requires beating the current durable **59.33k tok/s / 4433.8 ms** band
or preserving same-band throughput with lower XProf step time and unchanged HBM.

## [2026-06-12] loop-iteration | cc5 v072 scheduler resource sharing on explicit-SiLU HLO: refuted (59,306 host-best, 4437.3 ms; same HLO)

Workload `alekseyv-qwen3-cc5-jax-v072-rshare` completed cleanly on np0 with
finite loss, but regressed below the current durable band. Worker summaries:
`gke-tpu-0d0013c1-9rkl` reported **59,306 tok/s / 7,413 tok/s/chip / 42.5%
old MFU**, while `gke-tpu-0d0013c1-l5jf` reported **59,229 tok/s / 7,404
tok/s/chip / 42.5% old MFU**.

XProf
`2026-06-12-qwen3-jax-v072-v067-resource-sharing/2026_06_12_15_45_27`
reported **4437.3 ms** average step time, **0.2 ms** stddev, **66.4% MXU**,
**0.6%** idle, and unchanged peak HBM **29.03 GiB / 31.25 GiB** with
**2.2202 GiB** free. Top buckets were same-family but slower:
convolution fusion **56,930.473 ms / 53.3%**, custom-call
**29,433.840 ms / 27.6%**, and loop fusion **10,727.079 ms / 10.0%**.

Fresh train-step HLO in `/tmp/qwen3-v072-hlo` was byte-identical to v068/v071:
SHA256 `84df522a5e6dfdd158288d856af82a5b24506c30db8c0fb29dfb46c952378a38`,
**19,895,815 bytes / 119,353 lines**, with counts `all-gather=4628`,
`all-reduce=158`, `reduce-scatter=1766`, `async-start=394`,
`async-done=394`, `convolution=2106`, `custom-call=2256`, `copy=9918`,
`fusion=22711`, `dot_general=7122`, `splash=2211`, `tokamax=2`,
`jvp_jit_silu__=0`.

Verdict: refuted. Resource sharing changes neither HLO nor memory and worsens
runtime. Keep resource sharing disabled/default.

## [2026-06-12] loop-iteration | cc5 v072 launched: current explicit-SiLU stack with scheduler resource sharing

Launching workload `alekseyv-qwen3-cc5-jax-v072-rshare` as a current-stack
latency-hiding scheduler isolate. v071 showed that disabling active SparseCore
offload tracing is an identical-HLO tie with **59,329 tok/s** and **4432.8 ms**,
so the carried runtime remains the explicit-SiLU RS3+AR2 stack with normal
tracing. This run keeps the same image `qwen3-8b-jax:v067-explicit-silu`, np0
placement, fresh compile cache, no-scan remat, activation sharding, Tokamax
Splash max-logit 30, MaxText CE, batch size **4** per chip / global batch
**32**, all-gather on async collective fusion, SparseCore
reduce-scatter/all-reduce offload, SparseCore collective aggregator, RS latency
multiplier **3**, AR latency multiplier **2**, selective resources, scoped VMEM
**100352**, shared-memory limit **90**, and latency-hiding scheduler rerun
**3**.

The only intended runtime change is enabling scheduler resource sharing:

```text
--xla_latency_hiding_scheduler_resource_sharing=true
```

Run/cache paths:
`gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-12-qwen3-jax-v072-v067-resource-sharing`
and
`gs://<your-bucket>/autoresearch/qwen3_cc5/jax_lane_cache_v072_resource_sharing`.

Support requires beating the current durable **59.33k tok/s / 4433.8 ms** band
or preserving same-band throughput with lower XProf step time and unchanged HBM.

## [2026-06-12] loop-iteration | cc5 v071 no-offload-tracing on explicit-SiLU HLO: refuted/tie (59,329 host-best, 4432.8 ms; same HLO)

Workload `alekseyv-qwen3-cc5-jax-v071-notrace` completed cleanly on np0 with
finite loss. Worker summaries were same-band: `gke-tpu-0d0013c1-9rkl`
reported **59,329 tok/s / 7,416 tok/s/chip / 42.5% old MFU**, while
`gke-tpu-0d0013c1-l5jf` reported **59,309 tok/s / 7,414 tok/s/chip / 42.5%
old MFU**.

XProf
`2026-06-12-qwen3-jax-v071-v067-no-offload-tracing/2026_06_12_15_32_26`
reported **4432.8 ms** average step time, **1.7 ms** stddev, **66.8% MXU**,
**0.3%** idle, and unchanged peak HBM **29.03 GiB / 31.25 GiB** with
**2.2202 GiB** free. The main buckets stayed in the v068-v070 band:
convolution fusion **56,920.506 ms / 53.5%**, custom-call approximately
**29.43 s / 27.7%**, and data formatting **2,900.706 ms / 2.7%**.

Fresh train-step HLO in `/tmp/qwen3-v071-hlo` was byte-identical to v068:
SHA256 `84df522a5e6dfdd158288d856af82a5b24506c30db8c0fb29dfb46c952378a38`,
**19,895,815 bytes / 119,353 lines**, with counts `all-gather=4628`,
`all-reduce=158`, `reduce-scatter=1766`, `async-start=394`,
`async-done=394`, `convolution=2106`, `custom-call=2256`, `copy=9918`,
`fusion=22711`, `dot_general=7122`, `splash=2211`, `tokamax=2`,
`jvp_jit_silu__=0`.

Verdict: refuted/tie. Disabling active offload tracing does not improve the
current explicit-SiLU stack. Keep the normal tracing-enabled RS3+AR2 flags.

## [2026-06-12] loop-iteration | cc5 v071 launched: current explicit-SiLU stack without SparseCore offload tracing

Launching workload `alekseyv-qwen3-cc5-jax-v071-notrace` as a current-stack
runtime/profiling isolate. v070 showed that AR default changes the HLO hash but
regresses to **4435.5 ms**, so the carried runtime remains the v067/v068
explicit-SiLU stack with RS3+AR2. This run keeps the same image
`qwen3-8b-jax:v067-explicit-silu`, np0 placement, fresh compile cache,
no-scan remat, activation sharding, Tokamax Splash max-logit 30, MaxText CE,
batch size **4** per chip / global batch **32**, all-gather on async
collective fusion, SparseCore reduce-scatter/all-reduce offload, SparseCore
collective aggregator, RS latency multiplier **3**, AR latency multiplier
**2**, selective resources, scoped VMEM **100352**, shared-memory limit **90**,
and latency-hiding scheduler rerun **3**.

The only intended runtime change is disabling active offload tracing:

```text
--xla_tpu_enable_all_reduce_offload_tracing=false
--xla_tpu_enable_reduce_scatter_offload_tracing=false
```

Run/cache paths:
`gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-12-qwen3-jax-v071-v067-no-offload-tracing`
and
`gs://<your-bucket>/autoresearch/qwen3_cc5/jax_lane_cache_v071_no_offload_tracing`.

Support requires beating the current **59.3k tok/s / 4433.8-4433.9 ms** band
or preserving same-band throughput with lower XProf overhead and unchanged HBM.

## [2026-06-12] loop-iteration | cc5 v070 AR default on explicit-SiLU HLO: refuted (59,314 host-best, 4435.5 ms)

Workload `alekseyv-qwen3-cc5-jax-v070-ard` completed cleanly on np0 with
finite loss. It reran the v067/v068 explicit-SiLU shmem90 stack with a fresh
compile cache and omitted only
`--xla_tpu_sparse_core_all_reduce_latency_multiplier=2`. Worker summaries were
below the current high side: `gke-tpu-0d0013c1-l5jf` reported
**59,300 tok/s / 7,413 tok/s/chip / 42.5% old MFU**, while
`gke-tpu-0d0013c1-9rkl` reported **59,314 tok/s / 7,414 tok/s/chip / 42.5%
old MFU**.

XProf
`2026-06-12-qwen3-jax-v070-v067-ar-default/2026_06_12_15_17_45`
reported **4435.5 ms** average step time, **1.0 ms** stddev, **66.9% MXU**,
**0.2%** idle, and unchanged peak HBM **29.03 GiB / 31.25 GiB** with
**2.2202 GiB** free. Top buckets did not improve versus v068/v069:
convolution fusion **56,959.127 ms / 53.5%**, custom-call
**29,433.462 ms / 27.7%**, and loop fusion **10,719.414 ms / 10.1%**.

Fresh train-step HLO in `/tmp/qwen3-v070-hlo` changed hash but not aggregate
surface: SHA256
`9405450af843e65afe4e721de51b043ab33c4c01a1b3faedd39cc60a675349af`,
**19,895,815 bytes / 119,353 lines**, with counts `all-gather=4628`,
`all-reduce=158`, `reduce-scatter=1766`, `async-start=394`,
`async-done=394`, `convolution=2106`, `custom-call=2256`, `copy=9918`,
`fusion=22711`, `dot_general=7122`, `splash=2211`, `tokamax=2`,
`jvp_jit_silu__=0`.

Verdict: refuted. AR default repeats the older v262/v274 pattern on the
current stack: valid changed schedule, same memory, no profile win, lower
throughput. Keep all-reduce SparseCore latency multiplier `2`.

## [2026-06-12] loop-iteration | cc5 v070 launched: AR default on current explicit-SiLU lean HLO

Launching workload `alekseyv-qwen3-cc5-jax-v070-ard` as a current-stack
all-reduce latency isolate. v069 refuted the async all-gather concurrency cap:
same HLO, same memory, **4433.9 ms** XProf. This run keeps the v067/v068
explicit-SiLU image
`qwen3-8b-jax:v067-explicit-silu`, np0 placement, fresh compile cache,
no-scan remat, activation sharding, Tokamax Splash max-logit 30, MaxText CE,
batch size **4** per chip / global batch **32**, all-gather on async
collective fusion, SparseCore reduce-scatter/all-reduce offload, SparseCore
collective aggregator, RS latency multiplier **3**, selective resources,
scoped VMEM **100352**, shared-memory limit **90**, and latency-hiding
scheduler rerun **3**.

The only intended runtime change is omitting:

```text
--xla_tpu_sparse_core_all_reduce_latency_multiplier=2
```

Run/cache paths:
`gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-12-qwen3-jax-v070-v067-ar-default`
and
`gs://<your-bucket>/autoresearch/qwen3_cc5/jax_lane_cache_v070_ar_default`.

Support requires beating the current **59.3k tok/s / 4433.8-4433.9 ms** band
or preserving same-band throughput with a lower train-step profile bucket and
unchanged HBM. A changed HLO with no throughput/profile win repeats the older
v262/v274 AR-default refutation and keeps AR latency multiplier `2`.

## [2026-06-12] loop-iteration | cc5 v069 async all-gather cap 1: refuted/tie (59,332 host-best, 4433.9 ms; same HLO)

Workload `alekseyv-qwen3-cc5-jax-v069-ag1` completed cleanly on np0 with
finite loss. It reran the v067/v068 explicit-SiLU shmem90 stack and changed
only the runtime flag `--xla_max_concurrent_async_all_gathers=1`. Worker
summaries landed in the existing band: `gke-tpu-0d0013c1-9rkl` reported
**59,332 tok/s / 7,416 tok/s/chip / 42.5% old MFU**, and
`gke-tpu-0d0013c1-l5jf` reported **59,301 tok/s / 7,413 tok/s/chip / 42.5%
old MFU**.

XProf
`2026-06-12-qwen3-jax-v069-v067-async-allgather-cap1/2026_06_12_14_59_35`
reported **4433.9 ms** average step time, **1.3 ms** stddev, **66.8% MXU**,
**0.3%** idle, and unchanged peak HBM **29.03 GiB / 31.25 GiB** with
**2.2202 GiB** free. Op profile shape was unchanged: convolution fusion
**56,939.055 ms / 53.5%**, Splash/custom-call **29,434.380 ms / 27.6%**, and
loop fusion **10,725.883 ms / 10.1%**.

Fresh train-step HLO in `/tmp/qwen3-v069-hlo` was byte-identical to v067/v068:
SHA256 `84df522a5e6dfdd158288d856af82a5b24506c30db8c0fb29dfb46c952378a38`,
**19,895,815 bytes / 119,353 lines**, with counts `all-gather=4628`,
`all-reduce=158`, `reduce-scatter=1766`, `async-start=394`,
`async-done=394`, `convolution=2106`, `custom-call=2256`, `copy=9918`,
`fusion=22711`, `dot_general=7122`, `splash=2211`, `tokamax=2`,
`jvp_jit_silu__=0`.

Verdict: refuted/tie. The all-gather concurrency cap is not a missing lever on
the current lean shmem90 HLO family. Do not carry
`--xla_max_concurrent_async_all_gathers`.

## [2026-06-12] loop-iteration | cc5 v069 launched: async all-gather cap 1 on current lean shmem90 HLO

Launching workload `alekseyv-qwen3-cc5-jax-v069-ag1` to test a single runtime
scheduler flag on the current v067/v068 explicit-SiLU shmem90 stack. v068
proved the explicit-SiLU HLO is stable but only ties v064's **4433.8 ms** band.
This run keeps the same image
`qwen3-8b-jax:v067-explicit-silu`, np0 placement, fresh compile cache, no-scan
remat, activation sharding, Tokamax Splash max-logit 30, MaxText CE, batch
size **4** per chip / global batch **32**, RS3+AR2 SparseCore offload,
selective resources, scoped VMEM **100352**, shared-memory limit **90**, and
latency-hiding scheduler rerun **3**. The only intended runtime change is:

```text
--xla_max_concurrent_async_all_gathers=1
```

Run/cache paths:
`gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-12-qwen3-jax-v069-v067-async-allgather-cap1`
and
`gs://<your-bucket>/autoresearch/qwen3_cc5/jax_lane_cache_v069_ag1`.

Support requires beating the current durable **59.3k tok/s / 4433.8 ms** band
or preserving same-band throughput with a lower all-gather/custom-fusion
profile bucket. A changed HLO with exposed direct all-gather time or any clean
slowdown refutes the probe.

## [2026-06-12] loop-iteration | cc5 v068 explicit SiLU rerun: tie / not confirmed (59,318 host-best, 4433.8 ms)

Workload `alekseyv-qwen3-cc5-jax-v068-silur` reran the exact v067 image on np0
with a fresh compile cache. It completed cleanly with finite loss. Worker
summaries split because host `gke-tpu-0d0013c1-l5jf` had a post-profile
step-15 outlier (**4672.1 ms / 56,108 tok/s**): that host averaged
**59,103 tok/s / 7,388/chip / 42.4% old MFU**, while
`gke-tpu-0d0013c1-9rkl` averaged **59,318 tok/s / 7,415/chip / 42.5% old
MFU**. Steady steps outside the outlier stayed in the **59.3k** band.

XProf
`2026-06-12-qwen3-jax-v068-v067-explicit-silu-rerun/2026_06_12_14_43_32`
reported **4433.8 ms** average step time, **1.0 ms** stddev, **66.9% MXU**,
and peak HBM **29.03 GiB / 31.25 GiB** with **2.2202 GiB** free. Op profile
reported train-step convolution fusion **56,929.609 ms / 53.5%**, custom fusion
**3,848.180 ms / 3.6%**, and idle **0.3%**.

Fresh train-step HLO was byte-identical to v067: SHA256
`84df522a5e6dfdd158288d856af82a5b24506c30db8c0fb29dfb46c952378a38`,
**19,895,815 bytes / 119,353 lines**, with counts `all-gather=4628`,
`all-reduce=158`, `reduce-scatter=1766`, `async-start=394`,
`async-done=394`, `convolution=2106`, `custom-call=2256`, `copy=9918`,
`fusion=22711`, `dot_general=7122`, `splash=2211`, `tokamax=2`,
`jvp_jit_silu__=0`.

Verdict: tie / not confirmed as a throughput win. The explicit SiLU algebra is
a reliable semantics-preserving HLO cleanup, but the v068 XProf matched v064's
**4433.8 ms** instead of reproducing v067's **4432.5 ms** high-side sample.
Continue from v064/v067-equivalent performance and look for a larger source or
runtime-axis movement.

## [2026-06-12] loop-iteration | cc5 v067 explicit SwiGLU SiLU algebra: provisional support (59,347 tok/s / 4432.5 ms)

Workload `alekseyv-qwen3-cc5-jax-v067-silu` completed cleanly on np0 with a
small current-best result. Worker0 summary reported **59,347 tok/s**,
**7,418 tok/s/chip**, and **42.6% old MFU**. XProf
`2026-06-12-qwen3-jax-v067-v037-explicit-silu/2026_06_12_14_29_48` and
`/2026_06_12_14_29_49` both reported **4432.5 ms** average step time,
**1.1 ms** stddev, **66.8% / 67.0% MXU**, and peak HBM
**29.03 GiB / 31.25 GiB** with **2.2202 GiB** free. Op profile remained in the
same shape: convolution fusion **28,399.104 ms / 53.3%**, Splash custom-calls
**14,713.642 ms / 27.6%**, loop fusion **5,354.498 ms / 10.1%**, and idle
**0.3%**.

Fresh train-step HLO was copied to `/tmp/qwen3-v067-hlo` with SHA256
`84df522a5e6dfdd158288d856af82a5b24506c30db8c0fb29dfb46c952378a38`,
**19,895,815 bytes / 119,353 lines**. Counts stayed aligned with v064 for the
main families: `all-gather=4628`, `all-reduce=158`, `reduce-scatter=1766`,
`async-start=394`, `async-done=394`, `convolution=2106`, `custom-call=2256`,
`copy=9918`, `fusion=22711`, `dot_general=7122`, `splash=2211`, `tokamax=2`.
The intended source rewrite eliminated `jvp_jit_silu__` from **576** to **0**
without introducing explicit `sigmoid` or `logistic` names.

Verdict: provisional support. v067 beats v064 by **+21 tok/s** and **-1.3 ms**
while preserving the lean HLO family, but the movement is small enough to need
an exact fresh-cache rerun before treating explicit SiLU algebra as the durable
frontier.

## [2026-06-12] loop-iteration | cc5 v067 launched: explicit SwiGLU SiLU algebra on v064 baseline

Launching workload `alekseyv-qwen3-cc5-jax-v067-silu` to test a source-level
MLP fusion perturbation on the v064 live baseline. The thin image
`qwen3-8b-jax:v067-explicit-silu`
(`sha256:d752d2d9720ddc54602c2476689a40c20d322386c47339f047dbf7efa60ede21`) is
derived from `qwen3-8b-jax:v258-mlp-up-first` and changes only
`Qwen3MLP.__call__`: `jax.nn.silu(g) * u` becomes
`(g * jax.nn.sigmoid(g)) * u`. CPU MLP equivalence against the base file passed
with `max_abs_diff=0.0` and `allclose_1e-6=True`.

Runtime uses the v064/v037 stack with preferred np0 placement, fresh compile
cache
`gs://<your-bucket>/autoresearch/qwen3_cc5/jax_lane_cache_v067_silu`,
no-scan/remat, activation sharding, Tokamax Splash max-logit 30, MaxText CE,
batch size **4** per chip / global batch **32**, RS3+AR2 SparseCore offload,
selective resources, scoped VMEM **100352**, shared-memory limit **90**, and
latency-hiding scheduler rerun **3**. Profile/HLO path:
`gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-12-qwen3-jax-v067-v037-explicit-silu`.

Support requires beating v064's **59,326 tok/s / 4433.8 ms** band or preserving
same-band throughput with a useful reduction in MLP/convolution-fusion cost.

## [2026-06-12] loop-iteration | cc5 v066 np1 placement: refuted (58,948 tok/s / 4463.2 ms; same HLO)

Workload `alekseyv-qwen3-cc5-jax-v066-np1` completed cleanly on np1 nodes
`gke-tpu-01f93eb9-43vk` and `gke-tpu-01f93eb9-ghj6`. Fresh-cache np1 improved
over the v063 cache-hit control, but remained below v064/np0. Worker0 summary
reported **58,948 tok/s**, **7,369 tok/s/chip**, and **42.3% old MFU**. XProf
`2026-06-12-qwen3-jax-v066-v037-np1-placement/2026_06_12_14_14_28` reported
**4463.2 ms** average step time, **1.3 ms** stddev, **66.7% MXU**, and peak HBM
**29.03 GiB / 31.25 GiB** with **2.2202 GiB** free. Op profile reported
convolution fusion **57,034.0 ms / 53.3%**, Splash custom-calls
**29,412.0 ms / 27.5%**, and loop fusion **10,713.0 ms / 10.0%**.

The fresh train-step HLO was copied to `/tmp/qwen3-v066-hlo` and is identical
to v064/v065: SHA256
`81e3c3c58b3fbfa12c0b203bdc9d0df428913fad3462ea5552e36cd07dcd5a96`,
**19,915,660 bytes / 119,349 lines**, with counts `all-gather=4628`,
`all-reduce=158`, `reduce-scatter=1766`, `async-start=394`, `async-done=394`,
`convolution=2106`, `custom-call=2256`, `copy=9918`, `fusion=22711`,
`dot_general=7122`, `splash=2211`, `tokamax=2`, `jvp_jit_silu__=576`.

Verdict: refuted. np0 remains the preferred placement for the live frontier:
v064/np0 **59,326 tok/s / 4433.8 ms**, v066/np1 **58,948 / 4463.2**, v065/np4
**58,568 / 4487.0**. Close placement probing and resume source-level work from
v064.

## [2026-06-12] loop-iteration | cc5 v066 launched: exact v064 fresh-cache control pinned to nodepool np1

Launching workload `alekseyv-qwen3-cc5-jax-v066-np1` as the fair np1 placement
control. v063 was slow on np1 (**58,963 tok/s / 4461.2 ms**) but used the
shared compile cache and emitted no fresh train-step HLO, while v064 on np0
with a unique cache recovered **59,326 tok/s / 4433.8 ms**. v066 keeps the
v064/v037 stack fixed and pins only
`cloud.google.com/gke-nodepool=<your-cluster>-np-1`, with unique
cache
`gs://<your-bucket>/autoresearch/qwen3_cc5/jax_lane_cache_v066_np1`.
Profile/HLO path:
`gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-12-qwen3-jax-v066-v037-np1-placement`.

Support requires recovering v064's same-HLO step-time band. If np1 remains
slower, close placement probing and continue source-level work from the np0
v064 baseline.

## [2026-06-12] loop-iteration | cc5 v065 np4 placement: refuted (58,568 tok/s / 4487.0 ms; same HLO)

Workload `alekseyv-qwen3-cc5-jax-v065-np4` completed cleanly on np4 nodes
`gke-tpu-32cb1c36-2tqg` and `gke-tpu-32cb1c36-fjrb`, but regressed below the
v064 live baseline. Worker0 summary reported **58,568 tok/s**,
**7,321 tok/s/chip**, and **42.0% old MFU**. XProf
`2026-06-12-qwen3-jax-v065-v037-np4-placement/2026_06_12_14_02_43` and
`/2026_06_12_14_02_44` reported **4487.0 ms** average step time, **1.7 ms**
stddev, **65.9% / 67.5% MXU**, and peak HBM **29.03 GiB / 31.25 GiB** with
**2.2202 GiB** free. Host `gke-tpu-32cb1c36-fjrb` op profile reported
convolution fusion **28,180.5 ms / 52.3%**, Splash custom-calls
**14,703.2 ms / 27.3%**, loop fusion **5,338.7 ms / 9.9%**, and idle **0.4%**.

The fresh train-step HLO was copied to `/tmp/qwen3-v065-hlo` and is identical
to v064: SHA256 `81e3c3c58b3fbfa12c0b203bdc9d0df428913fad3462ea5552e36cd07dcd5a96`,
**19,915,660 bytes / 119,349 lines**, with counts `all-gather=4628`,
`all-reduce=158`, `reduce-scatter=1766`, `async-start=394`, `async-done=394`,
`convolution=2106`, `custom-call=2256`, `copy=9918`, `fusion=22711`,
`dot_general=7122`, `splash=2211`, `tokamax=2`, `jvp_jit_silu__=576`.

Verdict: refuted. np4 is slower for the exact same HLO and memory envelope; do
not use this placement for the frontier. Prefer the np0 placement used by
v036/v037/v064 when capacity permits.

## [2026-06-12] loop-iteration | cc5 v065 launched: exact v064 fresh-cache control pinned to nodepool np4

Launching workload `alekseyv-qwen3-cc5-jax-v065-np4` as an exact v064/v037
fresh-cache control pinned to nodepool
`<your-cluster>-np-4`. v064 recovered the lean HLO
(`81e3c3c5...`) and the current **59,326 tok/s / 4433.8 ms** live baseline, but
the same optimized train-step SHA historically reported **60.3k / 4358 ms**.
This run tests whether same-HLO runtime placement can recover part of that
gap. Runtime flags remain unchanged: no-scan/remat, activation sharding,
Tokamax Splash max-logit 30, MaxText CE, batch size **4** per chip / global
batch **32**, RS3+AR2 SparseCore offload, selective resources, scoped VMEM
**100352**, shared-memory limit **90**, and latency-hiding scheduler rerun
**3**. Unique compile cache:
`gs://<your-bucket>/autoresearch/qwen3_cc5/jax_lane_cache_v065_np4`.
Profile/HLO path:
`gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-12-qwen3-jax-v065-v037-np4-placement`.

Support requires the same lean HLO family and profile/throughput above v064.
If it ties or regresses, close placement and resume source-level work from
v064.

## [2026-06-12] loop-iteration | cc5 v064 fresh-cache v037 control: supported as live baseline (59,326 tok/s / 4433.8 ms)

Workload `alekseyv-qwen3-cc5-jax-v064-freshcache` completed cleanly with finite
loss after forcing a unique compile-cache directory. Worker0 summary reported
**59,326 tok/s**, **7,416 tok/s/chip**, and **42.5% old MFU**. XProf
`2026-06-12-qwen3-jax-v064-v037-fresh-cache/2026_06_12_13_49_23` and
`/2026_06_12_13_49_24` reported **4433.8 ms** average step time, **1.6 ms**
stddev, **67.0% / 66.8% MXU**, and peak HBM **29.03 GiB / 31.25 GiB** with
**2.2202 GiB** free. Host `gke-tpu-0d0013c1-9rkl` op profile is still dominated
by convolution fusion (**28,542.0 ms / 53.7%**), Splash custom-calls
(**14,720.8 ms / 27.7%**), and loop fusion (**5,365.9 ms / 10.1%**).

The fresh train-step HLO was copied to `/tmp/qwen3-v064-hlo` with SHA256
`81e3c3c58b3fbfa12c0b203bdc9d0df428913fad3462ea5552e36cd07dcd5a96`,
**19,915,660 bytes / 119,349 lines**. Fresh counts:
`all-gather=4628`, `all-reduce=158`, `reduce-scatter=1766`,
`async-start=394`, `async-done=394`, `convolution=2106`, `custom-call=2256`,
`copy=9918`, `fusion=22711`, `dot_general=7122`, `splash=2211`, `tokamax=2`,
`jvp_jit_silu__=576`.

Verdict: supported as the current live baseline. v064 confirms the v063
regression was cache/runtime-path sensitive, not a source-code regression.
The remaining gap to historical v273/v277 (**60.3k / 4358 ms**) is not HLO
text: v064 has the same SHA but lower live throughput. Next probes should keep
fresh-cache isolation and focus on runtime placement/scheduler behavior or
source changes that do not select the larger `copy=22117/fusion=30896` family.

## [2026-06-12] loop-iteration | cc5 v064 launched: exact v037 control with unique compile cache

Launching workload `alekseyv-qwen3-cc5-jax-v064-freshcache` as an exact
unmodified v258/v037 control with a unique persistent compile-cache directory:
`gs://<your-bucket>/autoresearch/qwen3_cc5/jax_lane_cache_v064_fresh`.
The purpose is to force fresh HLO emission after v063 hit the shared cache,
produced no fresh HLO, and landed at **58,963 tok/s / 4461.2 ms** instead of the
prior **59,330 tok/s / 4433.0 ms** frontier. Runtime flags otherwise remain the
v037 stack: no-scan/remat, activation sharding, Tokamax Splash max-logit 30,
MaxText CE, batch size **4** per chip / global batch **32**, RS3+AR2 SparseCore
offload, selective resources, scoped VMEM **100352**, shared-memory limit
**90**, and latency-hiding scheduler rerun **3**. Profile/HLO path:
`gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-12-qwen3-jax-v064-v037-fresh-cache`.

Support requires fresh HLO plus recovery toward v036/v037's **59,330 tok/s /
4433.0 ms** band. If it stays near **58.9k / 4460 ms**, the live baseline is
lower than the carried frontier and the next source experiments should compare
against v063/v064 rather than the stale high-side control.

## [2026-06-12] loop-iteration | cc5 v063 exact v037 control: refuted as 59.33k reproduction (58,963 tok/s / 4461.2 ms)

Workload `alekseyv-qwen3-cc5-jax-v063-v037rerun` completed cleanly with finite
loss, but did not reproduce the prior v036/v037 **59,330 tok/s / 4433.0 ms**
band. Worker0 summary reported **58,963 tok/s**, **7,370 tok/s/chip**, and
**42.3% old MFU**. XProf
`2026-06-12-qwen3-jax-v063-v037-frontier-rerun/2026_06_12_13_36_40` reported
**4461.2 ms** average step time, **1.2 ms** stddev, **66.6% MXU**, and peak HBM
**29.03 GiB / 31.25 GiB** with **2.2202 GiB** free. The op profile remains
dominated by convolution fusion (**57,030.6 ms / 53.2%**), Splash custom-calls
(**29,412.2 ms / 27.5%**), and loop fusion (**10,713.5 ms / 10.0%**).

No fresh optimized HLO was found in the run's `hlo/` prefix, consistent with a
persistent compile-cache hit. Verdict: the current live environment now lands
in the same **58.9k / 4460 ms** band as v061/v062, so those source probes are
not clearly slower than the live control. Next action is an exact frontier
rerun with a unique compile-cache directory to force fresh HLO before more
source edits.

## [2026-06-12] loop-iteration | cc5 v063 launched: exact v037 frontier rerun control

Launched workload `alekseyv-qwen3-cc5-jax-v063-v037rerun` as an exact
unmodified-image control after v061 and v062 each selected the same slower
large-HLO family. The image is the carried frontier base,
`qwen3-8b-jax:v258-mlp-up-first` (`sha256:3a08bd533913...`), with no code
changes. Runtime flags are the v037 stack: no-scan/remat, activation sharding,
Tokamax Splash max-logit 30, MaxText CE entrypoint, batch size **4** per chip /
global batch **32**, RS3+AR2 SparseCore offload, selective resources, scoped
VMEM **100352**, shared-memory limit **90**, and latency-hiding scheduler rerun
**3**. Profile/HLO path:
`gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-12-qwen3-jax-v063-v037-frontier-rerun`.

Healthy reproduction should land near v036/v037's **59,330 tok/s / 4433.0 ms**
band and recover the tighter current-HLO family. If it instead matches the
slower v061/v062 large-HLO family, the next step should treat the environment
or cache state as part of the current performance gap before more source edits.

## [2026-06-12] loop-iteration | cc5 v062 final hidden FSDP constraint: refuted (58,944 tok/s / 4463.5 ms)

Workload `alekseyv-qwen3-cc5-jax-v062-finalhidden` completed cleanly with
finite loss, but regressed below the v036/v037 current-cluster frontier. Worker0
summary reported **58,944 tok/s**, **7,368 tok/s/chip**, and **42.3% old MFU**.
XProf `2026-06-12-qwen3-jax-v062-v037-final-hidden-fsdp/2026_06_12_13_29_25`
reported **4463.5 ms** average step time, **1.4 ms** stddev, **66.7% MXU**, and
peak HBM **29.03 GiB / 31.25 GiB** with **2.2202 GiB** free.

The optimized train-step HLO was copied to `/tmp/qwen3-v062-hlo` with SHA256
`5a51ec39665a854cad511367b69bd400f6956a034cd295edfa9029489632349a`,
**19,915,837 bytes / 119,352 lines**. Counts match v061's slower large-HLO
family: `all-gather=6997`, `all-reduce=165`, `reduce-scatter=2774`,
`convolution=3081`, `custom-call=3849`, `copy=22117`, `fusion=30896`,
`dot_general=7302`, `splash=2861`, `tokamax=2`, `jvp_jit_silu__=576`.

Verdict: refuted. The current-stack final hidden-state constraint does not
improve CE/lm-head layout or HBM and again selects a slower schedule. Do not
carry.

## [2026-06-12] loop-iteration | cc5 v062 launched: current-stack final hidden FSDP constraint retest

Launched workload `alekseyv-qwen3-cc5-jax-v062-finalhidden` to retest the
final hidden-state FSDP activation constraint on the current no-scan
SparseCore/MaxText-CE stack. The earlier v164 scan-frontier probe refuted this
boundary, but the current stack has a different lm-head/CE and collective
schedule. The thin image `qwen3-8b-jax:v062-final-hidden-fsdp-current`
(`sha256:c562e776...`) is derived from `qwen3-8b-jax:v258-mlp-up-first` and
changes only `Qwen3ForCausalLM.__call__`: it inserts `hidden = _sac(hidden)`
after `hidden = self.model(...)` and before the `return_hidden`/`lm_head`
branch. `py_compile` passed and the source diff is exactly that inserted line.

Runtime flags are the v037 carried stack: no-scan/remat, activation sharding,
Tokamax Splash max-logit 30, MaxText CE entrypoint, batch size **4** per chip /
global batch **32**, RS3+AR2 SparseCore offload, selective resources, scoped
VMEM **100352**, shared-memory limit **90**, and latency-hiding scheduler rerun
**3**. Profile/HLO path:
`gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-12-qwen3-jax-v062-v037-final-hidden-fsdp`.

Support requires beating v037's **59,330 tok/s / 4433.0 ms** band, or a
same-band result with lower CE/lm-head layout/collective cost and no HBM
regression.

## [2026-06-12] loop-iteration | cc5 v061 MLP multiply operand order: refuted (58,948 tok/s / 4460.5 ms; larger HLO)

Workload `alekseyv-qwen3-cc5-jax-v061-mlpmul` completed cleanly with finite
loss, but regressed below the v036/v037 frontier. Worker0 summary reported
**58,948 tok/s**, **7,368 tok/s/chip**, and **42.3% old MFU**. XProf
`2026-06-12-qwen3-jax-v061-v037-mlp-mul-order/2026_06_12_13_18_44` reported
**4460.5 ms** average step time, **0.4 ms** stddev, **66.7% MXU**, and peak HBM
**29.03 GiB / 31.25 GiB** with **2.2202 GiB** free.

The optimized train-step HLO was copied to `/tmp/qwen3-v061-hlo` with SHA256
`ad313e21729d6b4a8c360f50b4ab5fc9b2a17e69779a8c004c486a655a72ef06`,
**19,915,775 bytes / 119,351 lines**. Counts shifted to the larger
old-v273-style structure: `all-gather=6997`, `all-reduce=165`,
`reduce-scatter=2774`, `convolution=3081`, `custom-call=3849`, `copy=22117`,
`fusion=30896`, `dot_general=7302`, `splash=2861`, `tokamax=2`,
`jvp_jit_silu__=576`.

Source derivation was clean: the v061 file differs from the base image by only
the MLP multiply line, and local CPU equivalence was exact. Verdict: refuted.
Do not carry the operand-order swap; it perturbs compiler structure in the
wrong direction without reducing HBM or improving the convolution-fusion bucket.

## [2026-06-12] loop-iteration | cc5 v061 launched: MLP SwiGLU multiply operand-order probe

Launched workload `alekseyv-qwen3-cc5-jax-v061-mlpmul` to test a narrow
source-level MLP schedule perturbation after v060 showed that CE/logits VJP
changes can break the HBM envelope. The thin image
`qwen3-8b-jax:v061-mlp-mul-order` (`sha256:cf397f1...`) is derived from
`qwen3-8b-jax:v258-mlp-up-first` and changes only `Qwen3MLP.__call__`: the
SwiGLU multiply becomes `up * silu(gate)` instead of `silu(gate) * up`. The
projection evaluation order, checkpoint names, sharding, remat policy,
MaxText/T5X CE, Splash attention, and all backend flags remain fixed. Local CPU
equivalence on a small MLP config was exact (`max_abs_diff=0.0`,
`exact_equal=True`).

Runtime flags are the v037 carried stack: no-scan/remat, activation sharding,
Tokamax Splash max-logit 30, MaxText CE entrypoint, batch size **4** per chip /
global batch **32**, RS3+AR2 SparseCore offload, selective resources, scoped
VMEM **100352**, shared-memory limit **90**, and latency-hiding scheduler rerun
**3**. Profile/HLO path:
`gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-12-qwen3-jax-v061-v037-mlp-mul-order`.

Support requires beating v037's **59,330 tok/s / 4433.0 ms** band, or a
near-tie with lower MLP convolution-fusion time and no HBM regression.

## [2026-06-12] loop-iteration | cc5 v060 sparse-label CE: refuted/invalid compile HBM OOM (39.19 GiB)

Workload `alekseyv-qwen3-cc5-jax-v060-sparsece` failed during TPU compile
before step 0. The sparse-label CE custom VJP preserved local CPU equivalence
against the dense MaxText/T5X CE path, but on TPU it broke the current v037
memory envelope: XLA required **39.19 GiB** of program HBM on a **31.25 GiB**
device and exceeded capacity by **7.95 GiB**. The dominant allocations were
the final logits path: **18.55 GiB** `f32[4,8192,151936]` and **9.27 GiB**
`bf16[4,8192,151936]` in the same fused lm-head/logits computation.

Verdict: refuted / invalid for the current graph. This did not remove the
logits bottleneck; it lengthened or exposed the full-logits live range. Do not
carry sparse-label CE forward unless it is part of a true streamed/fused
lm-head+CE path that avoids materializing full `[batch, seq, vocab]` logits.

## [2026-06-12] loop-iteration | cc5 v060 launched: sparse-label MaxText CE custom VJP

Launched workload `alekseyv-qwen3-cc5-jax-v060-sparsece` to test a narrower
CE/lm-head-adjacent source change. The thin image
`qwen3-8b-jax:v060-sparse-label-ce` is derived from
`qwen3-8b-jax:v258-mlp-up-first` and changes only `train.py`: it adds an
environment-gated sparse-label CE custom VJP, enabled by
`USE_SPARSE_LABEL_CE=1`, while leaving the dense MaxText/T5X CE path as the
default. The new forward computes `logsumexp(logits) - logits[label]`; the
backward computes softmax probabilities and subtracts the incoming gradient at
the sparse label index with indexed scatter-add. CPU equivalence versus the
existing dense-target MaxText CE passed (`loss_max_abs_diff=0.0`,
`grad_max_abs_diff=1.49e-08`).

Runtime flags are the v037 carried stack: no-scan/remat, activation sharding,
Tokamax Splash max-logit 30, MaxText CE entrypoint, batch size **4** per chip /
global batch **32**, RS3+AR2 SparseCore offload, selective resources, scoped
VMEM **100352**, shared-memory limit **90**, and latency-hiding scheduler rerun
**3**. Profile/HLO path:
`gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-12-qwen3-jax-v060-v037-sparse-label-ce`.

Support requires beating v037's **59,330 tok/s / 4433.0 ms** band, or a
same-band result with a smaller CE/loop-fusion footprint and no MXU/HBM
regression.

## [2026-06-12] loop-iteration | cc5 v059 fused K/V projection: refuted close-neutral (59,177 tok/s / 4444.5 ms)

Workload `alekseyv-qwen3-cc5-jax-v059-kv` completed cleanly. It briefly
matched or slightly exceeded the current-cluster frontier on early steady steps
(step 2 **59,394 tok/s**, step 3 **59,340 tok/s**), but settled below v036/v037
over the full run. Worker summary reported **59,177 tok/s**,
**7,397 tok/s/chip**, and **42.4% old MFU** (about **48.2% MaxText-style
MFU**). XProf
`2026-06-12-qwen3-jax-v059-v037-fused-kv-proj/2026_06_12_13_01_55` reported
**4444.5 ms** step time, **66.7% MXU**, and peak HBM **29.01 GiB** with
**2.2382 GiB** free.

The optimized train-step HLO is
`cbfbfbccc6607450269118ae0e6f16476412af89fc3b562cc709d46aca720d95`,
**18,723,015 bytes / 111,915 lines**, with text counts `all-gather=6137`,
`reduce-scatter=2378`, `convolution=2609`, `custom-call=3561`, `fusion=28019`,
and `dot_general=6295`. Verdict: refuted / close neutral. K/V-only fusion is
much less harmful than full QKV fusion but does not beat v037's **59,330 tok/s
/ 4433.0 ms**. Close the attention input-projection fusion axis for now.

## [2026-06-12] loop-iteration | cc5 v059 launched: fused attention K/V projection on v037 frontier

Launched workload `alekseyv-qwen3-cc5-jax-v059-kv` as a narrower follow-up to
v058. Full QKV projection fusion reduced HLO size but regressed to **56,604
tok/s / 4654.1 ms**, so v059 keeps the Q projection unchanged and only
concatenates K/V weights for one combined K/V `dot_general`, then splits back to
K and V while preserving the existing `key_proj` and `value_proj` checkpoint
names. Local CPU equivalence on a small Qwen3 attention config passed exactly
(`max_abs_diff=0.0`, `allclose0=True`).

Runtime flags are the v037 carried stack: no-scan/remat, activation sharding,
Tokamax Splash max-logit 30, MaxText/T5X CE, batch size **4** per chip / global
batch **32**, RS3+AR2 SparseCore offload, selective resources, scoped VMEM
**100352**, shared-memory limit **90**, and latency-hiding scheduler rerun
**3**. Profile/HLO path:
`gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-12-qwen3-jax-v059-v037-fused-kv-proj`.

Support requires beating v037's **59,330 tok/s / 4433.0 ms** band, or a
same-band result with a clear reduction in attention projection footprint that
does not lower MXU utilization.

## [2026-06-12] loop-iteration | cc5 v058 fused QKV projection: refuted (56,604 tok/s / 4654.1 ms; smaller HLO but slower schedule)

Workload `alekseyv-qwen3-cc5-jax-v058-qkv` completed cleanly but regressed the
v036/v037 current-cluster frontier. Worker summary reported **56,604 tok/s**,
**7,075 tok/s/chip**, and **40.6% old MFU** (about **46.2% MaxText-style MFU**).
XProf `2026-06-12-qwen3-jax-v058-v037-fused-qkv-proj/2026_06_12_12_53_00`
reported **4654.1 ms** step time, **64.0% MXU**, and peak HBM **28.97 GiB**
with **2.2749 GiB** free on the available host trace.

The optimized train-step HLO became smaller (`52604a87685c61374f8f1f8a31a5941ce05ad9f930184129171b59681e70960f`,
**17,098,781 bytes / 101,543 lines**) with text counts `all-gather=5067`,
`reduce-scatter=2005`, `convolution=2173`, `custom-call=2916`,
`fusion=24875`, and `dot_general=5188`, but the schedule was slower: op profile
still spends **51.2%** in convolution fusion and **26.4%** in Splash custom calls.
Verdict: refuted. Do not carry full QKV fusion forward; one narrower K/V-only
probe remains plausible because it avoids disturbing the larger Q path.

## [2026-06-12] loop-iteration | cc5 v058 launched: fused attention QKV projection on v037 frontier

Launched workload `alekseyv-qwen3-cc5-jax-v058-qkv` as a structural attention
source probe on the current v037 frontier. The thin image
`qwen3-8b-jax:v058-fused-qkv-proj` is derived from
`qwen3-8b-jax:v258-mlp-up-first` and changes only `Qwen3Attention.__call__`:
`q_proj`, `k_proj`, and `v_proj` weights are concatenated on the output axis,
one `dot_general` is computed, and the result is split back into Q, K, and V.
The existing projection checkpoint names remain on the split tensors because
v055 showed those names steer a better schedule. Local CPU equivalence on a
small Qwen3 attention config passed exactly (`max_abs_diff=0.0`).

Runtime flags are the v037 carried stack: no-scan/remat, activation sharding,
Tokamax Splash max-logit 30, MaxText/T5X CE, batch size **4** per chip / global
batch **32**, RS3+AR2 SparseCore offload, selective resources, scoped VMEM
**100352**, shared-memory limit **90**, and latency-hiding scheduler rerun
**3**. Profile/HLO path:
`gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-12-qwen3-jax-v058-v037-fused-qkv-proj`.

Support requires beating v037's **59,330 tok/s / 4433.0 ms** band, or a
same-band result with a clear reduction in the convolution-fusion/HLO dot
footprint that motivates a narrower follow-up.

## [2026-06-12] loop-iteration | cc5 v057 shared-memory limit 91: refuted (58,970 tok/s / 4461.5 ms; same tight HLO as v056)

Workload `alekseyv-qwen3-cc5-jax-v057-shm91` completed cleanly but did not
beat the v036/v037 frontier. Worker0 reported **58,947 tok/s / 7,368
tok/s/chip / 42.3% old MFU** and worker1 reported **58,970 tok/s / 7,371
tok/s/chip / 42.3% old MFU**. XProf
`2026-06-12-qwen3-jax-v057-v037-shmem91/2026_06_12_12_37_36` reports
**4461.5 ms** step time, **66.7% MXU**, and peak HBM **29.03 GiB** with
**2.2202 GiB** free.

The optimized train-step HLO is byte-identical to v056:
`81e3c3c58b3fbfa12c0b203bdc9d0df428913fad3462ea5552e36cd07dcd5a96`,
**19,915,660 bytes / 119,349 lines**, with text counts
`all-gather=6997`, `all-reduce=165`, `reduce-scatter=2774`,
`async-start=394`, `async-done=394`, `convolution=3081`, `custom-call=3849`,
`copy=22117`, `fusion=30896`, `dot_general=7302`, `splash=2861`, `tokamax=2`,
`jvp_jit_silu__=576`.

Verdict: refuted. 91% already selects the high-side slower schedule rather
than improving the supported 90% point. Keep
`--xla_tpu_scheduler_percent_shared_memory_limit=90` and close the high-side
shared-memory axis.

## [2026-06-12] loop-iteration | cc5 v057 launched: v037 frontier with shared-memory limit 91

Launched workload `alekseyv-qwen3-cc5-jax-v057-shm91` as the threshold check
between the supported 90% shared-memory point and the refuted 92% tight-HBM
schedule. It keeps the v037 stack fixed and changes only
`--xla_tpu_scheduler_percent_shared_memory_limit=90` to `91`: image
`qwen3-8b-jax:v258-mlp-up-first`, no-scan/remat, activation sharding,
Tokamax Splash max-logit 30, MaxText/T5X CE, batch size 4, seq8192, TP1/FSDP8,
scoped VMEM 100352, selective resources, rerun3, and RS3+AR2 SparseCore
collective offload. Run root:
`gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-12-qwen3-jax-v057-v037-shmem91`.

Support requires beating v037's **59,330 tok/s / 4433.0 ms** band or at least
preserving the low-HBM-pressure 90% schedule with a same-band speedup. If 91%
also selects the tight v056/v039 memory envelope, close the high-side
shared-memory axis.

## [2026-06-12] loop-iteration | cc5 v056 shared-memory limit 92: refuted (58,977 tok/s / 4461.7 ms; tight HBM)

Workload `alekseyv-qwen3-cc5-jax-v056-shm92` completed cleanly but did not
beat the v036/v037 frontier. Worker0 reported **58,959 tok/s / 7,370
tok/s/chip / 42.3% old MFU** and worker1 reported **58,977 tok/s / 7,372
tok/s/chip / 42.3% old MFU**. XProf
`2026-06-12-qwen3-jax-v056-v037-shmem92/2026_06_12_12_25_36` reports
**4461.7 ms** step time, **66.6% MXU**, and peak HBM **30.93 GiB** with only
**0.3181 GiB** free.

The optimized train-step HLO is
`81e3c3c58b3fbfa12c0b203bdc9d0df428913fad3462ea5552e36cd07dcd5a96`,
**19,915,660 bytes / 119,349 lines**, with text counts
`all-gather=6997`, `all-reduce=165`, `reduce-scatter=2774`,
`async-start=394`, `async-done=394`, `convolution=3081`, `custom-call=3849`,
`copy=22117`, `fusion=30896`, `dot_general=7302`, `splash=2861`, `tokamax=2`,
`jvp_jit_silu__=576`.

Verdict: refuted. 92% moves into the tight old-v273-style memory envelope and
is slower than v036/v037 (**59,330 tok/s / 4433.0 ms**). Do not carry 92%;
test 91% once to locate the 90→tight-schedule threshold, then close this
high-side shared-memory axis if it also regresses.

## [2026-06-12] loop-iteration | cc5 v056 launched: v037 frontier with shared-memory limit 92

Launched workload `alekseyv-qwen3-cc5-jax-v056-shm92` as a fine-grained
high-side shared-memory-limit probe around the supported v036/v037 point. It
keeps the v037 stack fixed and changes only
`--xla_tpu_scheduler_percent_shared_memory_limit=90` to `92`: image
`qwen3-8b-jax:v258-mlp-up-first`, no-scan/remat, activation sharding,
Tokamax Splash max-logit 30, MaxText/T5X CE, batch size 4, seq8192, TP1/FSDP8,
scoped VMEM 100352, selective resources, rerun3, and RS3+AR2 SparseCore
collective offload. Run root:
`gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-12-qwen3-jax-v056-v037-shmem92`.

Rationale: the coarse bracket found 85 slower and 95 slower/tighter on HBM,
while 90 became the current-cluster frontier. A 92% limit may keep the 90%
schedule's memory margin while giving the scheduler a little more shared-memory
latitude. Support requires beating v037's **59,330 tok/s / 4433.0 ms** band
with normal loss and valid HBM.

## [2026-06-12] loop-iteration | cc5 v055 remove attention projection checkpoint names: refuted (58,977 tok/s / 4463.0 ms)

Workload `alekseyv-qwen3-cc5-jax-v055-noattnnames` completed cleanly and
refuted the attention-name removal probe. Worker0 reported **58,977 tok/s /
7,372 tok/s/chip / 42.3% old MFU** and worker1 reported **58,954 tok/s / 7,369
tok/s/chip / 42.3% old MFU**. XProf
`2026-06-12-qwen3-jax-v055-v037-remove-attn-proj-names/2026_06_12_12_11_04`
reports **4463.0 ms** step time, **66.4% MXU**, and peak HBM **29.03 GiB**
with **2.2202 GiB** free.

The optimized train-step HLO is
`a090c01fb3b2871541ebdb5f19e89c64f50a73f7575e3ebf927251ae13e42e59`,
**19,915,660 bytes / 119,349 lines**. Text counts move back toward the larger
old-v273 family: `all-gather=6997`, `all-reduce=165`,
`reduce-scatter=2774`, `async-start=394`, `async-done=394`,
`convolution=3081`, `custom-call=3849`, `copy=22117`, `fusion=30896`,
`dot_general=7302`, `splash=2861`, `tokamax=2`, `jvp_jit_silu__=576`.

Verdict: refuted. Throughput is below the v036/v037 frontier
(**59,330 tok/s / 4433.0 ms**) and the HLO is structurally worse. Keep the
attention projection `checkpoint_name` wrappers; they are useful compiler
steering even under the `nothing_saveable` remat policy.

## [2026-06-12] loop-iteration | cc5 v055 launched: remove unused attention projection checkpoint names on v037 frontier

Launched workload `alekseyv-qwen3-cc5-jax-v055-noattnnames` as a narrow
attention/source-layout probe on the current v037 frontier. The thin image
`qwen3-8b-jax:v055-remove-attn-proj-names` was built from
`qwen3-8b-jax:v258-mlp-up-first` and changes only `Qwen3Attention.__call__`:
`query_proj`, `key_proj`, `value_proj`, and `out_proj` are no longer wrapped in
`checkpoint_name`. Projection order, QK norm, RoPE, Splash dispatch, MLP code,
decoder-layer input naming, optimizer, and loss are unchanged. Image digest:
`sha256:9c30f4f7dda3e8a69d6c75bcf5cb7e40627b0f5339bbcc7a2cead2436b567e75`.

Runtime flags match v037/v053: no-scan/remat with no offload, activation
sharding, MaxText CE, Tokamax Splash max-logit 30, batch size 4, seq8192,
TP1/FSDP8, scoped VMEM 100352, RS3+AR2 SparseCore collective offload,
selective resources, rerun3, and shared-memory limit 90. Run root:
`gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-12-qwen3-jax-v055-v037-remove-attn-proj-names`.

Support requires beating the current-cluster v036/v037 frontier
(**59,330 tok/s / 4433.0 ms**) or matching it with materially lower attention
custom-call/copy/fusion time. A slower clean completion refutes the source
layout change.

## [2026-06-12] loop-iteration | cc5 v054 fixed scan-unroll probe: invalid/refuted (XLA async-start shape mismatch)

Workload `alekseyv-qwen3-cc5-jax-v054-unrollfix` fixed the stale-image/package
problem from the broken `v034-unr` attempts: it reached fake dataloader
creation and entered the first jitted train step. It then failed during XLA
post-optimization before any training step completed:
`jax.errors.JaxRuntimeError: INTERNAL: during context [post-optimization]:
The async-start expects the shape of operand 0 to match the async shape at
index {0}`, comparing
`bf16[18,2,4,8192,32,128]{5,3,4,2,1,0:T(8,128)(2,1)}` with the same shape
carrying `S(5)` in the layout. The JobSet exited with code 1. No XProf profile
or `after_optimizations` HLO was produced; the HLO directory contains only the
pre/post-crash setup artifacts under
`gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-12-qwen3-jax-v054-scan-unroll-fixed/hlo_a/`.

Verdict: invalid/refuted. `JAX_SCAN_UNROLL=2` is not valid with the current
scan/offload-attention plus async-collective flag stack. Do not carry scan
unroll on this stack; any future repair would need to isolate the async
collective configuration first, and should be treated as a compiler-workaround
probe rather than a supported optimization.

## [2026-06-12] loop-iteration | cc5 v054 launched: corrected scan-unroll probe after stale/broken v034 image

Launched workload `alekseyv-qwen3-cc5-jax-v054-unrollfix` to rerun the scan
unroll experiment attempted by `alekseyv-qwen3-cc5-jax-v034-unr` and
`alekseyv-qwen3-cc5-jax-v034-unr-b`. Those two ad-hoc attempts were invalid:
both phase A and phase B failed before training with
`ModuleNotFoundError: No module named 'data'`, consistent with a stale or
incomplete image being used on the nodes.

The corrected image
`<your-registry>/torchtitan-images/qwen3-8b-jax:v054-scan-unroll-fixed`
was pushed at digest
`sha256:bc8d7950aa89ae8bebb689649fe397bd8764608a41a066324fcf8c6de771e836`.
The JobSet forces `imagePullPolicy: Always`, `cd /app/trainer`, explicit
`PYTHONPATH=/app/trainer`, and `/opt/venv/bin/python -u train.py`. Phase A sets
`JAX_SCAN_UNROLL=2` and profiles steps 12-14; phase B sets
`JAX_SCAN_UNROLL=4`. Both arms keep the scan/offload-attention stack from the
broken unroll probe: batch size 4, sequence length 8192, TP1/FSDP8, Splash,
Tokamax CE `chunked_xla`, scoped VMEM 98304, and shared-memory limit 100.
Run root:
`gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-12-qwen3-jax-v054-scan-unroll-fixed`.

## [2026-06-12] loop-iteration | cc5 v053 remove MLP `mlpwi` checkpoint names: refuted (58,976 tok/s / 4463.7 ms)

Workload `alekseyv-qwen3-cc5-jax-v053-nomlpwi` completed cleanly and refuted
the source-layout hypothesis. Worker0 reported **58,956 tok/s / 7,369
tok/s/chip / 42.3% old MFU** and worker1 reported **58,976 tok/s / 7,372
tok/s/chip / 42.3% old MFU**. XProf
`2026-06-12-qwen3-jax-v053-v037-remove-mlpwi-names/2026_06_12_11_50_32`
reports **4463.7 ms** step time, **66.7% MXU**, and peak HBM **29.03 GiB**
with **2.2202 GiB** free. The optimized train-step HLO is
`bed1332e27eab6acbdd164754cc1914618bbd1848c10cc7357282c7d968bef8b`,
**19,915,660 bytes / 119,349 lines**, with the same instruction-family counts
as v037: `all-gather=4628`, `all-reduce=158`, `reduce-scatter=1766`,
`async-start=394`, `async-done=394`, `convolution=2106`, `custom-call=2256`,
`copy=9918`, `fusion=22711`, `dot_general=7122`, `splash=2211`, `tokamax=2`,
`jvp_jit_silu__=576`.

Verdict: refuted. Removing the `mlpwi` names changes only hash-level details,
not the compiled surface that matters, and is slower than the current-cluster
v036/v037 frontier (**59,330 tok/s / 4433.0 ms**). Do not carry.

## [2026-06-12] loop-iteration | cc5 v053 launched: remove unused MLP `mlpwi` checkpoint names on v037 frontier

Launched workload `alekseyv-qwen3-cc5-jax-v053-nomlpwi` as a narrow
MLP/source-layout probe on top of the current v037 base image
`qwen3-8b-jax:v258-mlp-up-first`. The thin image
`qwen3-8b-jax:v053-remove-mlpwi-names` changes only
`Qwen3MLP.__call__`: `up_proj(x)` and `gate_proj(x)` are no longer wrapped in
`checkpoint_name(..., "mlpwi")`; up-proj-first order and all math are preserved.

Rationale: under the carried runtime (`--use_remat=True --offload_remat=False`)
the layer remat policy is `nothing_saveable`, so `mlpwi` names are not consumed
by the save/offload policy. If the name primitives are still acting as barriers
in the hot MLP/SwiGLU HLO region, removing them may reduce convolution-fusion
time without changing semantics. Runtime flags otherwise match v037 exactly:
no-scan/remat, activation sharding, Tokamax Splash max-logit 30, MaxText/T5X CE,
batch size **4** per chip / global batch **32**, RS3+AR2 SparseCore offload,
selective resources, scoped VMEM **100352**, shared-memory limit **90**, and
latency-hiding scheduler rerun **3**.

Profile/HLO path:
`gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-12-qwen3-jax-v053-v037-remove-mlpwi-names`.

## [2026-06-12] loop-iteration | cc5 v052 pinned-host attention offload shmem80: refuted (40,056 tok/s / 6571.4 ms)

Workload `alekseyv-qwen3-cc5-jax-v052-attnoff80` completed cleanly and proved
the pinned-host attention offload family can be made to fit, but only with a
large performance regression. Worker0 summary was **40,045 tok/s / 5,006
tok/s/chip / 28.7% old MFU**. Worker1 summary was **40,056 tok/s / 5,007
tok/s/chip / 28.7% old MFU**. XProf
`2026-06-12-qwen3-jax-v052-v050-attn-offload-shmem80/2026_06_12_11_31_42`
reports **6571.4 ms** average step time, **9.6 ms** stddev, **43.6% MXU**, and
**29.53 GiB / 31.25 GiB** peak HBM with **1.7194 GiB** free. This is far below
the v037 frontier (**59,330 tok/s / 4433.0 ms**) and below the fresh MaxText
baseline.

The optimized train-step HLO was copied to `/tmp/qwen3-v052-hlo`. Modules
`0264` and `0267` are byte-identical with SHA256
`991ef67c567c6a0e9c4a18537b4d36a34db9a7710acad73eb3283b7f87eb6e80`,
**14,579,321 bytes / 86,760 lines**. Counts:
`all-gather=2225`, `all-reduce=153`, `reduce-scatter=1766`,
`async-start=287`, `async-done=287`, `convolution=1718`,
`custom-call=1200`, `copy=12428`, `fusion=11901`, `dot_general=4354`,
`splash=2211`, `tokamax=2`, `jvp_jit_silu__=576`.

Verdict: refuted. The offload policy reduces graph size and memory enough to
compile at shmem80, but the host offload/copy path destroys utilization. Close
the remat save/offload axis for the current CE/logits structure. Further gains
need a structural fix for the logits/CE live range or a real MLP/CE kernel, not
more on-device/offload save-set variants.

## [2026-06-12] loop-iteration | cc5 v052 launched: v050 pinned-host attention offload with shared-memory limit 80

Launched workload `alekseyv-qwen3-cc5-jax-v052-attnoff80` as the next low-side
shared-memory bracket for the pinned-host attention offload HLO. It keeps the
v050/v051 image and trainer flags unchanged (`--offload_remat=True`) and changes
only `--xla_tpu_scheduler_percent_shared_memory_limit=85` to **80**.

Profile/HLO path:
`gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-12-qwen3-jax-v052-v050-attn-offload-shmem80`.

Rationale: v051 reduced the HBM overage to only **233.74 MiB** while preserving
the same semantics and batch size. Another small shared-memory reduction may be
enough to compile and produce the first usable profile for this offload family.
Success requires clean compile/run; throughput is expected to be weak, but a
fit would let us measure whether attention-offload reduces recompute enough to
justify further tuning.

## [2026-06-12] loop-iteration | cc5 v051 pinned-host attention offload shmem85: invalid/refuted, closer near-fit (31.47 GiB total HBM)

Workload `alekseyv-qwen3-cc5-jax-v051-attnoff85` failed during first-step
compile on both workers, but the low-side shared-memory bracket moved in the
right direction. Total HBM dropped from v050's **32.52 GiB** to **31.47 GiB**,
missing the **31.25 GiB** v6e device by only **233.74 MiB**. Program HBM was
**25.75 GiB**, arguments/live outputs remained **5.72 GiB**, and HLO temp was
**25.67 GiB**. The largest allocations were unchanged in kind: the **9.27 GiB**
bf16 `[4,8192,151936]` logits-path fusion and two **1.16 GiB** vocab-weight
all-gathers.

Verdict: invalid/refuted at shmem85, but the bracket is productive. Continue
one step lower to shmem80 before closing the offload memory-scheduler axis.

## [2026-06-12] loop-iteration | cc5 v051 launched: v050 pinned-host attention offload with shared-memory limit 85

Launched workload `alekseyv-qwen3-cc5-jax-v051-attnoff85` as the low-side
shared-memory bracket for the near-fitting v050 offload HLO. The image and
trainer flags are unchanged from v050 (`--offload_remat=True` with pinned-host
offload for `decoder_layer_input`, `query_proj`, `key_proj`, `value_proj`, and
`out_proj`). The only runtime change is
`--xla_tpu_scheduler_percent_shared_memory_limit=90` to **85**.

Profile/HLO path:
`gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-12-qwen3-jax-v051-v050-attn-offload-shmem85`.

Rationale: v050 reduced HLO temp to **26.71 GiB** but total HBM still reached
**32.52 GiB**, exceeding v6e HBM by **1.27 GiB** after arguments/live outputs.
The baseline shmem85 point was slower, but this offload HLO has a different
memory assignment; a lower shared-memory limit may recover enough HBM to compile
and produce a profile. Success requires clean compile/run and a throughput or
profile signal that justifies further offload tuning.

## [2026-06-12] loop-iteration | cc5 v050 pinned-host attention remat offload: invalid/refuted, but near-fit (32.52 GiB total HBM)

Workload `alekseyv-qwen3-cc5-jax-v050-attnoff` failed during first-step compile
on both workers. Unlike v048/v049, pinned-host offload changed the HLO and
reduced program temp substantially, but it still did not fit: **32.52 GiB** HBM
used on a **31.25 GiB** device, exceeding capacity by **1.27 GiB**. Program HBM
was **26.79 GiB**, arguments/live outputs added **5.72 GiB**, and HLO temp was
**26.71 GiB**. The top allocation remained the **9.27 GiB** bf16
`[4,8192,151936]` logits-path fusion, followed by two **1.16 GiB** vocab-weight
all-gathers.

The optimized OOM HLO was copied to `/tmp/qwen3-v050-oom-hlo` with SHA256
`966a963ac3a0801ad78b91843ba080b394ada66cf6871360e355363e7c88a2bd`,
**14,822,115 bytes / 88,217 lines**. Counts:
`all-gather=2343`, `all-reduce=153`, `reduce-scatter=1766`,
`async-start=273`, `async-done=273`, `convolution=1735`,
`custom-call=1248`, `copy=12366`, `fusion=12403`, `dot_general=4467`,
`splash=2211`, `tokamax=2`, `jvp_jit_silu__=576`.

Verdict: invalid/refuted at shmem90, but this is a near-fit rather than a
complete dead end. Continue with a narrow memory-scheduler bracket on the v050
offload HLO before abandoning pinned-host attention offload.

## [2026-06-12] loop-iteration | cc5 v050 launched: pinned-host attention remat offload on v037 frontier

Launched workload `alekseyv-qwen3-cc5-jax-v050-attnoff` as the memory-feasible
counterpart to v048/v049's on-device remat-save failures. It reuses the
`qwen3-8b-jax:v049-save-attn-proj-only` image but flips only the runtime flag
from `--offload_remat=False` to `--offload_remat=True`. In that branch, each
layer checkpoint uses `save_and_offload_only_these_names` for
`decoder_layer_input`, `query_proj`, `key_proj`, `value_proj`, and `out_proj`
with `offload_src="device"` and `offload_dst="pinned_host"`.

Runtime flags otherwise remain the v037 carried stack: no-scan/remat,
activation sharding, Tokamax Splash max-logit 30, MaxText/T5X CE, batch size
**4** per chip / global batch **32**, RS3+AR2 SparseCore offload, selective
resources, scoped VMEM **100352**, shared-memory limit **90**, and
latency-hiding scheduler rerun **3**. Profile/HLO path:
`gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-12-qwen3-jax-v050-v037-pinned-host-attn-offload`.

Rationale: v048/v049 prove on-device save policies are invalid under the
current CE/logits live range, but host offload may fit and can determine whether
attention activation saves reduce recompute enough to justify a more selective
or lower-overhead offload design. Success requires clean compile/run and
throughput above v037's **59,330 tok/s / 4433.0 ms** frontier; a slower result
still closes this offload axis for the current frontier.

## [2026-06-12] loop-iteration | cc5 v049 attention-projection-only remat save policy: invalid/refuted (byte-identical OOM to v048)

Workload `alekseyv-qwen3-cc5-jax-v049-attnsave` failed during first-step compile
on both workers with the same HBM signature as v048: **44.38 GiB** program HBM
required on a **31.25 GiB** device, exceeding capacity by **13.13 GiB**. The
top allocation remained the **9.27 GiB** bf16 `[4,8192,151936]` logits-path
fusion, followed by repeated **1.16 GiB** vocab-weight all-gathers.

The optimized OOM HLO was copied to `/tmp/qwen3-v049-oom-hlo` with SHA256
`5281ad518d3b2a1d186bd6c82698dd96b20139146ab70eb57c387634990286c7`,
**14,832,528 bytes / 88,229 lines**. It is byte-identical to the v048 OOM HLO,
and the memory-usage report is also byte-identical
(`6d6593cf6678d984162a9a83363d5be6188d88109ebc4f554207da3afe88540d`).
Counts in both OOM HLOs: `all-gather=2341`, `all-reduce=158`,
`reduce-scatter=1766`, `async-start=280`, `async-done=280`,
`convolution=1650`, `custom-call=1260`, `copy=11380`, `fusion=12554`,
`dot_general=4498`, `splash=2211`, `tokamax=2`, `jvp_jit_silu__=576`.

Verdict: invalid/refuted. Removing `mlpwo` from the on-device save set does not
change the compiled OOM graph. The current frontier cannot use
`save_only_these_names(...)` inside layer remat without triggering the bad
logits live range; continue only with offload or a CE/logit structural change.

## [2026-06-12] loop-iteration | cc5 v049 launched: attention-projection-only remat save policy on v037 frontier

Launched workload `alekseyv-qwen3-cc5-jax-v049-attnsave` as the narrower
follow-up to the v048 compile OOM. The thin image
`qwen3-8b-jax:v049-save-attn-proj-only` is derived from the same v047/v048
model fork and changes only the non-offload remat save set: when
`--use_remat=True --offload_remat=False`, each layer checkpoint now uses
`jax.checkpoint_policies.save_only_these_names("query_proj", "key_proj",
"value_proj", "out_proj")`. The `mlpwo` save from v048 is deliberately removed
because v048 pushed HBM temp to **44.38 GiB** and exposed a **9.27 GiB**
logits-path allocation.

Runtime flags are unchanged from v037/v048: no-scan/remat, activation sharding,
Tokamax Splash max-logit 30, MaxText/T5X CE, batch size **4** per chip / global
batch **32**, RS3+AR2 SparseCore offload, selective resources, scoped VMEM
**100352**, shared-memory limit **90**, and latency-hiding scheduler rerun
**3**. Profile/HLO path:
`gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-12-qwen3-jax-v049-v037-save-attn-proj-only`.

Rationale: if saving all selected dot outputs is too memory-heavy, saving only
attention projection outputs may still reduce recompute in the attention path
without forcing the MLP-output/logits live range into the OOM schedule. Success
requires compiling cleanly and beating v037's **59,330 tok/s / 4433.0 ms**
frontier, or producing a same-band profile with lower attention/convolution
fusion time and acceptable peak HBM.

## [2026-06-12] loop-iteration | cc5 v048 MaxText save_dot_except_mlpwi remat policy: invalid/refuted (compile HBM OOM, 44.38 GiB)

Workload `alekseyv-qwen3-cc5-jax-v048-savedot` failed during first-step compile
on both workers. The non-offload remat policy changed from `nothing_saveable` to
`save_only_these_names("query_proj", "key_proj", "value_proj", "out_proj",
"mlpwo")`, but this overran the current v037 memory envelope:
`RESOURCE_EXHAUSTED`, **44.38 GiB** program HBM required on a **31.25 GiB**
device, exceeding capacity by **13.13 GiB**. The top allocation was a
**9.27 GiB** bf16 `[4,8192,151936]` logits-path fusion, followed by repeated
**1.16 GiB** vocab-weight all-gathers.

Verdict: invalid/refuted for the current v037 shard/layout stack. The full
MaxText-style on-device save set cannot be carried with batch 4/global 32 and
only about **2.22 GiB** free HBM. A narrower follow-up may still test whether
attention projection saves alone reduce recompute without forcing the vocab
logits live range into an OOM schedule.

## [2026-06-12] loop-iteration | cc5 v048 launched: MaxText save_dot_except_mlpwi remat policy on v037 frontier

Launched workload `alekseyv-qwen3-cc5-jax-v048-savedot` to test a real remat
policy change after v047 showed MLP checkpoint naming alone is a no-op for
structure. The thin image `qwen3-8b-jax:v048-save-dot-except-mlpwi` is derived
from v047's MaxText-style MLP names and changes only the non-offload remat
policy: when `--use_remat=True --offload_remat=False`, each layer checkpoint
uses `jax.checkpoint_policies.save_only_these_names("query_proj", "key_proj",
"value_proj", "out_proj", "mlpwo")` instead of `nothing_saveable`. This mirrors
MaxText's `save_dot_except_mlpwi`: save selected dot outputs on device, but
still recompute the large MLP input projections.

Runtime flags are unchanged from v037/v047: no-scan/remat, activation sharding,
Tokamax Splash max-logit 30, MaxText/T5X CE, batch size **4** per chip / global
batch **32**, RS3+AR2 SparseCore offload, selective resources, scoped VMEM
**100352**, shared-memory limit **90**, and latency-hiding scheduler rerun
**3**. Profile/HLO path:
`gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-12-qwen3-jax-v048-v037-save-dot-except-mlpwi`.

Rationale: v037 has only about **2.22 GiB** free HBM, so this may compile/load
OOM; if it fits, it should reduce recomputation in attention/out/MLP-output
dot paths and may cut the dominant convolution-fusion bucket. Success requires
clean loss and throughput above v037's **59,330 tok/s / 4433.0 ms** frontier,
or a same-band result with lower convolution-fusion time and acceptable peak
HBM.

## [2026-06-12] loop-iteration | cc5 v047 MaxText-style MLP checkpoint names: refuted (58,962 tok/s / 4465.2 ms)

Workload `alekseyv-qwen3-cc5-jax-v047-mlpnames` completed cleanly as a narrow
MLP remat-name probe on the confirmed v037 frontier. Worker0 summary was
**58,845 tok/s / 7,356 tok/s/chip / 42.2% old MFU**. Worker1 summary was
**58,962 tok/s / 7,370 tok/s/chip / 42.3% old MFU**, approximately **48.08%
MaxText-style MFU**. XProf
`2026-06-12-qwen3-jax-v047-v037-maxtext-mlp-names/2026_06_12_10_57_42`
reports **4465.2 ms** average step time, **1.3 ms** stddev, **66.6% MXU**, and
unchanged **29.03 GiB / 31.25 GiB** peak HBM with **2.2202 GiB** free. This is
below v037's **59,330 tok/s / 4433.0 ms** frontier and below v046.

The optimized train-step HLO was copied to `/tmp/qwen3-v047-hlo` with SHA256
`cc98981f93ee31317e56f90673a34cfcff386967d74eaa0ce36a8f4ed3f70870`,
**19,915,660 bytes / 119,349 lines**. Structural counts are identical to
v037: `all-gather=4628`, `all-reduce=158`, `reduce-scatter=1766`,
`async-start=394`, `async-done=394`, `convolution=2106`,
`custom-call=2256`, `copy=9918`, `fusion=22711`, `dot_general=7122`,
`splash=2211`, `tokamax=2`, `jvp_jit_silu__=576`. The naming change altered
the HLO hash but not the useful graph structure or memory envelope.

Verdict: refuted. Do not carry MaxText-style MLP checkpoint names under the
current `nothing_saveable` remat policy. The result closes the low-blast-radius
MLP naming probe; further progress likely requires changing actual remat/offload
work, CE/logit structure, or a real MLP/CE kernel.

## [2026-06-12] loop-iteration | cc5 v047 launched: MaxText-style MLP checkpoint names on current v037 frontier

Launched workload `alekseyv-qwen3-cc5-jax-v047-mlpnames` as a narrow MLP
schedule probe on the confirmed v037 frontier. The thin image
`qwen3-8b-jax:v047-maxtext-mlp-names` is derived directly from base image
`qwen3-8b-jax:v258-mlp-up-first` (`sha256:3a08bd533913...`) and changes only
`Qwen3MLP.__call__` checkpoint labels: `up_proj` becomes `mlpwi_1`,
`gate_proj` becomes `mlpwi_0`, and the `down_proj` output is named `mlpwo`.
This mirrors MaxText/T5X naming while preserving the v258 up-projection-first
evaluation order and the carried q/k/v attention order. CPU equivalence against
HF passed in `/tmp/qwen3-v047-eq` with forward/loss and all gradient comparisons
OK.

Runtime flags are unchanged from v037: no-scan/remat, activation sharding,
Tokamax Splash max-logit 30, MaxText/T5X CE, batch size **4** per chip / global
batch **32**, RS3+AR2 SparseCore offload, selective resources, scoped VMEM
**100352**, shared-memory limit **90**, and latency-hiding scheduler rerun
**3**. Profile/HLO path:
`gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-12-qwen3-jax-v047-v037-maxtext-mlp-names`.

Rationale: v046 reduced collectives/HLO size but slowed the dominant
convolution-fusion and custom-fusion buckets, so the next low-blast-radius
source probe targets MLP remat/schedule anchors directly. Success requires clean
loss and throughput above v037's **59,330 tok/s / 4433.0 ms** frontier, or a
same-band result with lower convolution-fusion time that motivates a follow-up
offload-policy or MLP-kernel experiment.

## [2026-06-12] loop-iteration | cc5 v046 replicated RMSNorm params: refuted (59,067 tok/s / 4454.9 ms)

Workload `alekseyv-qwen3-cc5-jax-v046-repnorm` completed cleanly as a
current-stack retest of replicated RMSNorm parameter sharding. Worker0 summary
was **59,039 tok/s / 7,380 tok/s/chip / 42.3% old MFU**. Worker1 summary was
**59,067 tok/s / 7,383 tok/s/chip / 42.4% old MFU**, approximately **48.16%
MaxText-style MFU**. XProf
`2026-06-12-qwen3-jax-v046-v037-replicate-rmsnorm/2026_06_12_10_45_42`
reports **4454.9 ms** average step time, **0.8 ms** stddev, **66.6% MXU**, and
**29.03 GiB / 31.25 GiB** peak HBM with **2.2176 GiB** free. This is below
v037's **59,330 tok/s / 4433.0 ms** frontier.

The optimized train-step HLO was copied to `/tmp/qwen3-v046-hlo` with SHA256
`b0df41e79bb9a9e78b84e1076e052308a2558d88dfb0da33387206af0d69ff4a`,
**19,237,870 bytes / 114,351 lines**. It reduced several structural counts
relative to v037: `all-gather=4154` vs 4628, `async-start=295` vs 394,
`async-done=295` vs 394, `custom-call=2050` vs 2256, and `copy=8232` vs 9918,
while `reduce-scatter=1766`, `convolution=2106`, `splash=2211`, `tokamax=2`,
and `jvp_jit_silu__=576` stayed fixed. Despite the smaller graph, per-host
XProf comparison showed worse convolution-fusion and custom-fusion time than
v037, with unchanged HBM pressure.

Verdict: refuted. Do not carry replicated RMSNorm params. The smaller HLO loses
schedule/overlap quality and does not improve the memory envelope.

## [2026-06-12] loop-iteration | cc5 v046 launched: replicated RMSNorm params on current v037 frontier

Launched workload `alekseyv-qwen3-cc5-jax-v046-repnorm` as a current-stack
retest of the old replicated-RMSNorm HLO observation. The thin image
`qwen3-8b-jax:v046-replicate-rmsnorm-current` is derived from
`qwen3-8b-jax:v258-mlp-up-first` and changes only the sharding entries for
`model.layers.*.input_layernorm.weight`,
`model.layers.*.post_attention_layernorm.weight`, and `model.norm.weight` from
FSDP-sharded to replicated `()`. QK-norm weights were already replicated; all
large matmul, embedding, MLP, lm-head, CE, and attention sharding remains fixed.

Runtime flags are the v037 carried stack: no-scan/remat, activation sharding,
Tokamax Splash max-logit 30, MaxText/T5X CE, batch size **4** per chip / global
batch **32**, RS3+AR2 SparseCore offload, selective resources, scoped VMEM
**100352**, shared-memory limit **90**, and latency-hiding scheduler rerun
**3**. Profile/HLO path:
`gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-12-qwen3-jax-v046-v037-replicate-rmsnorm`.

Rationale: v165/v166 and v176 refuted replicated RMSNorm on older scan/v169
frontiers, but the current no-scan v037 stack has a much larger unrolled HLO and
different SparseCore/MLP scheduling. Success requires clean loss and throughput
above v037's **59,330 tok/s / 4433.0 ms** frontier, or a near-tie with a clear
all-gather/HLO reduction that motivates a paired current-stack follow-up.

## [2026-06-12] loop-iteration | cc5 v045 CE backward residual recompute: refuted (58,683 tok/s / 4484.0 ms; no memory relief)

Workload `alekseyv-qwen3-cc5-jax-v045-cerec` completed cleanly as a MaxText/T5X
CE custom-VJP residual-contract probe on the confirmed v037 frontier. The thin
image changed only `train.py`'s `cross_entropy_with_logits` VJP: the forward
pass saves `logits`, `targets`, `z_loss`, `max_logit`, `sum_exp`, and `log_z`,
and the backward pass recomputes `shifted`, `exp_shifted`, and `log_softmax`
instead of carrying the full f32 residual tensors.

Worker0 summary was **58,655 tok/s / 7,332 tok/s/chip / 42.1% old MFU**.
Worker1 summary was **58,683 tok/s / 7,335 tok/s/chip / 42.1% old MFU**,
approximately **47.85% MaxText-style MFU**. XProf
`2026-06-12-qwen3-jax-v045-v037-ce-recompute-bwd/2026_06_12_10_34_45`
reports **4484.0 ms** average step time, **0.6 ms** stddev, **66.7% MXU**, and
unchanged **29.03 GiB / 31.25 GiB** peak HBM with **2.2202 GiB** free. It is
slower than both v037 (**59,330 tok/s / 4433.0 ms**) and v044.

The optimized train-step HLO was copied to `/tmp/qwen3-v045-hlo` with SHA256
`79946913a40c30769e2b25631b4af5ef42a0be74c052163fa4d4ac2a4511386e`,
**19,915,690 bytes / 119,349 lines**. Counts match v036/v037/v044:
`all-gather=4628`, `all-reduce=158`, `reduce-scatter=1766`,
`async-start=394`, `async-done=394`, `convolution=2106`,
`custom-call=2256`, `copy=9918`, `fusion=22711`, `dot_general=7122`,
`splash=2211`, `tokamax=2`, `jvp_jit_silu__=576`. The flagfile is identical
to v044's. The CE residual-contract rewrite changed the HLO hash by one byte
but did not reduce runtime memory or structural counts; the extra recomputation
only adds runtime drag.

Verdict: refuted. Do not carry CE residual recompute. The CE target-cotangent
no-op was already closed in v178, and this result closes the smaller residual
contract variant for the current v037 stack.

## [2026-06-12] loop-iteration | cc5 v045 launched: CE backward residual recompute on v037 frontier

Launched workload `alekseyv-qwen3-cc5-jax-v045-cerec` as a MaxText/T5X CE
custom-VJP structural probe on the confirmed v037 frontier. The thin image
`qwen3-8b-jax:v045-ce-recompute-bwd` is derived from
`qwen3-8b-jax:v258-mlp-up-first` and changes only `train.py`'s
`cross_entropy_with_logits` VJP residual: the forward pass now saves
`logits`, `targets`, `z_loss`, `max_logit`, `sum_exp`, and `log_z`, while the
backward pass recomputes `shifted`, `exp_shifted`, and `log_softmax` rather
than carrying full f32 residual tensors from forward to backward.

Runtime flags are the v037 carried stack: no-scan/remat, activation sharding,
Tokamax Splash max-logit 30, MaxText/T5X CE, batch size **4** per chip / global
batch **32**, RS3+AR2 SparseCore offload, selective resources, scoped VMEM
**100352**, shared-memory limit **90**, and latency-hiding scheduler rerun
**3**. Profile/HLO path:
`gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-12-qwen3-jax-v045-v037-ce-recompute-bwd`.

Rationale: v178 showed the target cotangent/log-softmax path is already DCE'd,
but the CE VJP still has a separate opportunity to trade recomputation for a
smaller residual contract. Success requires clean loss and throughput above
v037's **59,330 tok/s / 4433.0 ms** frontier, or a same-band result with lower
HLO/runtime memory that reopens a batch-size or sharding follow-up.

## [2026-06-12] loop-iteration | cc5 v044 attention value projection first: refuted (58,969 tok/s / 4463.5 ms)

Workload `alekseyv-qwen3-cc5-jax-v044-attnv` completed cleanly as the first
post-bracket structural source-order probe on the confirmed v037 frontier. The
thin image changed only `Qwen3Attention.__call__` so `value_proj` is computed
before `query_proj` and `key_proj`; all arithmetic, checkpoint tags, sharding,
Tokamax Splash, MaxText/T5X CE, RS3+AR2 SparseCore offload, scoped VMEM
**100352**, shared-memory limit **90**, and scheduler rerun **3** stayed fixed.

Worker0 summary was **58,948 tok/s / 7,369 tok/s/chip / 42.3% old MFU**.
Worker1 summary was **58,969 tok/s / 7,371 tok/s/chip / 42.3% old MFU**,
approximately **48.08% MaxText-style MFU**. XProf
`2026-06-12-qwen3-jax-v044-v037-attn-v-first/2026_06_12_10_21_43` reports
**4463.5 ms** average step time, **0.4 ms** stddev, **66.6% MXU**, **0.3%**
idle, and unchanged **29.03 GiB / 31.25 GiB** peak HBM with **2.2202 GiB**
free. This is below v037's **59,330 tok/s / 4433.0 ms** frontier.

The optimized train-step HLO was copied to `/tmp/qwen3-v044-hlo` with SHA256
`9112bdc32d5ca4df1fb88515ac5c01af64fb4596a5760346c5c955f1bcd7e178`,
**19,915,689 bytes / 119,349 lines**. Counts match v036/v037:
`all-gather=4628`, `all-reduce=158`, `reduce-scatter=1766`,
`async-start=394`, `async-done=394`, `convolution=2106`,
`custom-call=2256`, `copy=9918`, `fusion=22711`, `dot_general=7122`,
`splash=2211`, `tokamax=2`, `jvp_jit_silu__=576`. The source order changed
the HLO hash but not the useful structural mix or HBM envelope.

Verdict: refuted. Do not carry value-first attention ordering. The next
structural probe should target CE/lm-head or MLP residuals rather than more
attention projection ordering.

## [2026-06-12] loop-iteration | cc5 v044 launched: attention value projection before q/k on v037 frontier

Launched workload `alekseyv-qwen3-cc5-jax-v044-attnv` as the first
post-bracket structural source-order probe on the confirmed v037 frontier. The
thin image `qwen3-8b-jax:v044-attn-v-first` is derived from
`qwen3-8b-jax:v258-mlp-up-first` and changes only `Qwen3Attention.__call__`:
`value_proj` is computed before `query_proj` and `key_proj`, with the same
`checkpoint_name(..., "value_proj")` tag, same transpose, same Splash call, and
same arithmetic.

Runtime flags are the v037 carried stack: no-scan/remat, activation sharding,
Tokamax Splash max-logit 30, MaxText/T5X CE, batch size **4** per chip / global
batch **32**, RS3+AR2 SparseCore offload, selective resources, scoped VMEM
**100352**, shared-memory limit **90**, and latency-hiding scheduler rerun
**3**. Profile/HLO path:
`gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-12-qwen3-jax-v044-v037-attn-v-first`.

Rationale: v040-v043 closed the local VMEM and scheduler-rerun brackets without
beating v037. This tests whether a semantics-preserving producer-order change
near attention can alter XLA scheduling or HLO layout enough to recover the
current-cluster gap to the historical v273 band. Success requires clean loss and
throughput above v037's **59,330 tok/s / 4433.0 ms** frontier, or a same-band
runtime with materially improved profile evidence; otherwise the value-first
attention-order direction is refuted.

## [2026-06-12] loop-iteration | cc5 v043 scheduler rerun 2 on shmem90: refuted (58,968 tok/s / 4461.9 ms; byte-identical HLO)

Workload `alekseyv-qwen3-cc5-jax-v043-rerun2` completed cleanly as the low-side
latency-hiding scheduler rerun bracket on the confirmed v036/v037 shmem90
frontier. It kept image, model shape, no-scan/remat, activation sharding,
Tokamax Splash, MaxText/T5X CE, RS3+AR2 SparseCore offload, selective
resources, scoped VMEM **100352**, shared-memory limit **90**, and the same
profile window fixed. The only changed flag was
`--xla_latency_hiding_scheduler_rerun=3` to **2**.

Worker0 summary was **58,942 tok/s / 7,368 tok/s/chip / 42.3% old MFU**.
Worker1 summary was **58,968 tok/s / 7,371 tok/s/chip / 42.3% old MFU**,
approximately **48.08% MaxText-style MFU**. XProf
`2026-06-12-qwen3-jax-v043-v037-shmem90-rerun2/2026_06_12_10_10_04` reports
**4461.9 ms** average step time, **0.1 ms** stddev, **66.7% MXU**, and
**29.03 GiB / 31.25 GiB** peak HBM with **2.2202 GiB** free. This is slightly
faster than v042 but still below v037's **59,330 tok/s / 4433.0 ms** frontier.

The optimized train-step HLO was copied to `/tmp/qwen3-v043-hlo` with SHA256
`81e3c3c58b3fbfa12c0b203bdc9d0df428913fad3462ea5552e36cd07dcd5a96`,
**19,915,660 bytes / 119,349 lines**, byte-identical to v036/v037/v042. HLO
text counts also match v036: `all-gather=4628`, `all-reduce=158`,
`reduce-scatter=1766`, `async-start=394`, `async-done=394`,
`convolution=2106`, `custom-call=2256`, `copy=9918`, `fusion=22711`,
`dot_general=7122`, `splash=2211`, `tokamax=2`, `jvp_jit_silu__=576`. The
flagfile diff against v036 contains only the rerun-count change from **3** to
**2**.

Verdict: refuted. With v042/v043, the current shmem90 rerun-count bracket is
closed: rerun **2** and **4** are same-HLO slower than rerun **3**. Keep
rerun **3** and stop scalar rerun sweeps unless a structural code change
changes the HLO neighborhood.

## [2026-06-12] loop-iteration | cc5 v042 scheduler rerun 4 on shmem90: refuted (58,965 tok/s / 4462.4 ms; byte-identical HLO)

Workload `alekseyv-qwen3-cc5-jax-v042-rerun4` completed cleanly as a bounded
current-stack scheduler interaction test. It kept image, model shape,
no-scan/remat, activation sharding, Tokamax Splash, MaxText/T5X CE, RS3+AR2
SparseCore offload, selective resources, scoped VMEM **100352**,
shared-memory limit **90**, and the same profile window fixed. The only changed
flag was `--xla_latency_hiding_scheduler_rerun=3` to **4**.

Worker0 summary was **58,936 tok/s / 7,367 tok/s/chip / 42.3% old MFU**.
Worker1 summary was **58,965 tok/s / 7,371 tok/s/chip / 42.3% old MFU**,
approximately **48.08% MaxText-style MFU**. XProf
`2026-06-12-qwen3-jax-v042-v037-shmem90-rerun4/2026_06_12_09_58_46` reports
**4462.4 ms** average step time, **0.4 ms** stddev, **66.6% MXU**, and
**29.03 GiB / 31.25 GiB** peak HBM with **2.2202 GiB** free. This is below
v037's **59,330 tok/s / 4433.0 ms** frontier.

The optimized train-step HLO was copied to `/tmp/qwen3-v042-hlo` with SHA256
`81e3c3c58b3fbfa12c0b203bdc9d0df428913fad3462ea5552e36cd07dcd5a96`,
**19,915,660 bytes / 119,349 lines**, byte-identical to v036/v037. HLO text
counts also match v036: `all-gather=4628`, `all-reduce=158`,
`reduce-scatter=1766`, `async-start=394`, `async-done=394`,
`convolution=2106`, `custom-call=2256`, `copy=9918`, `fusion=22711`,
`dot_general=7122`, `splash=2211`, `tokamax=2`, `jvp_jit_silu__=576`. The
flagfile diff against v036 contains only the rerun-count change from **3** to
**4**.

Verdict: refuted. Rerun4 is same-HLO runtime drag on this frontier. Run the
low-side rerun2 bracket once, then close the rerun-count axis if it does not
beat v037.

## [2026-06-12] loop-iteration | cc5 v041 scoped VMEM 99840 on shmem90: refuted (58,875 tok/s / 4468.6 ms)

Workload `alekseyv-qwen3-cc5-jax-v041-vm99840` completed cleanly as the
low-side scoped-VMEM interaction test on the confirmed v036/v037 shmem90
frontier. It kept image, model shape, no-scan/remat/activation sharding,
Tokamax Splash, MaxText/T5X CE, RS3+AR2 SparseCore offload, selective
resources, shared-memory limit **90**, and the same profile window fixed. The
only changed flag was `--xla_tpu_scoped_vmem_limit_kib=100352` to **99840**.

Worker0 summary was **58,860 tok/s / 7,358 tok/s/chip / 42.2% old MFU**.
Worker1 summary was **58,875 tok/s / 7,359 tok/s/chip / 42.2% old MFU**,
approximately **48.00% MaxText-style MFU**. XProf
`2026-06-12-qwen3-jax-v041-v037-shmem90-vmem99840/2026_06_12_09_46_48`
reports **4468.6 ms** average step time, **0.5 ms** stddev, **66.5% MXU**, and
**29.03 GiB / 31.25 GiB** peak HBM with **2.2147 GiB** free. This is slower
than v037's **59,330 tok/s / 4433.0 ms / 29.03 GiB** without improving memory.

The optimized train-step HLO was copied to `/tmp/qwen3-v041-hlo` with SHA256
`2d0f67ced58d725527e998e9e8492f74e2b4024d454fcf431932a44def3b2637`,
**19,906,164 bytes / 119,252 lines**. HLO text counts: `all-gather=4628`,
`all-reduce=158`, `reduce-scatter=1766`, `async-start=382`,
`async-done=382`, `convolution=2106`, `custom-call=2248`, `copy=9923`,
`fusion=22711`, `dot_general=7121`, `splash=2211`, `tokamax=2`,
`jvp_jit_silu__=576`. The flagfile diff against v036 contains only the
scoped-VMEM change from **100352** to **99840**.

Verdict: refuted. Together v040/v041 close the local scoped-VMEM interaction
around the shmem90 frontier: both high-side **100864** and low-side **99840**
change HLO and regress. Keep **100352**.

## [2026-06-12] loop-iteration | cc5 v040 scoped VMEM 100864 on shmem90: refuted (58,940 tok/s / 4465.2 ms)

Workload `alekseyv-qwen3-cc5-jax-v040-vm100864` completed cleanly as the
high-side scoped-VMEM interaction test on the confirmed v036/v037 shmem90
frontier. It kept image, model shape, no-scan/remat/activation sharding,
Tokamax Splash, MaxText/T5X CE, RS3+AR2 SparseCore offload, selective
resources, and shared-memory limit **90** fixed. The only changed flag was
`--xla_tpu_scoped_vmem_limit_kib=100352` to **100864**.

The run regressed on both workers: worker0 summary was **58,915 tok/s / 7,364
tok/s/chip / 42.2% old MFU**, and worker1 summary was **58,940 tok/s / 7,368
tok/s/chip / 42.3% old MFU**, approximately **48.05% MaxText-style MFU**. XProf
split into per-host runs
`2026-06-12-qwen3-jax-v040-v037-shmem90-vmem100864/2026_06_12_09_33_15` and
`2026-06-12-qwen3-jax-v040-v037-shmem90-vmem100864/2026_06_12_09_33_16`, both
reporting **4465.2 ms** average step time and **0.7 ms** stddev. MXU was
**67.0%** on one host and **66.4%** on the other. Peak HBM stayed at **29.03
GiB / 31.25 GiB**, with **2.2201 GiB** free, so the slowdown is schedule/HLO
choice rather than capacity.

The optimized train-step HLO was copied to `/tmp/qwen3-v040-hlo` with SHA256
`6e82cf40ff59806506583187b0d32c3242ae906e8aa6a6f49a2e725aa2c747a6`,
**19,915,428 bytes / 119,347 lines**. HLO text counts: `all-gather=4628`,
`all-reduce=158`, `reduce-scatter=1766`, `async-start=394`,
`async-done=394`, `convolution=2106`, `custom-call=2256`, `copy=9915`,
`fusion=22711`, `dot_general=7122`, `splash=2211`, `tokamax=2`,
`jvp_jit_silu__=576`. The flagfile diff against v036 contains only the
scoped-VMEM change from **100352** to **100864**.

Verdict: refuted. Keep **100352** as the current carried scoped-VMEM setting on
the shmem90 frontier unless the low-side interaction point beats it.

## [2026-06-12] loop-iteration | cc5 v039 shared-memory limit 95: refuted (58,987 tok/s / 4460.7 ms; shmem90 remains best)

Workload `alekseyv-qwen3-cc5-jax-v039-shm95` completed cleanly as the high-side
shared-memory-limit bracket. It changed only
`--xla_tpu_scheduler_percent_shared_memory_limit=90` to **95**.

Worker0 summary was **58,871 tok/s / 7,359 tok/s/chip / 42.2% old MFU**.
Worker1 summary was **58,987 tok/s / 7,373 tok/s/chip / 42.3% old MFU**,
approximately **48.09% MaxText-style MFU**. XProf profiles split into per-host
runs:
`2026-06-12-qwen3-jax-v039-v037-shmem95/2026_06_12_09_20_55` and
`2026-06-12-qwen3-jax-v039-v037-shmem95/2026_06_12_09_20_56`, both reporting
**4460.7 ms** average step time and **1.1 ms** stddev. MXU was **67.0%** on one
host and **66.0%** on the other. Peak HBM was **30.93 GiB / 31.25 GiB**, with
only **0.318 GiB** free. This is slower and much tighter on HBM than v037's
**59,330 tok/s / 4433.0 ms / 29.03 GiB**.

The optimized train-step HLO was copied to `/tmp/qwen3-v039-hlo` with SHA256
`81e3c3c58b3fbfa12c0b203bdc9d0df428913fad3462ea5552e36cd07dcd5a96`,
**19,915,660 bytes / 119,349 lines**, byte-identical to v036. HLO text counts
match v036: `all-gather=4628`, `all-reduce=158`, `reduce-scatter=1766`,
`async-start=394`, `async-done=394`, `convolution=2106`,
`custom-call=2256`, `copy=9918`, `fusion=22711`, `dot_general=7122`,
`splash=2211`, `tokamax=2`, `jvp_jit_silu__=576`. The flagfile diff against
v036 contains only the shared-memory-limit change from **90** to **95**.

Verdict: refuted. Together v038/v039 bracket the current shared-memory schedule:
**85** changes HLO and regresses, **95** keeps the HLO but restores the 30.93
GiB heap cliff and regresses. Keep **90** as the confirmed current-cluster
frontier.

## [2026-06-12] loop-iteration | cc5 v039 launched: v037 frontier with shared-memory limit 95

Launched workload `alekseyv-qwen3-cc5-jax-v039-shm95` as the high-side bracket
of the confirmed v036/v037 shmem90 frontier. It keeps the v037 stack fixed and
changes only `--xla_tpu_scheduler_percent_shared_memory_limit=90` to **95**.

Rationale: v038 showed **85** is too low, regressing to **58,781 tok/s /
4474.8 ms** and increasing peak HBM. Test whether a slightly higher cap keeps
the useful shmem90 schedule while recovering any constrained shared-memory
placement. Success requires beating or matching v037's **59,330 tok/s /
4433.0 ms** with equal or lower HBM; otherwise keep **90**.

## [2026-06-12] loop-iteration | cc5 v038 shared-memory limit 85: refuted (58,781 tok/s / 4474.8 ms; slower and higher HBM)

Workload `alekseyv-qwen3-cc5-jax-v038-shm85` completed cleanly as the lower
shared-memory-limit bracket of the confirmed v036/v037 frontier. It changed
only `--xla_tpu_scheduler_percent_shared_memory_limit=90` to **85**.

The run regressed on both workers. Worker0 summary: **58,757 tok/s / 7,345
tok/s/chip / 42.1% old MFU**. Worker1 summary: **58,781 tok/s / 7,348
tok/s/chip / 42.1% old MFU**. XProf
`2026-06-12-qwen3-jax-v038-v037-shmem85/2026_06_12_09_08_10` reports
**4474.8 ms** average step time, **0.7 ms** stddev, **66.2% MXU**, **0.3%**
idle time, and **29.95 GiB / 31.25 GiB** peak HBM with **1.30 GiB** free. That
is worse than v037's **59,330 tok/s / 4433.0 ms / 29.03 GiB**.

The optimized train-step HLO was copied to `/tmp/qwen3-v038-hlo` with SHA256
`ed0eb632a76be13aeec8e8190b2646aa267715607da1724785437866f21b1a2c`,
**18,145,628 bytes / 109,500 lines**. HLO text counts: `all-gather=3843`,
`all-reduce=158`, `reduce-scatter=1766`, `async-start=449`,
`async-done=449`, `convolution=2087`, `custom-call=2040`, `copy=10619`,
`fusion=18285`, `dot_general=6339`, `splash=2211`, `tokamax=2`,
`jvp_jit_silu__=576`. The flagfile diff against v036 contains only the
shared-memory-limit change from **90** to **85**.

Verdict: refuted. The lower cap selects a materially different, slower schedule
and also raises peak HBM. Keep **90** unless the high-side bracket at **95**
beats it.

## [2026-06-12] loop-iteration | cc5 v038 launched: v037 frontier with shared-memory limit 85

Launched workload `alekseyv-qwen3-cc5-jax-v038-shm85` as a one-flag bracket of
the confirmed v036/v037 shared-memory schedule. It keeps the v037 stack fixed:
image `qwen3-8b-jax:v258-mlp-up-first`, no scan, remat, activation sharding,
Tokamax Splash max-logit 30, MaxText/T5X CE, batch size **4** per chip / global
batch **32**, RS3+AR2 SparseCore offload, selective resources, scoped VMEM
100352, and the same profile window. The only changed flag is
`--xla_tpu_scheduler_percent_shared_memory_limit=90` to **85**.

Rationale: shmem90 reproducibly improved the current v030 exact stack
(**59,330 tok/s / 4433.0 ms**) and freed HBM versus the 100% schedule. Test
whether a slightly lower shared-memory cap further improves the memory schedule
or starts to starve useful fusion/shared-memory placement. Success requires
beating v037's **59,330 tok/s / 4433.0 ms** band with normal loss and valid
profile/HLO; otherwise keep **90**.

## [2026-06-12] loop-iteration | cc5 v036/v037 shared-memory limit 90 confirmed: supported (59,330 tok/s / 48.37% MaxText-style MFU)

Workload `alekseyv-qwen3-cc5-jax-v036-shm90` improved the fresh v030
current-cluster frontier by changing only
`--xla_tpu_scheduler_percent_shared_memory_limit=100` to **90**. Worker0
summary was **59,314 tok/s / 7,414 tok/s/chip / 42.5% old MFU** and worker1
summary was **59,336 tok/s / 7,417 tok/s/chip / 42.5% old MFU**, approximately
**48.38% MaxText-style MFU**.

The exact rerun `alekseyv-qwen3-cc5-jax-v037-shm90r` confirmed the signal:
worker0 summary **59,299 tok/s / 7,412 tok/s/chip / 42.5% old MFU**, worker1
summary **59,330 tok/s / 7,416 tok/s/chip / 42.5% old MFU**, approximately
**48.37% MaxText-style MFU**. XProf
`2026-06-12-qwen3-jax-v037-v036-shmem90-rerun/2026_06_12_08_56_08` reports
**4433.0 ms** average step time, **0.1 ms** stddev, **66.7% MXU**, **0.3%**
idle time, and **29.03 GiB / 31.25 GiB** peak HBM with **2.22 GiB** free. This
beats the fresh v030 exact stack's **58,997 tok/s / 4459.0 ms / 48.1%
MaxText-style MFU** band and remains ahead of the fresh MaxText Qwen3 baseline
at **45.4-45.7% MFU**.

The v036 optimized train-step HLO was copied to `/tmp/qwen3-v036-hlo` with
SHA256 `81e3c3c58b3fbfa12c0b203bdc9d0df428913fad3462ea5552e36cd07dcd5a96`,
**19,915,660 bytes / 119,349 lines**. HLO text counts:
`all-gather=4628`, `all-reduce=158`, `reduce-scatter=1766`,
`async-start=394`, `async-done=394`, `convolution=2106`,
`custom-call=2256`, `copy=9918`, `fusion=22711`, `dot_general=7122`,
`splash=2211`, `tokamax=2`, `jvp_jit_silu__=576`. The flagfile diff against
v030 is clean and contains only the shared-memory-limit change from **100** to
**90**.

Verdict: supported and carried as the current fresh native-JAX frontier. Next
probe should bracket the shared-memory limit locally, starting below 90, because
the gain appears to be a memory-schedule effect rather than a code/HLO change.

## [2026-06-12] loop-iteration | cc5 v036 launched: current v030 no-scan frontier with shared-memory limit 90

Launched workload `alekseyv-qwen3-cc5-jax-v036-shm90` as a current-cluster
batch-4 retest of the historical shmem90 memory-pressure schedule. It returns
to the fresh best v030 image and shape: `v258-mlp-up-first`, no scan, remat,
activation sharding, Tokamax Splash max-logit 30, MaxText/T5X CE, batch size
**4** per chip / global batch **32**, RS3+AR2 SparseCore offload, selective
resources, scoped VMEM 100352, and shared-memory limit changed from **100** to
**90**.

Rationale: historical v277 showed shmem90 could reduce peak HBM by about
**1.90 GiB** at batch 4 but slightly regressed throughput. Since today's exact
v273/v030 rerun is about **1.7-2.2%** slower than the historical run despite
byte-identical HLO/flags, retest whether the current runtime prefers the
lower-HBM schedule. Success requires beating the fresh v030 **58,997 tok/s /
4459.0 ms** band with normal loss and valid profile/HLO.

## [2026-06-12] loop-iteration | cc5 v035 lm-head vocab-axis batch 5 shmem90: invalid/refuted (compile OOM, 32.17 GiB)

Workload `alekseyv-qwen3-cc5-jax-v035-lmv90` failed during train-step
compilation on both workers with `RESOURCE_EXHAUSTED`. It kept v034's patched
vocab-axis `lm_head.weight` FSDP image and direct no-scan batch-5 MaxText-CE
runtime fixed, changing only `--xla_tpu_scheduler_percent_shared_memory_limit`
from **100** to **90**.

The flag did not materially reduce the batch-5 program. XLA reported
**32.17 GiB / 31.25 GiB** used, exceeding capacity by **945.68 MiB**, versus
v034 at **32.18 GiB** and **952.96 MiB** over capacity. Total HBM usage was at
least **32.42 GiB**: **26.45 GiB** program, **5.72 GiB** arguments, and
**260 MiB** reserved. The largest temp remained the full-vocabulary
`bf16[5,8192,151936]` logits allocation at **11.59 GiB**.

The optimized train-step HLO was copied to `/tmp/qwen3-v035-hlo`. SHA256 is
`3921313675d37d06bb031ccd360adb78710f2e95320c65f6dd9865507d8eee90`,
with **18,630,902 bytes / 112,210 lines**. Memory report SHA256 is
`8c9090c1f2a88be9a44f95ef5fdc5206892d5e3f1d29667c21c02cefb4a0ba74`.
Text counts: `all-gather=6131`, `all-reduce=159`, `reduce-scatter=2776`,
`async-start=429`, `async-done=429`, `convolution=3006`,
`custom-call=3644`, `copy=21513`, `fusion=27645`, `dot_general=6755`,
`splash=2861`, `tokamax=2`, `jvp_jit_silu__=576`.

Verdict: invalid/refuted. Shared-memory 90 changes the HLO shape but not the
dominant memory term. Direct batch 5 with MaxText CE remains blocked until the
lm-head+CE path stops materializing full logits, or until a materially
different sharding/CE implementation is available.

## [2026-06-12] loop-iteration | cc5 v035 launched: v034 batch 5 plus shared-memory limit 90

Launched workload `alekseyv-qwen3-cc5-jax-v035-lmv90` as a one-flag follow-up
to v034. It uses the same vocab-axis `lm_head.weight` FSDP image and the same
no-scan batch-5 MaxText-CE runtime, but changes only
`--xla_tpu_scheduler_percent_shared_memory_limit=100` to `90`.

Rationale: v034 reduced the direct batch-5 compile OOM from v278's **33.10
GiB** to **32.18 GiB**, missing capacity by **952.96 MiB**. Historical v277
showed shared-memory limit 90 can free about **1.90 GiB** at batch 4 on the
same no-scan frontier, although with some runtime regression. Success requires
compiling/running batch 5 cleanly; throughput can then decide whether the
extra batch work offsets the memory-pressure schedule.

## [2026-06-12] loop-iteration | cc5 v034 lm-head vocab-axis batch 5: invalid/refuted (compile OOM, 32.18 GiB; logits still full)

Workload `alekseyv-qwen3-cc5-jax-v034-lmvbs5` failed during train-step
compilation on both workers with `RESOURCE_EXHAUSTED`. The code patch was the
intended single mechanism: image
`qwen3-8b-jax:v6e8-qwen3-8b-jax-20260612-v034-lmhead-vocab-bs5` changed only
`lm_head.weight` sharding from `("tp", "fsdp")` to `("fsdp", None)` on top of
the v258 MLP-up-first image. The runtime used no scan, remat, activation
sharding, Tokamax Splash max-logit 30, MaxText/T5X CE, batch size **5** per
chip / global batch **40**, RS3+AR2 SparseCore offload, selective resources,
scoped VMEM 100352, and shared-memory limit 100.

The OOM improved but did not clear. XLA reported **32.18 GiB / 31.25 GiB**
used, exceeding capacity by **952.96 MiB**. Total HBM usage was at least
**32.43 GiB**: **26.45 GiB** program, **5.72 GiB** arguments, and **260 MiB**
reserved. The largest temp remained the full-vocabulary logits allocation
`bf16[5,8192,151936]` at **11.59 GiB**, so vocab-axis lm-head sharding did not
partition the MaxText CE logits.

The optimized train-step HLO was copied to `/tmp/qwen3-v034-hlo`. SHA256 is
`ead19dcd076f54ec36ffa195881da365dbe2cdc72669ffb756f25284a9a90177`,
with **19,323,341 bytes / 116,413 lines**. Memory report SHA256 is
`caf22bd7b3688517cd4ca65791bbb422938c05e48e77f18408ac19bf53d7a644`.
Text counts: `all-gather=6515`, `all-reduce=159`, `reduce-scatter=2776`,
`async-start=426`, `async-done=426`, `convolution=3107`,
`custom-call=3848`, `copy=22895`, `fusion=29007`, `dot_general=7016`,
`splash=2861`, `tokamax=2`, `jvp_jit_silu__=576`.

Verdict: invalid/refuted as a standalone memory fix. The sharding change saves
roughly **0.9 GiB** versus the previous direct batch-5 OOM, but the full logits
remain materialized. Run v035 with shared-memory limit 90 as the only cheap
follow-up; if that fails or is too slow, direct batch scaling needs a real
streaming/fused CE path rather than more lm-head sharding.

## [2026-06-12] loop-iteration | cc5 v034 launched: no-scan MaxText-CE batch 5 with vocab-axis lm_head FSDP

Launched workload `alekseyv-qwen3-cc5-jax-v034-lmvbs5` to test the smallest
structural memory-relief change for direct batch 5 on the no-scan frontier.
The image
`qwen3-8b-jax:v6e8-qwen3-8b-jax-20260612-v034-lmhead-vocab-bs5` is derived
from `v258-mlp-up-first` and changes only the sharding rule for
`lm_head.weight`: from hidden-axis FSDP `("tp", "fsdp")` to vocab-axis FSDP
`("fsdp", None)`. The runtime otherwise uses the v030 best current-cluster
stack: no scan, remat, activation sharding, Tokamax Splash max-logit 30,
MaxText/T5X CE, selective resources, SparseCore AR/RS offload plus aggregator,
RS latency multiplier **3**, AR latency multiplier **2**, scoped VMEM 100352,
and shared-memory limit 100.

Rationale: direct batch-5 MaxText CE on the current sharding failed in v278
with a **33.10 GiB / 31.25 GiB** compile OOM dominated by an
**11.59 GiB** full-vocabulary logits temp. Vocab-axis lm-head sharding was slow
at batch 4 on an older stack, but it is the smallest semantics-preserving way
to test whether the logits temp can be partitioned enough to make batch 5 fit
without falling back to the slow scan/offload path. Success requires a clean
compile/run and throughput above the v030 fresh frontier's **58,997 tok/s**.

## [2026-06-12] loop-iteration | cc5 v030-agov while-double-buffer/offload-unrolled probe: invalid/refuted (retired flag + compile OOM)

Workload `alekseyv-qwen3-cc5-jax-v030-agov` was a two-arm offload/scan side
probe, not a no-scan frontier run. Arm A attempted scan + while-loop double
buffering with profile/HLO, but failed immediately on both workers after
LIBTPU reported a retired flag access. Arm B attempted unrolled/no-scan
`offload_attn` at batch size **4** and failed compile-time HBM OOM on both
workers: **67.67 GiB / 31.25 GiB**, exceeded by **36.42 GiB**. The largest
temps were the expected attention/logit-scale unrolled intermediates.

Verdict: invalid/refuted. Do not carry while-loop double buffering or unrolled
`offload_attn` from this job. The result further supports the split seen in
other runs: scan/offload fits but is slow, while unrolled/no-scan needs a
structural CE/lm-head memory reduction before batch or remat experiments are
useful.

## [2026-06-12] loop-iteration | cc5 v033 RS2+AR2 current-cluster retest: refuted/tie-slower (58,986 tok/s; 4461.2 ms)

Workload `alekseyv-qwen3-cc5-jax-v033-rs2ar2` completed cleanly with
`EXIT_CODE=0`. It kept the v030 exact-v273 stack fixed and changed only
`--xla_tpu_sparse_core_reduce_scatter_latency_multiplier=3` to `2`; the
flagfile diff against v030 is exactly that one line. AR latency multiplier
remained `2`.

The run did not improve the current-cluster frontier. Worker0 summary:
**58,965 tok/s / 7,371 tok/s/chip / 42.3% old MFU**. Worker1 summary:
**58,986 tok/s / 7,373 tok/s/chip / 42.3% old MFU**, approximately
**48.1% MaxText-style MFU**. This is tied within noise but below v030's
**58,997 tok/s**.

XProf run `2026-06-12-qwen3-jax-v033-rs2-ar2/2026_06_12_08_23_34` reports
**4461.2 ms** average step time, **66.7% MXU**, and **0.3% idle**, essentially
the same as v030 at **4459.0 ms / 66.7% MXU**. Peak HBM is also effectively
unchanged: **30.93 GiB / 31.25 GiB**, leaving **0.3175 GiB** free.

The profiled train-step HLO was copied to `/tmp/qwen3-v033-hlo`. SHA256 is
`1f74e229fef548b2267fc3d9deddec66ca34e6062fbd37c6fa9cb31908001e7a`,
with **19,931,764 bytes / 119,437 lines**. Text counts:
`all-gather=7009`, `all-reduce=165`, `reduce-scatter=2774`,
`async-start=394`, `async-done=394`, `convolution=3077`,
`custom-call=3849`, `copy=22126`, `fusion=30947`, `dot_general=7309`,
`splash=2861`, `tokamax=2`, `jvp_jit_silu__=576`.

Verdict: do not carry RS2+AR2 on the current stack. The RS bracket is now
closed under the current cluster/runtime: RS2, RS3, and RS4 all land in the
same band, with RS3 still the measured best. Keep v030 as the fresh-run
frontier.

## [2026-06-12] loop-iteration | cc5 v033 launched: v030 stack with RS latency multiplier 2 / AR latency multiplier 2

Launched workload `alekseyv-qwen3-cc5-jax-v033-rs2ar2` to close the
current-cluster SparseCore reduce-scatter latency bracket. It keeps the v030
exact-v273 no-scan stack fixed: image `qwen3-8b-jax:v258-mlp-up-first`,
batch size **4** per chip / global batch **32**, seq **8192**, remat,
activation sharding, Tokamax Splash max-logit 30, MaxText/T5X CE, selective
resources, SparseCore AR/RS offload, aggregator, shared-memory limit 100, and
offload tracing. The only intended change is
`--xla_tpu_sparse_core_reduce_scatter_latency_multiplier=3 -> 2`; AR latency
multiplier remains `2`.

Rationale: v032 showed RS4+AR2 is slightly slower and tighter on HBM than RS3.
This lower-side probe tests whether the current runtime prefers the older
RS2+AR2 schedule. Success requires beating v030's **58,997 tok/s /
4459.0 ms** band with normal loss and valid profile/HLO.

## [2026-06-12] loop-iteration | cc5 v029-bs5 offload-attn/scan batch-5: refuted (48,085 tok/s; async update overhead)

Workload `alekseyv-qwen3-cc5-jax-v029-bs5` completed cleanly with
`EXIT_CODE=0`. This run used the offload-attn + scan image
`qwen3-8b-jax:v6e8-qwen3-8b-jax-20260612-v024-offload`, batch size **5** per
chip / global batch **40**, seq **8192**, Tokamax Splash, Tokamax chunked-XLA
CE, and the 25-flag scheduler stack. It was an offload/scan batch-scaling
probe, not the no-scan MaxText-CE v030 frontier.

It fits, but is too slow. Worker0 summary: **48,076 tok/s /
6,010 tok/s/chip / 34.5% old MFU**. Worker1 summary: **48,085 tok/s /
6,011 tok/s/chip / 34.5% old MFU**, approximately **39.2% MaxText-style MFU**.
Despite the larger batch, this is far below the no-scan v030 frontier
(**58,997 tok/s / 48.1% MaxText-style MFU**) and does not improve over the
batch-4 offload/scan variants in aggregate throughput.

XProf run `2026-06-12-qwen3-jax-v029-bs5/2026_06_12_08_01_13` reports
**54.6% MXU**, **0.2% idle**, and peak HBM **27.80 GiB / 31.25 GiB**, leaving
**3.44 GiB** free. Op profile shows convolution fusion
**75,505.6 ms / 46.1%**, custom-call/Splash **43,549.1 ms / 26.6%**, and
async-done **18,973.3 ms / 11.6%**, mostly
`dynamic-update-slice-done.2`.

The profiled train-step HLO was copied to `/tmp/qwen3-v029-bs5-hlo`. SHA256 is
`26861ea4c667766b6df24f35cb0e70f3f055fcea5d93bd43844564422021c460`,
with **1,368,380 bytes / 8,150 lines**. Text counts:
`all-gather=328`, `all-reduce=73`, `reduce-scatter=20`, `async-start=36`,
`async-done=36`, `convolution=183`, `custom-call=137`, `copy=1317`,
`fusion=1807`, `dot_general=369`, `splash=111`, `tokamax=340`,
`dynamic-update-slice=135`.

Verdict: do not carry scan/offload batch 5. It confirms that offload/scan can
make larger batches fit, but the dynamic-update-slice completion overhead and
lower MXU erase the batch-size benefit. Batch scaling still needs a structural
CE/lm-head memory reduction on the no-scan stack, not scan/offload.

## [2026-06-12] loop-iteration | cc5 v032 RS4+AR2 current-cluster retest: refuted/tie-slower (58,971 tok/s; 4462.0 ms)

Workload `alekseyv-qwen3-cc5-jax-v032-rs4ar2` completed cleanly with
`EXIT_CODE=0`. It kept the v030 exact-v273 stack fixed and changed only
`--xla_tpu_sparse_core_reduce_scatter_latency_multiplier=3` to `4`; the
flagfile diff against v030 is exactly that one line. AR latency multiplier
remained `2`.

The run did not improve the current-cluster frontier. Worker0 summary:
**58,948 tok/s / 7,369 tok/s/chip / 42.3% old MFU**. Worker1 summary:
**58,971 tok/s / 7,371 tok/s/chip / 42.3% old MFU**, approximately
**48.1% MaxText-style MFU**. This is below v030's **58,997 tok/s** and tied
within noise to v031's no-profile **58,993 tok/s**.

XProf run `2026-06-12-qwen3-jax-v032-rs4-ar2/2026_06_12_08_09_37` reports
**4462.0 ms** average step time, **66.6% MXU**, and **0.3% idle**, versus v030
at **4459.0 ms / 66.7% MXU**. Peak HBM is worse: **31.16 GiB / 31.25 GiB**,
leaving only **0.0896 GiB** free, versus v030's **30.93 GiB** peak and
**0.3181 GiB** free.

The profiled train-step HLO was copied to `/tmp/qwen3-v032-hlo`. SHA256 is
`a92469ed51218d7814d863f5e78b678967b81a836d9a558a1d9c966cc4a206e0`,
with **19,901,683 bytes / 119,264 lines**. Text counts:
`all-gather=6988`, `all-reduce=165`, `reduce-scatter=2774`,
`async-start=391`, `async-done=391`, `convolution=3083`,
`custom-call=3846`, `copy=22124`, `fusion=30859`, `dot_general=7296`,
`splash=2861`, `tokamax=2`, `jvp_jit_silu__=576`.

Verdict: do not carry RS4+AR2 on the current stack. It changes schedule/HLO,
tightens HBM, and is slightly slower. Keep the v030 RS3+AR2 exact-v273 stack
as the fresh-run frontier.

## [2026-06-12] loop-iteration | cc5 v032 launched: v030 stack with RS latency multiplier 4 / AR latency multiplier 2

Launched workload `alekseyv-qwen3-cc5-jax-v032-rs4ar2` as the next
single-variable scheduler probe on the current best no-scan stack. It keeps
the v030 exact-v273 image, model shape, no-scan/remat/activation-sharding
runtime, Tokamax Splash max-logit 30, MaxText/T5X CE, SparseCore AR/RS offload,
aggregator, selective resources, shared-memory limit 100, and offload tracing
fixed. The only intended change is
`--xla_tpu_sparse_core_reduce_scatter_latency_multiplier=3 -> 4`, while
`--xla_tpu_sparse_core_all_reduce_latency_multiplier=2` stays fixed.

Rationale: historical v268/v269 RS4+AR2 was a same-band alternative to the
later RS3+AR2 v272/v273 frontier. Because exact v273 HLO/flag parity now runs
about **1.7-2.2%** slower on the current cluster/runtime, this retests whether
the current environment prefers the RS4 latency model. Success requires a real
move below the v030 **4459 ms / 58,997 tok/s** band with normal loss and no
compile/runtime regression.

## [2026-06-12] loop-iteration | cc5 v031 exact-v273 no-profile/no-HLO: refuted as instrumentation explanation (58,993 tok/s)

Workload `alekseyv-qwen3-cc5-jax-v031-noprof` completed cleanly with
`EXIT_CODE=0`. It used the exact v030/v273 no-scan stack, image, shape, and
`LIBTPU_INIT_ARGS`, but removed both instrumentation sources: no `XLA_FLAGS`
HLO dump and no `--profile_dir`/XProf capture. This isolates whether the
current 59.0k tok/s band is caused by dump/trace overhead rather than runtime
variance.

The result was effectively identical to v030. Worker0 summary:
**58,991 tok/s / 7,374 tok/s/chip / 42.3% old MFU**. Worker1 summary:
**58,993 tok/s / 7,374 tok/s/chip / 42.3% old MFU**, approximately
**48.1% MaxText-style MFU**. Steps 2-5 briefly reached **59.1-59.2k tok/s**,
then settled into the same **58.9k** band seen in v029/v030.

Verdict: instrumentation is not the explanation for the historical v273 delta.
The current fresh best remains v030 at **58,997 tok/s** by summary, with v031
statistically tied but intentionally lacking XProf/HLO artifacts. Continue
optimizing from the no-scan v030 stack; do not spend more runs on profile/HLO
overhead isolation.

## [2026-06-12] loop-iteration | cc5 v028-spblk block-size probe: refuted (48,355 tok/s; over-offload signature remains)

Workload `alekseyv-qwen3-cc5-jax-v028-spblk` completed cleanly with
`EXIT_CODE=0`. This was a two-phase probe on the offload-attn + scan +
Tokamax chunked-XLA CE branch using image
`qwen3-8b-jax:v6e8-qwen3-8b-jax-20260612-v024-offload`, batch size **4** per
chip, seq **8192**, `fsdp=8,tp=1`, and the broader scheduler flag stack.

Phase A used symmetric **1024** Splash blocks. It reached **47,132 tok/s /
5,891 tok/s/chip / 33.8% old MFU**. Phase B used all-**2048** blocks to match
the MaxText block shape and was profiled; it reached **48,355 tok/s /
6,044 tok/s/chip / 34.7% old MFU**, approximately **39.5% MaxText-style MFU**.
That is a real improvement over phase A but still only ties the v026/v028
save-attn frontier and remains far below the no-scan v030 frontier
(**58,997 tok/s / 48.1% MaxText-style MFU**).

XProf run `2026-06-12-qwen3-jax-v028-spblk/2026_06_12_07_51_46` reports
**54.8% MXU**, **0.2% idle**, and peak HBM **23.72 GiB / 31.25 GiB**, leaving
**7.53 GiB** free. Device time is dominated by convolution fusion
**60,065.4 ms / 46.1%**, custom-call/Splash **34,689.7 ms / 26.6%**,
async-done **15,001.9 ms / 11.5%** mostly from
`dynamic-update-slice-done.2`, and loop fusion **10,039.0 ms / 7.7%**.

The profiled phase-B train-step HLO was copied to `/tmp/qwen3-v028-spblk-hlo`.
SHA256 is `83e7be94a7543d3a65362673255219392012ca9790e4caa11c5b3d0c5a012871`,
with **1,358,560 bytes / 8,111 lines**. Text counts:
`all-gather=323`, `all-reduce=73`, `reduce-scatter=20`, `async-start=40`,
`async-done=40`, `convolution=183`, `custom-call=142`, `copy=1231`,
`fusion=1785`, `dot_general=372`, `splash=111`, `tokamax=340`,
`dynamic-update-slice=135`.

Verdict: do not continue the offload-attn/scan block-size branch for the main
frontier. The all-2048 block shape reduces some attention cost, but the branch
still leaves excessive free HBM and spends too much time in async
dynamic-update-slice completion. Future work should either stay on the stronger
no-scan v030 stack or make a structural CE/lm-head memory change that enables
batch scaling without scan/offload overhead.

## [2026-06-12] loop-iteration | cc5 v030 exact old-v273 flagfile parity: supported for parity, refuted for old-speed recovery (58,997 tok/s)

Workload `alekseyv-qwen3-cc5-jax-v030-v273exact` completed cleanly with
`EXIT_CODE=0`. This run kept the v029/old-v273 model, image, shape, and
training stack fixed, then restored the three remaining flagfile differences:
`--xla_tpu_enable_all_reduce_offload_tracing=true`,
`--xla_tpu_enable_reduce_scatter_offload_tracing=true`, and
`--xla_tpu_scheduler_percent_shared_memory_limit=100`.

The optimized train-step HLO is still byte-identical to old v273 and v029:
SHA256 `81e3c3c58b3fbfa12c0b203bdc9d0df428913fad3462ea5552e36cd07dcd5a96`,
**19,915,660 bytes / 119,349 lines**. The train-step flagfile now also
diffs empty against old v273, with flagfile SHA256
`def115529fb823bacfffd3deec2eca631762b9c8560f01f93166ede16432c32f`.

Throughput did not recover the old 60.3k tok/s band. Worker0 summary:
**58,977 tok/s / 7,372 tok/s/chip / 42.3% old MFU**. Worker1 summary:
**58,997 tok/s / 7,375 tok/s/chip / 42.3% old MFU**, approximately
**48.1% MaxText-style MFU** after applying the MaxText-style FLOP reporting
normalization. Fresh MaxText base-b remains **6,956-6,998 tok/s/chip /
45.4-45.7% MFU**, so the native no-scan stack still leads MaxText by roughly
**5.4% tok/s/chip** on the fresh runs.

XProf run `2026-06-12-qwen3-jax-v030-v273-exactflags/2026_06_12_07_46_16`
reports **4459.0 ms** average step time, **66.7% MXU**, **0.3% idle**, and
peak HBM **30.93 GiB / 31.25 GiB** with **0.3181 GiB** free. Op profile:
convolution fusion **56,946.1 ms / 53.2%**, custom-call/Splash
**29,402.7 ms / 27.5%**, and loop fusion **10,711.0 ms / 10.0%**.

Verdict: the v029-to-old-v273 flag differences were not the source of the
current-cluster slowdown. The remaining gap to historical v273
(**60,346 tok/s / 7,543 tok/s/chip / 49.2% MaxText-style MFU**) is most likely
runtime/cluster variance rather than a model, HLO, or flag reconstruction
issue. Current fresh best is v030 at **58,997 tok/s / 48.1% MaxText-style
MFU**.

## [2026-06-12] loop-iteration | cc5 v030 launched: v029 plus exact old-v273 tracing/shared-memory flagfile values

Launched workload `alekseyv-qwen3-cc5-jax-v030-v273exact` as the direct
follow-up to v029. Model/image/shape are unchanged from v029 and old v273:
`qwen3-8b-jax:v258-mlp-up-first`, no-scan, remat, activation sharding,
Tokamax Splash max-logit 30, MaxText/T5X CE, batch size **4** per chip, seq
**8192**, mesh `fsdp=8,tp=1`. The only intended changes from v029 are the
three flagfile differences needed to match old v273:
`--xla_tpu_enable_all_reduce_offload_tracing=true`,
`--xla_tpu_enable_reduce_scatter_offload_tracing=true`, and
`--xla_tpu_scheduler_percent_shared_memory_limit=100`.

Expected signal: HLO should remain SHA256
`81e3c3c58b3fbfa12c0b203bdc9d0df428913fad3462ea5552e36cd07dcd5a96`. If
throughput returns toward **60.3k tok/s / 4358 ms**, one of the three runtime
flags was material despite byte-identical HLO. If it stays near v029's
**59.0k tok/s / 4460 ms**, the remaining old-v273 delta is likely cluster or
runtime-environment variance rather than model/HLO.

## [2026-06-12] loop-iteration | cc5 v029 v273 rerun under current root: supported but slower than old v273 (58,976 tok/s; exact HLO)

Workload `alekseyv-qwen3-cc5-jax-v029-v273r` completed cleanly with
`EXIT_CODE=0`. It reran the old full-throughput v273 native-JAX stack under
the current `qwen3_cc5` profile root: no-scan, remat, activation sharding,
Tokamax Splash with max-logit 30, MaxText/T5X CE, MLP up-proj-first image,
batch size **4** per chip / global batch **32**, SparseCore RS/AR offload with
aggregator, collective-matmul modes `none`, selective resources, RS latency
multiplier `3`, and AR latency multiplier `2`.

The run is valid and still beats the fresh MaxText baseline on normalized
MFU, but it did not reproduce old v273 throughput. Worker0 summary:
**58,954 tok/s / 7,369 tok/s/chip / 42.3% old MFU**. Worker1 summary:
**58,976 tok/s / 7,372 tok/s/chip / 42.3% old MFU**, approximately
**48.1% MaxText-style MFU**. Old v273 was **60,346 tok/s / 7,543 tok/s/chip /
43.3% old MFU**, about **49.2% MaxText-style MFU**.

XProf run `2026-06-12-qwen3-jax-v029-v273-rerun/2026_06_12_07_33_39` reports
**4460.5 ms** average step time and **66.7% MXU**, versus old v273's
**4358.1 ms / 68.2% MXU**. Peak HBM is unchanged at
**30.93 GiB / 31.25 GiB** with **0.3181 GiB** free.

The optimized train-step HLO is byte-identical to old v273:
SHA256 `81e3c3c58b3fbfa12c0b203bdc9d0df428913fad3462ea5552e36cd07dcd5a96`,
**19,915,660 bytes / 119,349 lines**. Text counts match v273 exactly:
`all-gather=6997`, `all-reduce=165`, `reduce-scatter=2774`,
`async-start=394`, `async-done=394`, `convolution=3081`,
`custom-call=3849`, `fusion=30896`, `copy=22117`, `tokamax=2`,
`jvp_jit_silu__=576`.

Flagfile comparison against old v273 showed three non-HLO-affecting runtime
differences: old v273 had `xla_tpu_enable_all_reduce_offload_tracing=true`,
`xla_tpu_enable_reduce_scatter_offload_tracing=true`, and
`xla_tpu_scheduler_percent_shared_memory_limit=100`; v029 had the tracing flags
false and shared-memory limit 95. Next run v030 should restore exactly those
three flagfile values before concluding the remaining delta is cluster/runtime
noise.

## [2026-06-12] loop-iteration | cc5 v029 launched: v273 no-scan batch-4 SparseCore/MaxText-CE rerun under current profile root

Launched workload `alekseyv-qwen3-cc5-jax-v029-v273r` to rerun the confirmed
old native-JAX v273 frontier under the current `qwen3_cc5` xprof/HLO root.
This is the direct comparison run after MaxText profiling: image
`qwen3-8b-jax:v258-mlp-up-first`, `--use_scan=False`, `--use_remat=True`,
`--offload_remat=False`, `--use_splash=True`, `--use_maxtext_ce=True`,
`--shard_acts=True`, batch size **4** per chip, seq **8192**, mesh
`fsdp=8,tp=1`, `USE_TOKAMAX_SPLASH=1`, and `TOKAMAX_MAX_LOGIT_CONST=30.0`.
The reconstructed v273 XLA stack sets scoped VMEM `100352`, collective-matmul
modes `none`, async collective fusion with all-gather fuse and multiple steps,
selective resources, SparseCore RS/AR offload with aggregator, RS latency
multiplier `3`, and AR latency multiplier `2`.

Expected validation signal: throughput should reproduce the old
**60.3k tok/s / 7.54k tok/s/chip / 43.3% old MFU** band, which normalizes to
about **49.2% MaxText-style MFU**, ahead of the fresh MaxText Qwen3 run's
**45.4-45.7%**. XProf should be near **4.36 s** average step and **68% MXU**,
with HLO hash `81e3c3c58b3fbfa12c0b203bdc9d0df428913fad3462ea5552e36cd07dcd5a96`
if the flag reconstruction is exact.

## [2026-06-12] loop-iteration | cc5 v026-mtfl offload-attn + MaxText flag stack: refuted (47,455 tok/s; 53.4% MXU; overpaid memory relief)

Workload `alekseyv-qwen3-cc5-jax-v026-mtfl` completed cleanly with
`EXIT_CODE=0`. It tested the offload-attn bs3/scan image with the broad
MaxText-like scheduler flag stack. The best host summary was worker1:
**47,455 tok/s / 5,932 tok/s/chip / 34.0% old MFU**, approximately
**38.7% MaxText-style MFU**. Worker0 was **47,428 tok/s / 5,929 tok/s/chip**.
This does not beat the cc5 v026/v028 save-attn frontier
(**48,202-48,206 tok/s / 6,025-6,026 tok/s/chip**, about **39.4%**
MaxText-style MFU), and remains far below the old full-throughput v273
native-JAX frontier.

XProf run `2026-06-12-qwen3-jax-v026-mtfl/2026_06_12_07_21_13` reports
**1785.2 ms** average step time over the captured window, **53.4% MXU**, and
only **0.3%** idle. The op profile shows convolution fusion **45.7%**,
custom-call/Splash **26.2%**, async-done **11.0%** (mostly
dynamic-update-slice-done), and loop fusion **7.8%**. Peak HBM is only
**19.66 GiB / 31.25 GiB**, leaving **11.58 GiB** free, so the run recovered
far more memory than needed while paying runtime overhead.

The optimized train-step HLO copied to `/tmp/qwen3-v026-mtfl-hlo` has SHA256
`19e26936e1d118d35b85f9449e8aee464cb824b34e160e935d5667ada98a96bb` on all
four emitted train-step files, **1,379,179 bytes / 8,273 lines** per file.
Aggregate text counts across the four files: `all-gather=1300`,
`all-reduce=292`, `reduce-scatter=80`, `convolution=736`,
`custom-call=616`, `copy=5284`, `fusion=7188`, `dot_general=1496`,
`splash=444`, `tokamax=1360`.

Verdict: do not carry the offload-attn MaxText flag stack. The smaller HLO and
large HBM margin do not close the throughput gap; the useful next comparison is
an apples-to-apples rerun of the stronger no-scan batch-4 v273
SparseCore/MaxText-CE stack under the current `qwen3_cc5` profile root.

## [2026-06-12] loop-iteration | cc5 v028 explicit scheduler memory-pressure tracking: refuted/tie (byte-identical HLO; 48,202 tok/s)

Workload `alekseyv-qwen3-cc5-jax-v028-memtrack` made only the planned
`LIBTPU_INIT_ARGS` change from v026: `--xla_tpu_enable_scheduler_memory_pressure_tracking=true`
was added to the carried scoped VMEM limit. The run completed cleanly with
normal loss and `EXIT_CODE=0`, but did not beat v026. Worker0 summary:
**48,184 tok/s / 6,023 tok/s/chip / 34.5% old MFU**. Worker1 summary:
**48,202 tok/s / 6,025 tok/s/chip / 34.6% old MFU**, approximately
**39.4% MaxText-style MFU**.

XProf run `2026-06-12-qwen3-jax-v028-memtrack/2026_06_12_07_12_13` is the
same performance class as v026: **583.5 ms** average step time, **50.4% MXU**,
and **30.92 GiB / 31.25 GiB** peak HBM with **0.3278 GiB** free. The optimized
train-step HLO is byte-identical to v026 across both hosts:
SHA256 `3636a8d7753a3d15777c76a5fed628d4761b4d3fbb7d78349f5ead3672ac39a0`
(35,678,206 bytes / 203,044 lines per host), with the same aggregate text
counts across the two identical files: `dot_general=96748`, `fusion=73470`,
`copy=64332`, `all-gather=8592`, `all-reduce=310`, `reduce-scatter=4`,
`custom-call=9228`, `splash=8118`, `tokamax=692`.

Verdict: do not carry explicit memory-pressure tracking as a meaningful change.
Keep v026 as the cc5 save-attn/Tokamax frontier by throughput, and pivot back
to the stronger no-scan SparseCore/MaxText-CE stack rather than continuing
scheduler micro-tuning on this slower HLO family.

## [2026-06-12] loop-iteration | cc5 v028 launched: v026 + explicit scheduler memory-pressure tracking

Launched workload `alekseyv-qwen3-cc5-jax-v028-memtrack` from the v026
frontier and changed only `LIBTPU_INIT_ARGS` from
`--xla_tpu_scoped_vmem_limit_kib=81920` to
`--xla_tpu_scoped_vmem_limit_kib=81920
--xla_tpu_enable_scheduler_memory_pressure_tracking=true`. The v026 flagfile
shows TPU scheduler memory-pressure tracking was `auto`, so this is a narrow
test of whether making the scheduler explicitly memory-aware improves overlap
without the full v027 scheduler bundle's HBM blow-up. Runtime/model settings
stay fixed: bs1 seq8192, `JAX_REMAT_SKIP_EVERY=5`, `remat_policy=save_attn`,
Tokamax CE chunked XLA, Tokamax Splash base2/fused-reciprocal, and
`TOKAMAX_MAX_LOGIT_CONST=30.0`.

## [2026-06-12] loop-iteration | cc5 v025-off3s offload-attn bs3 + scheduler bundle: refuted (fits but slow)

Workload `alekseyv-qwen3-cc5-jax-v025-off3s` completed the two-phase
offload-attn probe. Phase 1, bs1/offload-attn, ran correctly but was slow:
**41,031 tok/s / 5,129 tok/s/chip / 29.4% old MFU**. Phase 2, bs3/offload-attn
with the v027 scheduler bundle, fit and completed with normal loss, but
reached only worker0 **43,026 tok/s / 5,378 tok/s/chip / 30.8% old MFU** and
worker1 **43,047 tok/s / 5,381 tok/s/chip / 30.9% old MFU**. Under the
MaxText-style reporting convention this is only about **35.1% MFU**, far below
v026's **48,206 tok/s / 6,026 tok/s/chip / 39.4% MaxText-style MFU**.

XProf run `2026-06-12-qwen3-jax-v025-off3s/2026_06_12_06_58_27` shows
**1966.0 ms** average step time and **48.5% MXU**. Peak HBM drops to
**27.34 GiB / 31.25 GiB** with **3.91 GiB** free, confirming offload-attn
recovers memory, but the profile adds a large async-done/dynamic-update-slice
bucket: async-done is **9.1%** of device time and data formatting is **3.8%**.
The bs3 train-step HLO is SHA256
`67aa166bfee9fb357cb6db7e843994bfdef394fe12e1affc071492090c5cc32f`
(2,353,812 bytes / 12,397 lines), with text counts
`dot_general=3309`, `fusion=2138`, `copy=1390`, `all-gather=164`,
`all-reduce=21`, `reduce-scatter=2`, `custom-call=217`, `splash=111`,
`tokamax=340`.

Verdict: do not carry `offload_attn`. It is a memory escape hatch, not a path
to closing the MaxText performance gap.

## [2026-06-12] loop-iteration | cc5 v027 v026 + scheduler/latency-hiding flag bundle: invalid (compile-time HBM OOM)

Workload `alekseyv-qwen3-cc5-jax-v027-tkmax30-sch` failed during train-step
compilation on both workers with `EXIT_CODE=1`. The full scheduler bundle
changed the schedule enough to exceed HBM before runtime; the top live temps
include repeated **192 MiB** dot-general activations inside async collective
fusions such as `fusion.24665`, `fusion.24672`, `fusion.24440`,
`fusion.24447`, and `fusion.24492`. This refutes carrying the whole old
scheduler/latency-hiding bundle on the tight cc5 save-attn frontier.

Verdict: keep v026 as the cc5 frontier and only test scheduler knobs one at a
time if they are memory-neutral.

## [2026-06-12] loop-iteration | cc5 v027 launched: v026 + scheduler/latency-hiding flag bundle

Launched workload `alekseyv-qwen3-cc5-jax-v027-tkmax30-sch` from the v026
frontier and added the old Qwen3 scheduler/latency-hiding LIBTPU bundle:
`--xla_tpu_enable_all_experimental_scheduler_features=true`,
`--xla_tpu_enable_scheduler_memory_pressure_tracking=true`,
`--xla_tpu_host_transfer_overlap_limit=24`,
`--xla_tpu_aggressive_opt_barrier_removal=ENABLED`,
`--xla_lhs_prioritize_async_depth_over_stall=ENABLED`,
`--xla_tpu_enable_ag_backward_pipelining=true`,
the loop-invariant chain flags, `--xla_max_concurrent_host_send_recv=100`,
`--xla_tpu_scheduler_percent_shared_memory_limit=100`, and
`--xla_latency_hiding_scheduler_rerun=2`. Runtime/model settings stay fixed:
bs1 seq8192, `JAX_REMAT_SKIP_EVERY=5`, `remat_policy=save_attn`,
Tokamax CE chunked XLA, Tokamax Splash base2/fused-reciprocal, and
`TOKAMAX_MAX_LOGIT_CONST=30.0`.

Rationale: v026 is faster but still only **50.4% MXU** with visible
data-formatting/copy overhead; this tests whether the scheduler bundle can
improve overlap without changing semantics.

## [2026-06-12] loop-iteration | cc5 v026 v025 + TOKAMAX_MAX_LOGIT_CONST=30.0: supported (new cc5 best, 48,206 tok/s / 34.6% old MFU)

Workload `alekseyv-qwen3-cc5-jax-v026-tkmax30` carried the v025 stack and
changed only `TOKAMAX_MAX_LOGIT_CONST=30.0`. It completed cleanly with
`EXIT_CODE=0` and normal loss. Worker0 summary: **48,185 tok/s / 6,023
tok/s/chip / 34.5% old MFU**. Worker1 summary: **48,206 tok/s / 6,026
tok/s/chip / 34.6% old MFU**. Under the MaxText-style reporting convention,
this is approximately **39.4% MFU** (`34.6 * 1.1376426309815402`).

XProf run `2026-06-12-qwen3-jax-v026-tkmax30/2026_06_12_06_48_09` confirms the
speedup: **583.7 ms** average step time and **50.4% MXU**, versus v025
**596.5 ms / 49.3% MXU**. Peak memory remains unchanged and tight:
**30.92 GiB / 31.25 GiB** with **0.328 GiB** free. The optimized train-step HLO
changed materially and is identical across hosts: SHA256
`3636a8d7753a3d15777c76a5fed628d4761b4d3fbb7d78349f5ead3672ac39a0`
(35,678,206 bytes / 203,044 lines per host). Across both identical dumped
files, text counts are `dot_general=96748`, `fusion=73470`, `copy=64332`,
`all-gather=8592`, `all-reduce=310`, `reduce-scatter=4`,
`custom-call=9228`, `splash=8118`, `tokamax=692`.

Verdict: carry `TOKAMAX_MAX_LOGIT_CONST=30.0`. It is the first clear cc5 gain
after v025 and closes part of the gap to MaxText while preserving memory fit.

## [2026-06-12] loop-iteration | cc5 v026 launched: v025 + Tokamax max-logit constant

Launched workload `alekseyv-qwen3-cc5-jax-v026-tkmax30` from the v025
frontier stack and changed only `TOKAMAX_MAX_LOGIT_CONST=30.0`, keeping
`USE_TOKAMAX_SPLASH=1`, `TOKAMAX_USE_BASE2_EXP=1`,
`TOKAMAX_FUSE_RECIPROCAL=1`, `JAX_REMAT_SKIP_EVERY=5`, bs1 seq8192,
`tokamax_ce_impl=chunked_xla`, and `remat_policy=save_attn`. Rationale:
the older Qwen3 lane carried the constant-max-logit Tokamax Splash path after
it improved the attention/custom-call frontier, but the current cc5 v025 stack
had not tested it. Profile target is late steps 40..42 under
`gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-12-qwen3-jax-v026-tkmax30`.

## [2026-06-12] loop-iteration | cc5 v024-off3 offload-attn bs3 probe: invalid/refuted (bs1 slow; bs3 compile OOM)

Workload `alekseyv-qwen3-cc5-jax-v024-off3` tested whether the new
`offload_attn` remat policy could free enough memory to run the MaxText-like
per-chip batch 3 shape in the native JAX trainer. Phase 1, bs1 seq8192,
completed with normal loss but was much slower than the save-attn frontier:
**34,111 tok/s aggregate / 4,264 tok/s/chip / 24.5% old MFU**, versus v025
**47,528 tok/s / 5,941 tok/s/chip / 34.1% old MFU**. Phase 2, bs3 seq8192,
failed during train-step compilation with HBM OOM:
**59.41 GiB / 31.25 GiB used**, exceeded by **28.16 GiB**. The largest temp
was the Tokamax CE backward broadcast `f32[4096,153600]` at **2.34 GiB**,
followed by the CE weight pad at **1.17 GiB** and many repeated **384 MiB**
`f32[3,8192,32,128]` attention remat copies.

Verdict: do not carry `offload_attn`; it both slows the viable bs1 target and
does not make bs3 feasible. HLO/OOM artifacts are under
`gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-12-qwen3-jax-v024-off3/hlo/`,
including `module_0092.jit_train_step...oom_intermediate_module.txt` and
`oom_top_memory_consuming_instructions`.

## [2026-06-12] loop-iteration | cc5 v025 partial-remat + Tokamax Splash perf knobs: supported (new cc5 best, 47,528 tok/s / 34.1% old MFU)

Workload `alekseyv-qwen3-cc5-jax-v025-prem5-tksp50` reran v024's partial-remat
plus Tokamax Splash perf-knob stack for 50 steps with late profiling. It kept
the v017 partial-remat image (`v6e8-qwen3-8b-jax-20260612-v017-partial-remat`),
`JAX_REMAT_SKIP_EVERY=5`, `USE_TOKAMAX_SPLASH=1`,
`TOKAMAX_USE_BASE2_EXP=1`, `TOKAMAX_FUSE_RECIPROCAL=1`, bs1 seq8192,
`tokamax_ce_impl=chunked_xla`, and `remat_policy=save_attn`.

The run completed cleanly with `EXIT_CODE=0` and normal loss. Worker0 summary:
**47,505 tok/s / 5,938 tok/s/chip / 34.1% old MFU**. Worker1 summary:
**47,528 tok/s / 5,941 tok/s/chip / 34.1% old MFU**. Under the MaxText-style
MFU counter from the 2026-06-12 formula audit, this is approximately
**38.8% MFU** (`34.1 * 1.1376426309815402`). This beats v017's prior cc5
full-window best (**47,096 tok/s / 5,887 tok/s/chip / 33.8% old MFU**) and
confirms v024's steady-step signal without the one-off slow step.

XProf run `2026-06-12-qwen3-jax-v025-prem5-tksp50/2026_06_12_06_34_01` shows
**596.5 ms** average step time, **49.3% MXU**, **1.0% idle**, and train-step
dominance at **99.0%** of device time. Top families: convolution fusion
**47.4%**, custom calls/Tokamax Splash attention about **23%** by the v024/v025
shape, and loop fusion about **14%**. Peak memory is nearly saturated:
**30.92 GiB / 31.25 GiB** (98.94%), with only **0.327 GiB** free. The v025 GCS
prefix contains profile artifacts but did not materialize HLO text, likely
because the train-step compile was cache-reused; use the byte-identical v024
HLO as the structural reference for this carried stack.

Verdict: carry v025 as the current cc5 frontier. Next probes should avoid
activation/temp-memory increases and focus on attention/runtime knobs or
memory-neutral scheduling.

## [2026-06-12] loop-iteration | cc5 v024 partial-remat + Tokamax Splash perf knobs: mixed/needs long-window confirmation

Workload `alekseyv-qwen3-cc5-jax-v024-prem5-tksp` branched from the v017
partial-remat image (`v6e8-qwen3-8b-jax-20260612-v017-partial-remat`) and
changed only the Tokamax Splash perf flags:
`USE_TOKAMAX_SPLASH=1`, `TOKAMAX_USE_BASE2_EXP=1`,
`TOKAMAX_FUSE_RECIPROCAL=1`, keeping `JAX_REMAT_SKIP_EVERY=5`, bs1 seq8192,
`tokamax_ce_impl=chunked_xla`, and `remat_policy=save_attn`.

Result is mixed but promising. The run completed with `EXIT_CODE=0` and normal
loss. Worker0 summary was **46,524 tok/s / 5,815 tok/s/chip / 33.4% old MFU**
and worker1 summary was **46,580 tok/s / 5,823 tok/s/chip / 33.4% old MFU**,
below v017's full-window **47,096 tok/s / 5,887 tok/s/chip** because one
measured non-profile step slowed to **35,985 tok/s**. Normal steady steps,
however, were **47,345-47,465 tok/s** before profile and **47,416-47,438
tok/s** after profile, i.e. **5,918-5,930 tok/s/chip**, above v017.

XProf run `2026-06-12-qwen3-jax-v024-prem5-tksp/2026_06_12_06_25_55` improved
the profiled step-time neighborhood: **598.1 ms / 49.3% MXU / 1.1% idle** vs
v022 **599.4 ms / 49.2% MXU** and v017 **600.5 ms / 49.0% MXU**. Op profile is
train-step dominated (**98.9%**), with convolution fusion **47.3%**,
custom-call **23.0%**, and loop fusion **14.3%**. HLO is identical across hosts:
SHA256 `5bc5575d3267cd029ae4be85d2a8f255eccacf62df3e4902cecf0ef3deac9e6e`
(35,713,022 bytes / 203,044 lines), with counts `dot_general=47186`,
`fusion=24737`, `copy=13971`, `all-gather=2819`, `all-reduce=153`,
`reduce-scatter=2`, `custom-call=2794`, `splash=2609`, `tokamax=177`.

Verdict: do not carry from the 20-step summary alone, but the mechanism is
likely positive. Launched v025 as the same config with 50 train steps and late
profile to dilute the one slow step and determine whether the steady-state
gain is real.

## [2026-06-12] analysis | MaxText rerun shows MFU accounting gap; native reporting normalized

Reran MaxText bs3 recipe as `alekseyv-qwen3-maxtext-v007-bs3r` and analyzed
fresh XProf/HLO. MaxText reproduced the prior recipe: profile
`2026_06_12_06_07_07`, **3611.7 ms**, **62.2% MXU**, **0.3% idle**, with
steady post-profile steps at **409.2-410.0 TFLOP/s/device** and
**6,831-6,845 tok/s/device**. HLO is byte-identical to v006
(`598fe616e4031e66c7a0709c0e0dfd04bec2566041590378c0a4fb1ea931cf84`).

Formula audit found the MFU discrepancy: MaxText reports **1472.20
TFLOPs/device-step** at bs3 seq8192 by counting attention as full non-causal
QK/AV matmul cost. The native Qwen3 trainers used triangular causal attention,
which gives **1294.08 TFLOPs/device-step**. Updated
`jax/train.py` and `torchax/train.py` to use the MaxText-style reporting
counter and label it explicitly. This is reporting-only; model execution is
unchanged. Under this convention, the old v273 full-throughput native-JAX
frontier is about **49.2% MFU** at **7,543 tok/s/chip**. A subsequent MaxText
`base-b` run under `qwen3_cc5` improved the MaxText reference to about
**45.4-45.7% MFU / 6,956-6,998 tok/s/chip**, still below v273 on throughput and
normalized MFU. The current `qwen3_cc5` bs1 stability lane remains separate;
its best recent run is v017/v022 in the **5,813-5,887 tok/s/chip** band.

## [2026-06-05] loop-iteration | v279-no-scan-sparsecore-rs-ar-rs3-ar2-mlp-up-proj-first-selective-resources-tokamax-ce-bs5 on 8B/v6e-8: refuted (56,318 tok/s / 40.4% MFU; Tokamax CE batch 5 fits but is slow)

Workload `alekseyv-q3-v279-tkce-bs5` reused the confirmed v273
RS3+AR2 selective-resources frontier, switched only from MaxText CE to
Tokamax Mosaic TPU streaming CE, and raised per-chip batch from `4` to `5`
(global batch 40). It completed cleanly with finite loss ending at **12.0496**
and `EXIT_CODE=0` on both workers. Worker0 averaged **56,318 tok/s /
7,040 tok/s/chip / 40.4% MFU** over steps 2-19; worker1 averaged **56,306
tok/s / 7,038 tok/s/chip / 40.4% MFU**. XProf run
`2026-06-05-qwen3-jax-v279-no-scan-sparsecore-rs-ar-rs3-ar2-mlp-up-proj-first-selective-resources-tokamax-ce-bs5/2026_06_05_18_23_40`
shows **5829.3 ms** step time, **64.4% MXU**, and **29.50 GiB / 31.25 GiB**
peak HBM with **1.7418 GiB** free. Device idle is only **0.1%**; top op
families are convolution fusion **46.0%**, custom-call **37.7%**, loop fusion
**9.7%**, data formatting **2.8%**, and custom fusion **1.7%**. The optimized
train-step HLO has SHA256
`f205e693bc93ac8db8e11c71199fbe2b562d62cdebe6d3c6410cf9924ebe77bb`
(19,922,224 bytes / 119,222 lines) and confirms Tokamax CE custom calls
(`PallasMosaicTpuLinearSoftmaxCrossEntropyLoss` count 13, `tokamax` count 20).
Do not carry Tokamax CE: it avoids the v278 logits OOM and beats older v245
slightly, but remains far below the v273 MaxText-CE batch-4 frontier
(**60,346 tok/s / 43.3% MFU**).

## [2026-06-05] loop-iteration | v278-no-scan-sparsecore-rs-ar-rs3-ar2-mlp-up-proj-first-selective-resources-bs5-shmem90 on 8B/v6e-8: invalid/refuted (batch-5 MaxText-CE compile OOM; logits temp 11.59 GiB)

Workload `alekseyv-q3-v278-bs5-shmem90` reused the v277 lower-HBM schedule
and changed only trainer batch size from `4` to `5` (global batch 40). Both
pods failed during `jit_train_step` compilation with `RESOURCE_EXHAUSTED`:
XLA used **33.10 GiB / 31.25 GiB** HBM and exceeded capacity by **1.86 GiB**.
Total HBM usage was at least **33.36 GiB**: **27.38 GiB** program,
**5.72 GiB** arguments, and **260 MiB** reserved. The largest temp was the
full-vocabulary logits allocation `bf16[5,8192,151936]` at **11.59 GiB**
(`fusion.4976.remat` from the forward/backward lm-head `dot_general`).
No XProf was produced. The optimized train-step HLO was emitted with SHA256
`ab88cab233d7fabd595e00652e22e66f18f26c37a849847de761e41db7e85d9a`
(17,671,203 bytes / 106,145 lines, `tokamax` count 2), plus OOM memory reports.
Close direct batch 5 on the current MaxText-CE runtime family; next work needs
a structural streaming/fused lm-head+CE path before batch scaling can be
revisited.

## [2026-06-05] loop-iteration | v277-no-scan-sparsecore-rs-ar-rs3-ar2-mlp-up-proj-first-selective-resources-shmem90 on 8B/v6e-8: refuted (60,294 tok/s / 43.2% MFU; 2.22 GiB free HBM but 4361.9 ms XProf)

Workload `alekseyv-q3-v277-shmem90` reused the confirmed v273 RS3+AR2
selective-resources frontier and lowered only
`--xla_tpu_scheduler_percent_shared_memory_limit` from `100` to `90`. It
completed cleanly with normal loss ending at **12.0459** and `EXIT_CODE=0`.
Worker0 averaged **60,294 tok/s / 7,537 tok/s/chip / 43.2% MFU** over steps
2-19; worker1 averaged **60,269 tok/s / 7,534 tok/s/chip / 43.2% MFU**. XProf
run
`2026-06-05-qwen3-jax-v277-no-scan-sparsecore-rs-ar-rs3-ar2-mlp-up-proj-first-selective-resources-shmem90/2026_06_05_17_49_23`
shows **4361.9 ms** step time, **68.1% MXU**, and **29.03 GiB / 31.25 GiB**
peak HBM with **2.2202 GiB** free. The optimized train-step HLO is
byte-identical to v273 with SHA256
`81e3c3c58b3fbfa12c0b203bdc9d0df428913fad3462ea5552e36cd07dcd5a96`
(19,915,660 bytes / 119,349 lines, `tokamax` count 2). Do not carry shmem90:
the memory margin is real but the runtime profile regresses.

## [2026-06-05] loop-iteration | v276-no-scan-sparsecore-rs-ar-rs3-ar2-mlp-up-proj-first-selective-resources-shmem95 on 8B/v6e-8: refuted/tie (60,344 tok/s / 43.3% MFU; byte-identical HLO; 4358.5 ms XProf)

Workload `alekseyv-q3-v276-shmem95` reused the confirmed v273 RS3+AR2
selective-resources frontier and lowered only
`--xla_tpu_scheduler_percent_shared_memory_limit` from `100` to `95`. It
completed cleanly with normal loss ending at **12.0459** and `EXIT_CODE=0`.
Worker0 averaged **60,344 tok/s / 7,543 tok/s/chip / 43.3% MFU** over steps
2-19; worker1 averaged **60,325 tok/s / 7,541 tok/s/chip / 43.3% MFU**. XProf
run
`2026-06-05-qwen3-jax-v276-no-scan-sparsecore-rs-ar-rs3-ar2-mlp-up-proj-first-selective-resources-shmem95/2026_06_05_17_36_25`
shows **4358.5 ms** step time, **68.2% MXU**, and **30.93 GiB / 31.25 GiB**
peak HBM with **0.3181 GiB** free. The optimized train-step HLO is
byte-identical to v273 with SHA256
`81e3c3c58b3fbfa12c0b203bdc9d0df428913fad3462ea5552e36cd07dcd5a96`
(19,915,660 bytes / 119,349 lines, `tokamax` count 2). Do not carry shmem95;
keep shared-memory limit `100`.

## [2026-06-05] loop-iteration | v275-no-scan-sparsecore-rs-ar-rs3-ar2-mlp-up-proj-first-selective-resources-vmem100864 on 8B/v6e-8: refuted (60,311 tok/s / 43.2% MFU; 4360.2 ms XProf)

Workload `alekseyv-q3-v275-vmem100864` reused the confirmed v273 RS3+AR2
selective-resources frontier and raised only scoped VMEM from `100352` to
`100864` KiB. It completed cleanly with normal loss ending at **12.0459** and
`EXIT_CODE=0`. Worker0 averaged **60,284 tok/s / 7,536 tok/s/chip / 43.2%
MFU** over steps 2-19; worker1 averaged **60,311 tok/s / 7,539 tok/s/chip /
43.2% MFU**. XProf run
`2026-06-05-qwen3-jax-v275-no-scan-sparsecore-rs-ar-rs3-ar2-mlp-up-proj-first-selective-resources-vmem100864/2026_06_05_17_22_51`
shows **4360.2 ms** step time, **68.1% MXU**, and **30.93 GiB / 31.25 GiB**
peak HBM with **0.3181 GiB** free. The optimized train-step HLO changed to
SHA256 `6e82cf40ff59806506583187b0d32c3242ae906e8aa6a6f49a2e725aa2c747a6`
(19,915,428 bytes / 119,347 lines, `tokamax` count 2), with the same main
collective/matmul/custom-call counts as v273 and six fewer `copy` text matches.
Do not carry VMEM `100864`; keep scoped VMEM `100352` on the v273 frontier.

## [2026-06-05] loop-iteration | v274-no-scan-sparsecore-rs-ar-rs3-mlp-up-proj-first-selective-resources-ar-default on 8B/v6e-8: refuted (60,334 tok/s / 43.3% MFU; 4359.4 ms XProf)

Workload `alekseyv-q3-v274-sr-rs3-ard` reused the confirmed v273 RS3 selective
resources frontier and omitted only
`--xla_tpu_sparse_core_all_reduce_latency_multiplier=2`. It completed cleanly
with normal loss ending at **12.0459** and `EXIT_CODE=0`. Worker0 averaged
**60,334 tok/s / 7,542 tok/s/chip / 43.3% MFU** over steps 2-19; worker1
averaged **60,315 tok/s / 7,539 tok/s/chip / 43.2% MFU**. XProf run
`2026-06-05-qwen3-jax-v274-no-scan-sparsecore-rs-ar-rs3-mlp-up-proj-first-selective-resources-ar-default/2026_06_05_17_07_37`
shows **4359.4 ms** step time, **68.1% MXU**, and **30.93 GiB / 31.25 GiB**
peak HBM with **0.3181 GiB** free. The optimized train-step HLO changed to
SHA256 `361b10fd3f580e2829ae2a660ad81c4cf38bc62bfbcf0da033214495b2a5c44e`
(19,915,660 bytes / 119,349 lines, `tokamax` count 2), with the same aggregate
counts as v273. Do not carry AR default; keep AR latency multiplier `2`.

## [2026-06-05] loop-iteration | v273-no-scan-sparsecore-rs-ar-rs3-ar2-mlp-up-proj-first-selective-resources-rerun on 8B/v6e-8: supported (60,346 tok/s / 43.3% MFU; confirmed lower-HBM RS3+AR2 selective frontier)

Workload `alekseyv-q3-v273-sr-rs3-r` exactly reran v272. It completed cleanly
with normal loss ending at **12.0459** and `EXIT_CODE=0`. Worker0 averaged
**60,346 tok/s / 7,543 tok/s/chip / 43.3% MFU** over steps 2-19; worker1
averaged **60,322 tok/s / 7,540 tok/s/chip / 43.3% MFU**. XProf run
`2026-06-05-qwen3-jax-v273-no-scan-sparsecore-rs-ar-rs3-ar2-mlp-up-proj-first-selective-resources-rerun/2026_06_05_16_54_18`
shows **4358.1 ms** step time, **68.2% MXU**, and **30.93 GiB / 31.25 GiB**
peak HBM with **0.3181 GiB** free. The optimized train-step HLO is byte-identical
to v272 with SHA256
`81e3c3c58b3fbfa12c0b203bdc9d0df428913fad3462ea5552e36cd07dcd5a96`
(19,915,660 bytes / 119,349 lines, `tokamax` count 2). Carry RS3+AR2 selective
resources as the confirmed safer frontier: MFU remains the same rounded **43.3%**
band as v269, but memory headroom is confirmed at +0.2285 GiB over v269.

## [2026-06-05] loop-iteration | v272-no-scan-sparsecore-rs-ar-rs3-ar2-mlp-up-proj-first-selective-resources on 8B/v6e-8: supported provisional tie (60,351 tok/s / 43.3% MFU; 4358.2 ms XProf; lower HBM)

Workload `alekseyv-q3-v272-sr-rs3-ar2` reused the v269 selective-resources
frontier and changed only RS latency multiplier from `4` to `3`. It completed
cleanly with normal loss ending at **12.0459** and `EXIT_CODE=0`. Worker0
averaged **60,318 tok/s / 7,540 tok/s/chip / 43.2% MFU** over steps 2-19;
worker1 averaged **60,351 tok/s / 7,544 tok/s/chip / 43.3% MFU**. XProf run
`2026-06-05-qwen3-jax-v272-no-scan-sparsecore-rs-ar-rs3-ar2-mlp-up-proj-first-selective-resources/2026_06_05_16_39_14`
shows **4358.2 ms** step time, **68.1% MXU**, and **30.93 GiB / 31.25 GiB**
peak HBM with **0.3181 GiB** free. The optimized train-step HLO changed to
SHA256 `81e3c3c58b3fbfa12c0b203bdc9d0df428913fad3462ea5552e36cd07dcd5a96`
(19,915,660 bytes / 119,349 lines, `tokamax` count 2). Treat v272 as a
provisional memory-headroom tie: it preserves the v269 throughput band and
improves memory headroom materially, but the speed delta is too small to carry
without an exact rerun.

## [2026-06-05] loop-iteration | v271-no-scan-sparsecore-rs-ar-rs5-ar2-mlp-up-proj-first-selective-resources on 8B/v6e-8: invalid/refuted (program-load HBM failure)

Workload `alekseyv-q3-v271-sr-rs5-ar2` reused the v269 selective-resources
frontier and changed only RS latency multiplier from `4` to `5`. It failed
before step 0 while loading `jit_train_step` with `RESOURCE_EXHAUSTED`:
XLA attempted an additional **2.12 GiB** allocation with only **995.89 MiB**
free. The container exited with `EXIT_CODE=1`; no XProf was produced. The
optimized train-step HLO changed to SHA256
`a88aac126db1ab1d3d04db6f74b572b2f51091637084b82d470246003e9a7d8a`
(19,874,947 bytes / 119,032 lines, `tokamax` count 2), with fewer
all-gathers and async starts/dones than v269 but more copies and Silu JVP
markers. Static memory report lists **29.14 GiB** total bytes used and a
**23.38 GiB** preallocated temp, but runtime loading still exhausts HBM. Close
the RS5 high-side probe under selective resources and keep RS4.

## [2026-06-05] loop-iteration | v270-no-scan-sparsecore-rs-ar-rs4-ar2-mlp-up-proj-first-selective-resources-vmem99840 on 8B/v6e-8: refuted (60,116 tok/s / 43.1% MFU; 4374.8 ms XProf)

Workload `alekseyv-q3-v270-sr-vmem99840` reused the v269 selective-resources
frontier and changed only scoped VMEM from `100352` to `99840` KiB. It
completed cleanly with normal loss ending at **12.0463** and `EXIT_CODE=0`.
Worker0 averaged **60,116 tok/s / 7,514 tok/s/chip / 43.1% MFU** over steps
2-19; worker1 averaged **60,091 tok/s / 7,511 tok/s/chip / 43.1% MFU**. XProf
run
`2026-06-05-qwen3-jax-v270-no-scan-sparsecore-rs-ar-rs4-ar2-mlp-up-proj-first-selective-resources-vmem99840/2026_06_05_16_13_28`
shows **4374.8 ms** step time, **67.9% MXU**, and **31.13 GiB / 31.25 GiB**
peak HBM with **0.117 GiB** free. The optimized train-step HLO changed to
SHA256 `ed58458efdb96ad67a71a3ea3a837fc38446f66da69d5a50c9df0ef675c87caa`
(19,903,924 bytes / 119,288 lines, `tokamax` count 2). The HBM headroom gain
over v269 is only **0.0274 GiB**, not enough to offset the throughput and
profile regression, so keep scoped VMEM `100352`.

## [2026-06-05] loop-iteration | v269-no-scan-sparsecore-rs-ar-rs4-ar2-mlp-up-proj-first-selective-resources-rerun on 8B/v6e-8: confirmed supported (60,338 tok/s / 43.3% MFU; 4359.6 ms XProf)

Workload `alekseyv-q3-v269-mlp-up-rs4-ar2-sr-r` exactly reran v268. It
completed cleanly with normal loss ending at **12.0459** and `EXIT_CODE=0`.
Worker0 averaged **60,318 tok/s / 7,540 tok/s/chip / 43.2% MFU** over steps
2-19; worker1 averaged **60,338 tok/s / 7,542 tok/s/chip / 43.3% MFU**. Best
observed post-profile step was worker0 step 15 at **60,437 tok/s**. XProf run
`2026-06-05-qwen3-jax-v269-no-scan-sparsecore-rs-ar-rs4-ar2-mlp-up-proj-first-selective-resources-rerun/2026_06_05_15_58_24`
shows **4359.6 ms** step time, **68.2% MXU**, and **31.16 GiB / 31.25 GiB**
peak HBM with **0.0896 GiB** free. The optimized train-step HLO is
byte-identical to v268 with SHA256
`a92469ed51218d7814d863f5e78b678967b81a836d9a558a1d9c966cc4a206e0`
(19,901,683 bytes / 119,264 lines, `tokamax` count 2). Carry selective
resources as part of the confirmed frontier, but treat the frontier as
memory-tight.

## [2026-06-05] loop-iteration | v268-no-scan-sparsecore-rs-ar-rs4-ar2-mlp-up-proj-first-selective-resources on 8B/v6e-8: provisional supported (60,336 tok/s / 43.3% MFU; 4359.7 ms XProf)

Workload `alekseyv-q3-v268-mlp-up-rs4-ar2-sr` reused the v264 MLP
up-proj-first + RS4+AR2 image/stack and added only
`--xla_latency_hiding_scheduler_enable_selective_resources=true`. It completed
cleanly with normal loss ending in the expected band and `EXIT_CODE=0`.
Worker0 averaged **60,315 tok/s / 7,539 tok/s/chip / 43.2% MFU** over steps
2-19; worker1 averaged **60,336 tok/s / 7,542 tok/s/chip / 43.3% MFU**. Best
observed post-profile step was worker1 step 15 at **60,450 tok/s**. XProf run
`2026-06-05-qwen3-jax-v268-no-scan-sparsecore-rs-ar-rs4-ar2-mlp-up-proj-first-selective-resources/2026_06_05_15_43_13`
shows **4359.7 ms** step time, **68.2% MXU**, and **31.16 GiB / 31.25 GiB**
peak HBM. The optimized train-step HLO changed to SHA256
`a92469ed51218d7814d863f5e78b678967b81a836d9a558a1d9c966cc4a206e0`
(19,901,683 bytes / 119,264 lines, `tokamax` count 2), with
`--xla_latency_hiding_scheduler_enable_selective_resources='true'` confirmed
in the flagfile. Treat this as a provisional new high-water mark, but require
an exact rerun before carrying it because free HBM margin is only about
**0.09 GiB**.

## [2026-06-05] loop-iteration | v267-no-scan-sparsecore-rs-ar-rs4-ar2-mlp-up-proj-first-layer-scheduler on 8B/v6e-8: invalid/refuted (compile HBM 66.54 GiB)

Workload `alekseyv-q3-v267-mlp-up-rs4-ar2-ls` reused the v264 MLP
up-proj-first + RS4+AR2 image/stack and added only
`--xla_tpu_enable_latency_hiding_layer_scheduler=true` while keeping the
SparseCore collective aggregator enabled. Both workers failed during train-step
compile before step 0 with `RESOURCE_EXHAUSTED`: **66.54 GiB / 31.25 GiB** HBM
used, exceeding capacity by **35.29 GiB**. HLO temp alone was **66.46 GiB**,
dominated by repeated **768 MiB** rematerialized `bf16[4,8192,12288]`
dot-general temps in checkpoint/remat MLP work. No XProf was produced. The
optimized train-step HLO was dumped with SHA256
`02275c00be2740bf453d7fec2fdae2dfac9181c67b3f43e86be518a16c547bfa`
(16,418,809 bytes / 98,751 lines, `tokamax` count 2), textually smaller than
v264 but with many more copies and an impossible memory schedule. Keep the
latency-hiding layer scheduler disabled on the current frontier.

## [2026-06-05] loop-iteration | v266-no-scan-sparsecore-rs-ar-rs4-mlp-up-proj-first-ar-default on 8B/v6e-8: refuted/tie (59,942 tok/s / 43.0% MFU; 4388.5 ms XProf)

Workload `alekseyv-q3-v266-mlp-up-rs4-ar1` reused the v264 MLP up-proj-first
image, kept RS latency multiplier `4`, and removed only AR latency multiplier
`2`. It completed cleanly with normal loss ending at **12.0458** and
`EXIT_CODE=0`. Worker0 averaged **59,942 tok/s / 7,493 tok/s/chip / 43.0%
MFU** over steps 2-19; worker1 averaged **59,917 tok/s / 7,490 tok/s/chip /
43.0% MFU**. Best observed post-profile step was worker1 step 15 at
**60,045 tok/s**. XProf run
`2026-06-05-qwen3-jax-v266-no-scan-sparsecore-rs-ar-rs4-mlp-up-proj-first-ar-default/2026_06_05_15_19_00`
shows **4388.5 ms** step time, **67.8% MXU**, and **30.93 GiB** peak HBM. The
optimized train-step HLO changed hash to SHA256
`524d65de4503a2c4ae5aad00ebe173c6377e0f66256348db7e2a21a18a44b2c1`
(19,531,345 bytes / 116,454 lines, `tokamax` count 2), but aggregate op counts
match v264. This does not beat v264's confirmed **59,954 tok/s / 4387.7 ms**
band, so keep AR latency multiplier `2`.

## [2026-06-05] loop-iteration | v265-no-scan-sparsecore-rs-ar-rs5-ar2-mlp-up-proj-first on 8B/v6e-8: refuted/tie (59,955 tok/s / 43.0% MFU; 4387.6 ms XProf)

Workload `alekseyv-q3-v265-mlp-up-rs5-ar2` reused the v264 MLP up-proj-first
image and changed only RS latency multiplier from `4` to `5`, keeping AR
latency multiplier `2` and the rest of the Tokamax/SparseCore stack fixed. It
completed cleanly with normal loss ending at **12.0458** and `EXIT_CODE=0`.
Worker0 averaged **59,955 tok/s / 7,494 tok/s/chip / 43.0% MFU** over steps
2-19; worker1 averaged **59,927 tok/s / 7,491 tok/s/chip / 43.0% MFU**. Best
observed post-profile step was worker0 step 15 at **60,061 tok/s**. XProf run
`2026-06-05-qwen3-jax-v265-no-scan-sparsecore-rs-ar-rs5-ar2-mlp-up-proj-first/2026_06_05_15_05_08`
shows **4387.6 ms** step time, **67.9% MXU**, and **31.0 GiB** peak HBM. The
optimized train-step HLO changed to SHA256
`f5cb5c9c74e10596bddc5a6247e2d58cb162dfc7e2147fe90615cb03b97bdee5`
(19,519,075 bytes / 116,419 lines, `tokamax` count 2), with fewer
all-gathers/copies but more async starts/dones and custom-calls than v264. This
does not beat v264's confirmed **59,954 tok/s / 4387.7 ms** band and has
slightly higher HBM, so keep RS4+AR2 as the frontier.

## [2026-06-05] loop-iteration | v264-no-scan-sparsecore-rs-ar-rs4-ar2-mlp-up-proj-first-rerun on 8B/v6e-8: confirmed supported (59,954 tok/s / 43.0% MFU; 4387.7 ms XProf)

Workload `alekseyv-q3-v264-mlp-up-rs4-ar2-r` exactly reran v263. It completed
cleanly with normal loss ending at **12.0458** and `EXIT_CODE=0`. Worker0
averaged **59,954 tok/s / 7,494 tok/s/chip / 43.0% MFU** over steps 2-19;
worker1 averaged **59,933 tok/s / 7,492 tok/s/chip / 43.0% MFU**. Best
observed post-profile step was worker1 step 15 at **60,066 tok/s**. XProf run
`2026-06-05-qwen3-jax-v264-no-scan-sparsecore-rs-ar-rs4-ar2-mlp-up-proj-first-rerun/2026_06_05_14_51_03`
shows **4387.7 ms** step time, **67.9% MXU**, and **30.94 GiB** peak HBM. The
optimized train-step HLO is byte-identical to v263 with SHA256
`09b78e1542bfb74338809203b5a8cb1e7cc4a30c311e26432affa9b25ca1d9a3`
(19,531,349 bytes / 116,454 lines, `tokamax` count 2). This confirms the
RS4+AR2 MLP up-proj-first schedule as the new native-JAX frontier, replacing
v259/v260 RS3+AR2. Carry RS latency multiplier `4` plus AR latency multiplier
`2`.

## [2026-06-05] loop-iteration | v263-no-scan-sparsecore-rs-ar-rs4-ar2-mlp-up-proj-first on 8B/v6e-8: provisional supported (59,947 tok/s / 43.0% MFU; 4387.5 ms XProf)

Workload `alekseyv-q3-v263-mlp-up-rs4-ar2` reused the v258 MLP up-proj-first
image and changed only RS latency multiplier from `3` to `4`, keeping AR
latency multiplier `2` and the rest of the Tokamax/SparseCore stack fixed. It
completed cleanly with normal loss ending at **12.0458** and `EXIT_CODE=0`.
Worker0 averaged **59,947 tok/s / 7,493 tok/s/chip / 43.0% MFU** over steps
2-19; worker1 averaged **59,926 tok/s / 7,491 tok/s/chip / 43.0% MFU**. Best
observed post-profile step was worker1 step 15 at **60,045 tok/s**. XProf run
`2026-06-05-qwen3-jax-v263-no-scan-sparsecore-rs-ar-rs4-ar2-mlp-up-proj-first/2026_06_05_14_37_45`
shows **4387.5 ms** step time, **67.8% MXU**, and **30.94 GiB** peak HBM. The
optimized train-step HLO changed to SHA256
`09b78e1542bfb74338809203b5a8cb1e7cc4a30c311e26432affa9b25ca1d9a3`
(19,531,349 bytes / 116,454 lines, `tokamax` count 2) with fewer all-gathers
(6886 vs v260's 6923), fewer async starts/dones (364 vs 372), fewer
custom-calls/fusions, and more copies. Treat RS4+AR2 on MLP up-proj-first as a
provisional new high-water mark, but require exact rerun before carrying it
because the full-window margin over v259 is only **21 tok/s**.

## [2026-06-05] loop-iteration | v262-no-scan-sparsecore-rs-ar-rs3-mlp-up-proj-first-ar-default on 8B/v6e-8: refuted/tie (59,918 tok/s / 43.0% MFU; 4391.2 ms XProf)

Workload `alekseyv-q3-v262-mlp-up-rs3-ar1` kept the confirmed v260/v261 MLP
up-proj-first + RS3 stack fixed and removed only
`--xla_tpu_sparse_core_all_reduce_latency_multiplier=2`, returning all-reduce
latency to default. It completed cleanly with normal loss ending at **12.0460**
and `EXIT_CODE=0`. Worker0 averaged **59,918 tok/s / 7,490 tok/s/chip / 43.0%
MFU** over steps 2-19; worker1 averaged **59,900 tok/s / 7,487 tok/s/chip /
42.9% MFU**. Best observed post-profile step was worker0 step 15 at
**60,024 tok/s**. XProf run
`2026-06-05-qwen3-jax-v262-no-scan-sparsecore-rs-ar-rs3-mlp-up-proj-first-ar-default/2026_06_05_14_24_01`
shows **4391.2 ms** step time, **67.8% MXU**, and **30.93 GiB** peak HBM. The
optimized train-step HLO changed to SHA256
`22d309bd3f749edd48134a95832505670527871f796e8708cd47142b45d737de`
(19,580,274 bytes / 116,730 lines, `tokamax` count 2), but operation counts
match v260. The diff is localized to async all-gather fusion scheduling/typing
around rematerialized MLP dot-general and RMSNorm/mul work. This does not beat
v259's best full-window **59,926 tok/s** or v260's **4389.7 ms** XProf step
time, so keep AR latency multiplier `2`.

## [2026-06-05] loop-iteration | v261-no-scan-sparsecore-rs-ar-rs3-ar2-mlp-up-proj-first-no-offload-tracing on 8B/v6e-8: refuted/tie (59,901 tok/s / 42.9% MFU; 4390.8 ms XProf)

Workload `alekseyv-q3-v261-mlp-up-rs3-notr` kept the confirmed v260 MLP
up-proj-first + RS3+AR2 stack fixed and changed only active SparseCore
reduce-scatter/all-reduce offload tracing from `true` to `false`. It completed
cleanly with normal loss ending at **12.0460** and `EXIT_CODE=0`. Worker0
averaged **59,901 tok/s / 7,488 tok/s/chip / 42.9% MFU** over steps 2-19;
worker1 averaged **59,875 tok/s / 7,484 tok/s/chip / 42.9% MFU**. The best
observed post-profile step was **59,980 tok/s**. XProf run
`2026-06-05-qwen3-jax-v261-no-scan-sparsecore-rs-ar-rs3-ar2-mlp-up-proj-first-no-offload-tracing/2026_06_05_14_09_16`
shows **4390.8 ms** step time, **67.8% MXU**, and **30.94 GiB** peak HBM. The
optimized train-step HLO is byte-identical to v259/v260 with SHA256
`57b4add6fe5afe1d02e0423086570ab999f02a2f6185bf935a208f7928d4b26a`
(19,580,278 bytes / 116,730 lines, `tokamax` count 2). Profile artifacts
shrank from v260's **1,883,729,274 bytes** to **295,699,891 bytes** (about
**6.4x** smaller), but neither throughput nor XProf step time improved. Do not
carry no-offload-tracing as a performance lever; use it only when smaller
diagnostic profiles are more important than detailed offload traces.

## [2026-06-05] loop-iteration | v260-no-scan-sparsecore-rs-ar-rs3-ar2-mlp-up-proj-first-rerun on 8B/v6e-8: confirmed same-band (59,903 tok/s / 43.0% MFU; 4389.7 ms XProf)

Workload `alekseyv-q3-v260-mlp-up-rs3-r` exactly reran v259. It completed
cleanly with normal loss ending at **12.0460** and `EXIT_CODE=0`. Worker0
averaged **59,828 tok/s / 7,478 tok/s/chip / 42.9% MFU** over steps 2-19;
worker1 averaged **59,903 tok/s / 7,488 tok/s/chip / 43.0% MFU**. Worker1's
best observed post-profile step was **60,030 tok/s**; worker0 had a single slow
post-profile step 15 (**58,322 tok/s**) that depressed its full-window average.
XProf run
`2026-06-05-qwen3-jax-v260-no-scan-sparsecore-rs-ar-rs3-ar2-mlp-up-proj-first-rerun/2026_06_05_13_53_40`
shows **4389.7 ms** step time, **67.8% MXU**, and **30.94 GiB** peak HBM.
The optimized train-step HLO is byte-identical to v259 with SHA256
`57b4add6fe5afe1d02e0423086570ab999f02a2f6185bf935a208f7928d4b26a`
(19,580,278 bytes / 116,730 lines, `tokamax` count 2). Carry the MLP
up-proj-first + RS3+AR2 schedule as the best current native-JAX frontier:
confirmed by HLO/profile and same-band train throughput, with v259 retaining
the best full-window summary (**59,926 tok/s / 43.0% MFU**).

## [2026-06-05] loop-iteration | v259-no-scan-sparsecore-rs-ar-rs3-ar2-mlp-up-proj-first on 8B/v6e-8: provisional supported (59,926 tok/s / 43.0% MFU; 4390.3 ms XProf)

Workload `alekseyv-q3-v259-mlp-up-rs3` reused the v258 MLP up-proj-first image
and changed only the runtime flag
`--xla_tpu_sparse_core_reduce_scatter_latency_multiplier` from `2` to `3`,
keeping AR latency multiplier `2` and the rest of the Tokamax/SparseCore stack
fixed. It completed cleanly with normal loss ending at **12.0460** and
`EXIT_CODE=0`. Worker0 averaged **59,896 tok/s / 7,487 tok/s/chip / 42.9%
MFU** over steps 2-19; worker1 averaged **59,926 tok/s / 7,491 tok/s/chip /
43.0% MFU**. The best observed post-profile step was worker1 step 15 at
**60,028 tok/s**. XProf run
`2026-06-05-qwen3-jax-v259-no-scan-sparsecore-rs-ar-rs3-ar2-mlp-up-proj-first/2026_06_05_13_40_10`
shows **4390.3 ms** step time, **67.7% MXU**, and **30.94 GiB** peak HBM.
The optimized train-step HLO is distinct from v258 and has SHA256
`57b4add6fe5afe1d02e0423086570ab999f02a2f6185bf935a208f7928d4b26a`
(19,580,278 bytes / 116,730 lines, `tokamax` count 2). Treat v259 as the new
provisional best same-band point; exact rerun before declaring it confirmed.

## [2026-06-05] loop-iteration | v258-no-scan-sparsecore-rs-ar-rs2-ar2-mlp-up-proj-first on 8B/v6e-8: supported-same-band (59,904 tok/s / 43.0% MFU; 4391.1 ms XProf)

Workload `alekseyv-q3-v258-mlp-up-first` changed only `Qwen3MLP.__call__`
source evaluation order, computing `up_proj` before `gate_proj` while keeping
the same params, `checkpoint_name(..., "mlpwi")` tags, and final
`jax.nn.silu(g) * u` math. It launched with the required frontier Tokamax env
(`USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`) and completed cleanly
with loss ending at **12.0458**. Worker0 averaged **59,904 tok/s / 7,488
tok/s/chip / 43.0% MFU** over steps 2-19; worker1 averaged **59,875 tok/s /
7,484 tok/s/chip / 42.9% MFU**. XProf run
`2026-06-05-qwen3-jax-v258-no-scan-sparsecore-rs-ar-rs2-ar2-mlp-up-proj-first/2026_06_05_13_25_35`
shows **4391.1 ms** step time, **67.6% MXU**, and **30.94 GiB** peak HBM.
The optimized train-step HLO is distinct from v256/v257 and has SHA256
`e413f9db56a30764ebd38a27dc3587ae3889261d16b56473026515e2c91f758a`
(19,544,295 bytes / 116,493 lines, `tokamax` count 2). Treat v258 as a
same-band supported scheduling probe with the best profiled step-time signal so
far, but not a decisive primary-metric breakthrough over the v230/v236
confirmed frontier band.

## [2026-06-05] loop-iteration | v257-no-scan-sparsecore-rs-ar-rs2-ar2-qknorm-after-transpose-tokamax on 8B/v6e-8: refuted (59,328 tok/s / 42.5% MFU)

Workload `alekseyv-q3-v257-qknorm-xpose` reruns the v254 QK-norm-after-transpose
image with the missing Tokamax Splash env restored:
`USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`. This is the valid test of
the QK-norm layout hypothesis; v254 itself is invalid because its optimized HLO
has `tokamax` count 0. v257 completed cleanly with valid loss ending at
**12.0458** and HLO `tokamax` count 2. Worker0 averaged **59,296 tok/s / 7,412
tok/s/chip / 42.5% MFU** over steps 2-19; worker1 averaged **59,328 tok/s /
7,416 tok/s/chip / 42.5% MFU**. XProf run
`2026-06-05-qwen3-jax-v257-no-scan-sparsecore-rs-ar-rs2-ar2-qknorm-after-transpose-tokamax/2026_06_05_13_08_02`
shows **4430.7 ms** step time, **67.1% MXU**, and **30.95 GiB** peak HBM.
The optimized train-step HLO is **19,575,537 bytes / 116,637 lines** with
SHA256 `bfd1c54cb811133970fc8a65fd4a266698062d67bac3d10db7b21b05f4bd7e07`.
The simple QK-norm-after-transpose source rewrite is a valid refutation under
the corrected runtime; keep the original QK-norm placement.

## [2026-06-05] loop-iteration | v256-no-scan-sparsecore-rs-ar-rs2-ar2-explicit-swiglu-tokamax on 8B/v6e-8: refuted (59,328 tok/s / 42.5% MFU)

Workload `alekseyv-q3-v256-expl-swiglu` reran the v255 explicit-SwiGLU image
with the missing Tokamax Splash env restored:
`USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`. It completed cleanly with
`EXIT_CODE=0`, finite loss ending at **12.0458**, and valid Tokamax HLO markers
(`tokamax` text count 2). Worker0 averaged **59,328 tok/s / 7,416 tok/s/chip /
42.5% MFU** over steps 2-19; worker1 averaged **59,301 tok/s / 7,413
tok/s/chip / 42.5% MFU**. XProf run
`2026-06-05-qwen3-jax-v256-no-scan-sparsecore-rs-ar-rs2-ar2-explicit-swiglu-tokamax/2026_06_05_12_52_05`
shows **4431.8 ms** step time, **67.1% MXU**, and **30.94 GiB** peak HBM.
The optimized train-step HLO is **19,518,444 bytes / 116,497 lines** with
SHA256 `d1ded2a43caada7f7ba021d557240abf94a0da749cc6b09f44382a8d2dfd798c`.
Explicit `g * sigmoid(g)` is a valid semantics-preserving rewrite, but it
regresses against the v230/v236 **59,749-59,750 tok/s / 42.9% MFU** frontier.
Keep `jax.nn.silu(g)` and close this MLP source-expression probe.

## [2026-06-05] loop-iteration | v255-no-scan-sparsecore-rs-ar-rs2-ar2-explicit-swiglu on 8B/v6e-8: invalid (missing Tokamax env)

Workload `alekseyv-q3-v255-expl-swiglu` tested the same explicit-SwiGLU source
rewrite but the XPK launch omitted `USE_TOKAMAX_SPLASH=1` and
`TOKAMAX_MAX_LOGIT_CONST=30.0`. The run completed with finite loss, reaching
worker0 **56,347 tok/s / 40.4% MFU** and worker1 **56,414 tok/s / 40.4% MFU**,
but the optimized HLO had `tokamax` count 0. Treat v255 as invalid and compare
only the corrected v256 rerun to the frontier.

## [2026-06-05] loop-iteration | v250-no-scan-sparsecore-rs-ar-rs2-ar2-levanter-ce-h512-v768 on 8B/v6e-8: refuted (57,792 tok/s / 41.4% MFU)

Workload `alekseyv-q3-v250-levce-v768` reused the v248 Levanter CE image and
changed only the CE block sizes from v249's `b=1024,h=512,v=1024` to
`b=1024,h=512,v=768`, one of Levanter's TPU/Pallas float32 autotune
candidates. It completed cleanly with `EXIT_CODE=0` on both workers and finite
loss ending at 12.0461. Worker0 averaged **57,792 tok/s** over steps 2-19;
worker1 averaged **57,777 tok/s**, both about **41.4% nominal MFU**. XProf run
`2026-06-05-qwen3-jax-v250-no-scan-sparsecore-rs-ar-rs2-ar2-levanter-ce-h512-v768/2026_06_05_11_28_55`
shows **4553.0 ms** step time, **66.2% MXU**, and **29.38 GiB** peak HBM. The
train-step HLO landed at
`gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-05-qwen3-jax-v250-no-scan-sparsecore-rs-ar-rs2-ar2-levanter-ce-h512-v768/hlo/xla-dump/module_0267.jit_train_step.cl_854318611.after_optimizations.txt`
with hash `430ec9f326f5ee6956871f7428c591fcbbf5b0cba2acf1334a52a828dd0273fa`
(19,562,599 bytes). `v768` reduces the forward CE custom-call to **582.7 ms /
1.1%**, but the generated fusion surface gets worse: convolution fusion rises
to **54.9%** and `select_add_fusion.2` is **2,357.1 ms / 4.3%**. Close
Levanter Pallas CE block-size tuning; v249 remains the best Levanter setting
but still trails the MaxText-CE frontier.

## [2026-06-05] loop-iteration | v249-no-scan-sparsecore-rs-ar-rs2-ar2-levanter-ce-default-blocks on 8B/v6e-8: refuted (58,427 tok/s / 41.9% MFU)

Workload `alekseyv-q3-v249-levce-def` reused the v248 Levanter CE image and
changed only the CE block sizes from `b=1024,h=256,v=512` to Levanter's
fallback/default `b=1024,h=512,v=1024` for the Qwen local CE shape
`B=32768,H=4096,V=151936`. It completed cleanly with `EXIT_CODE=0` on both
workers and finite loss ending at 12.0461. Worker0 averaged **58,427 tok/s**
over steps 2-19; worker1 averaged **58,401 tok/s**, both about **41.9% nominal
MFU**. XProf run
`2026-06-05-qwen3-jax-v249-no-scan-sparsecore-rs-ar-rs2-ar2-levanter-ce-default-blocks/2026_06_05_11_15_15`
shows **4499.4 ms** step time, **67.2% MXU**, and **29.38 GiB** peak HBM.
Levanter CE forward is **775.3 ms / 1.4%**. This recovers about 2.34k tok/s
over v248 and proves the smaller v248 block shape was poor, but it remains
below the v230/v236 MaxText CE frontier (**59,749-59,750 tok/s / 42.9% MFU**).
The intended GCS HLO dump did not land because `XLA_FLAGS` used a literal
`${RUN_DIR}` Kubernetes env value; use a concrete dump path in future launches.

## [2026-06-05] loop-iteration | v248-no-scan-sparsecore-rs-ar-rs2-ar2-levanter-ce on 8B/v6e-8: refuted (56,088 tok/s / 40.2% MFU)

Workload `alekseyv-q3-v248-levanterce` kept the material v230 RS2+AR2
SparseCore frontier fixed except replacing MaxText CE with a vendored Levanter
Pallas TPU fused CE path
(`--use_levanter_ce=True --use_maxtext_ce=False`,
`b=1024,h=256,v=512`). It completed cleanly with `EXIT_CODE=0` on both workers
and finite loss ending at 12.0459. Worker0 averaged **56,088 tok/s** over
steps 2-19; worker1 averaged **56,053 tok/s**, both about **40.2% nominal
MFU**. The optimized train-step HLO changed to
`db06c168df683b818ae806fc4e7a34f0b695373398a18634219d10c243847ab0`
(19,562,405 bytes, 116,384 lines) and includes the Levanter
`linear_softmax_cross_entropy_loss_{fwd,bwd}_pallas_mosaic_tpu` path. XProf run
`2026-06-05-qwen3-jax-v248-no-scan-sparsecore-rs-ar-rs2-ar2-levanter-ce/2026_06_05_11_00_51`
shows **4702.9 ms** step time, **63.5% MXU**, and **29.37 GiB** peak HBM.
Levanter CE forward is only **1,154.1 ms / 1.0%**, but the generated fusion
surface includes `select_add_fusion.2` at **6,913.5 ms / 6.1%** and high bytes
accessed. This improves the Tokamax CE branch slightly but remains well below
the v230/v236 MaxText CE frontier (**59,749-59,750 tok/s / 42.9% MFU**), so do
not carry Levanter CE at these block sizes.

## [2026-06-05] loop-iteration | v247-no-scan-sparsecore-rs-ar-rs2-ar2-no-offload-tracing on 8B/v6e-8: refuted/tie (59,748 tok/s / 42.9% MFU)

Workload `alekseyv-q3-v247-rs2ar2-notrace` kept the material v230 RS2+AR2
SparseCore frontier fixed except changing active reduce-scatter/all-reduce
SparseCore offload tracing to `false`. It completed cleanly with `EXIT_CODE=0`
on both workers and normal loss ending at 12.0459. The optimized train-step HLO
was byte-identical to v230/v242/v246
(`8906838f769ee38abcec2c0498a68eace061c517c362b19e7efa7e4cc5854278`,
19,925,715 bytes). Worker0 averaged **59,748 tok/s** over steps 2-19,
**59,795 tok/s** over steps 15-19, and **59,763 tok/s** excluding profiled
steps; worker1 averaged **59,721 tok/s**, **59,797 tok/s**, and **59,759
tok/s** respectively. XProf run
`2026-06-05-qwen3-jax-v247-no-scan-sparsecore-rs-ar-rs2-ar2-no-offload-tracing/2026_06_05_10_36_52`
shows **4402.2 ms** step time, **67.5% MXU**, **30.93 GiB** peak HBM, and the
same top op split as v246: convolution fusion **53.7%**, custom-call **27.8%**,
loop fusion **10.1%**. Offload tracing remains a diagnostics-size knob, not a
frontier lever; keep the v230/v236 frontier unchanged.

## [2026-06-05] loop-iteration | v246-no-scan-sparsecore-rs-ar-rs2-ar2-no-mem-pressure-tracking on 8B/v6e-8: refuted/tie (59,732 tok/s / 42.9% MFU)

Workload `alekseyv-q3-v246-nomemtrack` kept the material v230 RS2+AR2
SparseCore frontier fixed except changing
`--xla_tpu_enable_scheduler_memory_pressure_tracking=true` to `false`. It
completed cleanly with `EXIT_CODE=0` on both workers and normal loss ending at
12.0459. The optimized train-step HLO was byte-identical to v230/v242
(`8906838f769ee38abcec2c0498a68eace061c517c362b19e7efa7e4cc5854278`,
19,925,715 bytes). Worker0 averaged **59,732 tok/s** over steps 2-19,
**59,767 tok/s** over steps 15-19, and **59,744 tok/s** excluding profiled
steps; worker1 averaged **59,712 tok/s**, **59,767 tok/s**, and **59,744
tok/s** respectively. XProf run
`2026-06-05-qwen3-jax-v246-no-scan-sparsecore-rs-ar-rs2-ar2-no-mem-pressure-tracking/2026_06_05_10_24_02`
shows **4401.5 ms** step time, **67.5% MXU**, **30.93 GiB** peak HBM, and the
same top op split as the frontier band: convolution fusion **53.7%**,
custom-call **27.8%**, loop fusion **10.1%**. This does not beat v230/v236's
**59,749-59,750 tok/s** primary full-window frontier, so keep TPU scheduler
memory-pressure tracking enabled and treat scalar scheduler-memory flags as
closed on the current HLO.

## [2026-06-05] loop-iteration | v245-no-scan-sparsecore-rs-ar-rs2-ar2-tokamax-ce-bs5 on 8B/v6e-8: refuted (55,885 tok/s / 40.0% MFU)

Workload `alekseyv-q3-v245-tokamaxce-bs5` kept v244's post-SparseCore
Tokamax CE stack fixed except increasing per-chip batch from 4 to 5/global
batch from 32 to 40. It completed cleanly with `EXIT_CODE=0` on both workers
and normal loss ending at 12.0496. Worker0 averaged **55,885 tok/s** over
steps 2-19, **55,988 tok/s** over steps 15-19, and **55,897 tok/s** excluding
profiled steps; worker1 averaged **55,866 tok/s**, **55,987 tok/s**, and
**55,897 tok/s** respectively. The optimized train-step HLO hash was
`0ff06d31846d093b49a0690ea11c0fc37003bc030a39a51ec976e72796f26bb3`
(19,930,103 bytes). XProf split hosts into two one-host runs:
**5876.4 ms / 64.2% MXU** and **5888.2 ms / 63.4% MXU**, with **29.16 GiB**
peak HBM. Batch 5 fits and improves v244 by about 392 tok/s, but remains far
below the MaxText CE v230/v236 frontier. Close Tokamax CE batch scaling unless
the `linear_softmax_cross_entropy_loss_bwd_pallas_mosaic_tpu` cost is reduced.

## [2026-06-05] loop-iteration | v244-no-scan-sparsecore-rs-ar-rs2-ar2-tokamax-ce on 8B/v6e-8: refuted (55,493 tok/s / 39.8% MFU)

Workload `alekseyv-q3-v244-tokamaxce-sc` kept the material v230 RS2+AR2
SparseCore frontier fixed except replacing MaxText CE with Tokamax CE
(`--use_tokamax_ce=True --tokamax_ce_impl=mosaic_tpu --use_maxtext_ce=False`).
It completed cleanly with `EXIT_CODE=0` on both workers and normal loss ending
at 12.0524. Worker0 averaged **55,473 tok/s** over steps 2-19, **55,491
tok/s** over steps 15-19, and **55,507 tok/s** excluding profiled steps;
worker1 averaged **55,493 tok/s**, **55,491 tok/s**, and **55,507 tok/s**
respectively. The optimized train-step HLO changed to
`a76111c8e5f40ce1e82cc1f8ce28f0656e7c9f41f1a446c10b2f06edeed81d33`
(19,909,919 bytes). XProf run
`2026-06-05-qwen3-jax-v244-no-scan-sparsecore-rs-ar-rs2-ar2-tokamax-ce/2026_06_05_09_53_42`
shows **4745.4 ms** step time, **63.3% MXU**, and **27.02 GiB** peak HBM.
Tokamax CE opens about 3.9 GiB versus v230's 30.93 GiB, and SparseCore improves
old v201 Tokamax CE by about 1.2k tok/s, but the CE backward custom-call still
takes **10,181.2 ms / 8.9%** of profiled train-step time. Do not carry Tokamax
CE unless a later experiment specifically tunes/replaces that kernel cost.

## [2026-06-05] loop-iteration | v243-no-scan-sparsecore-rs-ar-rs2-ar2-shmem95 on 8B/v6e-8: refuted (59,342 tok/s / 42.6% MFU)

Workload `alekseyv-q3-v243-shmem95` kept the material v230 RS2+AR2 SparseCore
frontier fixed except lowering `--xla_tpu_scheduler_percent_shared_memory_limit`
from `100` to `95`. It completed cleanly with `EXIT_CODE=0` on both workers and
normal loss ending at 12.0459. The optimized train-step HLO was byte-identical
to v230/v242
(`8906838f769ee38abcec2c0498a68eace061c517c362b19e7efa7e4cc5854278`,
19,925,715 bytes), and peak HBM stayed at **30.93 GiB**. Worker0 averaged
**59,317 tok/s** over steps 2-19, **59,322 tok/s** over steps 15-19, and
**59,363 tok/s** excluding profiled steps; worker1 averaged **59,342 tok/s**,
**59,322 tok/s**, and **59,361 tok/s** respectively. XProf run
`2026-06-05-qwen3-jax-v243-no-scan-sparsecore-rs-ar-rs2-ar2-shmem95/2026_06_05_09_37_52`
shows **4432.6 ms** step time and **66.9% MXU**, well below v230's
**4400.7 ms / 67.5% MXU**. Close shmem95 as a post-SparseCore lever.

## [2026-06-05] loop-iteration | v242-no-scan-sparsecore-rs-ar-rs2-ar2-rerun4 on 8B/v6e-8: refuted/tie (59,740 tok/s / 42.9% MFU)

Workload `alekseyv-q3-v242-rerun4` kept the material v230 RS2+AR2 SparseCore
frontier fixed except changing `--xla_latency_hiding_scheduler_rerun=3` to
`4`. It completed cleanly with `EXIT_CODE=0` on both workers and normal loss
ending at 12.0459. The optimized train-step HLO was byte-identical to v230
(`8906838f769ee38abcec2c0498a68eace061c517c362b19e7efa7e4cc5854278`,
19,925,715 bytes). Worker0 averaged **59,740 tok/s** over steps 2-19,
**59,781 tok/s** over steps 15-19, and **59,750 tok/s** excluding profiled
steps; worker1 averaged **59,703 tok/s**, **59,780 tok/s**, and **59,752 tok/s**
respectively. XProf split the two hosts into adjacent one-host runs and showed
**4400.4 ms** step time, **66.9-67.9% MXU**, and **30.93 GiB** peak HBM. This
ties the profile band but does not materially beat v230/v236's
**59,749-59,750 tok/s** primary full-window frontier. Keep rerun `3`; treat
rerun count as closed unless it interacts with a later structural change.

## [2026-06-05] loop-iteration | v241-no-scan-sparsecore-rs-ar-rs2-ar2-vmem100864 on 8B/v6e-8: refuted (59,702 tok/s / 42.8% MFU)

Workload `alekseyv-q3-v241-rs2ar2-vmem100864` moved the high-side VMEM retune
from the RS3+AR2 same-band alternative back to the material v230 RS2+AR2
frontier: it kept RS latency multiplier `2`, AR latency multiplier `2`, and
changed only `--xla_tpu_scoped_vmem_limit_kib=100352` to `100864`. It
completed cleanly with `EXIT_CODE=0` on both workers and normal loss ending at
12.0459, but selected a distinct optimized train-step HLO
(`1932dd46ea5b84d5be8f0a627d6ace6b056438114bd1b21586750fb9dddd65e3`,
19,925,483 bytes). Worker0 averaged **59,679 tok/s** over steps 2-19,
**59,744 tok/s** over steps 15-19, and **59,715 tok/s** excluding profiled
steps; worker1 averaged **59,702 tok/s**, **59,745 tok/s**, and **59,714 tok/s**
respectively. XProf split hosts into timestamped runs; the representative
profile shows **4404.0 ms** step time, **66.9-68.0% MXU**, and **30.93 GiB**
peak HBM. This is below v230/v236's **59,749-59,750 tok/s / ~4401 ms** band.
Close immediate post-SparseCore VMEM retuning for both RS2+AR2 and RS3+AR2 and
keep scoped VMEM `100352`.

## [2026-06-05] loop-iteration | v240-no-scan-sparsecore-rs-ar-rs3-ar2-vmem100864 on 8B/v6e-8: refuted/near-tie (59,714 tok/s / 42.8% MFU)

Workload `alekseyv-q3-v240-vmem100864` kept the v236 RS3+AR2 SparseCore stack
fixed except raising `--xla_tpu_scoped_vmem_limit_kib` from `100352` to
`100864`. The first submission failed immediately from an invalid xpk command
wrapper (`((set ...))`), was deleted, and the same workload/run directory was
resubmitted with the command body unwrapped. The real run completed cleanly
with `EXIT_CODE=0` on both workers and normal loss ending at 12.0459. Worker0
averaged **59,689 tok/s** over steps 2-19, **59,758 tok/s** over steps 15-19,
and **59,728 tok/s** excluding profiled steps; worker1 averaged **59,714
tok/s**, **59,759 tok/s**, and **59,730 tok/s** respectively. XProf run
`2026-06-05-qwen3-jax-v240-no-scan-sparsecore-rs-ar-rs3-ar2-vmem100864/2026_06_05_09_01_58`
shows **4404.1 ms** step time, **67.4% MXU**, and **30.93 GiB** peak memory.
The optimized train-step HLO changed
(`774cf325442f03b866bd63d2955e5f2709bfa292642d319b54e8a2887066121b`,
19,909,379 bytes), but it did not beat v230/v236's **59,749-59,750 tok/s /
~4401 ms** band. Together with v239, close the immediate post-SparseCore
RS3+AR2 VMEM bracket and keep scoped VMEM `100352`.

## [2026-06-05] loop-iteration | v239-no-scan-sparsecore-rs-ar-rs3-ar2-vmem98304 on 8B/v6e-8: refuted (59,507 tok/s / 42.7% MFU)

Workload `alekseyv-q3-v239-vmem98304` kept the v236 RS3+AR2 SparseCore stack
fixed except lowering `--xla_tpu_scoped_vmem_limit_kib` from `100352` to
`98304`. It completed cleanly with `EXIT_CODE=0` on both workers and normal
loss ending at 12.0462, but selected a distinct optimized train-step HLO
(`6ca611a629704d2fbc70da3a9b34244b60a11c19d4418dade3ee3d86474734ef`,
19,838,938 bytes). Worker0 averaged **59,440 tok/s** over steps 2-19,
**59,237 tok/s** over steps 15-19, and **59,437 tok/s** excluding profiled
steps; worker1 averaged **59,507 tok/s**, **59,577 tok/s**, and **59,553 tok/s**
respectively. XProf split the two hosts into timestamped one-host runs; the
representative profile shows **4418.9 ms** step time, **66.7-67.7% MXU**, and
**30.89 GiB** peak HBM. The lower VMEM point reduces HLO size but slows the
schedule relative to v230/v236's **59,749-59,750 tok/s / ~4401 ms** band. Keep
scoped VMEM `100352` on the current frontier.

## [2026-06-05] loop-iteration | v238-no-scan-sparsecore-rs-ar-rs3-ar2-no-offload-tracing on 8B/v6e-8: refuted/tie (59,749 tok/s / 42.9% MFU)

Workload `alekseyv-q3-v238-notrace` kept the confirmed RS3+AR2 schedule fixed
but disabled active SparseCore reduce-scatter and all-reduce offload tracing.
It completed cleanly with `EXIT_CODE=0` and normal loss ending at 12.0459, and
emitted byte-identical optimized train-step HLO to v235/v236
(`d4d46797cc4508667e0aad8b0a75bf6485f046806db6ced027a159f89ee80b5c`,
19,909,611 bytes). Worker0 averaged **59,749 tok/s** over steps 2-19,
**59,796 tok/s** over steps 15-19, and **59,766 tok/s** excluding profiled
steps; worker1 averaged **59,730 tok/s**, **59,793 tok/s**, and **59,766 tok/s**
respectively. XProf run
`2026-06-05-qwen3-jax-v238-no-scan-sparsecore-rs-ar-rs3-ar2-no-offload-tracing/2026_06_05_08_34_43`
shows **4401.5 ms** step time, **67.5% MXU**, and **30.93 GiB** peak memory.
The tracing-off profile payload is much smaller, but runtime is only a
same-HLO tie; keep tracing choice as a diagnostics-size knob, not a performance
frontier lever.

## [2026-06-05] loop-iteration | v237-no-scan-sparsecore-rs-ar-rs4-ar2 on 8B/v6e-8: refuted (59,725 tok/s / 42.8% MFU)

Workload `alekseyv-q3-v237-rs4-ar2` kept the v236 RS3+AR2 stack fixed except
raising only `--xla_tpu_sparse_core_reduce_scatter_latency_multiplier` from
`3` to `4`. It completed cleanly with `EXIT_CODE=0` and normal loss ending at
12.0459, but selected a different optimized train-step HLO
(`6277ba8867f83440a8606ec916b3e2ab108955756ee053ee29d3e03d80fbba17`,
19,895,634 bytes). Worker0 averaged **59,725 tok/s** over steps 2-19,
**59,773 tok/s** over steps 15-19, and **59,741 tok/s** excluding profiled
steps; worker1 averaged **59,704 tok/s**, **59,776 tok/s**, and **59,745 tok/s**
respectively. XProf run
`2026-06-05-qwen3-jax-v237-no-scan-sparsecore-rs-ar-rs4-ar2/2026_06_05_08_22_46`
shows **4404.0 ms** step time, **67.5% MXU**, and **31.16 GiB** peak memory.
The high-side RS latency bracket is closed: RS4+AR2 raises memory and regresses
below the v230/v236 band.

## [2026-06-05] loop-iteration | v236-no-scan-sparsecore-rs-ar-rs3-ar2-rerun on 8B/v6e-8: supported/tie (59,750 tok/s / 42.9% MFU)

Workload `alekseyv-q3-v236-rs3-ar2-r` exactly reran v235's RS3+AR2
SparseCore latency-model setting. It completed cleanly with `EXIT_CODE=0` and
normal loss ending at 12.0459, and reproduced v235's optimized train-step HLO
byte-for-byte (`d4d46797cc4508667e0aad8b0a75bf6485f046806db6ced027a159f89ee80b5c`,
19,909,611 bytes). Worker0 averaged **59,750 tok/s** over steps 2-19,
**59,795 tok/s** over steps 15-19, and **59,765 tok/s** excluding profiled
steps; worker1 averaged **59,624 tok/s**, **59,456 tok/s**, and **59,650 tok/s**
respectively after runtime dips at step 12 and step 16. XProf run
`2026-06-05-qwen3-jax-v236-no-scan-sparsecore-rs-ar-rs3-ar2-rerun/2026_06_05_08_10_30`
shows **4401.1 ms** step time, **67.0% MXU**, and **30.93 GiB** peak memory.
This confirms RS3+AR2 as a same-band alternative schedule, but the measured
margin over v230's **59,749 tok/s / 4400.7 ms / 67.5% MXU** is only one tok/s
with slightly lower MXU. Treat the best MFU as unchanged at about **42.9%**.

## [2026-06-05] loop-iteration | v235-no-scan-sparsecore-rs-ar-rs3-ar2 on 8B/v6e-8: near-tie, rerun (59,747 tok/s / 42.9% MFU)

Workload `alekseyv-q3-v235-rs3-ar2` kept the v230 RS/AR SparseCore split and
AR latency multiplier `2`, but raised only the reduce-scatter SparseCore
latency multiplier from `2` to `3`. It completed cleanly with `EXIT_CODE=0`
and normal loss ending at 12.0459, selecting optimized train-step HLO
`d4d46797cc4508667e0aad8b0a75bf6485f046806db6ced027a159f89ee80b5c`
(19,909,611 bytes). Worker0 averaged **59,724 tok/s** over steps 2-19,
**59,789 tok/s** over steps 15-19, and **59,763 tok/s** excluding profiled
steps; worker1 averaged **59,747 tok/s**, **59,788 tok/s**, and **59,762 tok/s**
respectively. XProf run
`2026-06-05-qwen3-jax-v235-no-scan-sparsecore-rs-ar-rs3-ar2/2026_06_05_07_58_34`
shows **4400.9 ms** step time, **67.4% MXU**, and **30.93 GiB** peak memory.
This is just below v230's primary full-window average (**59,749 tok/s**) but
matches or slightly exceeds its late/no-trace windows, so treat RS3+AR2 as a
near-tie and run an exact rerun before closing it.

## [2026-06-05] loop-iteration | v234-no-scan-sparsecore-rs-ar-rs1-ar2 on 8B/v6e-8: refuted/tie (59,643 tok/s / 42.8% MFU)

Workload `alekseyv-q3-v234-rs1-ar2` kept the v230 RS/AR SparseCore split and
AR latency multiplier `2`, but lowered only the reduce-scatter SparseCore
latency multiplier from `2` to `1`. It completed cleanly with `EXIT_CODE=0`
and normal loss ending at 12.0459, but selected a distinct optimized train-step
HLO (`02d404f309bfe92d40d098b49274b39bfd5e41e42370f1ea5a9bf8110f50258b`,
19,956,545 bytes). Worker0 averaged **59,643 tok/s** over steps 2-19,
**59,683 tok/s** over steps 15-19, and **59,656 tok/s** excluding profiled
steps; worker1 averaged **59,618 tok/s**, **59,685 tok/s**, and **59,657 tok/s**
respectively. XProf run
`2026-06-05-qwen3-jax-v234-no-scan-sparsecore-rs-ar-rs1-ar2/2026_06_05_07_47_40`
shows **4408.3 ms** step time, **67.3% MXU**, and **30.93 GiB** peak memory.
This is close but below v230's **59,749 tok/s / 4400.7 ms / 67.5% MXU**
frontier. Keep RS latency multiplier `2` with AR latency multiplier `2`.

## [2026-06-05] loop-iteration | v233-no-scan-sparsecore-rs-ar-rs2-ar2-max-async-ag2 on 8B/v6e-8: refuted (58,311 tok/s / 41.8% MFU)

Workload `alekseyv-q3-v233-rs2-ar2-ag2` kept the confirmed v230 RS2+AR2
frontier fixed and added only `--xla_max_concurrent_async_all_gathers=2`. It
completed cleanly with `EXIT_CODE=0` and normal loss ending at 12.0459, but the
compiler selected a different optimized train-step HLO
(`94c2737b7e9dfa48715821d41919490b32113f315107bdc924d8dcbf714e01ac`,
17,784,311 bytes). Worker0 averaged **58,282 tok/s** over steps 2-19,
**58,334 tok/s** over steps 15-19, and **58,323 tok/s** excluding profiled
steps; worker1 averaged **58,311 tok/s**, **58,335 tok/s**, and **58,322 tok/s**
respectively. XProf run
`2026-06-05-qwen3-jax-v233-no-scan-sparsecore-rs-ar-rs2-ar2-max-async-ag2/2026_06_05_07_36_21`
shows **4508.1 ms** step time, **65.7% MXU**, **30.90 GiB** peak memory, and
direct `all-gather` exposed at **5,067.2 ms / 4.7%**. This confirms that async
all-gather concurrency caps are not the missing lever on the v230 frontier.
Do not carry `--xla_max_concurrent_async_all_gathers`.

## [2026-06-05] loop-iteration | v232-no-scan-sparsecore-rs-ar-rs2-ar2-max-async-ag1 on 8B/v6e-8: refuted/tie (59,730 tok/s / 42.9% MFU)

Workload `alekseyv-q3-v232-rs2-ar2-ag1` kept the confirmed v230 RS2+AR2
frontier fixed and added only `--xla_max_concurrent_async_all_gathers=1`. It
completed cleanly with `EXIT_CODE=0`, normal loss ending at 12.0459, and
byte-identical optimized train-step HLO to v229/v230
(`8906838f769ee38abcec2c0498a68eace061c517c362b19e7efa7e4cc5854278`,
19,925,715 bytes). Worker0 averaged **59,730 tok/s** over steps 2-19,
**59,766 tok/s** over steps 15-19, and **59,745 tok/s** excluding profiled
steps; worker1 averaged **59,707 tok/s**, **59,766 tok/s**, and **59,745 tok/s**
respectively. XProf run
`2026-06-05-qwen3-jax-v232-no-scan-sparsecore-rs-ar-rs2-ar2-max-async-ag1/2026_06_05_07_22_59`
shows **4402.4 ms** step time, **67.5% MXU**, and **30.93 GiB** peak memory.
The all-gather concurrency cap did not improve the same-HLO profile or
throughput, so do not carry it; keep v230's RS2+AR2 frontier.

## [2026-06-05] loop-iteration | v231-no-scan-sparsecore-rs-ar-rs-latency2-ar-latency3 on 8B/v6e-8: refuted (57,586 tok/s / 41.3% MFU)

Workload `alekseyv-q3-v231-rs2-ar3` kept confirmed RS latency multiplier `2`
but raised the all-reduce SparseCore latency multiplier from `2` to `3`. It
completed cleanly with `EXIT_CODE=0` and normal loss ending at 12.0459, but
selected a very different, much smaller optimized train-step HLO
(`3c4126a41ca2ac948d1e2ae3aced9ba66b47fae10d95381d1d2a8d2d99f20743`,
16,297,820 bytes) and regressed hard. Worker0 averaged **57,586 tok/s** over
steps 2-19, **57,625 tok/s** over steps 15-19, and **57,598 tok/s** excluding
profiled steps; worker1 averaged **57,563 tok/s**, **57,625 tok/s**, and
**57,598 tok/s** respectively. XProf run
`2026-06-05-qwen3-jax-v231-no-scan-sparsecore-rs-ar-rs-latency2-ar-latency3/2026_06_05_07_11_47`
shows **4565.7 ms** step time, **66.0% MXU**, **30.15 GiB** peak memory,
`all-gather` exposed at **3,184.9 ms / 2.9%**, and `async-done` rising to
**2,821.8 ms / 2.6%**. Do not carry AR latency `3`; keep v230's RS2+AR2
frontier.

## [2026-06-05] loop-iteration | v230-no-scan-sparsecore-rs-ar-rs-latency2-ar-latency2-rerun on 8B/v6e-8: supported (59,749 tok/s / 42.9% MFU)

Workload `alekseyv-q3-v230-rs2-ar2-r` exactly reran v229. It completed cleanly
with `EXIT_CODE=0`, normal loss ending at 12.0459, profile capture, and
byte-identical optimized train-step HLO to v229
(`8906838f769ee38abcec2c0498a68eace061c517c362b19e7efa7e4cc5854278`,
19,925,715 bytes). Worker0 averaged **59,749 tok/s** over steps 2-19,
**59,778 tok/s** over steps 15-19, and **59,762 tok/s** excluding profiled
steps; worker1 averaged **59,725 tok/s**, **59,779 tok/s**, and **59,764 tok/s**
respectively. XProf run
`2026-06-05-qwen3-jax-v230-no-scan-sparsecore-rs-ar-rs-latency2-ar-latency2-rerun/2026_06_05_06_59_59`
shows **4400.7 ms** step time, **67.5% MXU**, **30.93 GiB** peak memory, and
`async-done` **626.4 ms / 0.6%**. Carry RS2+AR2 latency-model flags as the
confirmed native-JAX frontier.

## [2026-06-05] loop-iteration | v229-no-scan-sparsecore-rs-ar-rs-latency2-ar-latency2 on 8B/v6e-8: provisional supported (59,738 tok/s / 42.9% MFU)

Workload `alekseyv-q3-v229-rs2-ar2` kept confirmed RS latency multiplier `2`
and added only `--xla_tpu_sparse_core_all_reduce_latency_multiplier=2`. It
completed cleanly with `EXIT_CODE=0`, normal loss ending at 12.0459, profile
capture, and a new optimized train-step HLO
(`8906838f769ee38abcec2c0498a68eace061c517c362b19e7efa7e4cc5854278`,
19,925,715 bytes). Worker0 averaged **59,738 tok/s** over steps 2-19,
**59,780 tok/s** over steps 15-19, and **59,752 tok/s** excluding profiled
steps; worker1 averaged **59,721 tok/s**, **59,782 tok/s**, and **59,754 tok/s**
respectively. XProf run
`2026-06-05-qwen3-jax-v229-no-scan-sparsecore-rs-ar-rs-latency2-ar-latency2/2026_06_05_06_49_16`
shows **4402.4 ms** step time, **67.5% MXU**, **30.93 GiB** peak memory, and
`async-done` **632.8 ms / 0.6%**. This is the best full-window result observed
so far and slightly exceeds v225's 59,732 tok/s high-water mark, but exact
rerun is required because the HLO changed.

## [2026-06-05] loop-iteration | v228-no-scan-sparsecore-rs-ar-rs-latency2-rerun on 8B/v6e-8: supported (59,681 tok/s / 42.9% MFU)

Workload `alekseyv-qwen3-v228-sc-rsar-rslat2-r` exactly reran v227. It
completed cleanly with `EXIT_CODE=0`, normal loss ending at 12.0459, profile
capture on both hosts, and byte-identical optimized train-step HLO to v227
(`247229cca7ccc50e7d695af1efe8f798dbc119cabe31b220617e3c7f7d3f2dd1`,
19,921,350 bytes). Worker0 averaged **59,681 tok/s** over steps 2-19,
**59,712 tok/s** over steps 15-19, and **59,693 tok/s** excluding profiled
steps; worker1 averaged **59,654 tok/s**, **59,719 tok/s**, and **59,697 tok/s**
respectively. XProf runs
`2026-06-05-qwen3-jax-v228-no-scan-sparsecore-rs-ar-rs-latency2-rerun/2026_06_05_06_37_35`
and `/2026_06_05_06_37_36` show **4405.3 ms** step time, **66.9-67.9% MXU**,
and **30.93 GiB** peak memory. Carry
`--xla_tpu_sparse_core_reduce_scatter_latency_multiplier=2` as the confirmed
frontier setting; best observed single run remains v225 at **59,732 tok/s**,
but the strongest confirmed repeatable stack is v227/v228 latency2.

## [2026-06-05] loop-iteration | v227-no-scan-sparsecore-rs-ar-rs-latency2 on 8B/v6e-8: provisional supported (59,661 tok/s / 42.9% MFU)

Workload `alekseyv-qwen3-v227-sc-rsar-rslat2` kept the confirmed v220/v221
collective split fixed and used
`--xla_tpu_sparse_core_reduce_scatter_latency_multiplier=2` instead of v225/v226's
multiplier `3`. It completed cleanly with `EXIT_CODE=0`, normal loss ending at
12.0459, profile capture, and a new optimized train-step HLO
(`247229cca7ccc50e7d695af1efe8f798dbc119cabe31b220617e3c7f7d3f2dd1`,
19,921,350 bytes). Worker0 averaged **59,640 tok/s** over steps 2-19,
**59,706 tok/s** over steps 15-19, and **59,675 tok/s** excluding profiled
steps; worker1 averaged **59,661 tok/s**, **59,704 tok/s**, and **59,673 tok/s**
respectively. XProf run
`2026-06-05-qwen3-jax-v227-no-scan-sparsecore-rs-ar-rs-latency2/2026_06_05_06_27_10`
shows **4406.7 ms** step time, **67.4% MXU**, **30.93 GiB** peak memory, and
`async-done` **722.5 ms / 0.7%**. This is a stronger and more stable result
than the v226 latency3 rerun and nearly matches v225's high-water mark; exact
rerun is required before carrying latency2 as confirmed.

## [2026-06-05] loop-iteration | v226-no-scan-sparsecore-rs-ar-rs-latency3-rerun on 8B/v6e-8: supported, weak confirmation (59,519 tok/s / 42.7% MFU; best observed remains v225 59,732 / 42.9%)

Workload `alekseyv-qwen3-v226-sc-rsar-rslat3-r` exactly reran v225. It
completed cleanly with `EXIT_CODE=0`, normal loss ending at 12.0459, profile
capture, and byte-identical optimized train-step HLO to v225
(`50dae614cf9c7f4cb48f50ed3bee40310a972e141d58476da21b951462d66b88`,
19,909,611 bytes). Worker0 averaged **59,519 tok/s** over steps 2-19,
**59,578 tok/s** over steps 15-19, and **59,544 tok/s** excluding profiled
steps; worker1 averaged **59,495 tok/s**, **59,579 tok/s**, and **59,542 tok/s**
respectively. XProf run
`2026-06-05-qwen3-jax-v226-no-scan-sparsecore-rs-ar-rs-latency3-rerun/2026_06_05_06_15_45`
shows **4422.3 ms** step time, **67.3% MXU**, **30.93 GiB** peak memory, and
`async-done` **577.3 ms / 0.5%**. This confirms the HLO/profile mechanism and a
small improvement over v220/v221, but it does not reproduce v225's 59,732 tok/s
high-water mark. Carry RS-latency3 as a marginal supported flag, while keeping
v225 as best observed and v226 as the conservative confirmed value.

## [2026-06-05] loop-iteration | v225-no-scan-sparsecore-rs-ar-rs-latency3 on 8B/v6e-8: provisional supported (59,732 tok/s / 42.9% MFU)

Workload `alekseyv-qwen3-v225-sc-rsar-rslat3` kept the confirmed v220/v221
collective split fixed and added only
`--xla_tpu_sparse_core_reduce_scatter_latency_multiplier=3`. It completed
cleanly with `EXIT_CODE=0`, normal loss ending at 12.0459, profile capture, and
a changed optimized train-step HLO
(`50dae614cf9c7f4cb48f50ed3bee40310a972e141d58476da21b951462d66b88`,
19,909,611 bytes, compiled memory 27.66 GiB). Worker0 averaged **59,713
tok/s** over steps 2-19, **59,757 tok/s** over steps 15-19, and **59,747 tok/s**
excluding profiled steps. Worker1 averaged **59,732 tok/s**, **59,756 tok/s**,
and **59,746 tok/s** respectively. XProf run
`2026-06-05-qwen3-jax-v225-no-scan-sparsecore-rs-ar-rs-latency3/2026_06_05_06_02_53`
shows **4402.9 ms** step time, **67.5% MXU**, **30.93 GiB** peak memory, and
`async-done` dropping to **620.8 ms / 0.6%** from v221's 1,093.0 ms / 1.0%.
This is a new provisional best at roughly **42.9% nominal MFU**, but the margin
is small and the HLO changed, so exact rerun is required before carrying it.

## [2026-06-05] loop-iteration | v224-no-scan-sparsecore-rs-ar-rs-v2 on 8B/v6e-8: refuted (59,204 tok/s / 42.5% MFU)

Workload `alekseyv-qwen3-v224-sc-rsar-rsv2` kept the confirmed v220/v221
frontier fixed and added only
`--xla_tpu_enable_sparse_core_reduce_scatter_v2=true`. It completed cleanly with
`EXIT_CODE=0`, normal loss ending at 12.0461, profile capture, and
byte-identical optimized train-step HLO to v220/v221
(`a33c7940efea974ff9a7dcf2704fa4d08f79e5d82e1821a59a553805ceca81d5`,
19,954,794 bytes). Throughput regressed: worker0 averaged **59,204 tok/s** over
steps 2-19, **59,241 tok/s** over steps 15-19, and **59,247 tok/s** excluding
profiled steps; worker1 averaged **59,133 tok/s** full-window with a slow late
outlier. XProf runs
`2026-06-05-qwen3-jax-v224-no-scan-sparsecore-rs-ar-rs-v2/2026_06_05_05_52_17`
and `/2026_06_05_05_52_18` show **4442.6 ms** step time, **66.8-66.9% MXU**,
and **30.93 GiB** peak memory, worse than v221's 4429.0 ms / 67.4%. The
reduce-scatter v2 flag is a runtime scheduling regression for this native-JAX
shape; do not carry it.

## [2026-06-05] loop-iteration | v223-no-scan-sparsecore-rs-ar-nd-reduce-scatter on 8B/v6e-8: invalid (compiler abort)

Workload `alekseyv-qwen3-v223-sc-rsar-ndrs` kept the confirmed v220/v221
frontier fixed and added only
`--xla_tpu_enable_sparse_core_collective_offload_nd_reduce_scatter=true`, a
newer recipe flag seen in TPU SparseCore stacks. It failed before step 0 on both
workers with `EXIT_CODE=134`. The compiler aborted in
`standalone_reduce_scatter_strategies.cc:1001` with
`Check failed: span_counts_for_colors_[i] == span_count`; the stack includes
`SinglePhaseStandaloneRSStrategy::GetSpanCount`, `VerifyReduceScatter`, and
`ReduceScatterLegalizer::IsValidReduceScatterForSparseCoreOffload`. No profile
or optimized train-step HLO was produced, but partial HLO/MLIR artifacts were
uploaded under
`gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-05-qwen3-jax-v223-no-scan-sparsecore-rs-ar-nd-reduce-scatter/hlo/xla-dump/`
(19.53 MiB). Do not carry ND reduce-scatter offload for this native-JAX shape.

## [2026-06-05] loop-iteration | v222-no-scan-sparsecore-rs-ar-no-aggregator on 8B/v6e-8: refuted (59,246 tok/s / 42.5% MFU)

Workload `alekseyv-qwen3-v222-sc-rsar-noagg` kept the confirmed v220/v221
collective split fixed but disabled only
`--xla_tpu_enable_sparse_core_collective_aggregator`. It completed cleanly with
`EXIT_CODE=0`, normal loss ending at 12.0459, profile capture, and
byte-identical optimized train-step HLO to v220/v221
(`a33c7940efea974ff9a7dcf2704fa4d08f79e5d82e1821a59a553805ceca81d5`,
19,954,794 bytes). Throughput regressed: worker0 averaged **59,246 tok/s** over
steps 2-19, **59,229 tok/s** over steps 15-19, and **59,261 tok/s** excluding
profiled steps; worker1 averaged **59,213 tok/s**, **59,231 tok/s**, and
**59,260 tok/s** respectively. XProf run
`2026-06-05-qwen3-jax-v222-no-scan-sparsecore-rs-ar-no-aggregator/2026_06_05_05_34_27`
shows **4439.1 ms** step time and **66.8% MXU**, worse than v221's 4429.0 ms /
67.4%. The main profile regressions are `async-done` rising to **1,340.1 ms /
1.3%** from v221's 1,093.0 ms / 1.0% and `custom fusion` rising to **3,355.9 ms
/ 3.1%** from 2,813.3 ms / 2.6%. Keep the SparseCore collective aggregator
enabled in the v220/v221 frontier.

## [2026-06-05] loop-iteration | v221-no-scan-sparsecore-rs-ar-only-rerun on 8B/v6e-8: supported (v220 confirmed; best remains 59,441 tok/s / 42.7% MFU)

Workload `alekseyv-qwen3-v221-sc-rsar-r` exactly reran v220: reduce-scatter
and all-reduce remained on SparseCore offload with the SparseCore collective
aggregator, while all-gather stayed on async collective fusion. It completed
cleanly with `EXIT_CODE=0`, normal loss ending at 12.0459, profile capture, and
byte-identical optimized train-step HLO to v220
(`a33c7940efea974ff9a7dcf2704fa4d08f79e5d82e1821a59a553805ceca81d5`,
19,954,794 bytes). Worker0 reported **59,435 tok/s** average over steps 2-19,
**59,478 tok/s** over steps 15-19, and **59,463 tok/s** excluding profiled
steps. Worker1 reported **59,406 tok/s** over steps 2-19, **59,478 tok/s** over
steps 15-19, and **59,462 tok/s** excluding profiled steps; worker1 step 12 was
slow during trace capture. XProf run
`2026-06-05-qwen3-jax-v221-no-scan-sparsecore-rs-ar-only-rerun/2026_06_05_05_19_51`
shows **4429.0 ms** step time, **67.4% MXU**, **30.93 GiB** peak memory,
`async-done` **1,093.0 ms / 1.0%**, and direct `all-gather` **225.5 ms / 0.2%**.
Carry the v220/v221 collective split as the confirmed native-JAX frontier, with
v220 remaining the best single full-window measurement at **59,441 tok/s /
42.7% MFU**. Next isolate: disable only the SparseCore collective aggregator on
the v220 split to test whether it still helps when all-gather is no longer
offloaded to SparseCore.

## [2026-06-05] loop-iteration | v220-no-scan-sparsecore-rs-ar-only on 8B/v6e-8: provisional supported (59,441 tok/s / 42.7% MFU; needs rerun)

Workload `alekseyv-qwen3-v220-sc-rsar` kept the confirmed v210 no-scan
SparseCore collective-aggregator frontier fixed but disabled only SparseCore
all-gather offload, returning all-gather to async collective fusion while
keeping reduce-scatter/all-reduce on SparseCore. It completed cleanly with
`EXIT_CODE=0`, normal loss ending at 12.0459, profile capture, and fresh HLO.
Worker0 reported **59,441 tok/s** average over steps 2-19,
**7,430 tok/s/chip**, and **42.7% nominal MFU**; worker1 reported
**59,424 tok/s**, **7,428 tok/s/chip**, and **42.7% MFU**. Best visible step
was **59,645 tok/s**.

XProf supports the small log-level gain: step time averaged **4428.4 ms** versus
v210's indexed **4433.3 ms**, MXU was **67.5%** versus v210's **67.4%**, peak
memory fell to **30.93 GiB** from **31.06 GiB**, and train-step program time
was **106,044.4 ms** versus v210's **106,155.5 ms**. The main profile change is
that `async-done` drops to **1,087.6 ms / 1.0%** from v210's 3,646.9 ms / 3.4%,
while direct `all-gather` appears as only **218.7 ms / 0.2%**. Optimized HLO
SHA256 `a33c7940efea974ff9a7dcf2704fa4d08f79e5d82e1821a59a553805ceca81d5`,
size **19,954,794 bytes**, compiled memory **27.66 GiB**. Treat as a
provisional new frontier and run an exact rerun before carrying.

## [2026-06-05] loop-iteration | v219-no-scan-sparsecore-ag-ar-only on 8B/v6e-8: refuted (58,269 tok/s / 41.8% MFU)

Workload `alekseyv-qwen3-v219-sc-agar` kept the confirmed v210 no-scan
SparseCore collective-aggregator frontier fixed but disabled only SparseCore
reduce-scatter offload, returning reduce-scatter to async collective fusion.
It completed cleanly with `EXIT_CODE=0`, normal loss, profile capture, and
fresh HLO, but regressed hard: worker0 reported **58,238 tok/s** aggregate,
**7,280 tok/s/chip**, and **41.8% MFU**; worker1 reported **58,269 tok/s**,
**7,284 tok/s/chip**, and **41.8% MFU**. XProf showed train-step program time
increasing to **108,018.8 ms** from v210's 106,155.5 ms, MXU falling to
**65.8%**, and new all-reduce-containing fusion buckets despite `convolution
fusion` shrinking to 52.5%. Optimized HLO SHA256
`5c28221460c5446a0888b89db036f6bc11f47e9cacce9c8416766009bb9555ca`, size
15,650,659 bytes, compiled memory **27.65 GiB**. Do not carry this ablation;
keep reduce-scatter SparseCore offload enabled. Together v218/v219 show both
all-reduce and reduce-scatter offload are needed; next cheap isolate is
all-gather offload.

## [2026-06-05] loop-iteration | v218-no-scan-sparsecore-ag-rs-only on 8B/v6e-8: refuted (59,359 tok/s / 42.6% MFU, below v210)

Workload `alekseyv-qwen3-v218-sc-agrs` kept the confirmed v210 no-scan
SparseCore collective-aggregator frontier fixed but disabled only SparseCore
all-reduce offload, returning all-reduce to async collective fusion. It
completed cleanly with `EXIT_CODE=0`, normal loss, profile capture, and fresh
HLO. Worker0 reported **59,333 tok/s** aggregate, **7,417 tok/s/chip**, and
**42.5% MFU**; worker1 reported **59,359 tok/s**, **7,420 tok/s/chip**, and
**42.6% MFU**. This is below the confirmed v210 band of 59,370-59,372 tok/s.
The optimized train-step HLO changed to SHA256
`eeb040b12d05e32196f22516547a22d84a9343d6681871d4ee7962c1a63308df`
with **27.66 GiB** compiled memory and 16,537,770 bytes of HLO text. XProf
showed the dominant buckets essentially unchanged (`convolution fusion`
57,710 ms / 54.2%), with `async-done` reduced from v210's 3,646 ms / 3.4% to
3,393 ms / 3.2% because the top all-reduce done entry disappeared. The async
tail reduction did not convert into throughput. Do not carry this ablation;
keep all-reduce SparseCore offload enabled. Next cheap isolate: disable only
reduce-scatter SparseCore offload while keeping all-gather and all-reduce
offloaded.

## [2026-06-05] local-candidate | v217-scan-custom-vjp-tiled-ce on 8B/v6e-8: invalid before launch (CPU equivalence drift)

Local fork `v217-scan-custom-vjp-tiled-ce` tried to fix v216's unrolled custom
VJP by expressing the vocab chunk loop as `lax.scan`. `py_compile` passed, but
the CPU equivalence gate failed against full CE: padded scan with toy
`V=17, chunk=4` had loss delta `0.0066075325`; dynamic-slice scan had the same
loss drift; and a divisible `V=16, chunk=4` test still drifted by
`0.0066699982`. Precision settings did not remove the mismatch. No TPU workload
was launched. Do not continue the scan-custom-VJP path without first restoring
strict local CE equivalence; move CE work to a faithful streaming/Pallas kernel
or pivot to a non-CE structural hypothesis. See
[v217 local invalid](experiments/2026-06-05-v217-scan-custom-vjp-tiled-ce-local-invalid.md).

## [2026-06-05] loop-iteration | v216-no-scan-custom-vjp-tiled-ce-bs4 on 8B/v6e-8: invalid (compile OOM, 38.12G HBM)

Workload `alekseyv-qwen3-v216-cvjpce-bs4` replaced the remat-only tiled CE
backward with a custom VJP that recomputes chunk logits and explicitly forms
`softmax - onehot`. Pre-run validation passed (`py_compile`; toy CPU CE
equivalence with loss delta `0.0`, hidden-grad max delta `0.00146484375`, and
lm-head-grad max delta `0.0009765625`). Train-step compilation still failed
before step 0 with `RESOURCE_EXHAUSTED`: XLA reported **38.12G used of 31.25G
HBM**, exceeded by **6.87G**. The custom VJP removed the saved predicate live
set (**0×`pred[32768,4096]`**) but expanded f32 chunk-dot temps
(**52×`f32[32768,4096]`**) and is worse than v214's 35.78G. Memory-report total
is **43.59 GiB**, with **37.83 GiB** preallocated temp. Optimized train-step
HLO: `module_0294.jit_train_step.cl_854318611.after_optimizations.txt`, 109,076
lines / 18,668,321 bytes, SHA256
`4b4277d8e930d9dc40c491334ffd8bffaec38ba4f2b33d9aa4bed537c1d6d8f6`; HLO prefix
`gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-05-qwen3-jax-v216-no-scan-custom-vjp-tiled-ce-bs4/hlo/xla-dump/`
has 4950 objects / 1,760,258,711 bytes. Do not carry unrolled Python custom
VJP; the local v217 `lax.scan` custom-VJP rewrite failed CPU equivalence, so the
cheap scan path is also closed unless equivalence is restored first.

## [2026-06-05] loop-iteration | v215-no-scan-remat-tiled-ce-chunk2048-bs4 on 8B/v6e-8: invalid (compile OOM, 54.36G HBM)

Workload `alekseyv-qwen3-v215-remt2k-bs4` kept the v214 rematerialized
vocab-tiled CE image and changed only `--tiled_ce_chunk_size=4096` to
`--tiled_ce_chunk_size=2048`. It failed before step 0 with
`RESOURCE_EXHAUSTED`: XLA reported **54.36G used of 31.25G HBM**, exceeded by
**23.11G**. This is much worse than v214's 35.78G. The memory report shows
**59.76 GiB** total bytes, **54.01 GiB** preallocated temp, and a larger chunk
live set: **132×`bf16[32768,2048]`**, **29×`f32[32768,2048]`**,
**74×`pred[32768,2048]`**, plus **150×`bf16[32768,4096]`**. Optimized
train-step HLO: `module_0294.jit_train_step.cl_854318611.after_optimizations.txt`,
141,911 lines / 23,370,047 bytes, SHA256
`4c5b3059980dcc98463a686c10a300c6d939f92c99d576d4710280dab036adb6`; HLO prefix
`gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-05-qwen3-jax-v215-no-scan-remat-tiled-ce-chunk2048-bs4/hlo/xla-dump/`
has 1226 objects / 149,773,227 bytes. Do not continue the remat-only
chunk-size sweep; move CE/logit-memory work to custom VJP or Pallas/streaming
CE.

## [2026-06-05] loop-iteration | v214-no-scan-remat-tiled-ce-bs4 on 8B/v6e-8: invalid (compile OOM, 35.78G HBM)

Workload `alekseyv-qwen3-v214-remtiled-bs4` tested the v213 native-JAX
vocab-tiled CE path with per-chunk max and sum/picked-logit computations wrapped
in `jax.checkpoint`. Pre-run validation passed (`py_compile`; toy CPU CE
equivalence with loss delta `0.0`, hidden-grad max delta `0.00390625`, and
lm-head-grad max delta `0.0`). Train-step compilation still failed before step 0
with `RESOURCE_EXHAUSTED`: XLA reported **35.78G used of 31.25G HBM**, exceeded
by **4.54G**. This improves v213 by about **1.21G**, but remains invalid.
The failed memory report shows **41.23 GiB** total bytes, **35.48 GiB**
preallocated temp, and still many chunk-shaped values:
**78×`bf16[32768,4096]`**, **15×`f32[32768,4096]`**, and
**37×`pred[32768,4096]`**. Optimized train-step HLO:
`module_0294.jit_train_step.cl_854318611.after_optimizations.txt`, 115,286
lines / 19,541,330 bytes, SHA256
`07adb0643ddb20b0f329f1d68a2dc05e32e1c40b28ccf1ce31d025f2d334976f`; HLO prefix
`gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-05-qwen3-jax-v214-no-scan-remat-tiled-ce-bs4/hlo/xla-dump/`
has 1220 objects / 122.93 MiB. Remat is not enough at chunk size 4096; use one
smaller chunk-size response probe, then move to custom VJP/Pallas if still far
from fitting.

## [2026-06-05] loop-iteration | v213-no-scan-tiled-ce-bs4 on 8B/v6e-8: invalid (compile OOM, 36.99G HBM)

Workload `alekseyv-qwen3-v213-tiledce-bs4` tested a native-JAX vocab-tiled CE
prototype on top of the v210 no-scan SparseCore-aggregator stack at seq8192,
per-chip batch 4/global batch 32. The code passed `py_compile` and a toy CPU
CE/gradient equivalence check against full CE, but train-step compilation
failed before step 0 with `RESOURCE_EXHAUSTED`: XLA reported **36.99G used of
31.25G HBM**, exceeded by **5.74G**. The memory report total is **42.44 GiB**,
with **36.69 GiB** preallocated temp. The mechanism removed the full-vocab
`bf16[4,8192,151936]` logits tensor, but autodiff retained many per-chunk
values instead: **83×`bf16[32768,4096]`**, **10×`f32[32768,4096]`**, and
**38×`pred[32768,4096]`** in the memory report. Optimized train-step HLO:
`module_0294.jit_train_step.cl_854318611.after_optimizations.txt`, 116,128
lines / 19,780,378 bytes, SHA256
`c645d622e27b78b2e72a2130aff9383274a02a06150f96ae3a246c44dcbf47dd`; HLO prefix
`gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-05-qwen3-jax-v213-no-scan-tiled-ce-bs4/hlo/`
has 903 objects / 116.14 MiB. Do not carry naive autodiffed tiled CE; future
CE/logit-memory work needs rematerialized chunks, custom VJP, or a Pallas
streaming CE that does not save chunk logits/masks.

## [2026-06-05] loop-iteration | v212-no-scan-sparsecore-aggregator-bs3-comparison on 8B/v6e-8: supported comparison (58,970 tok/s / 42.3% MFU beats MaxText v006 throughput)

Workload `alekseyv-qwen3-v212-noscan-scagg-bs3` reran the confirmed v210
no-scan SparseCore collective-aggregator frontier at MaxText's bs3/chip shape
(seq8192, global batch 24). It completed with `EXIT_CODE=0`, normal loss,
profile capture, and a fresh train-step HLO. Worker0 reported **58,970 tok/s**
aggregate, **7,371 tok/s/chip**, and **42.3% MFU**; worker1 reported
58,940 tok/s, 7,368 tok/s/chip, and 42.3% MFU. This is not a new frontier
because v210 bs4/chip remains faster at 59,370-59,372 tok/s, but it beats the
same-shape MaxText v006 reference of 55,064 tok/s / 6,883 tok/s/chip by about
**+7.1%**. Reported MFU stays below MaxText's 44.92%, confirming the MFU gap is
mostly trainer accounting rather than token-throughput performance.

## [2026-06-05] loop-iteration | v211-no-scan-sparsecore-concurrent-offload on 8B/v6e-8: refuted (59,268 tok/s / 42.5% MFU)

Workload `alekseyv-qwen3-v211-noscan-scconc` kept the confirmed v210 no-scan
SparseCore collective-aggregator frontier fixed and added only
`--xla_tpu_enable_concurrent_sparse_core_offloading=true`. It completed with
`EXIT_CODE=0`, normal loss, profile capture, and fresh train-step HLO, but
worker0 reported only **59,268 tok/s** aggregate, **7,408 tok/s/chip**, and
**42.5% MFU**; worker1 reported 59,230 tok/s, 7,404 tok/s/chip, and 42.5% MFU.
XProf showed **67.2% MXU**, `convolution fusion` still dominant at 54.1%, and
`async-done` slightly higher than v210 at 3.5%. The optimized train-step HLO
changed to SHA256 `ba6e2d7aa46de6265678865f1dc30d9645926b402ed667d8453b15a22ac75a41`
with **27.68 GiB** compiled memory, up from v210's 27.66 GiB. Do not carry
concurrent SparseCore offloading. See also the MaxText/native gap analysis:
`wiki/analyses/2026-06-05-qwen3-jax-maxtext-gap-analysis.md`.

## [2026-06-05] loop-iteration | v210-no-scan-sparsecore-aggregator-rerun3 on 8B/v6e-8: supported (confirms v208 aggregator band at 59,370 tok/s / 42.6% MFU)

Workload `alekseyv-qwen3-v210-noscan-scagg-r3` is the corrected exact
aggregator confirmation rerun after v209 failed before compile due invalid flag
names. It uses the accepted v208 flag names from the train-step flagfile. It
completed with `EXIT_CODE=0`, normal loss, profile capture, and a fresh
train-step HLO. Worker0 reported **59,370 tok/s** aggregate,
**7,421 tok/s/chip**, and **42.6% MFU**; worker1 reported 59,342 tok/s,
7,418 tok/s/chip, and 42.5% MFU. The optimized train-step HLO SHA256 was
`d67433daa35f81edbfbd0fb25d79bd4cd210fc9697bd39133cd50dfe11da9573`, identical
to v205/v208, with 27.66 GiB compiled memory. Carry the aggregator as a
marginal rerun-confirmed runtime scheduling win.

## [2026-06-05] loop-iteration | v209-no-scan-sparsecore-aggregator-rerun2 on 8B/v6e-8: invalid (launch flag typo)

Workload `alekseyv-qwen3-v209-noscan-scagg-r2` is an exact rerun of the v208
aggregator stack. v208 completed cleanly and reported a marginal new
best-worker average of 59,372 tok/s / 42.6% MFU, but the train-step HLO was
byte-identical to v205/v206. This rerun tests whether that small runtime lift
is reproducible before carrying the aggregator flag. It failed immediately
before compilation on both workers with `EXIT_CODE=1` because
`xla_tpu_enable_layer_wise_all_gather` and
`xla_tpu_enable_custom_call_combiner` are unknown to this libtpu build. No
performance signal; use v210.

## [2026-06-05] loop-iteration | v208-no-scan-sparsecore-aggregator-rerun on 8B/v6e-8: supported and rerun-confirmed (59,372 tok/s / 42.6% MFU)

Workload `alekseyv-qwen3-v208-noscan-scagg-r` is the clean relaunch of v207's
one-flag aggregator hypothesis after v207 was killed inconclusively before
train-step metrics. It keeps the confirmed v206 all-collective SparseCore
offload frontier fixed and changes only
`--xla_tpu_enable_sparse_core_collective_aggregator=false` to `true`. It
completed with `EXIT_CODE=0`, normal loss, profile capture, and a fresh
train-step HLO. Worker0 reported **59,372 tok/s** aggregate,
**7,422 tok/s/chip**, and **42.6% MFU**; worker1 reported 59,355 tok/s,
7,419 tok/s/chip, and 42.6% MFU. The optimized train-step HLO SHA256 was
`d67433daa35f81edbfbd0fb25d79bd4cd210fc9697bd39133cd50dfe11da9573`, identical
to v205, with 27.66 GiB compiled memory. v210 reproduced the band at
59,370 tok/s / 42.6% MFU, so carry as a marginal runtime-level win.

## [2026-06-05] loop-iteration | v207-no-scan-sparsecore-aggregator on 8B/v6e-8: inconclusive (killed before train-step metrics)

Workload `alekseyv-qwen3-v207-noscan-scagg` keeps the confirmed v206 no-scan
SparseCore all-collective offload frontier fixed and changes only
`--xla_tpu_enable_sparse_core_collective_aggregator=false` to `true`. v158 had
shown the aggregator was a no-op without SparseCore collective offload; this
tests whether it becomes useful now that all-gather, reduce-scatter, and
all-reduce are offloaded. It was killed before train-step metrics after an
ambiguous control-plane read showed no JobSet while XPK still showed quota
reserved; `xpk workload delete` then found and removed the JobSet. Treat v207
as inconclusive and use v208 as the actual aggregator test.

## [2026-06-05] loop-iteration | v206-no-scan-sparsecore-all-collectives-rerun on 8B/v6e-8: supported (confirms 59,351 tok/s / 42.6% MFU)

Workload `alekseyv-qwen3-v206-noscan-scall-r` is an exact rerun of the v205
no-scan SparseCore all-collective offload stack. v205 is a new once-supported
frontier at 59,338 tok/s / 42.5% MFU; this rerun tests whether the lift over
the v182/v183 58,288-58,330 tok/s / 41.8% MFU frontier is stable before
carrying the flag set as confirmed. It completed with `EXIT_CODE=0`, normal
loss, and matched the v205 band. Worker0 reported **59,351 tok/s** aggregate,
**7,419 tok/s/chip**, and **42.6% MFU**; worker1 reported 59,329 tok/s,
7,416 tok/s/chip, and 42.5% MFU. Profile files landed under
`2026-06-05-qwen3-jax-v206-no-scan-sparsecore-all-collectives-rerun/plugins/profile/2026_06_05_01_42_37/`
and `.../2026_06_05_01_42_38/`. No fresh HLO dump appeared because the run hit
the persistent compile cache; use the v205 train-step HLO SHA
`d67433daa35f81edbfbd0fb25d79bd4cd210fc9697bd39133cd50dfe11da9573` as the
confirmed program reference. Carry all-collective SparseCore offload as the
confirmed JAX frontier.

## [2026-06-05] loop-iteration | v205-no-scan-sparsecore-all-collectives on 8B/v6e-8: supported (new best 59,338 tok/s / 42.5% MFU)

Workload `alekseyv-qwen3-v205-noscan-scall` keeps the confirmed v183 no-scan
frontier fixed and changes only collective placement: disable async
collective-fusion subtypes for all-gather/reduce-scatter/all-reduce, then
enable SparseCore collective offload plus tracing for all three collective
families. This tests whether the Llama3-style all-collective SparseCore relay
is a portable structural lever for Qwen3's no-scan frontier. It completed
cleanly with `EXIT_CODE=0` on both workers and normal loss through step 19.
Worker0 reported **59,338 tok/s** aggregate, **7,417 tok/s/chip**, and
**42.5% MFU**; worker1 reported 59,319 tok/s, 7,415 tok/s/chip, and 42.5% MFU.
This beats the v182/v183 confirmed 58,288-58,330 tok/s / 41.8% MFU frontier by
about +1.7% throughput and +0.7 pp MFU. Profile files landed under
`2026-06-05-qwen3-jax-v205-no-scan-sparsecore-all-collectives/plugins/profile/2026_06_05_01_31_01/`
and `.../2026_06_05_01_31_02/`. The two optimized train-step HLOs were
byte-identical, SHA256
`d67433daa35f81edbfbd0fb25d79bd4cd210fc9697bd39133cd50dfe11da9573`, with
27.66 GiB compiled memory. SparseCore metadata includes repeated all-gather
offload factory messages about `SpmemWindow` offsets; keep that caveat in mind,
but carry the runtime result. Next step: exact rerun before adding another
mechanism.

## [2026-06-05] loop-iteration | v204-no-scan-maxtext-ce-bs5-shmem90 on 8B/v6e-8: invalid (batch 5 MaxText CE still OOMs at 32.10 GiB)

Workload `alekseyv-qwen3-v204-noscan-bs5-shmem90` tests the v203 no-scan
MaxText CE batch-5 shape with only
`--xla_tpu_scheduler_percent_shared_memory_limit=100` changed to `90`. v203
missed HBM by only 877 MiB, so this is a targeted lower-HBM schedule attempt;
support requires both fitting and beating the 58,288-58,330 tok/s batch-4
frontier. It failed before step 0 with `EXIT_CODE=1`: XLA reported a permanent
compile HBM OOM, using 32.10 GiB of 31.25 GiB and exceeding capacity by
876.30 MiB. Lowering scheduler shared memory did not materially change the
batch-5 memory envelope, so this direct MaxText CE batch-5 branch is closed
until a real lm-head/logit memory reduction is introduced.

## [2026-06-05] loop-iteration | v203-no-scan-maxtext-ce-bs5 on 8B/v6e-8: invalid (batch 5 MaxText CE OOMs at 32.10 GiB)

Workload `alekseyv-qwen3-v203-noscan-maxtextce-bs5` tests the confirmed v183
no-scan MaxText CE frontier with only `--batch_size=4` changed to
`--batch_size=5` (global batch 40). This is a high-risk near-fit probe:
v202 proved batch 5 can fit with Tokamax CE but is too slow at ~54.9k tok/s;
if the faster MaxText CE path fits at batch 5, it has a plausible route to
beat the 58,288-58,330 tok/s frontier. Compile HBM OOM closes this direct
batch-scaling branch. The job failed before step 0 with `EXIT_CODE=1`: XLA
reported a permanent compile HBM OOM, using 32.10 GiB of 31.25 GiB and
exceeding capacity by 877.10 MiB. Failed optimized HLO was 18,965,922 bytes
with SHA256 `741eb3be6ff0b4d55273479788cac32efc30cf51fd0e8cf6ceb36014c9bbcefd`;
memory report was 32.11 GiB total bytes used with a 26.35 GiB preallocated
temp. The largest top-memory allocation was an 11.59 GiB
`bf16[5,8192,151936]` full-vocab lm-head/logit temp from
`jit(train_step)/forward_backward/jvp()/dot_general`. Text counts were
all-gather 6,964, all-reduce 1,148, collective-permute 6, convolution 3,082,
custom-call 2,804, copy 21,780, fusion 31,399, `log_softmax` 0, `_one_hot` 11.
Direct MaxText CE batch-5 scaling is closed until a real memory reduction is
introduced.

## [2026-06-05] loop-iteration | v202-no-scan-tokamax-ce-bs5 on 8B/v6e-8: refuted (batch 5 fits but stays ~54.9k tok/s)

Workload `alekseyv-qwen3-v202-noscan-tokamaxce-bs5` tests the v201 no-scan
Tokamax CE stack with only `--batch_size=4` changed to `--batch_size=5`
(global batch 40). v201 lowered compiled memory to 22.50 GiB but reached only
about 54.3k tok/s at batch 4; this run tests whether that memory relief can
fit batch 5 and recover enough total throughput to beat the v182/v183 MaxText
CE no-scan frontier of 58,288-58,330 tok/s / 41.8% MFU. It completed cleanly
with `EXIT_CODE=0` and normal loss through step 19. Worker0 steady steps 2-19
averaged 54,915 tok/s (range 54,844-55,069), and worker1 averaged 54,905 tok/s
(range 54,684-55,068); final loss was 12.0496. Profile files were written
under `2026-06-05-qwen3-jax-v202-no-scan-tokamax-ce-bs5/plugins/profile/2026_06_05_00_43_21/`.
Trace JSON spans for `jit_train_step(3503685562718697460)` confirmed
`gke-tpu-964065d9-9vh8` at 5.977850 s / 54,816 tok/s and
`gke-tpu-964065d9-c180` at 5.972015 s / 54,869 tok/s. HLO proves the memory
envelope is valid: optimized train-step HLO was 19,048,795 bytes with SHA256
`e0dfbacf45659f1c3a07a30cd0736e4b80f16bcf7e401765015647d3341ccc7c`; memory
report was 25.62 GiB total bytes used with a 19.87 GiB preallocated temp. Text
counts were all-gather 7,038, all-reduce 1,147, collective-permute 6,
convolution 3,060, custom-call 2,804, copy 21,400, fusion 31,679, `tokamax`
32, `linear_softmax_cross_entropy_loss` 34, `log_softmax` 0, `_one_hot` 0.
Do not carry Tokamax CE batch scaling.

## [2026-06-05] loop-iteration | v201-no-scan-tokamax-ce on 8B/v6e-8: refuted (54.3k tok/s; memory relief but too slow)

Workload `alekseyv-qwen3-v201-noscan-tokamaxce` reused the v183 no-scan
frontier stack and changed only CE implementation from MaxText CE to
`--use_tokamax_ce=True --tokamax_ce_impl=mosaic_tpu --use_maxtext_ce=False`.
It compiled and reached training with normal loss through the captured profile
window; the cluster reaped the JobSet/pods before final stdout could be fetched,
so the final `EXIT_CODE` line is unavailable. Captured steady throughput was
about 54.3k tok/s: worker0 steps 7-14 ranged 54,294-54,368 tok/s, and worker1
steps 7-14 were in the same band except profiler-overhead step 12 at 53,904
tok/s. Profile files were written under
`2026-06-05-qwen3-jax-v201-no-scan-tokamax-ce/plugins/profile/2026_06_05_00_28_12/`.
Trace JSON spans for `jit_train_step(2566476259130850218)` confirmed
`gke-tpu-964065d9-9vh8` at 4.840374 s / 54,158 tok/s and
`gke-tpu-964065d9-c180` at 4.826154 s / 54,317 tok/s. Optimized train-step HLO
was 19,005,717 bytes with SHA256
`3453e887ee7ccf1ccab52e2503ad14362a8e6d50b2c77c944f1aa10b1edfd5ab`; memory
fell to 22.50 GiB with a 16.75 GiB preallocated temp. Text counts were
all-gather 4,655, all-reduce 960, collective-permute 3, convolution 2,088,
custom-call 1,474, copy 8,946, fusion 23,235, `tokamax` 20,
`linear_softmax_cross_entropy_loss` 16, `log_softmax` 0, `_one_hot` 0. Do not
carry Tokamax CE for batch-4 throughput; its memory relief may justify only a
separate batch-5 envelope test.

## [2026-06-05] loop-iteration | v200-no-scan-plain-integer-ce on 8B/v6e-8: invalid (plain CE OOMs at 48.39 GiB)

Workload `alekseyv-qwen3-v200-noscan-plaince` tested the current no-scan
frontier with MaxText CE disabled, leaving `--use_tokamax_ce=False` so the
trainer used its built-in integer-label `log_softmax + gather` CE. This is not
the v043/v044 full-logit custom-VJP branch; it changes only CE lowering from
the v183 runtime stack. The job failed before step 0 on both pods with
`EXIT_CODE=1`: XLA reported a permanent TPU compile HBM OOM, using 48.39 GiB
of 31.25 GiB and exceeding capacity by 17.14 GiB. The top two temps were both
18.55 GiB f32 full-vocab tensors: `f32[4,8192,151936]` from the lm-head
`dot_general` and `f32[32768,151936]` from `jvp(jit(log_softmax))/sub`.
Optimized failed train-step HLO was emitted at 19,057,269 bytes with SHA256
`a5fdbd91bf3adfd6f99006e784753b2ffd6f15c79fa544c3d00c2d0ce40a8869`; memory
report showed 54.12 GiB total bytes used and a 48.37 GiB preallocated temp.
Text counts were all-gather 4,655, all-reduce 960, collective-permute 3,
convolution 2,103, custom-call 1,472, copy 9,451, fusion 23,313,
`log_softmax` 41, `_one_hot` 0. Do not replace MaxText CE with plain integer
CE on this frontier; MaxText remains required to avoid f32 full-vocab
materialization.

## [2026-06-05] loop-iteration | v199-no-scan-rs-collective-matmul-auto on 8B/v6e-8: invalid (reduce-scatter auto OOMs at 35.87 GiB)

Workload `alekseyv-qwen3-v199-noscan-rsauto` completed the collective-matmul
isolate sweep by restoring only
`--xla_tpu_reduce_scatter_collective_matmul_mode=auto` while keeping
`--xla_tpu_all_gather_collective_matmul_mode=none`. The job failed before
step 0 on both pods with `EXIT_CODE=1`: XLA reported a permanent TPU compile
HBM OOM, using 35.87 GiB of 31.25 GiB and exceeding capacity by 4.62 GiB. The
first failing label was `fusion.10969 = fusion(fusion.7165,
async-collective-done.337)` under
`jit(train_step)/forward_backward/jvp()/dot_general`. Optimized train-step HLO
was emitted at 26,420,592 bytes with SHA256
`66a1b7df4c332ae916784705aa6e902bde041c4204c2fddc24a3cd987a856ce9`; memory
report was 35.88 GiB with a 30.12 GiB preallocated temp. Text counts were
all-gather 5,664, all-reduce 153, collective-permute 7,596, convolution 3,676,
custom-call 3,381, copy 34,627, fusion 33,024. Together with v197/v198, this
closes the no-scan collective-matmul mode sweep: keep both modes `none`.

## [2026-06-04] loop-iteration | v198-no-scan-ag-collective-matmul-auto on 8B/v6e-8: invalid (all-gather auto OOMs at 32.67 GiB)

Workload `alekseyv-qwen3-v198-noscan-agauto` isolated v197's
collective-matmul OOM by restoring only
`--xla_tpu_all_gather_collective_matmul_mode=auto` while keeping
`--xla_tpu_reduce_scatter_collective_matmul_mode=none`. The job failed before
step 0 on both pods with `EXIT_CODE=1`: XLA reported a permanent TPU compile
HBM OOM, using 32.67 GiB of 31.25 GiB and exceeding capacity by 1.42 GiB. The
failing labels included `fusion.9455.remat = fusion(async-collective-done, ...)`
under `jit(train_step)/forward_backward/jvp()/dot_general` and
`fusion = fusion(all-gather.6169.remat, copy-done.2039)`. Optimized train-step
HLO was emitted at 28,098,167 bytes with SHA256
`52ec4c05240fefa69a03e817b4050015f57edb0ef24df9a9a6020df151ab418f`; memory
report was 32.68 GiB with a 26.93 GiB preallocated temp. Text counts changed to
all-gather 5,544, all-reduce 1,148, collective-permute 9,221, convolution
3,956, custom-call 2,285, copy 30,383, fusion 36,308. Keep all-gather
collective-matmul mode `none`; next isolate reduce-scatter `auto` alone.

## [2026-06-04] loop-iteration | v197-no-scan-collective-matmul-auto on 8B/v6e-8: invalid (compile HBM OOM at 35.56 GiB)

Workload `alekseyv-qwen3-v197-noscan-cmauto` tested the retrospective's
collective-matmul dependency under the confirmed no-scan frontier by restoring
both XLA modes from `none` to `auto`:
`--xla_tpu_all_gather_collective_matmul_mode=auto` and
`--xla_tpu_reduce_scatter_collective_matmul_mode=auto`. The model image and
runtime command otherwise matched v183. The job failed before step 0 with
`EXIT_CODE=1`: XLA reported a permanent TPU compile HBM OOM, using 35.56 GiB of
31.25 GiB and exceeding capacity by 4.32 GiB. The failing label was
`fusion.14097 = fusion(... async-collective-done)` under
`jit(train_step)/forward_backward/jvp()/dot_general`. No profile was produced.
The collective-matmul `none` pair remains required for the current no-scan
batch-4/seq8192 shape; isolate all-gather vs reduce-scatter modes next.

## [2026-06-04] loop-iteration | v196-no-scan-no-layer-input-sac on 8B/v6e-8: refuted/near-tie (58,099 tok/s, 41.7% MFU; closes single-boundary sweep)

Workload `alekseyv-qwen3-v196-noscan-nolayersac` reused the numerically
equivalent `v175-no-layer-input-sac` image under the no-scan runtime stack,
removing only the decoder layer input `_sac` while keeping `--shard_acts=True`.
Containerized CPU equivalence was rerun before launch and passed: logits max
delta `2.682e-07`, loss delta `0`, all 25 gradients OK, max gradient delta
`7.823e-08`. The TPU run completed cleanly with normal loss ending at 12.0459
and `EXIT_CODE=0`, reaching worker0 58,099 tok/s average / 7,262 tok/s/chip /
41.7% MFU and worker1 58,083 tok/s average / 7,260 tok/s/chip / 41.6% MFU,
with a best observed post-warmup step of 58,251 tok/s. Profile files were
written under
`2026-06-04-qwen3-jax-v196-no-scan-no-layer-input-sac/plugins/profile/2026_06_04_23_33_01/`.
Device-side `jit_train_step(403931025898375493)` spans confirm the near-tie
regression: `gke-tpu-964065d9-9vh8` averaged 4.524858 s / 57,934 tok/s and
`gke-tpu-964065d9-c180` averaged 4.517163 s / 58,033 tok/s, below v183's
4.497 s class. Fresh optimized train-step HLO is 19,043,220 bytes with SHA256
`f94aebd8de00f0a1311ee3960d32851f1dfc262c9f8f4b4a056cfc3cc02e0dfc`, 27.66 GiB
memory, and text counts all-gather 7,039, all-reduce 1,148,
collective-permute 6, convolution 3,071, custom-call 2,806, copy 21,271,
fusion 31,686. Do not carry decoder layer input `_sac` removal under no-scan.
The single-boundary activation-constraint sweep is closed; only the MLP
intermediate `_sac` removal remains carried.

## [2026-06-04] loop-iteration | v195-no-scan-no-pre-MLP-norm-sac on 8B/v6e-8: refuted (56,654 tok/s, 40.6% MFU; boundary removal regresses)

Workload `alekseyv-qwen3-v195-noscan-nomlpnormsac` reused the numerically
equivalent `v174-no-pre-mlp-norm-sac` image under the no-scan runtime stack,
removing only the pre-MLP norm-output `_sac` while keeping `--shard_acts=True`.
Containerized CPU equivalence was rerun before classification and passed:
logits max delta `2.682e-07`, loss delta `0`, all 25 gradients OK, max gradient
delta `7.823e-08`. The TPU run completed cleanly with normal loss ending at
12.0459 and `EXIT_CODE=0`, but regressed to worker0 56,654 tok/s average /
7,082 tok/s/chip / 40.6% MFU and worker1 56,628 tok/s average / 7,079
tok/s/chip / 40.6% MFU, with a best observed post-warmup step of 57,007 tok/s.
Profile files were written under
`2026-06-04-qwen3-jax-v195-no-scan-no-pre-mlp-norm-sac/plugins/profile/2026_06_04_23_22_20/`.
Device-side `jit_train_step(14016405483169477827)` spans confirm the regression:
`gke-tpu-32cb1c36-gs53` averaged 4.663221 s / 56,215 tok/s and
`gke-tpu-32cb1c36-jj8f` averaged 4.650469 s / 56,369 tok/s. Fresh optimized
train-step HLO is 19,043,256 bytes with SHA256
`aa8c66042d4f0c579643e99c7a4e99a9b6291d73bf6f50ff9e095930e493a773`, 27.66 GiB
memory, and text counts all-gather 7,039, all-reduce 1,148,
collective-permute 6, convolution 3,071, custom-call 2,806, copy 21,271,
fusion 31,686. Do not carry the pre-MLP norm-output `_sac` removal under
no-scan.

## [2026-06-04] loop-iteration | v194-no-scan-no-pre-attn-norm-sac on 8B/v6e-8: refuted/near-tie (58,086 tok/s, 41.6% MFU; same high-count boundary-relaxation family)

Workload `alekseyv-qwen3-v194-noscan-noinnormsac` reused the numerically
equivalent `v173-no-pre-attn-norm-sac` image under the no-scan runtime stack,
removing only the pre-attention norm-output `_sac` while keeping
`--shard_acts=True`. The run completed cleanly with normal loss ending at
12.0459 and `EXIT_CODE=0`, reaching worker0 58,086 tok/s average / 7,261
tok/s/chip / 41.6% MFU and worker1 58,064 tok/s average / 7,258 tok/s/chip /
41.6% MFU, with a best observed steady step of 58,244 tok/s. Profile files were
written under
`2026-06-04-qwen3-jax-v194-no-scan-no-pre-attn-norm-sac/plugins/profile/2026_06_04_23_14_45/`.
Device-side `jit_train_step(403931025898375493)` spans confirm the near-tie
regression: `gke-tpu-964065d9-9vh8` averaged 4.528267 s / 57,891 tok/s and
`gke-tpu-964065d9-c180` averaged 4.517641 s / 58,027 tok/s. Fresh optimized
train-step HLO is 19,043,220 bytes with SHA256
`ad96ef695c73656ad5560cde57b941bfc1e0db58797e71143dc7cd4591360aba`, 27.66 GiB
memory, and text counts all-gather 7,039, all-reduce 1,148,
collective-permute 6, convolution 3,071, custom-call 2,806, copy 21,271,
fusion 31,686. Do not carry the pre-attention norm-output `_sac` removal under
no-scan.

## [2026-06-04] loop-iteration | v193-no-scan-no-attn-output-sac on 8B/v6e-8: refuted/near-tie (58,087 tok/s, 41.6% MFU; individual boundary still below frontier)

Workload `alekseyv-qwen3-v193-noscan-noattnsac` reused the numerically
equivalent `v172-no-attn-output-sac` image under the no-scan runtime stack,
removing only the attention-output `_sac` while keeping `--shard_acts=True`.
The run completed cleanly with normal loss ending at 12.0459 and `EXIT_CODE=0`,
reaching worker0 58,087 tok/s average / 7,261 tok/s/chip / 41.6% MFU and
worker1 58,075 tok/s average / 7,259 tok/s/chip / 41.6% MFU, with a best
observed steady step of 58,236 tok/s. Profile files were written under
`2026-06-04-qwen3-jax-v193-no-scan-no-attn-output-sac/plugins/profile/2026_06_04_23_06_48/`.
Device-side `jit_train_step(403931025898375493)` spans confirm the near-tie
regression: `gke-tpu-964065d9-9vh8` averaged 4.525476 s / 57,926 tok/s and
`gke-tpu-964065d9-c180` averaged 4.517539 s / 58,028 tok/s. Fresh optimized
train-step HLO is 19,043,256 bytes with SHA256
`7d7516976eb9198a5898be22fb06e0364d73a6a35fb16e1d1fc2fc8dec2410fe`, 27.66 GiB
memory, and text counts all-gather 7,039, all-reduce 1,148,
collective-permute 6, convolution 3,071, custom-call 2,806, copy 21,271,
fusion 31,686. Do not carry the attention-output `_sac` removal under no-scan.

## [2026-06-04] loop-iteration | v192-no-mlp-intermediate-sac-no-scan-no-shard-acts on 8B/v6e-8: refuted/near-tie (58,088 tok/s, 41.6% MFU; changed HLO below frontier)

Workload `alekseyv-qwen3-v192-noscan-noshardacts` removed all explicit
activation sharding constraints from the confirmed no-scan frontier by changing
only `--shard_acts=True` to `--shard_acts=False`. The run completed cleanly
with normal loss ending at 12.0459 and `EXIT_CODE=0`, reaching worker0 58,088
tok/s average / 7,261 tok/s/chip / 41.6% MFU and worker1 58,067 tok/s average /
7,258 tok/s/chip / 41.6% MFU, with a best observed steady step of 58,244 tok/s.
Profile files were written under
`2026-06-04-qwen3-jax-v192-no-mlp-sac-no-scan-no-shard-acts/plugins/profile/2026_06_04_22_58_44/`.
Device-side `jit_train_step(14016405483169477827)` spans confirm this is below
the frontier: `gke-tpu-964065d9-9vh8` averaged 4.527451 s / 57,901 tok/s and
`gke-tpu-964065d9-c180` averaged 4.517729 s / 58,026 tok/s, versus v183's
4.497 s class. Fresh optimized train-step HLO is 19,042,581 bytes with SHA256
`a05d8bbc890d3894410361052661d359e95898e9ae3365121b1d1426db227dc7`, 27.66 GiB
memory, and text counts all-gather 7,039, all-reduce 1,148,
collective-permute 6, convolution 3,071, custom-call 2,806, copy 21,271,
fusion 31,686. Do not carry wholesale `--shard_acts=False`; individual no-scan
activation-boundary probes remain the more targeted follow-up.

## [2026-06-04] loop-iteration | v191-no-mlp-intermediate-sac-no-scan-tokamax-exp-sched on 8B/v6e-8: refuted (57,516 tok/s, 41.2% MFU; changed HLO below frontier)

Workload `alekseyv-qwen3-v191-noscan-expsched` retested Tokamax Splash's
experimental scheduler on the confirmed no-scan frontier with
`TOKAMAX_USE_EXP_SCHED=1`. The run completed cleanly with normal loss ending at
12.0459 and `EXIT_CODE=0`, but reached only worker0 57,516 tok/s average /
7,189 tok/s/chip / 41.2% MFU and worker1 57,495 tok/s average / 7,187
tok/s/chip / 41.2% MFU, with a best observed steady step of 57,663 tok/s.
Profile files were written under
`2026-06-04-qwen3-jax-v191-no-mlp-sac-no-scan-tokamax-exp-sched/plugins/profile/2026_06_04_22_50_36/`.
Device-side `jit_train_step(6674237032253463936)` spans validate the
regression: `gke-tpu-964065d9-9vh8` averaged 4.572703 s / 57,328 tok/s and
`gke-tpu-964065d9-c180` averaged 4.562751 s / 57,453 tok/s. Fresh optimized
train-step HLO is 19,043,039 bytes with SHA256
`4a924f0315e80fecb00e1484b43911ac902f8112d8f447ec50baf75085aea731`, 27.66 GiB
memory, and text counts all-gather 7,039, all-reduce 1,148,
collective-permute 6, convolution 3,071, custom-call 2,806, copy 21,271,
fusion 31,686. Do not carry Tokamax experimental scheduling in the no-scan
frontier.

## [2026-06-04] loop-iteration | v190-no-mlp-intermediate-sac-no-scan-bkvdkv1024 on 8B/v6e-8: refuted (55,489 tok/s, 39.8% MFU; changed HLO but much slower)

Workload `alekseyv-qwen3-v190-noscan-bkvdkv1024` retested smaller backward DKV
Splash KV tiles on the confirmed no-scan frontier with
`SPLASH_BKV_DKV=1024 SPLASH_BKV_DKV_COMPUTE=1024`, leaving forward tiling at
the default `bkv=1024`. The run completed cleanly with normal loss ending at
12.0460 and `EXIT_CODE=0`, but regressed to 55,489 tok/s average / 6,936
tok/s/chip / 39.8% MFU on worker0, with best observed steady step only 55,648
tok/s. Profile files were written under
`2026-06-04-qwen3-jax-v190-no-mlp-sac-no-scan-bkvdkv1024/plugins/profile/2026_06_04_22_42_05/`.
Device-side `jit_train_step(14633774807237963760)` spans validate the
regression: `gke-tpu-964065d9-9vh8` averaged 4.739211 s / 55,314 tok/s and
`gke-tpu-964065d9-c180` averaged 4.730082 s / 55,421 tok/s. Fresh optimized
train-step HLO is 19,050,551 bytes with SHA256
`362c1236acda6c60a5cea43826813a9c8b6545f641d00632d9ac75ee87935ace`, 27.66 GiB
memory, and text counts all-gather 7,039, all-reduce 1,148,
collective-permute 6, convolution 3,071, custom-call 2,809, copy 21,331,
fusion 31,686. Do not carry smaller DKV KV tiles in the no-scan frontier.

## [2026-06-04] loop-iteration | v189-no-mlp-intermediate-sac-no-scan-bkv512-compute512 on 8B/v6e-8: refuted (56,837 tok/s, 40.8% MFU; changed HLO but more fusion/copy/custom-call pressure)

Workload `alekseyv-qwen3-v189-noscan-bkv512c512` reran the intended v188
hypothesis with the valid Tokamax Splash pair
`SPLASH_BKV=512 SPLASH_BKV_COMPUTE=512`. The run completed cleanly with normal
loss ending at 12.0459 and `EXIT_CODE=0`, but regressed to 56,837 tok/s average
/ 7,105 tok/s/chip / 40.8% MFU on worker0, with best observed steady step only
56,984 tok/s. Profile files were written under
`2026-06-04-qwen3-jax-v189-no-mlp-sac-no-scan-bkv512-compute512/plugins/profile/2026_06_04_22_33_58/`.
Device-side `jit_train_step(16781811678069776883)` spans validate the
regression: `gke-tpu-964065d9-9vh8` averaged 4.629782 s / 56,621 tok/s and
`gke-tpu-964065d9-c180` averaged 4.616597 s / 56,783 tok/s, far slower than the
v183 no-scan 4.497 s class. Fresh optimized train-step HLO is 19,042,499 bytes
with SHA256 `668dfe34f5eef201487104c076b15c8a7e999c5be616c7d26948fbf031509a1d`,
27.66 GiB memory, and text counts all-gather 7,039, all-reduce 1,148,
collective-permute 6, convolution 3,071, custom-call 2,806, copy 21,271,
fusion 31,686. Do not carry smaller forward Splash KV tiles in the no-scan
frontier.

## [2026-06-04] loop-iteration | v188-no-mlp-intermediate-sac-no-scan-bkv512 on 8B/v6e-8: invalid launch (Tokamax BKV/BKV_COMPUTE mismatch)

Workload `alekseyv-qwen3-v188-noscan-bkv512` attempted to retest smaller
forward Splash KV tiles on the confirmed no-scan frontier by setting
`SPLASH_BKV=512`. The launch was invalid: Tokamax Splash kept
`bkv_compute=1024` and rejected the configuration before train-step
compilation with `ValueError: bkv=512 must be a multiple of bkv_compute=1024`.
The job exited with `EXIT_CODE=1`, produced no profile, and emitted no
train-step HLO. This is not a performance result. Rerun the intended hypothesis
with both `SPLASH_BKV=512` and `SPLASH_BKV_COMPUTE=512`.

## [2026-06-04] loop-iteration | v187-no-mlp-intermediate-sac-no-scan-hto48 on 8B/v6e-8: refuted/tie (58,324 tok/s, 41.8% MFU; byte-identical HLO to frontier)

Workload `alekseyv-qwen3-v187-noscan-hto48` retested
`--xla_tpu_host_transfer_overlap_limit=48` on the confirmed no-scan frontier,
changing only that runtime overlap limit from 24. The run completed cleanly
with normal loss ending at 12.0459 and `EXIT_CODE=0`, reaching 58,324 tok/s
average / 7,290 tok/s/chip / 41.8% MFU, with a best observed step of 58,440
tok/s. Profile files were written under
`2026-06-04-qwen3-jax-v187-no-mlp-sac-no-scan-hto48/plugins/profile/2026_06_04_22_22_15/`.
Optimized train-step HLO is byte-identical to v182/v183/v186:
19,043,435 bytes with SHA256
`1ba7015c0111bd3ce0a8a9238ca32e25359e7f45c92033b921cf321da227dc48`, and memory
stays 27.66 GiB. The best step is still just below v182's 58,443 best observed
step, so this is a same-HLO tie/no-gain. Keep host-transfer overlap limit 24.

## [2026-06-04] loop-iteration | v186-no-mlp-intermediate-sac-no-scan-shmem95 on 8B/v6e-8: refuted/tie (58,104 tok/s, 41.7% MFU; byte-identical HLO to frontier)

Workload `alekseyv-qwen3-v186-noscan-shmem95` retested
`--xla_tpu_scheduler_percent_shared_memory_limit=95` on the confirmed no-scan
frontier, changing only that scheduler shared-memory cap from 100. The run
completed cleanly with normal loss ending at 12.0459 and `EXIT_CODE=0`, but
reached only 58,104 tok/s average / 7,263 tok/s/chip / 41.7% MFU, with a best
observed step of 58,210 tok/s. Profile files were written under
`2026-06-04-qwen3-jax-v186-no-mlp-sac-no-scan-shmem95/plugins/profile/2026_06_04_22_10_33/`.
Optimized train-step HLO is byte-identical to v182/v183:
19,043,435 bytes with SHA256
`1ba7015c0111bd3ce0a8a9238ca32e25359e7f45c92033b921cf321da227dc48`, and memory
stays 27.66 GiB. Do not carry shared-memory limit 95; it is a same-HLO
no-gain/regression on the current no-scan frontier.

## [2026-06-04] loop-iteration | v185-no-mlp-intermediate-sac-no-scan-vmem99328 on 8B/v6e-8: refuted (57,815 tok/s, 41.5% MFU; lower no-scan VMEM changes HLO but regresses)

Workload `alekseyv-qwen3-v185-noscan-vmem99328` retuned scoped VMEM on the
confirmed v182/v183 no-scan frontier, changing only
`--xla_tpu_scoped_vmem_limit_kib=100352` to `99328`. An initial v185 JobSet was
accidentally submitted to a stale Europe v5p kubectl context and remained
pending; it was deleted before running, v6e credentials were refreshed, and the
workload was resubmitted to the correct
`<your-gke-context>` context. The v6e run
completed cleanly with normal loss ending at 12.0460 and `EXIT_CODE=0`, but
regressed to 57,815 tok/s average / 7,227 tok/s/chip / 41.5% MFU, with a best
observed step of only 58,004 tok/s. Profile files were written under
`2026-06-04-qwen3-jax-v185-no-mlp-sac-no-scan-vmem99328/plugins/profile/2026_06_04_21_59_51/`.
Fresh optimized train-step HLO changed to 19,043,751 bytes with SHA256
`6e904d246b2893d10d93b978661be51bdaf2aa04f4ee5d1305fadfb99a971d2f`, and memory
remained 27.66 GiB. Do not carry VMEM 99328. Together with v184's no-gain
100864 result, keep scoped VMEM 100352 for the confirmed no-scan schedule.

## [2026-06-04] loop-iteration | v184-no-mlp-intermediate-sac-no-scan-vmem100864 on 8B/v6e-8: refuted/tie (58,311 tok/s worker0 avg, 58,421 best, 41.8% MFU; distinct HLO but no frontier gain)

Workload `alekseyv-qwen3-v184-noscan-vmem100864` retuned scoped VMEM on the
confirmed v182/v183 no-scan frontier, changing only
`--xla_tpu_scoped_vmem_limit_kib=100352` to `100864`. It completed cleanly on
worker0 with normal loss ending at 12.0459 and `EXIT_CODE=0`, reaching 58,311
tok/s average / 7,289 tok/s/chip / 41.8% MFU, with a best observed step of
58,421 tok/s. The worker1 pod was cleaned up before delayed stdout capture, but
profile traces were written under
`2026-06-04-qwen3-jax-v184-no-mlp-sac-no-scan-vmem100864/plugins/profile/2026_06_04_21_50_04/`.
Device-side `jit_train_step` spans show `g30s` average 4.498196 s / 58,278
tok/s and `rkr2` average 4.510664 s / 58,116 tok/s including a profile-edge
span; excluding that edge, `rkr2` is also in the 4.498 s class. Fresh optimized
train-step HLO changed slightly to 19,043,156 bytes with SHA256
`8997b4c6b68aa332204162465ff4801465786cb170952864b3614ad637fcf16b`, versus
v182's `1ba7015c...` 19,043,435-byte HLO, and memory stayed essentially flat at
27.65 GiB. Since throughput remains inside the confirmed no-scan band and below
v182's 58,443 best observed step, do not carry VMEM 100864; keep 100352 KiB.

## [2026-06-04] loop-iteration | v183-no-mlp-intermediate-sac-no-scan-rerun on 8B/v6e-8: supported (profile-derived 58,288-58,291 tok/s steady avg, 58,302 best span; no-scan frontier confirmed)

Workload `alekseyv-qwen3-v183-noscan-rerun` exactly reran v182 by reusing the
`v169-no-mlp-sac` image with `--use_scan=False` and all other runtime flags,
shape, environment, and XLA flags fixed. The JobSet and pods were cleaned up
before delayed log capture, so stdout throughput and final loss lines were not
recoverable. The profile artifacts were written under
`2026-06-04-qwen3-jax-v183-no-mlp-sac-no-scan-rerun/plugins/profile/2026_06_04_21_37_36/`
and contain device-side `jit_train_step(403931025898375493)` spans on both
hosts. Trace-derived steady timing is 4.497381 s / 58,288 tok/s on `g30s` and
4.497194 s / 58,291 tok/s on `rkr2` after excluding a single `rkr2`
profiler-window edge span; best per-device spans are 58,302 and 58,301 tok/s.
No fresh HLO dump was emitted, consistent with persistent-cache replay of v182's
19,043,435-byte unrolled no-scan HLO
(`1ba7015c0111bd3ce0a8a9238ca32e25359e7f45c92033b921cf321da227dc48`). Per user
request, CPU equivalence was rerun in the v169 dependency image and passed:
logits max delta `2.682e-07`, loss delta `0`, all 25 gradients OK, max gradient
delta `7.823e-08`. Carry `--use_scan=False` as the confirmed frontier on top of
the v169 no-MLP-intermediate-sac schedule; account for much larger HLO and
compile/warmup cost.

## [2026-06-04] loop-iteration | v182-no-mlp-intermediate-sac-no-scan on 8B/v6e-8: provisional supported (58,330 tok/s best-worker avg, 58,443 best, 41.8% MFU; unrolled graph now fits)

Workload `alekseyv-qwen3-v182-noscan-mlpsac` reused the confirmed
`v169-no-mlp-sac` image and changed only `--use_scan=True` to
`--use_scan=False`, keeping remat, tokamax Splash, MaxText CE, activation
sharding, batch/sequence shape, and the v170 XLA flag stack fixed. The old v042
no-scan OOM did not reproduce on this newer schedule family. v182 completed
cleanly on both workers with normal loss ending at 12.0459 and `EXIT_CODE=0`,
reaching worker0 58,307 tok/s / 7,288 tok/s/chip / 41.8% MFU and worker1
58,330 tok/s / 7,291 tok/s/chip / 41.8% MFU. Best observed steady step was
58,443 tok/s, above v169/v170's 58,051 best observed and 57,935-57,947
repeated-average band. Compile/warmup is much heavier: step 0 took 134.6 s and
step 1 took 10.7 s, but steady steps settle near 4.49 s. Profile files were
written under
`2026-06-04-qwen3-jax-v182-no-mlp-sac-no-scan/plugins/profile/2026_06_04_21_31_37/`.
Fresh optimized train-step HLO is the expected unrolled program: 19,043,435
bytes with SHA256
`1ba7015c0111bd3ce0a8a9238ca32e25359e7f45c92033b921cf321da227dc48`; text
counts are all-gather 4,656, all-reduce 960, collective-permute 3, convolution
2,096, custom-call 1,476, copy 9,401, fusion 18,276. Memory report rises to
27.66 GiB from v169's 26.86 GiB but still fits. Treat v182 as provisional
supported and rerun exactly before carrying the no-scan frontier.

## [2026-06-04] loop-iteration | v181-no-mlp-intermediate-sac-tokamax-ce on 8B/v6e-8: refuted (53,280 tok/s best-worker avg, 38.2% MFU; lower HBM but CE custom-call path too slow)

Workload `alekseyv-qwen3-v181-tokamaxce-mlpsac` reused the confirmed
`v169-no-mlp-sac` image and changed only the CE implementation from MaxText CE
to `--use_tokamax_ce=True --tokamax_ce_impl=mosaic_tpu --use_maxtext_ce=False`.
The run completed cleanly with finite normal loss ending at 12.0526 and
`EXIT_CODE=0`, reaching worker0 53,280 tok/s / 6,660 tok/s/chip / 38.2% MFU and
worker1 53,265 tok/s / 6,658 tok/s/chip / 38.2% MFU. Best observed steady step
was 53,515 tok/s. This is much better than old v031 Tokamax CE (45,461 tok/s /
32.6% MFU) because the surrounding schedule is better, but still far below
v169/v170's 57,935-57,947 repeated-average band and 58,051 best observed.
Profile files were written under
`2026-06-04-qwen3-jax-v181-no-mlp-sac-tokamax-ce/plugins/profile/2026_06_04_21_20_51/`.
Fresh train-step HLO dumps `module_0109` and `module_0308` are byte-identical at
791,173 bytes with SHA256
`caca667e8d8e01a478379131a36d943f4007cbd428cdfab5db7ee5b3d3b71ac3`; text
counts are all-gather 139, all-reduce 36, collective-permute 3, convolution 58,
custom-call 43, copy 423, fusion 558, with `tokamax` metadata present. The
memory report drops to 23.15 GiB from v169's 26.86 GiB, but the CE custom-call
path remains too slow. Do not carry Tokamax CE.

## [2026-06-04] loop-iteration | v180-no-mlp-intermediate-sac-offload-remat on 8B/v6e-8: refuted (49,761 tok/s best-worker avg, 35.7% MFU; offload overhead dominates)

Workload `alekseyv-qwen3-v180-offload-mlpsac` reused the confirmed
`v169-no-mlp-sac` image and changed only the runtime remat policy to
`--use_remat=True --offload_remat=True`. Per user request, CPU equivalence was
rerun in the v169 dependency image with the checked-out JAX lane mounted:
`==== EQUIVALENCE PASS ====`, logits max delta `2.682e-07`, loss delta `0`, all
25 gradients matched, max gradient delta `7.823e-08`. The TPU run completed
cleanly with normal loss ending at 12.0457 and `EXIT_CODE=0`, but regressed to
worker0 49,761 tok/s / 6,220 tok/s/chip / 35.7% MFU and worker1 49,743 tok/s /
6,218 tok/s/chip / 35.7% MFU. Best observed steady step was 49,888 tok/s, far
below v169/v170's 58,051 best observed and 57,935-57,947 repeated-average band.
Profile files were written under
`2026-06-04-qwen3-jax-v180-no-mlp-sac-offload-remat/plugins/profile/2026_06_04_21_11_31/`.
Fresh optimized train-step HLO is 1,303,770 bytes with SHA256
`262e50f1fe353d98136e328b51018e20df05b4ee151ee27ecb6b1403020501c1`; text
counts are all-gather 226, all-reduce 62, collective-permute 3, convolution
110, custom-call 78, copy 694, fusion 964. The memory report rises to 54.71 GiB
total bytes used, with 31.50 GiB and 17.46 GiB preallocated temps, versus
v169's 26.86 GiB. Selective host offload avoids v179's compile-time 340.87 GiB
HBM OOM, but it is much slower; do not carry `--offload_remat=True`.

## [2026-06-04] loop-iteration | v179-no-mlp-intermediate-sac-no-remat on 8B/v6e-8: invalid/refuted (compile-time HBM OOM; 340.87 GiB required)

Workload `alekseyv-qwen3-v179-noremat-mlpsac` reused the confirmed
`v169-no-mlp-sac` image and changed only the runtime flag from
`--use_remat=True` to `--use_remat=False`. Startup confirmed the rest of the
stack was fixed and no `[remat]` line was printed. Both workers failed during
train-step compilation with `RESOURCE_EXHAUSTED`: XLA reported 340.87 GiB of
31.25 GiB HBM required, exceeded by 309.62 GiB, and `EXIT_CODE=1`. The memory
report shows 346.62 GiB total bytes used with a 340.87 GiB preallocated temp.
The largest allocations are repeated scan-stacked activations: six 27.00 GiB
`bf16[36,4,8192,12288]` temps plus multiple 18.00 GiB
`f32[36,4,8192,4096]` and `f32[36,4,8192,32,128]` temps. OOM HLO artifacts were
written under
`2026-06-04-qwen3-jax-v179-no-mlp-sac-no-remat/hlo/`; optimized HLO is 701,833
bytes with SHA256 `f4dd4b1813a5c635e443c93d58e1ee927ceaa819615757aa52e253be51d73950`
and text counts all-gather 59, all-reduce 36, collective-permute 3, convolution
61, custom-call 41, copy 272, fusion 436. Keep `--use_remat=True`; no-remat is
not viable without a new activation-memory mechanism.

## [2026-06-04] loop-iteration | v178-no-target-grad-maxtext-ce on 8B/v6e-8: refuted/no-op (57,770 tok/s best-worker avg, 41.4% MFU; CE target cotangent already DCE'd)

Workload `alekseyv-qwen3-v178-ce-notarget` changed only the MaxText CE custom
VJP to return `None` for the one-hot target cotangent and avoid saving
`log_softmax` for backward. CPU equivalence passed before launch: logits max
delta `2.682e-07`, loss delta `0`, all 25 gradients matched, max gradient delta
`7.823e-08`. The TPU run completed cleanly with normal loss ending at 12.0461
and `EXIT_CODE=0`, reaching worker0 57,770 tok/s / 7,221 tok/s/chip / 41.4% MFU
and worker1 57,749 tok/s / 7,219 tok/s/chip / 41.4% MFU. Best observed steady
step was 57,941 tok/s, below v169/v170's 58,051 best observed and not above the
confirmed repeated average band. Profile files were written under
`2026-06-04-qwen3-jax-v178-no-target-grad-maxtext-ce/plugins/profile/2026_06_04_20_58_50/`.
No HLO objects were emitted under the v178 prefix, and the short step-0 compile
behavior is consistent with persistent-cache reuse of the v169/v170 train-step
program. The original target-cotangent/log-softmax path was already dead-code
eliminated from the compiled training graph; do not carry this as a performance
mechanism.

## [2026-06-04] loop-iteration | v177-no-mlp-intermediate-sac-lmhead-replicated on 8B/v6e-8: refuted (56,449 tok/s best-worker avg, 40.5% MFU; lm-head replication still harmful)

Workload `alekseyv-qwen3-v177-lmrep-mlpsac` retested fully replicated
`lm_head.weight` on top of the confirmed v169 no-MLP-intermediate `_sac`
schedule. CPU equivalence passed before launch: logits max delta `2.682e-07`,
loss delta `0`, all 25 gradients matched, max gradient delta `7.823e-08`. The
TPU run completed cleanly on both workers with normal loss ending at 12.0461 and
`EXIT_CODE=0`, reaching worker0 56,449 tok/s / 7,056 tok/s/chip / 40.5% MFU and
worker1 56,423 tok/s / 7,053 tok/s/chip / 40.5% MFU. Best observed steady step
was only 56,839 tok/s, far below the confirmed v169/v170 frontier. Profile files
were written under
`2026-06-04-qwen3-jax-v177-no-mlp-sac-lmhead-repl/plugins/profile/2026_06_04_20_49_44/`.
Fresh optimized train-step HLO is 780,640 bytes with SHA256
`ac73f41c9703b91efddab337d55cbd99061bf718f14d92ec718904e7006de4ae`; text
counts are all-gather 136, all-reduce 36, collective-permute 3, convolution 66,
custom-call 41, copy 310, fusion 752. Full lm-head replication remains a
wall-time regression even in the v169 schedule family; do not carry it.

## [2026-06-04] loop-iteration | v176-no-mlp-intermediate-sac-replicate-rmsnorm on 8B/v6e-8: refuted (57,718 tok/s best-worker avg, 41.4% MFU; fewer all-gathers but slower wall time)

Workload `alekseyv-qwen3-v176-repnorm-mlpsac` retested the v165/v166
replicated-RMSNorm parameter sharding lever on top of the confirmed v169
no-MLP-intermediate `_sac` schedule. CPU equivalence passed before launch:
logits max delta `2.682e-07`, loss delta `0`, all 25 gradients matched, max
gradient delta `7.823e-08`. The TPU run completed cleanly on both workers with
normal loss ending at 12.0461 and `EXIT_CODE=0`, reaching worker0 57,718 tok/s /
7,215 tok/s/chip / 41.4% MFU and worker1 57,702 tok/s / 7,213 tok/s/chip /
41.4% MFU. Best observed steady step was 57,868 tok/s, below the confirmed
v169/v170 frontier band of 57,935-57,947 tok/s repeated average and 58,051 best
observed. Profile files were written under
`2026-06-04-qwen3-jax-v176-no-mlp-sac-replicate-rmsnorm/plugins/profile/2026_06_04_20_41_16/`.
Fresh optimized train-step HLO is 778,035 bytes with SHA256
`d6839ab4eaed88451aa554ba3b0a53deda0386a08f3a9ae4e5eee9fa6c05ca0f`; text
counts are all-gather 116, all-reduce 36, collective-permute 3, convolution 66,
custom-call 41, copy 324, fusion 753. The all-gather reduction from v169's 140
does not improve wall time, so do not carry replicated RMSNorm params in the
v169 schedule family.

## [2026-06-04] loop-iteration | v156-shard-acts-s8k-bs4-tokamax-splash-collective-matmul-none-vmem100352-bkv512 on 8B/v6e-8: refuted (56,541 tok/s best worker avg, 40.5% MFU; smaller forward KV tiles regress)

Workload `alekseyv-qwen3-v156-bs4-bkv512` kept the confirmed v154 scoped-VMEM 100352 frontier fixed and added only `SPLASH_BKV=512 SPLASH_BKV_COMPUTE=512`. The Splash config confirmed `bq=2048 bkv=512 bq_dkv=2048 bkv_dkv=2048`. It completed cleanly on both workers with normal loss ending at 12.0458, but regressed to worker0 56,541 tok/s / 7,068 tok/s/chip / 40.5% MFU and worker1 56,521 tok/s / 7,065 tok/s/chip / 40.5% MFU. Profile files were written at `2026-06-04-qwen3-jax-v156-shard-acts-s8k-bs4-tokamax-splash-collective-matmul-none-vmem100352-bkv512/plugins/profile/2026_06_04_17_49_22`, but the local XProf server was unavailable after restart. Fresh optimized train-step HLO changed to 783,923 bytes with content hash `9b5c23ed2a0035b7352e87ce0a5dcbbd3e0c89bf45c6d5a66dc4b9e061f044f4`; compiled memory remained 26.86 GiB. Do not carry `SPLASH_BKV=512`; restore default forward KV tiling and keep scoped VMEM 100352.

## [2026-06-04] loop-iteration | v155-shard-acts-s8k-bs4-tokamax-splash-collective-matmul-none-vmem101376 on 8B/v6e-8: refuted/no-gain (57,588 tok/s worker0 avg, 41.3% MFU; same HLO as v153)

Workload `alekseyv-qwen3-v155-bs4-vmem101376` kept the confirmed v154 scoped-VMEM 100352 frontier fixed and changed only `--xla_tpu_scoped_vmem_limit_kib=101376`. Worker0 completed cleanly with normal loss ending at 12.0463, but reached only 57,588 tok/s / 7,198 tok/s/chip / 41.3% MFU, below the confirmed v153/v154 band. The JobSet/pods were removed before worker1 logs could be fetched, but profile files for both TPU hosts were written at `2026-06-04-qwen3-jax-v155-shard-acts-s8k-bs4-tokamax-splash-collective-matmul-none-vmem101376/plugins/profile/2026_06_04_17_40_29`; immediate XProf HTTP/MCP queries still returned no-xplane/500. Fresh optimized train-step HLO was emitted and is byte-identical to v153: 783,949 bytes, content hash `3734fae066fc35c4b9c6c4cce9b0015413ab4513c34a09ba33a0e1862bff3a3d`, compiled memory 26.86 GiB, with the flagfile confirming `xla_tpu_scoped_vmem_limit_kib=101376`. Do not carry 101376; keep 100352 as the confirmed frontier and move to a new mechanism rather than continuing the high-side VMEM bracket.

## [2026-06-04] loop-iteration | v154-shard-acts-s8k-bs4-tokamax-splash-collective-matmul-none-vmem100352-rerun on 8B/v6e-8: supported (57,754 tok/s best worker avg, 41.4% MFU; scoped VMEM 100352 confirmed)

Workload `alekseyv-qwen3-v154-bs4-vmem100352r` exactly reran v153's scoped-VMEM 100352 setting on the batch-4/global-32 collective-matmul-none stack. It completed cleanly on both workers with normal loss ending at 12.0461, reaching worker0 57,754 tok/s / 7,219 tok/s/chip / 41.4% MFU and worker1 57,724 tok/s / 7,215 tok/s/chip / 41.4% MFU, with best observed steady steps of 57,862 and 57,857 tok/s. Profile files were written at `2026-06-04-qwen3-jax-v154-shard-acts-s8k-bs4-tokamax-splash-collective-matmul-none-vmem100352-rerun/plugins/profile/2026_06_04_17_29_23`, but immediate XProf HTTP/MCP indexing returned no-xplane/500 despite the GCS `.xplane.pb` files being present. No fresh HLO dump appeared because this exact rerun hit the persistent compile cache; use v153's train-step program `10728347668315821976`, 783,949-byte HLO, content hash `3734fae066fc35c4b9c6c4cce9b0015413ab4513c34a09ba33a0e1862bff3a3d`, and 26.86 GiB compiled memory report. Carry scoped VMEM 100352 as the confirmed frontier. Next test the upper midpoint 101376 KiB between confirmed 100352 and refuted 102400.

## [2026-06-04] loop-iteration | v153-shard-acts-s8k-bs4-tokamax-splash-collective-matmul-none-vmem100352 on 8B/v6e-8: provisional supported (57,757 tok/s best worker avg, 41.4% MFU; midpoint recovers and improves beyond 98304)

Workload `alekseyv-qwen3-v153-bs4-vmem100352` kept the confirmed v151 scoped-VMEM 98304 frontier fixed after v152's 102400 overshoot and changed only `--xla_tpu_scoped_vmem_limit_kib=100352`. It completed cleanly on both workers with normal loss ending at 12.0461, reaching worker0 57,757 tok/s / 7,220 tok/s/chip / 41.4% MFU and worker1 57,734 tok/s / 7,217 tok/s/chip / 41.4% MFU, with best observed steady steps of 57,886 and 57,888 tok/s. XProf run `2026-06-04-qwen3-jax-v153-shard-acts-s8k-bs4-tokamax-splash-collective-matmul-none-vmem100352/2026_06_04_17_22_28` shows total time 109,331.3 ms, step average 1954.6 ms, MXU 65.6%, convolution fusion 52.1%, data formatting 2.6%, custom fusion 1.7%, and idle 0.3%. Fresh train-step HLO changes program hash to `10728347668315821976`; optimized HLO is 783,949 bytes, content hash `3734fae066fc35c4b9c6c4cce9b0015413ab4513c34a09ba33a0e1862bff3a3d`, and compiled memory is 26.86 GiB. Treat v153 as provisional supported and rerun exactly before carrying scoped VMEM 100352.

## [2026-06-04] loop-iteration | v152-shard-acts-s8k-bs4-tokamax-splash-collective-matmul-none-vmem102400 on 8B/v6e-8: refuted (57,410 tok/s best worker avg, 41.2% MFU; 102400 is a valid upper overshoot)

Workload `alekseyv-qwen3-v152-bs4-vmem102400` kept the confirmed v151 scoped-VMEM 98304 frontier fixed and changed only `--xla_tpu_scoped_vmem_limit_kib=102400`. It completed cleanly on both workers with normal loss ending at 12.0461, but regressed to worker0 57,410 tok/s / 7,176 tok/s/chip / 41.2% MFU and worker1 57,385 tok/s / 7,173 tok/s/chip / 41.1% MFU, with best observed steady steps of 57,516 and 57,513 tok/s. XProf run `2026-06-04-qwen3-jax-v152-shard-acts-s8k-bs4-tokamax-splash-collective-matmul-none-vmem102400/2026_06_04_17_09_36` shows total time 110,002.1 ms, step average 1966.8 ms, MXU 65.2%, convolution fusion 52.3%, custom-call approximately 26.7%, DKV 14.1%, forward residuals 12.6% combined, data formatting 2.6%, custom fusion 1.6%, and idle 0.3%. Fresh train-step HLO changes program hash to `12059987095245408810`; optimized HLO is 782,971 bytes, content hash `ed58479a073ffd6c56449ddaf4da1cd5799d781d25011f602b379bf57fcb897f`, and compiled memory is 26.86 GiB. Do not carry scoped VMEM 102400; keep 98304 as the frontier. Next probe the midpoint 100352 KiB to locate the high-side edge.

## [2026-06-04] loop-iteration | v151-shard-acts-s8k-bs4-tokamax-splash-collective-matmul-none-vmem98304-rerun on 8B/v6e-8: supported (57,572 tok/s best worker avg, 41.3% MFU; scoped VMEM 98304 repeats and becomes the frontier)

Workload `alekseyv-qwen3-v151-bs4-vmem98304r` exactly reran v150's scoped-VMEM 98304 setting on the batch-4/global-32 collective-matmul-none stack. It completed cleanly on both workers with normal loss ending at 12.0460, reaching worker0 57,572 tok/s / 7,196 tok/s/chip / 41.3% MFU and worker1 57,546 tok/s / 7,193 tok/s/chip / 41.3% MFU, with best observed steady steps of 57,707 and 57,704 tok/s. XProf run `2026-06-04-qwen3-jax-v151-shard-acts-s8k-bs4-tokamax-splash-collective-matmul-none-vmem98304-rerun/2026_06_04_16_58_05` shows total time 109,695.4 ms, step average 1961.2 ms, MXU 65.3%, custom-call 26.8%, DKV 14.1%, forward residuals 12.6% combined, convolution fusion 52.2%, loop fusion 10.5%, all-reduce-scatter fusion 3.4%, data formatting 2.6%, custom fusion 1.5%, and idle 0.3%. XProf reports the same train-step program as v150, `6628478088121256461`; no fresh HLO was emitted due to persistent compile cache, so use v150's 787,060-byte HLO, content hash `40d497c2bad650f5da214fd82b9636af17633b718094f8561a4b5a9ba7c2fdd9`, and 26.86 GiB memory report. Carry scoped VMEM 98304 as the new current frontier. Next test 102400 KiB to continue the upper bracket.

## [2026-06-04] loop-iteration | v150-shard-acts-s8k-bs4-tokamax-splash-collective-matmul-none-vmem98304 on 8B/v6e-8: provisional supported (57,578 tok/s best worker avg, 41.3% MFU; 98304 gives a large step-time drop)

Workload `alekseyv-qwen3-v150-bs4-vmem98304` kept the confirmed v149 scoped-VMEM 94208 frontier fixed and changed only `--xla_tpu_scoped_vmem_limit_kib=98304`. It completed cleanly on both workers with normal loss ending at 12.0460, reaching worker0 57,578 tok/s / 7,197 tok/s/chip / 41.3% MFU and worker1 57,561 tok/s / 7,195 tok/s/chip / 41.3% MFU, with best observed steady steps of 57,682 and 57,685 tok/s. XProf run `2026-06-04-qwen3-jax-v150-shard-acts-s8k-bs4-tokamax-splash-collective-matmul-none-vmem98304/2026_06_04_16_52_56` shows total time 109,644.8 ms, step average 1959.9 ms, MXU 65.3%, custom-call 26.8%, DKV 14.1%, forward residuals 12.7% combined, convolution fusion 52.3%, loop fusion 10.5%, all-reduce-scatter fusion 3.4%, data formatting 2.6%, custom fusion 1.5%, and idle 0.3%. Fresh train-step HLO changes program hash to `6628478088121256461`; optimized HLO is 787,060 bytes, content hash `40d497c2bad650f5da214fd82b9636af17633b718094f8561a4b5a9ba7c2fdd9`, and compiled memory is 26.86 GiB. Treat v150 as provisional supported and rerun exactly before carrying scoped VMEM 98304.

## [2026-06-04] loop-iteration | v149-shard-acts-s8k-bs4-tokamax-splash-collective-matmul-none-vmem94208-rerun on 8B/v6e-8: supported (56,768 tok/s best worker avg, 40.7% MFU; scoped VMEM 94208 repeats and becomes the frontier)

Workload `alekseyv-qwen3-v149-bs4-vmem94208r` exactly reran v148's scoped-VMEM 94208 setting on the batch-4/global-32 collective-matmul-none stack. It completed cleanly on both workers with normal loss ending at 12.0460, reaching worker0 56,768 tok/s / 7,096 tok/s/chip / 40.7% MFU and worker1 56,743 tok/s / 7,093 tok/s/chip / 40.7% MFU, with best observed steady steps of 56,856 tok/s on both workers. XProf run `2026-06-04-qwen3-jax-v149-shard-acts-s8k-bs4-tokamax-splash-collective-matmul-none-vmem94208-rerun/2026_06_04_16_43_17` shows total time 111,236.3 ms, step average 1988.9 ms, MXU 65.1%, custom-call 26.4%, DKV 13.9%, forward residuals 12.5% combined, convolution fusion 53.2%, loop fusion 10.3%, all-reduce-scatter fusion 3.4%, data formatting 2.6%, custom fusion 1.3%, and idle 0.3%. XProf reports the same train-step program as v148, `15797032912653574046`; no fresh HLO was emitted due to persistent compile cache, so use v148's 779,591-byte HLO, content hash `e78989b8aa0a3c726e47566c51f7fd42cd5d218e129b2bf0bff2fc5b4f51fc93`, and 26.87 GiB memory report. Carry scoped VMEM 94208 as the new current frontier. Next test 98304 KiB to continue the upper bracket.

## [2026-06-04] loop-iteration | v148-shard-acts-s8k-bs4-tokamax-splash-collective-matmul-none-vmem94208 on 8B/v6e-8: provisional supported (56,773 tok/s best worker avg, 40.7% MFU; higher VMEM point improves again)

Workload `alekseyv-qwen3-v148-bs4-vmem94208` kept the confirmed v147 scoped-VMEM 90112 frontier fixed and changed only `--xla_tpu_scoped_vmem_limit_kib=94208`. It completed cleanly on both workers with normal loss ending at 12.0460, reaching worker0 56,773 tok/s / 7,097 tok/s/chip / 40.7% MFU and worker1 56,750 tok/s / 7,094 tok/s/chip / 40.7% MFU, with best observed steady steps of 56,904 and 56,902 tok/s. XProf run `2026-06-04-qwen3-jax-v148-shard-acts-s8k-bs4-tokamax-splash-collective-matmul-none-vmem94208/2026_06_04_16_37_52` shows total time 111,227.2 ms, step average 1988.6 ms, MXU 65.3%, custom-call 26.4%, DKV 13.9%, forward residuals 12.5% combined, convolution fusion 53.2%, loop fusion 10.3%, all-reduce-scatter fusion 3.4%, data formatting 2.6%, custom fusion 1.3%, and idle 0.3%. Fresh train-step HLO changes program hash to `15797032912653574046`; optimized HLO is 779,591 bytes, content hash `e78989b8aa0a3c726e47566c51f7fd42cd5d218e129b2bf0bff2fc5b4f51fc93`, and compiled memory is 26.87 GiB. Treat v148 as provisional supported and rerun exactly before carrying scoped VMEM 94208.

## [2026-06-04] loop-iteration | v147-shard-acts-s8k-bs4-tokamax-splash-collective-matmul-none-vmem90112-rerun on 8B/v6e-8: supported (56,644 tok/s best worker avg, 40.6% MFU; scoped VMEM 90112 repeats and becomes the frontier)

Workload `alekseyv-qwen3-v147-bs4-vmem90112r` exactly reran v146's scoped-VMEM 90112 setting on the batch-4/global-32 collective-matmul-none stack. It completed cleanly on both workers with normal loss ending at 12.0460, reaching worker0 56,644 tok/s / 7,081 tok/s/chip / 40.6% MFU and worker1 56,618 tok/s / 7,077 tok/s/chip / 40.6% MFU, with best observed steady steps of 56,785 and 56,784 tok/s. XProf run `2026-06-04-qwen3-jax-v147-shard-acts-s8k-bs4-tokamax-splash-collective-matmul-none-vmem90112-rerun/2026_06_04_16_28_01` shows total time 111,507.5 ms, step average 1993.9 ms, MXU 64.8%, custom-call 26.3%, DKV 13.9%, forward residuals 12.4% combined, convolution fusion 53.5%, loop fusion 10.3%, all-reduce-scatter fusion 3.4%, data formatting 2.6%, custom fusion 1.1%, and idle 0.3%. XProf reports the same train-step program as v146, `13912291835542571918`; no fresh HLO was emitted due to persistent compile cache, so use v146's 787,246-byte HLO, content hash `eec5d587c0d511d4a7c6eb30dbdacbe4563121cc1321870f957d686be63b0ea6`, and 26.87 GiB memory report. Carry scoped VMEM 90112 as the new current frontier. Next test a higher scoped-VMEM point to bracket the upper side.

## [2026-06-04] loop-iteration | v146-shard-acts-s8k-bs4-tokamax-splash-collective-matmul-none-vmem90112 on 8B/v6e-8: provisional supported (56,635 tok/s best worker avg, 40.6% MFU; upper VMEM point slightly improves frontier)

Workload `alekseyv-qwen3-v146-bs4-vmem90112` kept the confirmed v144/v145 scoped-VMEM 86016 frontier fixed and changed only `--xla_tpu_scoped_vmem_limit_kib=90112`. It completed cleanly on both workers with normal loss ending near 12.0460, reaching worker0 56,635 tok/s / 7,079 tok/s/chip / 40.6% MFU and worker1 56,624 tok/s / 7,078 tok/s/chip / 40.6% MFU, with best observed steady steps of 56,763 and 56,764 tok/s. XProf run `2026-06-04-qwen3-jax-v146-shard-acts-s8k-bs4-tokamax-splash-collective-matmul-none-vmem90112/2026_06_04_16_19_46` shows total time 111,433.5 ms, step average 1990.9 ms, MXU 64.9%, custom-call 26.3%, DKV 13.9%, forward residuals 12.4% combined, convolution fusion 53.5%, loop fusion 10.3%, all-reduce-scatter fusion 3.4%, and idle 0.2%. Fresh train-step HLO changes program hash to `13912291835542571918`; optimized HLO is 787,246 bytes, content hash `eec5d587c0d511d4a7c6eb30dbdacbe4563121cc1321870f957d686be63b0ea6`, and compiled memory is 26.87 GiB. Treat v146 as provisional supported and rerun exactly before carrying scoped VMEM 90112.

## [2026-06-04] loop-iteration | v145-shard-acts-s8k-bs4-tokamax-splash-collective-matmul-none-vmem86016-rerun on 8B/v6e-8: supported (56,603 tok/s best worker avg, 40.6% MFU; scoped VMEM 86016 repeats above old frontier)

Workload `alekseyv-qwen3-v145-bs4-vmem86016r` exactly reran v144's scoped-VMEM 86016 setting on the v127 simplified batch-4/global-32 collective-matmul-none stack. It completed cleanly on both workers with normal loss ending at 12.0458, reaching worker0 56,603 tok/s / 7,075 tok/s/chip / 40.6% MFU and worker1 56,574 tok/s / 7,072 tok/s/chip / 40.6% MFU, with best observed steady steps of 56,708 and 56,697 tok/s. XProf run `2026-06-04-qwen3-jax-v145-shard-acts-s8k-bs4-tokamax-splash-collective-matmul-none-vmem86016-rerun/2026_06_04_16_10_05` shows total time 111,550.1 ms, step average 1994.4 ms, MXU 64.0%, custom-call 26.3%, DKV 13.9%, forward residuals 12.4% combined, convolution fusion 53.7%, loop fusion 10.5%, all-reduce-scatter fusion 3.4%, and idle 0.3%. The train-step program remains `11563614913035853403`; no fresh HLO was emitted due to persistent compile cache, so use v144's 787,753-byte HLO, content hash `aa02d3286a36812ae812c94b82e0bd8fb733cc06e99abf6db17d42aa896cf594`, and 26.88 GiB memory report. Carry scoped VMEM 86016 as the new current frontier. Next test the upper bracket 90112 KiB from this new frontier.

## [2026-06-04] loop-iteration | v144-shard-acts-s8k-bs4-tokamax-splash-collective-matmul-none-vmem86016 on 8B/v6e-8: provisional supported (56,609 tok/s best worker avg, 40.6% MFU; upper VMEM improves average and best step)

Workload `alekseyv-qwen3-v144-bs4-vmem86016` restored the v127 simplified batch-4/global-32 collective-matmul-none stack and changed only `--xla_tpu_scoped_vmem_limit_kib=81920` to `86016`, after v143 refuted the lower 73728 cap. It completed cleanly on both workers with normal loss ending at 12.0458, reaching worker0 56,609 tok/s / 7,076 tok/s/chip / 40.6% MFU and worker1 56,591 tok/s / 7,074 tok/s/chip / 40.6% MFU, with best observed steady steps of 56,728 and 56,727 tok/s. This beats v123's 56,599 best observed step and the v127/v136-v142 best-worker average band. XProf run `2026-06-04-qwen3-jax-v144-shard-acts-s8k-bs4-tokamax-splash-collective-matmul-none-vmem86016/2026_06_04_16_04_23` shows total time 111,492.2 ms, step average 1992.6 ms, MXU 64.2%, custom-call 26.3%, DKV 13.9%, forward residuals 12.4% combined, convolution fusion 53.7%, loop fusion 10.5%, all-reduce-scatter fusion 3.4%, and idle 0.2%. Fresh train-step HLO changes program hash to `11563614913035853403`; optimized HLO is 787,753 bytes, content hash `aa02d3286a36812ae812c94b82e0bd8fb733cc06e99abf6db17d42aa896cf594`, and memory remains 26.88 GiB. Treat v144 as provisional supported and rerun exactly before carrying scoped VMEM 86016 as the durable frontier.

## [2026-06-04] loop-iteration | v143-shard-acts-s8k-bs4-tokamax-splash-collective-matmul-none-vmem73728 on 8B/v6e-8: refuted (55,810 tok/s best worker avg, 40.0% MFU; lower VMEM changes HLO and regresses)

Workload `alekseyv-qwen3-v143-bs4-vmem73728` restored the v127 simplified batch-4/global-32 collective-matmul-none stack after the shared-memory sweep, set shared-memory limit back to 100, and changed only `--xla_tpu_scoped_vmem_limit_kib=81920` to `73728`. It completed cleanly on both workers with normal loss ending near 12.0460, but regressed to worker0 55,810 tok/s / 6,976 tok/s/chip / 40.0% MFU and worker1 55,791 tok/s / 6,974 tok/s/chip / 40.0% MFU, with best observed steady steps only 55,909 and 55,903 tok/s. XProf run `2026-06-04-qwen3-jax-v143-shard-acts-s8k-bs4-tokamax-splash-collective-matmul-none-vmem73728/2026_06_04_15_46_45` shows total time 113,085.9 ms, step average 2021.3 ms, MXU 63.3%, custom-call 26.0%, DKV 13.7%, forward residuals 12.3% combined, convolution fusion 53.7%, loop fusion 10.5%, all-reduce-scatter fusion 3.5%, and idle 0.3%. Fresh train-step HLO changes program hash to `5130325203490098187`; optimized HLO grows to 792,825 bytes, content hash `b973254c94f02a50011375ed7283dd9f28824adbce31186ab89eb21c3d08c9e1`, and memory rises slightly to 26.89 GiB. Do not carry scoped VMEM 73728; next test upper-side scoped VMEM 86016 once in the same post-cmnone family.

## [2026-06-04] loop-iteration | v142-shard-acts-s8k-bs4-tokamax-splash-collective-matmul-none-shmem90 on 8B/v6e-8: refuted/tie (56,452 tok/s best worker avg, 40.5% MFU; shared-memory 90 changes schedule but no gain)

Workload `alekseyv-qwen3-v142-bs4-shmem90` kept the v127 simplified batch-4/global-32 collective-matmul-none stack fixed, restored shared-memory 100's other baseline flags after v141, and changed only `--xla_tpu_scheduler_percent_shared_memory_limit=90`. It completed cleanly on both workers with normal loss ending at 12.0461, reaching worker0 56,452 tok/s / 7,057 tok/s/chip / 40.5% MFU and worker1 56,426 tok/s / 7,053 tok/s/chip / 40.5% MFU, with best observed steady steps of 56,550 and 56,556 tok/s. XProf run `2026-06-04-qwen3-jax-v142-shard-acts-s8k-bs4-tokamax-splash-collective-matmul-none-shmem90/2026_06_04_15_36_10` shows total time 111,840.6 ms, step average 2000.0 ms, MXU 64.1%, custom-call 26.2%, DKV 13.8%, forward residuals 12.4% combined, convolution fusion 53.8%, loop fusion 10.4%, all-reduce-scatter fusion 3.6%, and idle 0.3%. Fresh HLO changes train-step program hash to `8671917837109355706`; optimized HLO is still 787,978 bytes, content hash is `149d09d651b30397ae9474d92bfada657c6d99896fac2686ff580b16b328df5a`, and memory remains 26.88 GiB. A text diff against v141 shows scheduling reorder around copy/all-reduce placement and async slice/custom-call staging, not a math change. Do not carry shared-memory 90; close the post-cmnone shared-memory scalar sweep. Next retest scoped VMEM 73728 on the batch-4 collective-matmul-none family to see whether the newer HLO reacts differently from the old v068/v071 VMEM bracket.

## [2026-06-04] loop-iteration | v141-shard-acts-s8k-bs4-tokamax-splash-collective-matmul-none-shmem95 on 8B/v6e-8: refuted/tie (56,455 tok/s best worker avg, 40.5% MFU; shared-memory 95 keeps frontier HLO but no gain)

Workload `alekseyv-qwen3-v141-bs4-shmem95` kept the v127 simplified batch-4/global-32 collective-matmul-none stack fixed, restored scheduler rerun 3 after v140, and changed only `--xla_tpu_scheduler_percent_shared_memory_limit=95`. It completed cleanly on both workers with normal loss ending at 12.0461, reaching worker0 56,455 tok/s / 7,057 tok/s/chip / 40.5% MFU and worker1 56,429 tok/s / 7,054 tok/s/chip / 40.5% MFU, with best observed steady steps of 56,567 and 56,560 tok/s. XProf run `2026-06-04-qwen3-jax-v141-shard-acts-s8k-bs4-tokamax-splash-collective-matmul-none-shmem95/2026_06_04_15_23_31` shows total time 111,830.8 ms, step average 1999.4 ms, MXU 64.1%, custom-call 26.2%, DKV 13.8%, forward residuals 12.4% combined, convolution fusion 53.8%, loop fusion 10.4%, all-reduce-scatter fusion 3.4%, and idle 0.3%. Fresh HLO confirms the train-step program hash remains `16296087512277873197`; optimized HLO is 787,978 bytes with content hash `3efb113cd72f8c9d08804e6cec46cf9373d83c06fae34ae9e3c27468e6b83fdf`, memory is 26.88 GiB, and the flagfile confirms `xla_tpu_scheduler_percent_shared_memory_limit=95`. The XProf memory-profile endpoint failed for this run, but the HLO memory report and op profile match the surrounding frontier runs. Do not carry shared-memory 95; next test the stronger shared-memory cap 90 on the same v127 baseline.

## [2026-06-04] loop-iteration | v140-shard-acts-s8k-bs4-tokamax-splash-collective-matmul-none-rerun4 on 8B/v6e-8: refuted/tie (56,445 tok/s best worker avg, 40.5% MFU; scheduler rerun4 keeps frontier HLO but no gain)

Workload `alekseyv-qwen3-v140-bs4-rerun4` kept the v127 simplified batch-4/global-32 collective-matmul-none stack fixed, restored `--xla_max_concurrent_host_send_recv=100` after the host send/recv sweep, and changed only `--xla_latency_hiding_scheduler_rerun=4`. It completed cleanly on both workers with normal loss ending at 12.0461, reaching worker0 56,445 tok/s / 7,056 tok/s/chip / 40.5% MFU and worker1 56,424 tok/s / 7,053 tok/s/chip / 40.5% MFU, with best observed steady steps of 56,549 on both workers. XProf run `2026-06-04-qwen3-jax-v140-shard-acts-s8k-bs4-tokamax-splash-collective-matmul-none-rerun4/2026_06_04_15_13_34` shows total time 111,824.4 ms, MXU 64.1%, custom-call 26.2%, DKV 13.8%, forward residuals 12.4% combined, convolution fusion 53.8%, loop fusion 10.4%, all-reduce-scatter fusion 3.4%, idle 0.3%, and peak HBM 26.91 GiB. Fresh HLO confirms the train-step program hash remains `16296087512277873197`; optimized HLO is 787,978 bytes with content hash `3efb113cd72f8c9d08804e6cec46cf9373d83c06fae34ae9e3c27468e6b83fdf`, memory is 26.88 GiB, and the flagfile confirms `xla_latency_hiding_scheduler_rerun=4`. Do not carry rerun4; keep rerun3. Next test a mild scheduler shared-memory cap, 95, on the same v127 baseline.

## [2026-06-04] loop-iteration | v139-shard-acts-s8k-bs4-tokamax-splash-collective-matmul-none-host-sendrecv64 on 8B/v6e-8: refuted/tie (56,439 tok/s best worker avg, 40.5% MFU; sendrecv64 keeps frontier HLO but regresses)

Workload `alekseyv-qwen3-v139-bs4-sendrecv64` kept the v127 simplified batch-4/global-32 collective-matmul-none stack fixed and changed only `--xla_max_concurrent_host_send_recv=64`, restoring the baseline host-transfer overlap 24 and undoing v138's 128 setting. It completed cleanly on both workers with normal loss ending at 12.0461, reaching worker0 56,439 tok/s / 7,055 tok/s/chip / 40.5% MFU and worker1 56,416 tok/s / 7,052 tok/s/chip / 40.4% MFU, with best observed steady steps of 56,553 and 56,557 tok/s. XProf run `2026-06-04-qwen3-jax-v139-shard-acts-s8k-bs4-tokamax-splash-collective-matmul-none-host-sendrecv64/2026_06_04_15_03_14` shows total time 111,872.8 ms, MXU 64.1%, custom-call 26.2%, DKV 13.8%, forward residuals 12.4% combined, convolution fusion 53.8%, loop fusion 10.4%, all-reduce-scatter fusion 3.4%, idle 0.3%, and peak HBM 26.91 GiB. Fresh HLO confirms the train-step program hash remains `16296087512277873197`; optimized HLO is 787,978 bytes with content hash `3efb113cd72f8c9d08804e6cec46cf9373d83c06fae34ae9e3c27468e6b83fdf`, memory is 26.88 GiB, and the flagfile confirms `xla_max_concurrent_host_send_recv=64`. Do not carry 64; close the host send/recv scalar sweep and keep 100. Next test scheduler rerun4 on the same v127 baseline.

## [2026-06-04] loop-iteration | v138-shard-acts-s8k-bs4-tokamax-splash-collective-matmul-none-host-sendrecv128 on 8B/v6e-8: refuted/tie (56,447 tok/s best worker avg, 40.5% MFU; sendrecv128 keeps frontier HLO but no gain)

Workload `alekseyv-qwen3-v138-bs4-sendrecv128` kept the v127 simplified batch-4/global-32 collective-matmul-none stack fixed, restored `--xla_tpu_host_transfer_overlap_limit=24`, and changed only `--xla_max_concurrent_host_send_recv=128`. It completed cleanly on both workers with normal loss ending at 12.0461, reaching worker0 56,447 tok/s / 7,056 tok/s/chip / 40.5% MFU and worker1 56,428 tok/s / 7,053 tok/s/chip / 40.5% MFU, with best observed steady steps of 56,550 on both workers. XProf run `2026-06-04-qwen3-jax-v138-shard-acts-s8k-bs4-tokamax-splash-collective-matmul-none-host-sendrecv128/2026_06_04_14_52_26` shows total time 111,860.2 ms, MXU 64.1%, custom-call 26.2%, DKV 13.8%, forward residuals 12.4% combined, convolution fusion 53.8%, loop fusion 10.4%, all-reduce-scatter fusion 3.4%, idle 0.3%, and peak HBM 26.91 GiB. Fresh HLO confirms the train-step program hash remains `16296087512277873197`; optimized HLO is 787,978 bytes with content hash `3efb113cd72f8c9d08804e6cec46cf9373d83c06fae34ae9e3c27468e6b83fdf`, memory is 26.88 GiB, and the flagfile confirms `xla_max_concurrent_host_send_recv=128`. Do not carry 128; next test the symmetric lower value, 64, from the same v127 baseline.

## [2026-06-04] loop-iteration | v137-shard-acts-s8k-bs4-tokamax-splash-collective-matmul-none-host-transfer-overlap48-rerun on 8B/v6e-8: refuted/tie (56,456 tok/s best worker avg, 40.5% MFU; overlap48 rerun does not robustly improve)

Workload `alekseyv-qwen3-v137-bs4-hto48r` exactly reran v136's overlap48 setting on the v127 simplified batch-4/global-32 collective-matmul-none stack. It completed cleanly on both workers with normal loss ending at 12.0461, reaching worker0 56,456 tok/s / 7,057 tok/s/chip / 40.5% MFU and worker1 56,383 tok/s / 7,048 tok/s/chip / 40.4% MFU, with best observed steady steps of 56,565 and 56,566 tok/s. XProf run `2026-06-04-qwen3-jax-v137-shard-acts-s8k-bs4-tokamax-splash-collective-matmul-none-host-transfer-overlap48-rerun/2026_06_04_14_40_22` shows total time 112,110.6 ms, MXU 63.7%, custom-call 26.2%, DKV 13.8%, forward residuals 12.4% combined, convolution fusion 53.7%, loop fusion 10.4%, all-reduce-scatter fusion 3.4%, idle 0.5%, and peak HBM 26.91 GiB. The train-step program hash remains `16296087512277873197`; no fresh HLO objects were emitted because the run hit the persistent compilation cache, so use v136's byte-identical frontier HLO evidence at 787,978 bytes with content hash `3efb113cd72f8c9d08804e6cec46cf9373d83c06fae34ae9e3c27468e6b83fdf` and memory 26.88 GiB. Do not carry overlap48; restore host-transfer overlap 24 and next test `--xla_max_concurrent_host_send_recv=128` on the same v127 baseline.

## [2026-06-04] loop-iteration | v136-shard-acts-s8k-bs4-tokamax-splash-collective-matmul-none-host-transfer-overlap48 on 8B/v6e-8: provisional/tie (56,456 tok/s best worker avg, 40.5% MFU; overlap48 keeps frontier HLO, noise-level +3 tok/s over v127)

Workload `alekseyv-qwen3-v136-bs4-hto48` kept the v127 simplified batch-4/global-32 collective-matmul-none stack fixed and changed only `--xla_tpu_host_transfer_overlap_limit=48`. It completed cleanly on both workers with normal loss ending at 12.0461, reaching worker0 56,456 tok/s / 7,057 tok/s/chip / 40.5% MFU and worker1 56,423 tok/s / 7,053 tok/s/chip / 40.5% MFU, with best observed steady steps of 56,589 and 56,585 tok/s. XProf run `2026-06-04-qwen3-jax-v136-shard-acts-s8k-bs4-tokamax-splash-collective-matmul-none-host-transfer-overlap48/2026_06_04_14_33_34` shows total time 111,876.0 ms, MXU 64.1%, custom-call 26.2%, DKV 13.8%, forward residuals 12.4% combined, convolution fusion 53.8%, loop fusion 10.4%, all-reduce-scatter fusion 3.4%, idle 0.3%, and peak HBM 26.91 GiB. The train-step program hash remains `16296087512277873197`; optimized HLO is byte-identical to v123/v126/v127/v135 at 787,978 bytes with content hash `3efb113cd72f8c9d08804e6cec46cf9373d83c06fae34ae9e3c27468e6b83fdf`, and memory is 26.88 GiB. Because the average gain is only +3 tok/s versus v127 and the best observed step is still below v123, treat this as a provisional tie and rerun overlap48 before carrying it.

## [2026-06-04] loop-iteration | v135-shard-acts-s8k-bs4-tokamax-splash-collective-matmul-none-host-transfer-overlap12 on 8B/v6e-8: refuted/tie (56,436 tok/s best worker avg, 40.5% MFU; host-transfer overlap12 keeps frontier HLO but no gain)

Workload `alekseyv-qwen3-v135-bs4-hto12` kept the v127 simplified batch-4/global-32 collective-matmul-none stack fixed and changed only `--xla_tpu_host_transfer_overlap_limit=12`, restoring ACF multiple-steps to true after v134. It completed cleanly on both workers with normal loss ending at 12.0461, reaching worker0 56,436 tok/s / 7,055 tok/s/chip / 40.5% MFU and worker1 56,412 tok/s / 7,052 tok/s/chip / 40.4% MFU, with best observed steady steps of 56,558 on both workers. XProf run `2026-06-04-qwen3-jax-v135-shard-acts-s8k-bs4-tokamax-splash-collective-matmul-none-host-transfer-overlap12/2026_06_04_14_22_31` shows total time 111,865.1 ms, MXU 64.1%, custom-call 26.2%, DKV 13.8%, forward residuals 12.4% combined, convolution fusion 53.8%, loop fusion 10.4%, all-reduce-scatter fusion 3.4%, idle 0.3%, and peak HBM 26.91 GiB. The train-step program hash remains `16296087512277873197`; optimized HLO is byte-identical to v123/v126/v127 at 787,978 bytes with content hash `3efb113cd72f8c9d08804e6cec46cf9373d83c06fae34ae9e3c27468e6b83fdf`, and memory is 26.88 GiB. Do not carry overlap12; next test the opposite host-transfer direction, overlap48, on the same v127 baseline.

## [2026-06-04] loop-iteration | v134-shard-acts-s8k-bs4-tokamax-splash-collective-matmul-none-no-acf-multistep on 8B/v6e-8: refuted (56,294 tok/s best worker avg, 40.4% MFU; ACF multiple-steps remains required)

Workload `alekseyv-qwen3-v134-bs4-noacfms` kept the v127 simplified batch-4/global-32 collective-matmul-none stack fixed and changed only `--xla_tpu_enable_async_collective_fusion_multiple_steps=false`, leaving base ACF, all-gather ACF, and reduce-scatter ACF enabled. It completed cleanly on both workers with normal loss ending at 12.0461, but regressed to worker0 56,294 tok/s / 7,037 tok/s/chip / 40.4% MFU and worker1 56,269 tok/s / 7,034 tok/s/chip / 40.3% MFU, with best observed steady steps of 56,425 and 56,424 tok/s. XProf run `2026-06-04-qwen3-jax-v134-shard-acts-s8k-bs4-tokamax-splash-collective-matmul-none-no-acf-multistep/2026_06_04_14_08_09` shows total time 112,174.7 ms, MXU 64.0%, custom-call 26.3%, DKV 13.8%, forward residuals 12.4% combined, convolution fusion 54.4%, loop fusion 10.4%, all-reduce-scatter fusion 3.4%, idle 0.3%, and peak HBM 26.91 GiB. The train-step program hash changed to `14153491495991100093`; optimized HLO is 641,277 bytes with content hash `8a67b912870aa4546fc59e132fe9add8fa85086d4eea9e8b2c3d360ccf13dfcb`, memory is 26.88 GiB, and the flagfile confirms only the ACF multiple-steps subtype was false. Keep multiple-step ACF enabled; the post-cmnone ACF ablation sweep is closed. Next revalidate host-transfer overlap limit 12 on the simplified v127 baseline.

## [2026-06-04] loop-iteration | v133-shard-acts-s8k-bs4-tokamax-splash-collective-matmul-none-no-acf-reducescatter on 8B/v6e-8: supported/tie (56,440 tok/s best worker avg, 40.5% MFU; reduce-scatter ACF subtype false keeps frontier HLO)

Workload `alekseyv-qwen3-v133-bs4-noacfrs` kept the v127 simplified batch-4/global-32 collective-matmul-none stack fixed and changed only `--xla_tpu_enable_async_collective_fusion_fuse_reduce_scatter=false`, leaving base ACF, all-gather ACF, and multiple-step ACF enabled. It completed cleanly on both workers with normal loss ending at 12.0461, reaching worker0 56,440 tok/s / 7,055 tok/s/chip / 40.5% MFU and worker1 56,414 tok/s / 7,052 tok/s/chip / 40.4% MFU, with best observed steady steps of 56,561 and 56,557 tok/s. XProf run `2026-06-04-qwen3-jax-v133-shard-acts-s8k-bs4-tokamax-splash-collective-matmul-none-no-acf-reducescatter/2026_06_04_13_56_07` shows total time 111,893.1 ms, MXU 64.1%, custom-call 26.2%, DKV 13.8%, forward residuals 12.4% combined, convolution fusion 53.8%, loop fusion 10.4%, all-reduce-scatter fusion 3.4%, idle 0.3%, and peak HBM 26.91 GiB. The train-step program hash remains `16296087512277873197`; optimized HLO is byte-identical to v123/v126/v127 at 787,978 bytes with content hash `3efb113cd72f8c9d08804e6cec46cf9373d83c06fae34ae9e3c27468e6b83fdf`, and memory is 26.88 GiB. This supports reduce-scatter ACF subtype redundancy after collective-matmul-none, but not a throughput gain; continue from v127 with only multiple-step ACF disabled.

## [2026-06-04] loop-iteration | v132-shard-acts-s8k-bs4-tokamax-splash-collective-matmul-none-no-acf-allgather on 8B/v6e-8: refuted (55,479 tok/s best worker avg, 39.8% MFU; all-gather ACF subtype still required)

Workload `alekseyv-qwen3-v132-bs4-noacfag` kept the v127 simplified batch-4/global-32 collective-matmul-none stack fixed and changed only `--xla_tpu_enable_async_collective_fusion_fuse_all_gather=false`, leaving base ACF, reduce-scatter ACF, and multiple-step ACF enabled. It completed cleanly on both workers with normal loss ending at 12.0462, but regressed to worker0 55,479 tok/s / 6,935 tok/s/chip / 39.8% MFU and worker1 55,462 tok/s / 6,933 tok/s/chip / 39.8% MFU, with best observed steady steps of 55,547 and 55,538 tok/s. XProf run `2026-06-04-qwen3-jax-v132-shard-acts-s8k-bs4-tokamax-splash-collective-matmul-none-no-acf-allgather/2026_06_04_13_43_58` shows total time 113,699.9 ms, MXU 63.3%, custom-call 25.9%, DKV 13.6%, forward residuals 12.2% combined, convolution fusion 52.3%, loop fusion 10.2%, explicit all-gather 4.0%, idle 0.2%, and peak HBM 27.10 GiB. The train-step program hash changed to `16650495805246986607`, matching v131's no-base-ACF program; optimized HLO is 632,630 bytes with content hash `49e2af3ecfcb130fedcaa85842d312728e433a38dd3fd89aab1b1793a6e7631f`, memory is 27.07 GiB, and the flagfile confirms only the all-gather ACF subtype was false. Keep all-gather ACF enabled; next revalidate only the reduce-scatter ACF subtype on the post-collective-matmul-none stack.

## [2026-06-04] loop-iteration | v131-shard-acts-s8k-bs4-tokamax-splash-collective-matmul-none-no-base-async-collective-fusion on 8B/v6e-8: refuted (55,482 tok/s best worker avg, 39.8% MFU; base ACF still required after collective-matmul-none)

Workload `alekseyv-qwen3-v131-bs4-noacf` kept the v127 simplified batch-4/global-32 collective-matmul-none stack fixed and changed only `--xla_tpu_enable_async_collective_fusion=false`. It completed cleanly on both workers with normal loss ending at 12.0462, but regressed to worker0 55,482 tok/s / 6,935 tok/s/chip / 39.8% MFU and worker1 55,459 tok/s / 6,932 tok/s/chip / 39.8% MFU, with best observed steady steps of 55,530 and 55,533 tok/s. XProf run `2026-06-04-qwen3-jax-v131-shard-acts-s8k-bs4-tokamax-splash-collective-matmul-none-no-base-async-collective-fusion/2026_06_04_13_31_02` shows total time 113,739.4 ms, MXU 63.2%, custom-call 25.9%, DKV 13.6%, forward residuals 12.2% combined, convolution fusion 52.3%, loop fusion 10.2%, idle 0.3%, and peak HBM 27.10 GiB. The train-step program hash changed to `16650495805246986607`; optimized HLO changed to 632,630 bytes with content hash `49e2af3ecfcb130fedcaa85842d312728e433a38dd3fd89aab1b1793a6e7631f`, memory rose to 27.07 GiB, and the flagfile confirms base async collective fusion was false while subtype flags remained true. Keep base ACF enabled; next revalidate only the all-gather ACF subtype on the post-collective-matmul-none stack.

## [2026-06-04] loop-iteration | v130-shard-acts-s8k-bs4-tokamax-splash-collective-matmul-none-no-ag-backward-pipelining on 8B/v6e-8: refuted/tie (56,449 tok/s best worker avg, 40.5% MFU; AG backward pipelining false keeps same HLO but no gain)

Workload `alekseyv-qwen3-v130-bs4-noagpipe` kept the v127 simplified batch-4/global-32 collective-matmul-none stack fixed and changed only `--xla_tpu_enable_ag_backward_pipelining=false`. It completed cleanly on both workers with normal loss ending at 12.0461, reaching worker0 56,449 tok/s / 7,056 tok/s/chip / 40.5% MFU and worker1 56,431 tok/s / 7,054 tok/s/chip / 40.5% MFU, with best observed steady steps of 56,570 and 56,571 tok/s. XProf run `2026-06-04-qwen3-jax-v130-shard-acts-s8k-bs4-tokamax-splash-collective-matmul-none-no-ag-backward-pipelining/2026_06_04_13_18_53` shows total time 111,852.9 ms, MXU 64.1%, custom-call 26.2%, DKV 13.8%, forward residuals 12.4% combined, convolution fusion 53.8%, loop fusion 10.4%, idle 0.3%, and peak HBM 26.91 GiB. The train-step program hash remains `16296087512277873197`; optimized HLO is byte-identical to v123/v126/v127/v128/v129 at 787,978 bytes with content hash `3efb113cd72f8c9d08804e6cec46cf9373d83c06fae34ae9e3c27468e6b83fdf`, and the flagfile confirms AG backward pipelining was false. Do not carry this ablation; return to v127 and next revalidate base async collective fusion on the post-collective-matmul-none stack.

## [2026-06-04] loop-iteration | v129-shard-acts-s8k-bs4-tokamax-splash-collective-matmul-none-no-customcall-vmem-adjust on 8B/v6e-8: refuted/tie (56,449 tok/s best worker avg, 40.5% MFU; custom-call VMEM adjust false keeps same HLO but no gain)

Workload `alekseyv-qwen3-v129-bs4-noccvmem` kept the v127 simplified batch-4/global-32 collective-matmul-none stack fixed and changed only `--xla_tpu_enable_tpu_custom_call_scoped_vmem_adjustments=false`. It completed cleanly on both workers with normal loss ending at 12.0461, reaching worker0 56,430 tok/s / 7,054 tok/s/chip / 40.5% MFU and worker1 56,449 tok/s / 7,056 tok/s/chip / 40.5% MFU, with best observed steps of 56,587 tok/s on both workers. XProf run `2026-06-04-qwen3-jax-v129-shard-acts-s8k-bs4-tokamax-splash-collective-matmul-none-no-customcall-vmem-adjust/2026_06_04_13_06_38` shows total time 111,859.6 ms, MXU 64.1%, custom-call 26.2%, DKV 13.8%, forward residuals 12.4% combined, convolution fusion 53.8%, loop fusion 10.4%, idle 0.3%, and peak HBM 26.91 GiB. The train-step program hash remains `16296087512277873197`; optimized HLO is byte-identical to v123/v126/v127/v128 at 787,978 bytes with content hash `3efb113cd72f8c9d08804e6cec46cf9373d83c06fae34ae9e3c27468e6b83fdf`, and the flagfile confirms custom-call scoped VMEM adjustment was false. Do not carry this ablation; return to v127 and next revalidate disabling AG backward pipelining on the post-collective-matmul-none stack.

## [2026-06-04] loop-iteration | v128-shard-acts-s8k-bs4-tokamax-splash-collective-matmul-none-no-memory-pressure-tracking on 8B/v6e-8: refuted/tie (56,448 tok/s, 40.5% MFU; memory-pressure tracking false keeps same HLO but no gain)

Workload `alekseyv-qwen3-v128-bs4-nomemtrack` kept the v127 simplified batch-4/global-32 collective-matmul-none stack fixed and changed only `--xla_tpu_enable_scheduler_memory_pressure_tracking=false`. It completed cleanly on both workers with normal loss ending at 12.0461, reaching worker0 56,448 tok/s / 7,056 tok/s/chip / 40.5% MFU and worker1 56,428 tok/s / 7,053 tok/s/chip / 40.5% MFU, with best observed steps of 56,592 and 56,597 tok/s. XProf run `2026-06-04-qwen3-jax-v128-shard-acts-s8k-bs4-tokamax-splash-collective-matmul-none-no-memory-pressure-tracking/2026_06_04_12_54_39` shows total time 111,850.3 ms, MXU 64.1%, custom-call 26.2%, DKV 13.8%, forward residuals 12.4% combined, convolution fusion 53.8%, loop fusion 10.4%, idle 0.3%, and peak HBM 26.91 GiB. The train-step program hash remains `16296087512277873197`; optimized HLO is byte-identical to v123/v126/v127 at 787,978 bytes with content hash `3efb113cd72f8c9d08804e6cec46cf9373d83c06fae34ae9e3c27468e6b83fdf`, and the flagfile confirms TPU memory-pressure tracking was false while sparse-core memory-pressure tracking remained true. Do not carry this ablation; return to v127 and next test whether custom-call scoped VMEM adjustments remain needed.

## [2026-06-04] loop-iteration | v127-shard-acts-s8k-bs4-tokamax-splash-collective-matmul-none-no-all-experimental-scheduler on 8B/v6e-8: supported/tie (56,453 tok/s, 40.5% MFU; all-experimental scheduler bundle false keeps same HLO)

Workload `alekseyv-qwen3-v127-bs4-noallsched` kept the v126 simplified batch-4/global-32 collective-matmul-none stack fixed and changed only `--xla_tpu_enable_all_experimental_scheduler_features=false`. It completed cleanly on both workers with normal loss ending at 12.0461, reaching worker0 56,453 tok/s / 7,057 tok/s/chip / 40.5% MFU and worker1 56,437 tok/s / 7,055 tok/s/chip / 40.5% MFU, with best observed steps of 56,573 and 56,579 tok/s. XProf run `2026-06-04-qwen3-jax-v127-shard-acts-s8k-bs4-tokamax-splash-collective-matmul-none-no-all-experimental-scheduler/2026_06_04_12_41_32` shows total time 111,827.9 ms, MXU 64.1%, custom-call 26.2%, DKV 13.8%, forward residuals 12.4% combined, convolution fusion 53.8%, loop fusion 10.4%, all-reduce-scatter fusion 3.4%, idle 0.3%, and peak HBM 26.91 GiB. The train-step program hash remains `16296087512277873197`; optimized HLO is byte-identical to v123/v126 at 787,978 bytes with content hash `3efb113cd72f8c9d08804e6cec46cf9373d83c06fae34ae9e3c27468e6b83fdf`, and the flagfile confirms the all-experimental scheduler bundle was explicitly `false` while selective resources stayed enabled and layeragg stayed explicitly false. Treat v127 as an equivalent simplified stack, not a new observed-step frontier; next test should disable scheduler memory-pressure tracking from this baseline.

## [2026-06-04] loop-iteration | v126-shard-acts-s8k-bs4-tokamax-splash-collective-matmul-none-layeragg-false on 8B/v6e-8: supported/tie (56,448 tok/s, 40.5% MFU; explicit layeragg false keeps same HLO)

Workload `alekseyv-qwen3-v126-bs4-layerfalse` kept the confirmed batch-4/global-32 collective-matmul-none stack fixed, restored selective resources, and set only `--xla_tpu_enable_latency_hiding_layer_scheduler=false --xla_tpu_enable_sparse_core_collective_aggregator=false`. It completed cleanly on both workers with normal loss ending at 12.0461, reaching worker0 56,448 tok/s / 7,056 tok/s/chip / 40.5% MFU and worker1 56,428 tok/s / 7,054 tok/s/chip / 40.5% MFU. XProf run `2026-06-04-qwen3-jax-v126-shard-acts-s8k-bs4-tokamax-splash-collective-matmul-none-layeragg-false/2026_06_04_12_30_04` shows total time 111,845.3 ms, MXU 63.9%, custom-call 26.2%, DKV 13.8%, forward residuals 12.4% combined, convolution fusion 53.8%, loop fusion 10.4%, all-reduce-scatter fusion 3.4%, idle 0.3%, and peak HBM 26.91 GiB. The train-step program hash remains `16296087512277873197`; optimized HLO is byte-identical to v118/v123/v124 at 787,978 bytes with content hash `3efb113cd72f8c9d08804e6cec46cf9373d83c06fae34ae9e3c27468e6b83fdf`, and the flagfile confirms both layeragg flags were explicitly `false`. Use the simpler no-layeragg baseline for future scheduler probes; next test should disable the broad all-experimental scheduler bundle.

## [2026-06-04] loop-iteration | v125-shard-acts-s8k-bs4-tokamax-splash-layeragg-collective-matmul-none-no-selective-resources on 8B/v6e-8: refuted (56,387 tok/s, 40.4% MFU; changed HLO, slight regression)

Workload `alekseyv-qwen3-v125-bs4-noselres` kept the confirmed v123 batch-4/global-32 collective-matmul-none stack fixed and changed only `--xla_latency_hiding_scheduler_enable_selective_resources=false`. It completed cleanly on both workers with normal loss ending at 12.0458, but regressed to worker0 56,387 tok/s / 7,048 tok/s/chip / 40.4% MFU and worker1 56,369 tok/s / 7,046 tok/s/chip / 40.4% MFU. XProf run `2026-06-04-qwen3-jax-v125-shard-acts-s8k-bs4-tokamax-splash-layeragg-collective-matmul-none-no-selective-resources/2026_06_04_12_18_17` shows total time 111,948.0 ms, MXU 64.0%, custom-call 26.2%, DKV 13.8%, forward residuals 12.4% combined, convolution fusion 53.7%, loop fusion 10.5%, all-reduce-scatter fusion 3.4%, idle 0.2%, and peak HBM 26.91 GiB. The train-step program hash changed to `12520399088891659285`; optimized HLO is 788,403 bytes with content hash `c87d6be51ac696104580c7500f3970e2073a4533e20a1b34649b3cc9f2f87ba7`, HLO object count 1,619, and flagfile confirmation of `selective_resources=false`. Keep selective resources enabled. Because v124 only omitted the layeragg pair and the flagfile fell back to `auto`, the next test should set the layer-scheduler plus sparse-core aggregator pair explicitly to `false`.

## [2026-06-04] loop-iteration | v124-shard-acts-s8k-bs4-tokamax-splash-collective-matmul-none-no-layeragg on 8B/v6e-8: supported/tie (56,447 tok/s, 40.5% MFU; same HLO as v118/v123)

Workload `alekseyv-qwen3-v124-bs4-nolayeragg` kept the confirmed v123 batch-4/global-32 collective-matmul-none stack fixed and removed only the explicit layer-scheduler plus sparse-core collective aggregator flags. It completed cleanly on both workers with normal loss ending at 12.0461, reaching worker0 56,447 tok/s / 7,056 tok/s/chip / 40.5% MFU and worker1 56,429 tok/s / 7,054 tok/s/chip / 40.5% MFU, with best observed steps of 56,569 and 56,576 tok/s. XProf run `2026-06-04-qwen3-jax-v124-shard-acts-s8k-bs4-tokamax-splash-collective-matmul-none-no-layeragg/2026_06_04_12_05_51` shows total time 111,854.0 ms, MXU 64.1%, custom-call 26.2%, DKV 13.8%, forward residuals 12.4% combined, convolution fusion 53.8%, loop fusion 10.4%, all-reduce-scatter fusion 3.4%, idle 0.3%, and peak HBM 26.91 GiB with 4.34 GiB free. Fresh HLO dumps confirm the optimized train-step is byte-identical to v118/v123: 787,978 bytes, hash `3efb113cd72f8c9d08804e6cec46cf9373d83c06fae34ae9e3c27468e6b83fdf`, and train-step program hash `16296087512277873197`; the omitted flags appear as default `auto`. Treat the explicit layeragg pair as unnecessary for this final HLO but not performance-positive; v123 remains the named frontier. Next test should ablate selective resources from the same batch-4 collective-matmul-none stack.

## [2026-06-04] loop-iteration | v123-shard-acts-s8k-bs4-tokamax-splash-layeragg-collective-matmul-none-rerun on 8B/v6e-8: supported (56,448 tok/s, 40.5% MFU; confirms v118 batch-4 frontier)

Workload `alekseyv-qwen3-v123-bs4-rerun` exactly reran v118 after batch-5 scalar fit attempts failed. It completed cleanly on both workers with normal loss ending at 12.0461, reaching worker0 56,448 tok/s / 7,056 tok/s/chip / 40.5% MFU and worker1 56,432 tok/s / 7,054 tok/s/chip / 40.5% MFU, with best observed steps of 56,599 and 56,596 tok/s. XProf run `2026-06-04-qwen3-jax-v123-shard-acts-s8k-bs4-tokamax-splash-layeragg-collective-matmul-none-rerun/2026_06_04_11_52_08` shows total time 111,868.7 ms, MXU 64.0%, custom-call 26.2%, DKV 13.8%, forward residuals 12.4% combined, convolution fusion 53.8%, loop fusion 10.4%, all-reduce-scatter fusion 3.4%, idle 0.2%, and peak HBM 26.91 GiB with 4.34 GiB free. No new HLO objects were emitted, consistent with a cache hit; XProf reports the same train-step program hash as v118 (`16296087512277873197`), so use v118's 787,978-byte HLO and hash `3efb113cd72f8c9d08804e6cec46cf9373d83c06fae34ae9e3c27468e6b83fdf` as compiled-program evidence. Adopt batch 4/global 32 collective-matmul-none as the durable current best; next test should ablate the now-questionable layer-scheduler+sparse-core-aggregator pair on this confirmed stack.

## [2026-06-04] loop-iteration | v122-shard-acts-s8k-bs5-tokamax-splash-layeragg-collective-matmul-none-vmem65536 on 8B/v6e-8: invalid (batch 5 still OOM, +33.24 MiB; scalar VMEM path exhausted)

Workload `alekseyv-qwen3-v122-bs5-vmem65536` kept the v121 batch-5 scoped-VMEM fit attempt fixed and changed only scoped VMEM from 73728 to 65536 KiB. It failed during train-step compile on both workers with `RESOURCE_EXHAUSTED: XLA:TPU compile permanent error. Ran out of memory in memory space hbm. Used 31.28G of 31.25G hbm. Exceeded hbm capacity by 33.24M.` This improves v121 by only about 4 MiB and still does not fit. OOM artifacts were emitted under `module_0267.jit_train_step.cl_854318611`; optimized train-step text is 805,927 bytes, content hash `d824280a1d9a97f4e70db3fa5c68fa01dcf501fb1154b2399fe22a8a9196e2f8`, with 1,289 HLO objects. The flagfile confirms collective-matmul modes remain `NONE`, scheduler shared-memory limit is 100, and scoped VMEM is 65536. Retire scalar batch-5 fit attempts for now; the best valid frontier remains v118 batch 4/global 32 and should be rerun before banking.

## [2026-06-04] loop-iteration | v121-shard-acts-s8k-bs5-tokamax-splash-layeragg-collective-matmul-none-vmem73728 on 8B/v6e-8: invalid (batch 5 still OOM, +37.55 MiB; scoped VMEM helps only slightly)

Workload `alekseyv-qwen3-v121-bs5-vmem73728` restored scheduler shared-memory limit to 100 and changed only scoped VMEM from 81920 to 73728 KiB on the v119 batch-5 near-fit shape. It failed during train-step compile on both workers with `RESOURCE_EXHAUSTED: XLA:TPU compile permanent error. Ran out of memory in memory space hbm. Used 31.28G of 31.25G hbm. Exceeded hbm capacity by 37.55M.` This improves v119's OOM margin by only about 9 MiB and still does not fit. OOM artifacts were emitted under `module_0267.jit_train_step.cl_854318611`; optimized train-step text is 790,117 bytes, content hash `c3dcd9083be4b8a7db2664908e9bb95779f01b2d348dc70786c6d3e24309a79e`, with 1,290 HLO objects. The flagfile confirms collective-matmul modes remain `NONE`, scheduler shared-memory limit is 100, and scoped VMEM is 73728. Try one lower VMEM point, 65536 KiB; if that misses too, retire scalar VMEM tweaks for batch 5.

## [2026-06-04] loop-iteration | v120-shard-acts-s8k-bs5-tokamax-splash-layeragg-collective-matmul-none-shmem90 on 8B/v6e-8: refuted/invalid (shared-memory cap worsens batch 5 OOM to +1.12 GiB)

Workload `alekseyv-qwen3-v120-bs5-shmem90` kept the v119 batch-5 near-fit OOM shape fixed and changed only `--xla_tpu_scheduler_percent_shared_memory_limit=100` to `90`. It failed during train-step compile on both workers with `RESOURCE_EXHAUSTED: XLA:TPU compile permanent error. Ran out of memory in memory space hbm. Used 32.37G of 31.25G hbm. Exceeded hbm capacity by 1.12G.` This is materially worse than v119's 46.98 MiB OOM. OOM artifacts were emitted under `module_0267.jit_train_step.cl_854318611`; optimized train-step text is 726,125 bytes, content hash `1f3ef62bd5511ce5ac592a076ee65e5755d8f7fa5df3d17a5f2027d79b6a7644`, with 1,288 HLO objects. The flagfile confirms collective-matmul modes remain `NONE`, scoped VMEM remains 81920 KiB, and shared-memory limit is 90. Do not carry shmem90 for the batch-5 fit path; restore shared-memory limit 100 and test scoped VMEM reduction instead.

## [2026-06-04] loop-iteration | v119-shard-acts-s8k-bs5-tokamax-splash-layeragg-collective-matmul-none on 8B/v6e-8: invalid (batch 5 near-fit OOM, exceeds HBM by 46.98 MiB)

Workload `alekseyv-qwen3-v119-bs5-cmnone` kept the v118 batch-4 frontier fixed and changed only per-chip batch from 4 to 5, raising global batch from 32 to 40. It failed during train-step compile on both workers with `RESOURCE_EXHAUSTED: XLA:TPU compile permanent error. Ran out of memory in memory space hbm. Used 31.29G of 31.25G hbm. Exceeded hbm capacity by 46.98M.` No runtime profile exists. OOM artifacts were emitted under `module_0112.jit_train_step.cl_854318611`, including `oom_intermediate_module.txt` and `oom_top_memory_consuming_instructions`; the optimized train-step text is 790,290 bytes with content hash `67b888766a137eab085ae4c419a9dc7369e263b7ff770b53ef891bae618b1c0d`, 50 HLO objects, and the flagfile confirms both collective-matmul modes are `NONE`, scoped VMEM 81920 KiB, and scheduler shared-memory limit 100. Treat batch 5 as a near-fit OOM, not a broad rejection; next test should lower only scheduler shared-memory limit to 90 to see if a lower-HBM schedule can make batch 5 valid.

## [2026-06-04] loop-iteration | v118-shard-acts-s8k-bs4-tokamax-splash-layeragg-collective-matmul-none on 8B/v6e-8: supported (56,445 tok/s, 40.5% MFU; batch 4 fits but scales sublinearly)

Workload `alekseyv-qwen3-v118-bs4-cmnone` kept the confirmed v116/v117 collective-matmul-none frontier fixed and changed only per-chip batch from 3 to 4, raising global batch from 24 to 32. It completed cleanly on both workers with normal loss ending at 12.0461, reaching worker0 56,445 tok/s / 7,056 tok/s/chip / 40.5% MFU and worker1 56,418 tok/s / 7,052 tok/s/chip / 40.5% MFU, with best observed steps of 56,557 and 56,553 tok/s. XProf run `2026-06-04-qwen3-jax-v118-shard-acts-s8k-bs4-tokamax-splash-layeragg-collective-matmul-none/2026_06_04_11_11_24` shows total time 111,891.2 ms, MXU 64.1%, custom-call 26.2%, DKV 13.8%, forward residuals 12.4% combined, convolution fusion 53.8%, loop fusion 10.4%, all-reduce-scatter fusion down to 3.4% share with nearly flat absolute time, idle 0.3%, and peak HBM 26.91 GiB with 4.34 GiB free. The optimized train-step HLO is `module_0112.jit_train_step.cl_854318611.after_optimizations.txt`, 787,978 bytes, content hash `3efb113cd72f8c9d08804e6cec46cf9373d83c06fae34ae9e3c27468e6b83fdf`, 49 HLO objects, and train-step program hash `16296087512277873197`; the flagfile confirms both collective-matmul modes remain `NONE`. Carry batch 4 provisionally as the new throughput frontier, but treat the gain as small/sublinear; test batch 5/global 40 as a close-to-capacity fit probe.

## [2026-06-04] loop-iteration | v117-shard-acts-s8k-bs3-tokamax-splash-layeragg-collective-matmul-none-rerun on 8B/v6e-8: supported (56,041 tok/s best worker, 40.2% MFU; confirms v116 collective-matmul-none frontier)

Workload `alekseyv-qwen3-v117-cmnone-rerun` exactly reran v116 with the MaxText `DISABLE_COLLECTIVE_MATMUL` pair: `--xla_tpu_all_gather_collective_matmul_mode=none --xla_tpu_reduce_scatter_collective_matmul_mode=none`. It completed cleanly on both workers with normal loss ending at 12.0512, reaching worker0 56,041 tok/s / 7,005 tok/s/chip / 40.2% MFU and worker1 55,993 tok/s / 6,999 tok/s/chip / 40.1% MFU, with best observed steps of 56,151 tok/s on both workers. XProf run `2026-06-04-qwen3-jax-v117-shard-acts-s8k-bs3-tokamax-splash-layeragg-collective-matmul-none-rerun/2026_06_04_10_59_36` shows total time 84,645.3 ms, MXU 62.9%, custom-call 26.1%, DKV 13.8%, forward residuals 12.3% combined, convolution fusion 52.3%, loop fusion 10.1%, all-reduce-scatter fusion 4.4%, idle 0.4%, and peak HBM 22.21 GiB with 9.03 GiB free. No new HLO objects were emitted under the v117 run prefix, consistent with a persistent-cache hit; XProf reports the same train-step program hash as v116 (`16402808057163658277`), so use v116's 795,784-byte HLO/hash/flagfile as the compiled-program evidence. Adopt collective-matmul-none as confirmed and pivot to HBM-enabled batch scaling.

## [2026-06-04] loop-iteration | v116-shard-acts-s8k-bs3-tokamax-splash-layeragg-collective-matmul-none on 8B/v6e-8: supported/provisional (56,009 tok/s avg, 40.2% MFU; disabling collective-matmul modes is a large new frontier)

Workload `alekseyv-qwen3-v116-cmnone` kept the valid v090 qseq1 layeragg frontier fixed and added only the MaxText `DISABLE_COLLECTIVE_MATMUL` pair: `--xla_tpu_all_gather_collective_matmul_mode=none --xla_tpu_reduce_scatter_collective_matmul_mode=none`. It completed cleanly on both workers with normal loss ending at 12.0512, reaching 56,009 tok/s average / 7,001 tok/s/chip / 40.2% MFU, with best observed steps of 56,172 tok/s and 56,171 tok/s. This is far above v090's 51,033 tok/s / 36.6% MFU frontier. XProf run `2026-06-04-qwen3-jax-v116-shard-acts-s8k-bs3-tokamax-splash-layeragg-collective-matmul-none/2026_06_04_10_52_19` shows total time 84,576.4 ms, MXU 63.1%, custom-call 26.1%, DKV 13.8%, forward residuals 12.4% combined, convolution fusion 52.4%, loop fusion down to 10.1%, all-reduce-scatter fusion 4.4%, idle 0.3%, and peak HBM down to 22.21 GiB. The optimized train-step HLO shrinks to 795,784 bytes with content hash `ff53351f8c3f828d212665c2505f4f865079a5f42232a66c05bf345a35044a8b`, HLO prefix object count 1,289, approximate instruction-line count 3,848, and train-step program hash `16402808057163658277`; the flagfile confirms both collective-matmul modes are `NONE`. Adopt this pair provisionally and immediately rerun before banking the frontier.

## [2026-06-04] loop-iteration | v115-shard-acts-s8k-bs3-tokamax-splash-layeragg-no-acf-multistep on 8B/v6e-8: refuted/tie (51,023 tok/s, 36.6% MFU; ACF multiple-steps ablation changes HLO but does not improve frontier)

Workload `alekseyv-qwen3-v115-noacfms` kept the valid v090 qseq1 frontier fixed and changed only `--xla_tpu_enable_async_collective_fusion_multiple_steps=true` to `false`, preserving base async collective fusion plus all-gather and reduce-scatter fusion. It completed cleanly with normal loss ending at 12.0510, reaching worker0 50,998 tok/s / 36.6% MFU and worker1 51,023 tok/s / 36.6% MFU, below v090's 51,033 best-observed worker and only in the prior repeated frontier band. XProf run `2026-06-04-qwen3-jax-v115-shard-acts-s8k-bs3-tokamax-splash-layeragg-no-acf-multistep/2026_06_04_10_37_12` shows total time 92,836.8 ms, MXU 56.9%, custom-call 23.8%, DKV 12.5%, forward residuals 11.2% combined, convolution fusion 49.3%, loop fusion 18.0%, data formatting 4.0%, collective-permute-start 1.3%, and peak HBM 28.32 GiB. The train-step program hash changes to `6644865601235731379`, optimized HLO text shrinks to 1,188,888 bytes with 1,619 objects, and content hash is `e5cf4844bafcebfacbeb1ef3d883dc3fcd0368005d33d4a123afae4efc1fb657`. Keep multiple-step async collective fusion enabled; the ACF subtype sweep is closed.

## [2026-06-04] loop-iteration | v114-shard-acts-s8k-bs3-tokamax-splash-layeragg-no-acf-reducescatter on 8B/v6e-8: refuted/tie (51,008 tok/s, 36.6% MFU; reduce-scatter fusion subtype ablation does not improve frontier)

Workload `alekseyv-qwen3-v114-noacfrs` kept the valid v090 qseq1 frontier fixed and changed only `--xla_tpu_enable_async_collective_fusion_fuse_reduce_scatter=true` to `false`, preserving base async collective fusion, all-gather fusion, and multiple-steps fusion. It completed cleanly with normal loss ending at 12.0508, but reached only worker0 51,008 tok/s / 36.6% MFU and worker1 50,982 tok/s / 36.6% MFU, below v090's 51,033 best-observed and below the 51,026+ repeated v089/v090 frontier band. XProf run `2026-06-04-qwen3-jax-v114-shard-acts-s8k-bs3-tokamax-splash-layeragg-no-acf-reducescatter/2026_06_04_10_24_55` shows total time 92,807.7 ms, MXU 56.8%, custom-call 23.8%, DKV 12.5%, forward residuals 11.2% combined, convolution fusion 49.3%, loop fusion 18.0%, data formatting 4.1%, collective-permute-start 1.0%, and peak HBM 28.32 GiB. The train-step program hash is the normal frontier `13194376450853388418`; optimized HLO text is 1,375,271 bytes with 2,210 objects under the run prefix and content hash `796167499c5335993c3c6cc63486cb24875556369b39d8bdc1e70d30ebeb969c`. Keep reduce-scatter async collective fusion enabled.

## [2026-06-04] loop-iteration | v113-shard-acts-s8k-bs3-tokamax-splash-layeragg-no-acf-allgather on 8B/v6e-8: refuted (50,355 tok/s, 36.1% MFU; all-gather fusion subtype is required)

Workload `alekseyv-qwen3-v113-noacfag` kept the valid v090 qseq1 frontier fixed and changed only `--xla_tpu_enable_async_collective_fusion_fuse_all_gather=true` to `false`, leaving base async collective fusion, reduce-scatter fusion, and multiple-steps fusion enabled. It completed cleanly with normal loss ending at 12.0511, but reached only worker0 50,335 tok/s / 36.1% MFU and worker1 50,355 tok/s / 36.1% MFU, well below the v089/v090 frontier. XProf run `2026-06-04-qwen3-jax-v113-shard-acts-s8k-bs3-tokamax-splash-layeragg-no-acf-allgather/2026_06_04_10_11_19` shows total time 94,003.8 ms, MXU 56.1%, custom-call 23.5%, DKV 12.4%, forward residuals 11.1% combined, convolution fusion 48.3%, loop fusion 17.8%, data formatting 3.9%, all-gather 2.3%, and peak HBM down to 28.03 GiB. The train-step optimized HLO shrinks to 1,187,819 bytes with 1,619 objects, content hash `4cdb9bf855debf8559a8e2f93c316d993b555fdc92b80cd380116317e29cf41d`, and train-step program hash `16920192351436248925`. The lower-HBM schedule is not worth the exposed all-gather cost; keep all-gather async collective fusion enabled.

## [2026-06-04] loop-iteration | v112-shard-acts-s8k-bs3-tokamax-splash-layeragg-dq3 on 8B/v6e-8: refuted (50,853 tok/s, 36.5% MFU; DQ3 changes HLO but slows wall time)

Workload `alekseyv-qwen3-v112-dq3` kept the valid v090 qseq1 frontier fixed and added only `TOKAMAX_DQ_REDUCTION_STEPS=3`. It completed cleanly with normal loss ending at 12.0508, but reached only worker0 50,853 tok/s / 36.5% MFU and worker1 50,825 tok/s / 36.4% MFU, below v090's 51,033 best-observed and below the 51,026+ repeated v089/v090 frontier band. XProf run `2026-06-04-qwen3-jax-v112-shard-acts-s8k-bs3-tokamax-splash-layeragg-dq3/2026_06_04_09_56_36` shows total time 93,096.6 ms, MXU 56.6%, peak HBM 28.32 GiB, custom-call 23.3%, DKV 12.1%, forward residuals 11.2% combined, convolution fusion 49.1%, loop fusion 18.1%, and data formatting 4.0%. The train-step optimized HLO changes materially: 1,394,534 bytes, 32 objects under the HLO prefix, content hash `ebcd22220b014d8e255b05311311e539b3dd9923ffbdea299732c26249d978dd`, and train-step program hash `12596932080449751219`. The DQ3 path reduces the DKV custom-call bucket but loses more in convolution fusion; do not carry `TOKAMAX_DQ_REDUCTION_STEPS=3` forward.

## [2026-06-04] loop-iteration | v111-shard-acts-s8k-bs3-tokamax-splash-layeragg-vmem86016 on 8B/v6e-8: refuted (50,003 tok/s, 35.9% MFU; upper VMEM point changes schedule but regresses)

Workload `alekseyv-qwen3-v111-vmem86016` kept the valid v090 qseq1 frontier fixed and changed only `--xla_tpu_scoped_vmem_limit_kib=81920` to `86016`. It completed cleanly with normal loss ending at 12.0511, but reached only worker0 50,003 tok/s / 35.9% MFU and worker1 49,982 tok/s / 35.8% MFU, well below the v089/v090 frontier. XProf run `2026-06-04-qwen3-jax-v111-shard-acts-s8k-bs3-tokamax-splash-layeragg-vmem86016/2026_06_04_09_48_25` shows total time 94,659.4 ms, MXU 55.7%, custom-call 23.2%, DKV 12.3%, forward residuals 11.0%, loop fusion 18.9%, data formatting 4.0%, and peak HBM 28.32 GiB. The train-step program hash changes to `3111414773617791261`, optimized HLO text grows to 1,376,311 bytes, and object count changes to 1,615, so this is a real but slower schedule. Keep scoped VMEM at 81920 KiB.

## [2026-06-04] loop-iteration | v110-shard-acts-s8k-bs3-tokamax-splash-layeragg-rerun4 on 8B/v6e-8: refuted/tie (51,009 tok/s, 36.6% MFU; scheduler rerun4 does not improve layeragg frontier)

Workload `alekseyv-qwen3-v110-rerun4` kept the valid v090 qseq1 frontier fixed and changed only `--xla_latency_hiding_scheduler_rerun=3` to `4`. It completed cleanly with normal loss ending at 12.0508, but reached only worker0 51,009 tok/s / 36.6% MFU and worker1 50,992 tok/s / 36.6% MFU, below v090's 51,033 best-observed and below the 51,026+ repeated v089/v090 band. XProf run `2026-06-04-qwen3-jax-v110-shard-acts-s8k-bs3-tokamax-splash-layeragg-rerun4/2026_06_04_09_35_42` shows total time 92,743.7 ms, custom-call 23.8%, DKV 12.6%, forward residuals 11.2%, loop fusion 18.0%, data formatting 4.1%, peak HBM 28.32 GiB, and MXU 56.9%. The train-step optimized HLO text is 1,375,271 bytes with 1,619 HLO objects and the normal frontier program hash. Keep scheduler rerun at 3.

## [2026-06-04] loop-iteration | v109-shard-acts-s8k-bs3-tokamax-splash-layeragg-host-sendrecv128 on 8B/v6e-8: refuted/tie (51,008 tok/s, 36.6% MFU; higher host send/recv concurrency does not improve frontier)

Workload `alekseyv-qwen3-v109-hsr128` kept the valid v090 qseq1 frontier fixed and changed only `--xla_max_concurrent_host_send_recv=100` to `128`. It completed cleanly with normal loss ending at 12.0508, but reached only worker0 51,008 tok/s / 36.6% MFU and worker1 50,984 tok/s / 36.6% MFU, below v090's 51,033 best-observed and below the 51,026+ repeated v089/v090 band. XProf run `2026-06-04-qwen3-jax-v109-shard-acts-s8k-bs3-tokamax-splash-layeragg-host-sendrecv128/2026_06_04_09_23_09` shows total time 46,337.3 ms on `gke-tpu-964065d9-c180`, custom-call 23.8%, DKV 12.6%, forward residuals 11.2%, loop fusion 18.0%, data formatting 4.1%, and peak HBM 28.32 GiB. The train-step optimized HLO text is 1,375,271 bytes with 1,619 HLO objects and the normal frontier program hash. Keep the max concurrent host send/recv limit at 100.

## [2026-06-04] loop-iteration | v108-shard-acts-s8k-bs3-tokamax-splash-layeragg-host-transfer-overlap48 on 8B/v6e-8: refuted/tie (51,014 tok/s, 36.6% MFU; higher host-transfer overlap limit does not improve frontier)

Workload `alekseyv-qwen3-v108-hto48` kept the valid v090 qseq1 frontier fixed and changed only `--xla_tpu_host_transfer_overlap_limit=24` to `48`. It completed cleanly with normal loss ending at 12.0508, but reached only worker0 51,014 tok/s / 36.6% MFU and worker1 50,990 tok/s / 36.6% MFU, below v090's 51,033 best-observed and below the 51,026+ repeated v089/v090 band. XProf run `2026-06-04-qwen3-jax-v108-shard-acts-s8k-bs3-tokamax-splash-layeragg-host-transfer-overlap48/2026_06_04_09_11_38` shows total time 92,786.7 ms, custom-call 23.8%, DKV 12.5%, forward residuals 11.2%, loop fusion 18.0%, data formatting 4.1%, and peak HBM 28.32 GiB. HLO matches the normal frontier program hash and 1,375,271-byte train-step text. Keep host-transfer overlap limit at 24.

## [2026-06-04] loop-iteration | v107-shard-acts-s8k-bs3-tokamax-splash-layeragg-shmem95 on 8B/v6e-8: refuted/tie (51,012 tok/s, 36.6% MFU; mild shared-memory cap returns to normal HLO/profile)

Workload `alekseyv-qwen3-v107-shmem95` kept the valid v090 qseq1 frontier fixed and changed only `--xla_tpu_scheduler_percent_shared_memory_limit=100` to `95`. It completed cleanly with normal loss ending at 12.0508, but reached only worker0 51,012 tok/s / 36.6% MFU and worker1 50,988 tok/s / 36.6% MFU, below v090's 51,033 best-observed and below the 51,026+ repeated v089/v090 band. XProf run `2026-06-04-qwen3-jax-v107-shard-acts-s8k-bs3-tokamax-splash-layeragg-shmem95/2026_06_04_09_00_58` shows total time 92,763.4 ms, custom-call 23.8%, DKV 12.5%, forward residuals 11.2%, loop fusion 18.0%, data formatting 4.1%, and peak HBM 28.32 GiB. HLO returned to the normal frontier program hash and 1,375,271-byte train-step text, unlike v106's lower-HBM limit-90 schedule. Keep the shared-memory limit at 100 for throughput.

## [2026-06-04] loop-iteration | v106-shard-acts-s8k-bs3-tokamax-splash-layeragg-shmem90 on 8B/v6e-8: refuted/tie (51,000 tok/s, 36.6% MFU; lower shared-memory cap saves HBM but slows wall time)

Workload `alekseyv-qwen3-v106-shmem90` kept the valid v090 qseq1 frontier fixed and changed only `--xla_tpu_scheduler_percent_shared_memory_limit=100` to `90`. It completed cleanly with normal loss ending at 12.0508, but reached only worker0 51,000 tok/s / 36.6% MFU and worker1 50,970 tok/s / 36.5% MFU, below v090's 51,033 best-observed and below the 51,026+ repeated v089/v090 band. XProf run `2026-06-04-qwen3-jax-v106-shard-acts-s8k-bs3-tokamax-splash-layeragg-shmem90/2026_06_04_08_50_51` shows total time 92,816.9 ms, custom-call 23.8%, DKV 12.6%, forward residuals 11.2%, loop fusion 18.1%, data formatting 4.1%, and peak HBM 28.15 GiB with 3.09 GiB free. The train-step program hash changed and optimized HLO text grew to 1,375,670 bytes, so this was a real lower-HBM schedule, but wall time regressed. Keep the shared-memory limit at 100 for throughput.

## [2026-06-04] loop-iteration | v105-shard-acts-s8k-bs3-tokamax-splash-layeragg-host-sendrecv64 on 8B/v6e-8: refuted/tie (51,006 tok/s, 36.6% MFU; lower host send/recv concurrency does not improve frontier)

Workload `alekseyv-qwen3-v105-hsr64` kept the valid v090 qseq1 frontier fixed and changed only `--xla_max_concurrent_host_send_recv=100` to `64`. It completed cleanly with normal loss ending at 12.0508, but reached only worker0 51,006 tok/s / 36.6% MFU and worker1 50,994 tok/s / 36.6% MFU, below v090's 51,033 best-observed and below the 51,026+ repeated v089/v090 band. XProf run `2026-06-04-qwen3-jax-v105-shard-acts-s8k-bs3-tokamax-splash-layeragg-host-sendrecv64/2026_06_04_08_40_49` shows total time 92,735.3 ms, custom-call 23.8%, DKV 12.6%, forward residuals 11.2%, loop fusion 18.0%, data formatting 4.1%, and peak HBM 28.32 GiB. The train-step optimized HLO text is 1,375,271 bytes, matching v089/v090/v102-v104, and the pod command confirms `--xla_max_concurrent_host_send_recv=64`. Keep the max concurrent host send/recv limit at 100; the lower limit is an in-band tie/slight regression.

## [2026-06-04] loop-iteration | v104-shard-acts-s8k-bs3-tokamax-splash-layeragg-host-transfer-overlap12 on 8B/v6e-8: refuted/tie (51,007 tok/s, 36.6% MFU; lower host-transfer overlap limit does not improve frontier)

Workload `alekseyv-qwen3-v104-hto12` kept the valid v090 qseq1 frontier fixed and changed only `--xla_tpu_host_transfer_overlap_limit=24` to `12`. It completed cleanly with normal loss ending at 12.0508, but reached only worker0 51,007 tok/s / 36.6% MFU and worker1 50,964 tok/s / 36.5% MFU, below v090's 51,033 best-observed and below the 51,026+ repeated v089/v090 band. XProf run `2026-06-04-qwen3-jax-v104-shard-acts-s8k-bs3-tokamax-splash-layeragg-host-transfer-overlap12/2026_06_04_08_30_23` shows total time 92,910.3 ms, custom-call 23.7%, DKV 12.5%, forward residuals 11.2%, loop fusion 18.0%, data formatting 4.1%, and peak HBM 28.32 GiB. The train-step optimized HLO text is 1,375,271 bytes, matching v089/v090/v102/v103, and the pod command confirms `--xla_tpu_host_transfer_overlap_limit=12`. Keep the overlap limit at 24; the lower limit is an in-band tie/slight regression.

## [2026-06-04] loop-iteration | v103-shard-acts-s8k-bs3-tokamax-splash-layeragg-no-all-experimental-scheduler on 8B/v6e-8: refuted/tie (51,004 tok/s, 36.6% MFU; broad scheduler bundle ablation does not improve frontier)

Workload `alekseyv-qwen3-v103-noallsched` kept the valid v090 qseq1 frontier fixed and changed only `--xla_tpu_enable_all_experimental_scheduler_features=true` to `false`, preserving explicit scoped VMEM, custom-call VMEM adjustment, selective resources, layer scheduler, sparse-core collective aggregator, memory-pressure tracking, AG backward pipelining, async collective fusion, and rerun3. It completed cleanly with normal loss ending at 12.0508, but reached only worker0 51,004 tok/s / 36.6% MFU and worker1 50,971 tok/s / 36.5% MFU. XProf run `2026-06-04-qwen3-jax-v103-shard-acts-s8k-bs3-tokamax-splash-layeragg-no-all-experimental-scheduler/2026_06_04_08_18_27` shows total time 92,877.8 ms, custom-call 23.7%, DKV 12.5%, forward residuals 11.2%, loop fusion 18.0%, data formatting 4.1%, collective-permute-start 1.0%, and peak HBM 28.32 GiB. HLO regenerated with 1,839 objects and train-step text 1,375,271 bytes, matching v089/v090/v102. Keep the broad scheduler bundle enabled operationally; disabling it is an in-band tie/slight regression.

## [2026-06-04] loop-iteration | v102-shard-acts-s8k-bs3-tokamax-splash-layeragg-no-scheduler-memory-pressure on 8B/v6e-8: refuted/tie (51,019 tok/s, 36.6% MFU; memory-pressure tracking ablation does not improve frontier)

Workload `alekseyv-qwen3-v102-nomemtrack` kept the valid v090 qseq1 frontier fixed and changed only `--xla_tpu_enable_scheduler_memory_pressure_tracking=true` to `false`. It completed cleanly with normal loss ending at 12.0508, but reached only worker0 51,019 tok/s / 36.6% MFU and worker1 50,997 tok/s / 36.6% MFU, below v090's 51,033 best-observed and below the 51,026+ repeated v089/v090 band. XProf run `2026-06-04-qwen3-jax-v102-shard-acts-s8k-bs3-tokamax-splash-layeragg-no-scheduler-memory-pressure/2026_06_04_08_08_33` shows total time 92,773.1 ms, custom-call 23.8%, DKV 12.5%, forward residuals 11.2%, loop fusion 18.0%, data formatting 4.1%, collective-permute-start 1.0%, and peak HBM 28.32 GiB. HLO regenerated with 1,839 objects and train-step text 1,375,271 bytes, matching v089/v090. Keep scheduler memory-pressure tracking enabled; this ablation is an in-band tie/slight regression.

## [2026-06-04] loop-iteration | v101-shard-acts-s8k-bs3-tokamax-splash-layeragg-no-async-collective-fusion on 8B/v6e-8: refuted (50,361 tok/s, 36.1% MFU; base async collective fusion is needed)

Workload `alekseyv-qwen3-v101-noacf` kept the valid v090 qseq1 frontier fixed and changed only the broad async collective fusion enable from `--xla_tpu_enable_async_collective_fusion=true` to `false`. It completed cleanly with normal loss ending at 12.0511, but regressed to worker0 50,361 tok/s / 36.1% MFU and worker1 50,322 tok/s / 36.1% MFU, below v090's 51,033 best-observed and 51,026+ repeated band. XProf run `2026-06-04-qwen3-jax-v101-shard-acts-s8k-bs3-tokamax-splash-layeragg-no-async-collective-fusion/2026_06_04_07_58_11` shows total time 94,108.0 ms, custom-call 23.5%, DKV 12.4%, forward residuals 11.1%, loop fusion 17.7%, all-gather 2.3%, data formatting 3.9%, and peak HBM down to 28.03 GiB. HLO regenerated with 1,839 objects and train-step text 1,187,819 bytes. Keep base async collective fusion enabled; the HBM reduction is not worth the all-gather/wall-time regression.

## [2026-06-04] loop-iteration | v100-shard-acts-s8k-bs3-tokamax-splash-qseq2-conservative-softmax on 8B/v6e-8: invalid (54,766 tok/s, 39.3% nominal MFU; combined conservative qseq2 NaNs at step 0)

Workload `alekseyv-qwen3-v100-qseqsafe` kept the v090 scheduler frontier and fast `TOKAMAX_Q_SEQ_SHARDS=2 TOKAMAX_MAX_LOGIT_CONST=30.0` path fixed, disabling both `TOKAMAX_USE_BASE2_EXP=0` and `TOKAMAX_FUSE_RECIPROCAL=0`. It completed with exit code 0 but produced NaN loss from step 0 on both workers, reaching worker0 54,732 tok/s / 39.2% nominal MFU and worker1 54,766 tok/s / 39.3% nominal MFU. XProf run `2026-06-04-qwen3-jax-v100-shard-acts-s8k-bs3-tokamax-splash-qseq2-conservative-softmax/2026_06_04_07_46_13` shows total time 86,481.5 ms, custom-call 17.3%, DKV 13.4%, forward residuals 3.9%, data formatting 4.4%, and peak HBM 28.32 GiB. HLO/runtime metadata confirms `q_seq_shards=2`, `use_base2_exp=false`, `fuse_reciprocal=false`, and `max_logit_const=30.0`. The combined conservative qseq2 softmax path does not restore correctness; retire qseq2 runtime numeric-control probes and pivot back to valid qseq1 frontier work.

## [2026-06-04] loop-iteration | v099-shard-acts-s8k-bs3-tokamax-splash-qseq2-no-fuse-recip on 8B/v6e-8: invalid (54,909 tok/s, 39.4% nominal MFU; no-fuse-recip NaNs at step 0)

Workload `alekseyv-qwen3-v099-qseqnorecip` kept the v090 scheduler frontier and fast `TOKAMAX_Q_SEQ_SHARDS=2 TOKAMAX_MAX_LOGIT_CONST=30.0` path fixed, changing only `TOKAMAX_FUSE_RECIPROCAL=0`. It completed with exit code 0 but produced NaN loss from step 0 on both workers, reaching worker0 54,909 tok/s / 39.4% nominal MFU and worker1 54,882 tok/s / 39.4% nominal MFU. XProf run `2026-06-04-qwen3-jax-v099-shard-acts-s8k-bs3-tokamax-splash-qseq2-no-fuse-recip/2026_06_04_07_40_14` shows total time 86,222.3 ms, custom-call 17.1%, DKV 13.4%, forward residuals 3.7%, and peak HBM 28.32 GiB. HLO metadata confirms `fuse_reciprocal=false`, `use_base2_exp=true`, and `max_logit_const=30.0`. Disabling fused reciprocal alone does not fix qseq2 correctness and makes the invalidity earlier.

## [2026-06-04] loop-iteration | v098-shard-acts-s8k-bs3-tokamax-splash-qseq2-no-base2 on 8B/v6e-8: invalid (55,574 tok/s, 39.8% nominal MFU; no-base2 does not fix qseq2 NaNs)

Workload `alekseyv-qwen3-v098-qseqnob2` kept the v090 scheduler frontier and fast `TOKAMAX_Q_SEQ_SHARDS=2 TOKAMAX_MAX_LOGIT_CONST=30.0` path fixed, changing only `TOKAMAX_USE_BASE2_EXP=0`. It completed with exit code 0 but produced NaN loss from step 1 on both workers, reaching worker0 55,574 tok/s / 39.8% nominal MFU and worker1 55,549 tok/s / 39.8% nominal MFU. XProf run `2026-06-04-qwen3-jax-v098-shard-acts-s8k-bs3-tokamax-splash-qseq2-no-base2/2026_06_04_07_34_21` shows the invalid-fast profile: total time 85,186.2 ms, custom-call 17.7%, DKV 13.6%, forward residuals 4.0%, loop fusion 19.6%, and peak HBM 28.32 GiB. HLO was regenerated with 32 objects; Tokamax metadata confirms `use_base2_exp=false`, `fuse_reciprocal=true`, and `max_logit_const=30.0`. The qseq2 NaN is not isolated to base2 exp.

## [2026-06-04] loop-iteration | v097-shard-acts-s8k-bs3-tokamax-splash-qseq2-dense-mask on 8B/v6e-8: invalid (54,832 tok/s, 39.3% nominal MFU; dense mask does not fix qseq2 NaNs)

Workload `alekseyv-qwen3-v097-qseqdense` replaced Tokamax's lazy causal mask with a dense static `NumpyMask(make_causal_mask(...))`, while keeping `TOKAMAX_Q_SEQ_SHARDS=2`, `TOKAMAX_MAX_LOGIT_CONST=30.0`, and the v090 scheduler frontier. It completed with exit code 0 but produced NaN loss from step 1 on both workers, reaching worker0 54,832 tok/s / 39.3% nominal MFU and worker1 54,805 tok/s / 39.3% nominal MFU. XProf run `2026-06-04-qwen3-jax-v097-shard-acts-s8k-bs3-tokamax-splash-qseq2-dense-mask/2026_06_04_07_26_25` shows the invalid-fast profile: total time 86,305.3 ms, custom-call 18.6%, DKV 13.8%, forward residuals 4.8%, loop fusion 19.3%, and peak HBM 28.33 GiB. HLO was regenerated with 32 objects and train-step text 1,377,637 bytes; static mask operands appear in the Tokamax custom-call signatures. Dense masking is not the qseq2 correctness fix; do not continue mask-only qseq2 retries.

## [2026-06-04] loop-iteration | v096-shard-acts-s8k-bs3-tokamax-splash-sparsecore-aggregator-only on 8B/v6e-8: refuted/tie (51,011 tok/s, 36.6% MFU; aggregator alone does not explain v089/v090)

Workload `alekseyv-qwen3-v096-aggonly` kept the v074/v075 scheduler frontier fixed and added only `--xla_tpu_enable_sparse_core_collective_aggregator=true`, without explicitly enabling the layer scheduler. It completed cleanly with normal loss, but reached only worker0 51,011 tok/s / 36.6% MFU and worker1 50,989 tok/s / 36.6% MFU, below the old v074/v075 51,021 band and the v089/v090 repeated 51,026+ band. XProf run `2026-06-04-qwen3-jax-v096-shard-acts-s8k-bs3-tokamax-splash-sparsecore-aggregator-only/2026_06_04_07_11_26` shows MXU 56.8%, total time 92,799.7 ms, custom-call 23.8%, DKV 12.5%, forward residuals 11.2%, loop fusion 18.0%, and peak HBM 28.32 GiB. HLO was regenerated with 1,619 objects and train-step text 1,375,271 bytes; the flagfile confirms sparse-core aggregator true while the layer scheduler stayed auto. The v089/v090 gain requires the explicit layer-scheduler plus sparse-core-aggregator pair.

## [2026-06-04] loop-iteration | v095-shard-acts-s8k-bs3-tokamax-splash-layeragg-dependent-collectives on 8B/v6e-8: refuted/tie (51,018 tok/s, 36.6% MFU; no dependent-collectives interaction win)

Workload `alekseyv-qwen3-v095-layerdepcoll` kept the v090 paired layer-scheduler/sparse-core-aggregator frontier fixed and added only `--xla_tpu_enable_layer_scheduler_for_dependent_collectives=true`. It completed cleanly with normal loss, but reached only worker0 51,018 tok/s / 36.6% MFU and worker1 50,999 tok/s / 36.6% MFU, below the v089/v090 repeated 51,026+ band. XProf run `2026-06-04-qwen3-jax-v095-shard-acts-s8k-bs3-tokamax-splash-layeragg-dependent-collectives/2026_06_04_07_00_59` shows MXU 56.8%, total time 92,777.4 ms, custom-call 23.8%, DKV 12.5%, forward residuals 11.2%, loop fusion 18.0%, and peak HBM 28.32 GiB. HLO was regenerated with 1,619 objects and train-step text 1,375,271 bytes; the flagfile confirms dependent-collectives scheduling true with ILP/BRKGA/multi-compute/resource-serializing off/auto. Do not carry dependent collectives forward.

## [2026-06-04] loop-iteration | v094-shard-acts-s8k-bs3-tokamax-splash-layeragg-multicompute-overlap on 8B/v6e-8: refuted/tie (51,022 tok/s, 36.6% MFU; no layeragg interaction win)

Workload `alekseyv-qwen3-v094-layermulticomp` kept the v090 paired layer-scheduler/sparse-core-aggregator frontier fixed and added only `--xla_tpu_enable_multi_compute_overlap_in_layer_scheduler=true`. It completed cleanly with normal loss, but reached only worker0 51,022 tok/s / 36.6% MFU and worker1 50,996 tok/s / 36.6% MFU, below the v089/v090 repeated 51,026+ band. XProf run `2026-06-04-qwen3-jax-v094-shard-acts-s8k-bs3-tokamax-splash-layeragg-multicompute-overlap/2026_06_04_06_50_01` shows MXU 56.8%, total time 92,781.2 ms, custom-call 23.8%, DKV 12.5%, forward residuals 11.2%, loop fusion 18.0%, and peak HBM 28.32 GiB. HLO was regenerated with 1,619 objects and train-step text 1,375,271 bytes; the flagfile confirms multi-compute overlap true with ILP/BRKGA/resource-serializing off/auto. Do not carry multi-compute overlap forward.

## [2026-06-04] loop-iteration | v093-shard-acts-s8k-bs3-tokamax-splash-layeragg-resource-serializing on 8B/v6e-8: refuted/tie (50,977 tok/s, 36.6% MFU; less HBM but slower)

Workload `alekseyv-qwen3-v093-layerresserial` kept the v090 paired layer-scheduler/sparse-core-aggregator frontier fixed and added only `--xla_latency_hiding_scheduler_resource_serializing=true`. It completed cleanly with normal loss, but reached only worker0 50,977 tok/s / 36.6% MFU and worker1 50,961 tok/s / 36.5% MFU, below the v089/v090 repeated 51,026+ band. XProf run `2026-06-04-qwen3-jax-v093-shard-acts-s8k-bs3-tokamax-splash-layeragg-resource-serializing/2026_06_04_06_38_36` shows MXU 56.8%, total time 92,810.4 ms, custom-call 23.8%, DKV 12.5%, forward residuals 11.2%, loop fusion 18.1%, and peak HBM 28.03 GiB. HLO was regenerated with 1,619 objects and train-step text shrank to 1,303,468 bytes; the flagfile confirms resource serializing true with ILP/BRKGA auto. Do not carry resource serializing for throughput.

## [2026-06-04] loop-iteration | v092-shard-acts-s8k-bs3-tokamax-splash-layeragg-ilp-lhs on 8B/v6e-8: refuted/tie (51,016 tok/s, 36.6% MFU; ILP does not compose with layeragg frontier)

Workload `alekseyv-qwen3-v092-layerilp` kept the v090 paired layer-scheduler/sparse-core-aggregator frontier fixed and added only `--xla_tpu_enable_ilp_latency_hiding_scheduler=true`. It completed cleanly with normal loss, but reached only worker0 51,016 tok/s / 36.6% MFU and worker1 50,991 tok/s / 36.6% MFU, below the v089/v090 repeated 51,026+ band. XProf run `2026-06-04-qwen3-jax-v092-shard-acts-s8k-bs3-tokamax-splash-layeragg-ilp-lhs/2026_06_04_06_27_38` shows MXU 56.8%, total time 92,768.7 ms, custom-call 23.8%, DKV 12.5%, forward residuals 11.2%, loop fusion 18.0%, and peak HBM 28.32 GiB. HLO was regenerated with 1,615 objects and train-step text 1,375,271 bytes; the flagfile confirms layeragg pair plus ILP true and BRKGA auto. Do not carry ILP forward.

## [2026-06-04] loop-iteration | v091-shard-acts-s8k-bs3-tokamax-splash-layeragg-brkga-lhs on 8B/v6e-8: refuted/tie (51,020 tok/s, 36.6% MFU; BRKGA does not compose with layeragg frontier)

Workload `alekseyv-qwen3-v091-layerbrkga` kept the v090 paired layer-scheduler/sparse-core-aggregator frontier fixed and added only `--xla_tpu_enable_brkga_latency_hiding_scheduler=true`. It completed cleanly with normal loss, but reached only worker0 51,002 tok/s / 36.6% MFU and worker1 51,020 tok/s / 36.6% MFU, below the v089/v090 repeated 51,026+ band. XProf run `2026-06-04-qwen3-jax-v091-shard-acts-s8k-bs3-tokamax-splash-layeragg-brkga-lhs/2026_06_04_06_14_51` shows MXU 56.8%, total time 92,734.8 ms, custom-call 23.8%, DKV 12.6%, forward residuals 11.2%, loop fusion 18.0%, and peak HBM 28.32 GiB. HLO was regenerated with 1,619 objects and train-step text 1,375,271 bytes; the flagfile confirms BRKGA enabled on top of the v090 pair. Do not carry BRKGA forward.

## [2026-06-04] loop-iteration | v090-shard-acts-s8k-bs3-tokamax-splash-layer-scheduler-sparsecore-aggregator-rerun on 8B/v6e-8: supported (51,033 tok/s best worker, 36.6% MFU; confirms v089 pair)

Workload `alekseyv-qwen3-v090-layeragg-rerun` exactly reran v089 with `--xla_tpu_enable_latency_hiding_layer_scheduler=true --xla_tpu_enable_sparse_core_collective_aggregator=true` on top of the v074/v075 frontier stack. It completed cleanly with normal loss, reaching worker0 51,033 tok/s / 36.6% MFU and worker1 51,005 tok/s / 36.6% MFU. XProf run `2026-06-04-qwen3-jax-v090-shard-acts-s8k-bs3-tokamax-splash-layer-scheduler-sparsecore-aggregator-rerun/2026_06_04_06_03_39` shows MXU 56.8%, total time 92,767.7 ms, custom-call 23.8%, DKV 12.5%, forward residuals 11.2%, loop fusion 18.0%, and peak HBM 28.32 GiB. No HLO objects were written, consistent with a persistent-cache hit; use v089's HLO/flagfile for this exact compiled program. Adopt the paired layer-scheduler/sparse-core-aggregator flags as the new supported scheduler frontier, while treating the exact 51,033 tok/s high-water mark as small/noisy.

## [2026-06-04] loop-iteration | v089-shard-acts-s8k-bs3-tokamax-splash-layer-scheduler-sparsecore-aggregator on 8B/v6e-8: supported/provisional (51,026 tok/s best observed, 36.6% MFU; needs rerun)

Workload `alekseyv-qwen3-v089-layeragg` kept the confirmed v074/v075 scheduler frontier fixed and added the required pair `--xla_tpu_enable_latency_hiding_layer_scheduler=true --xla_tpu_enable_sparse_core_collective_aggregator=true`. It completed cleanly with normal loss, reaching worker0 51,026 tok/s / 36.6% MFU and worker1 51,000 tok/s / 36.6% MFU. XProf run `2026-06-04-qwen3-jax-v089-shard-acts-s8k-bs3-tokamax-splash-layer-scheduler-sparsecore-aggregator/2026_06_04_05_56_44` shows MXU 56.8%, total time 92,770.4 ms, custom-call 23.8%, DKV 12.5%, forward residuals 11.2%, loop fusion 18.0%, and peak HBM 28.32 GiB. HLO was regenerated with 1,619 objects and train-step text 1,375,271 bytes; the flagfile confirms layer scheduler and sparse-core collective aggregator enabled while ILP/BRKGA and collective-fusion variants stayed auto/off. This is a tiny new best-observed worker but not durable; rerun before adoption.

## [2026-06-04] loop-iteration | v088-shard-acts-s8k-bs3-tokamax-splash-latency-hiding-layer-scheduler on 8B/v6e-8: invalid (standalone layer scheduler requires sparse-core aggregator)

Workload `alekseyv-qwen3-v088-layerlhs` kept the confirmed v074/v075 scheduler frontier fixed and added only `--xla_tpu_enable_latency_hiding_layer_scheduler=true`. It failed during startup before train-step compile with `INVALID_ARGUMENT: Latency hiding layer scheduler requires sparse core collective aggregator to be enabled`, exit code 1. Only 17 startup HLO artifacts were written and no train-step HLO/profile exists. The standalone layer-scheduler flag is invalid; follow-up can test the explicit pair with `--xla_tpu_enable_sparse_core_collective_aggregator=true`.

## [2026-06-04] loop-iteration | v087-shard-acts-s8k-bs3-tokamax-splash-brkga-latency-hiding-scheduler on 8B/v6e-8: supported/tie (51,023 tok/s best worker, 36.6% MFU; no profile breakout)

Workload `alekseyv-qwen3-v087-brkgalhs` kept the confirmed v074/v075 scheduler frontier fixed and added only `--xla_tpu_enable_brkga_latency_hiding_scheduler=true`. It completed cleanly with normal loss, reaching worker0 51,023 tok/s / 36.6% MFU and worker1 50,997 tok/s / 36.6% MFU. XProf run `2026-06-04-qwen3-jax-v087-shard-acts-s8k-bs3-tokamax-splash-brkga-latency-hiding-scheduler/2026_06_04_05_42_06` shows MXU 56.8%, total time 92,790.1 ms, custom-call 23.8%, DKV 12.5%, forward residuals 11.2%, loop fusion 18.0%, and peak HBM 28.32 GiB. HLO was regenerated with 1,655 objects and train-step text 1,375,271 bytes; the flagfile confirms BRKGA LHS enabled while ILP and the v076-v085 collective/scheduler variants stayed auto/off. This is a valid in-band tie, not a durable frontier; it does not exceed v078's 51,025 tok/s best-observed worker.

## [2026-06-04] loop-iteration | v086-shard-acts-s8k-bs3-tokamax-splash-ilp-latency-hiding-scheduler on 8B/v6e-8: supported/tie (51,016 tok/s, 36.6% MFU; ILP LHS in-band but no frontier)

Workload `alekseyv-qwen3-v086-ilplhs` kept the confirmed v074/v075 scheduler frontier fixed and added only `--xla_tpu_enable_ilp_latency_hiding_scheduler=true`. It completed cleanly with normal loss, reaching worker0 51,016 tok/s / 36.6% MFU and worker1 50,989 tok/s / 36.6% MFU. XProf run `2026-06-04-qwen3-jax-v086-shard-acts-s8k-bs3-tokamax-splash-ilp-latency-hiding-scheduler/2026_06_04_05_31_51` shows MXU 56.8%, total time 92,818.8 ms, custom-call 23.8%, DKV 12.5%, forward residuals 11.2%, loop fusion 18.0%, and peak HBM 28.32 GiB. HLO was regenerated with 1,619 objects and train-step text 1,375,271 bytes; the flagfile confirms ILP LHS enabled while the v076-v085 collective/scheduler variants stayed off/auto. This is valid but not a new frontier.

## [2026-06-04] loop-iteration | v085-shard-acts-s8k-bs3-tokamax-splash-staged-collective-compute-pipelining on 8B/v6e-8: invalid (libtpu CollectiveComputeDecomposer SIGSEGV; exit 139)

Workload `alekseyv-qwen3-v085-stagedcc` kept the confirmed v074/v075 scheduler frontier fixed and added only `--xla_tpu_enable_staged_collective_compute_pipelining=true`. It failed during train-step compile before any measured step. The retained worker exited 139 with a SIGSEGV in `xla::jellyfish::fusion_util::IsDynamicUpdateSliceAligned`, called from `GetStagedAllGatherDotInfo`, `CollectiveComputeDecomposer::DecomposeAllGatherEinsums`, and `CollectiveComputeDecomposer::RunImpl`. Partial HLO artifacts were written with 1,600 objects, including train-step before-optimizations and a flagfile but no train-step after-optimizations. The flagfile confirms staged collective-compute pipelining forced true while collective-compute fusion stayed auto and the v076-v084 variants stayed off/auto. Do not carry this flag forward.

## [2026-06-04] loop-iteration | v084-shard-acts-s8k-bs3-tokamax-splash-async-start-done-only on 8B/v6e-8: refuted (50,656 tok/s, 36.3% MFU; start/done-only async fusion regresses)

Workload `alekseyv-qwen3-v084-startdone` kept the confirmed v074/v075 scheduler frontier fixed and added only `--xla_tpu_async_collective_fusion_with_start_done_only=true`, without carrying v076-v083 refuted/tie async scheduler variants. It completed cleanly with normal loss, but regressed to worker0 50,629 tok/s / 36.3% MFU and worker1 50,656 tok/s / 36.3% MFU. XProf run `2026-06-04-qwen3-jax-v084-shard-acts-s8k-bs3-tokamax-splash-async-start-done-only/2026_06_04_05_11_46` shows MXU 56.4%, total time 93,503.4 ms, custom-call 23.6%, DKV 12.5%, forward residuals 11.2%, loop fusion 17.7%, data formatting 3.9%, and peak HBM 28.32 GiB. HLO was regenerated with 1,619 objects and train-step text shrank to 1,215,911 bytes; the flagfile confirms start/done-only enabled while multiple-collective, kloop, Mosaic, resource-sharing, and collective-compute variants remained off/auto. Do not carry this flag forward.

## [2026-06-04] loop-iteration | v083-shard-acts-s8k-bs3-tokamax-splash-fuse-kloop-fusions on 8B/v6e-8: invalid (libtpu async all-gather emitter crash; exit 139)

Workload `alekseyv-qwen3-v083-kloopacf` kept the confirmed v074/v075 scheduler frontier fixed and added only `--xla_tpu_enable_async_collective_fusion_fuse_kloop_fusions=true`. It failed during train-step compile before any measured step. Worker 0 exited 139 with `async_all_gather_emitter.cc:52 Check failed: Init(false) is OK (INVALID_ARGUMENT: Unable to find constant of type: 3)`. Partial HLO artifacts were written, including `module_0267.jit_train_step.cl_854318611.after_optimizations.txt` at 1,664,434 bytes and 1,604 objects; the flagfile confirms kloop async-fusion forced true while multiple-collectives fusion and collective-compute fusion stayed off/auto. Do not carry this flag forward.

## [2026-06-04] loop-iteration | v082-shard-acts-s8k-bs3-tokamax-splash-fuse-multiple-collectives on 8B/v6e-8: supported/tie (51,021 tok/s, 36.6% MFU; no async multi-collective win)

Workload `alekseyv-qwen3-v082-fusemulti` kept the confirmed v074/v075 scheduler frontier fixed and added only `--xla_tpu_async_collective_fusion_fuse_multiple_collectives=true`, without carrying v081 collective-compute fusion or other refuted scheduler variants. It completed cleanly with normal loss, reaching worker0 50,995 tok/s / 36.6% MFU and worker1 51,021 tok/s / 36.6% MFU. XProf run `2026-06-04-qwen3-jax-v082-shard-acts-s8k-bs3-tokamax-splash-fuse-multiple-collectives/2026_06_04_04_47_42` shows MXU 56.8%, custom-call 23.8%, DKV 12.5%, forward residuals 11.2%, loop fusion 18.0%, collective-permute-start 884.1 ms / 1.0%, and peak HBM 28.32 GiB. HLO was regenerated with 2,218 objects and the flagfile confirms multiple-collective fusion enabled. This is a valid tie, not a new frontier.

## [2026-06-04] loop-iteration | v081-shard-acts-s8k-bs3-tokamax-splash-collective-compute-fusion on 8B/v6e-8: refuted (42,555 tok/s, 30.5% MFU; convolution fusion regresses hard)

Workload `alekseyv-qwen3-v081-ccfusion` kept the confirmed v074/v075 scheduler frontier fixed and added only `--xla_tpu_enable_collective_compute_fusion=true`. It completed cleanly with normal loss, but regressed to worker0 42,538 tok/s / 30.5% MFU and worker1 42,555 tok/s / 30.5% MFU. XProf run `2026-06-04-qwen3-jax-v081-shard-acts-s8k-bs3-tokamax-splash-collective-compute-fusion/2026_06_04_04_36_57` shows MXU falling to 47.4%, convolution fusion rising to 62,001.6 ms / 55.8%, custom-call 19.8%, loop fusion 16.5%, and peak HBM 28.34 GiB. HLO was regenerated with train-step text shrinking to 1,254,662 bytes and the flagfile confirms collective-compute fusion forced true. This flag severely hurts matmul/convolution efficiency; do not carry it forward.

## [2026-06-04] loop-iteration | v080-shard-acts-s8k-bs3-tokamax-splash-dependent-collectives-layer-scheduler on 8B/v6e-8: refuted/tie (51,017 tok/s, 36.6% MFU; no dependent-collective win)

Workload `alekseyv-qwen3-v080-depcoll` kept the confirmed v074/v075 scheduler frontier fixed and added only `--xla_tpu_enable_layer_scheduler_for_dependent_collectives=true`, without carrying v076 resource sharing, v077 Mosaic async collective fusion, or v078/v079 multi-compute overlap. It completed cleanly with normal loss, reaching worker0 50,997 tok/s / 36.6% MFU and worker1 51,017 tok/s / 36.6% MFU, below the confirmed 51,021 tok/s band. XProf run `2026-06-04-qwen3-jax-v080-shard-acts-s8k-bs3-tokamax-splash-dependent-collectives-layer-scheduler/2026_06_04_04_25_41` shows MXU 56.8%, custom-call 23.8%, DKV 12.5%, forward residuals 11.2%, loop fusion 18.0%, collective-permute-start 1.0%, and peak HBM 28.32 GiB. HLO was regenerated with 1,619 objects and the flagfile confirms dependent-collectives layer scheduling enabled. The flag is valid but does not improve the frontier.

## [2026-06-04] loop-iteration | v079-shard-acts-s8k-bs3-tokamax-splash-layer-multicompute-overlap-rerun on 8B/v6e-8: supported/tie (51,021 tok/s, 36.6% MFU; confirms in-band but not new frontier)

Workload `alekseyv-qwen3-v079-multicomp-rerun` exactly reran v078 with `--xla_tpu_enable_multi_compute_overlap_in_layer_scheduler=true`. It completed cleanly with normal loss, reaching worker0 51,021 tok/s / 36.6% MFU and worker1 51,009 tok/s / 36.6% MFU. XProf run `2026-06-04-qwen3-jax-v079-shard-acts-s8k-bs3-tokamax-splash-layer-multicompute-overlap-rerun/2026_06_04_04_14_25` shows MXU 56.7%, custom-call 23.8%, DKV 12.6%, forward residuals 11.2%, loop fusion 18.0%, and peak HBM 28.32 GiB. No HLO objects were written, consistent with a cache hit from v078. Multi-compute overlap is non-harmful and in the frontier band, but the v078 51,025 tok/s single-worker high-water mark did not repeat; keep the durable frontier at the 51,021 tok/s band.

## [2026-06-04] loop-iteration | v078-shard-acts-s8k-bs3-tokamax-splash-layer-multicompute-overlap on 8B/v6e-8: supported/provisional (51,025 tok/s, 36.6% MFU; tiny single-worker high-water mark)

Workload `alekseyv-qwen3-v078-multicompute` kept the confirmed v074/v075 scheduler frontier fixed and added only `--xla_tpu_enable_multi_compute_overlap_in_layer_scheduler=true`, without carrying v076 resource sharing or v077 Mosaic async collective fusion. It completed cleanly with normal loss, reaching worker0 51,025 tok/s / 36.6% MFU and worker1 51,001 tok/s / 36.6% MFU. XProf run `2026-06-04-qwen3-jax-v078-shard-acts-s8k-bs3-tokamax-splash-layer-multicompute-overlap/2026_06_04_04_08_02` shows MXU 56.8%, custom-call 23.8%, DKV 12.5%, forward residuals 11.2%, loop fusion 18.0%, and peak HBM 28.32 GiB. HLO was regenerated with 1,615 objects and the flagfile confirms multi-compute overlap enabled. This is a tiny best-worker improvement only; rerun before banking it.

## [2026-06-04] loop-iteration | v077-shard-acts-s8k-bs3-tokamax-splash-mosaic-async-collective-fusion on 8B/v6e-8: refuted (50,976 tok/s, 36.5% MFU; custom-call reclassified but wall time worsened)

Workload `alekseyv-qwen3-v077-mosaicacf` kept the confirmed v074/v075 scheduler frontier fixed and added only `--xla_tpu_enable_async_collective_fusion_with_mosaic_custom_call=true`, without carrying v076 resource sharing. It completed cleanly with normal loss, but reached only worker0 50,976 tok/s / 36.5% MFU and worker1 50,952 tok/s / 36.5% MFU. XProf run `2026-06-04-qwen3-jax-v077-shard-acts-s8k-bs3-tokamax-splash-mosaic-async-collective-fusion/2026_06_04_03_57_07` shows MXU 56.8%, custom-call 18.2%, custom fusion 6.3%, loop fusion 18.0%, data formatting 4.0%, and peak HBM 28.32 GiB. The flag was active and regenerated HLO (1,611 objects; train step 1,384,405 bytes), but it mostly moved work from custom-call into custom fusion and regressed wall time. Do not carry it forward.

## [2026-06-04] loop-iteration | v076-shard-acts-s8k-bs3-tokamax-splash-resource-sharing on 8B/v6e-8: refuted (51,017 tok/s, 36.6% MFU; no gain over v074/v075)

Workload `alekseyv-qwen3-v076-sharing` kept the confirmed v074/v075 scheduler frontier fixed and added only `--xla_latency_hiding_scheduler_resource_sharing=true`. It completed cleanly with normal loss, reaching worker0 50,980 tok/s / 36.6% MFU and worker1 51,017 tok/s / 36.6% MFU, just below the confirmed 51,021 tok/s best. XProf runs `2026-06-04-qwen3-jax-v076-shard-acts-s8k-bs3-tokamax-splash-resource-sharing/2026_06_04_03_44_03` and `.../2026_06_04_03_44_04` show MXU 56.7%, custom-call 23.7-23.8%, DKV 12.5-12.6%, forward residuals 11.2%, loop fusion 18.0%, and peak HBM 28.32 GiB. HLO was regenerated with 2,226 objects and the flagfile confirms resource sharing enabled, so this was a real schedule variant; keep selective resources but do not carry resource sharing forward.

## [2026-06-04] loop-iteration | v075-shard-acts-s8k-bs3-tokamax-splash-selective-resources-rerun on 8B/v6e-8: supported (51,021 tok/s, 36.6% MFU; confirms v074 frontier)

Workload `alekseyv-qwen3-v075-selective-rerun` exactly reran v074 with VMEM 81920, custom-call scoped VMEM adjustments, and selective resources enabled. It completed cleanly with normal loss, reaching worker0 50,991 tok/s / 36.6% MFU and worker1 51,021 tok/s / 36.6% MFU. XProf run `2026-06-04-qwen3-jax-v075-shard-acts-s8k-bs3-tokamax-splash-selective-resources-rerun/2026_06_04_03_33_18` shows MXU 56.7%, custom-call 23.8%, DKV 12.5%, forward residuals 11.2%, loop fusion 18.0%, and peak HBM 28.32 GiB. No new HLO objects were written, consistent with a cache hit. Carry the v074/v075 scheduler stack forward.

## [2026-06-04] loop-iteration | v074-shard-acts-s8k-bs3-tokamax-splash-selective-resources on 8B/v6e-8: supported (51,021 tok/s, 36.6% MFU; new scheduler frontier)

Workload `alekseyv-qwen3-v074-selective` kept the v072/v073 stack fixed and added only `--xla_latency_hiding_scheduler_enable_selective_resources=true`. It completed cleanly with normal loss, improving to worker0 51,013 tok/s / 36.6% MFU and worker1 51,021 tok/s / 36.6% MFU. XProf run `2026-06-04-qwen3-jax-v074-shard-acts-s8k-bs3-tokamax-splash-selective-resources/2026_06_04_03_26_50` shows MXU 56.8%, custom-call 23.8%, DKV 12.6%, forward residuals 11.2%, loop fusion 18.0%, and peak HBM 28.32 GiB. The op mix is similar to v072/v073 but profile total time drops; rerun this exact stack before declaring it durable.

## [2026-06-04] loop-iteration | v073-shard-acts-s8k-bs3-tokamax-splash-customcall-vmem-adjust-rerun on 8B/v6e-8: supported (50,789 tok/s, 36.4% MFU; confirms v072 band)

Workload `alekseyv-qwen3-v073-ccvmem-rerun` exactly reran v072 with VMEM 81920 and `--xla_tpu_enable_tpu_custom_call_scoped_vmem_adjustments=true`. It completed cleanly with normal loss, reaching worker0 50,779 tok/s / 36.4% MFU and worker1 50,789 tok/s / 36.4% MFU. XProf run `2026-06-04-qwen3-jax-v073-shard-acts-s8k-bs3-tokamax-splash-customcall-vmem-adjust-rerun/2026_06_04_03_03_54` shows MXU 56.4%, custom-call 23.7%, DKV 12.5%, forward residuals 11.2%, loop fusion 17.9%, and peak HBM 28.32 GiB. No new HLO objects were written, consistent with a persistent cache hit. Keep the custom-call scoped-VMEM flag enabled; the gain versus v068 is real enough to carry forward but very small.

## [2026-06-04] loop-iteration | v072-shard-acts-s8k-bs3-tokamax-splash-customcall-vmem-adjust on 8B/v6e-8: supported/provisional (50,793 tok/s, 36.4% MFU; tiny new frontier)

Workload `alekseyv-qwen3-v072-ccvmem` kept the v068 frontier fixed at VMEM 81920 and added only `--xla_tpu_enable_tpu_custom_call_scoped_vmem_adjustments=true`. It completed cleanly with normal loss, reaching worker0 50,793 tok/s / 36.4% MFU and worker1 50,767 tok/s / 36.4% MFU, a tiny improvement over v068's 50,790 / 50,763 tok/s. XProf run `2026-06-04-qwen3-jax-v072-shard-acts-s8k-bs3-tokamax-splash-customcall-vmem-adjust/2026_06_04_02_57_24` shows MXU 56.6%, custom-call 23.7%, DKV 12.5%, forward residuals 11.2%, loop fusion 18.0%, and peak HBM 28.32 GiB. Treat this as a provisional frontier and rerun the same configuration before banking the gain.

## [2026-06-04] loop-iteration | v071-shard-acts-s8k-bs3-tokamax-splash-vmem73728 on 8B/v6e-8: refuted (50,174 tok/s, 36.0% MFU; lower midpoint still below v068)

Workload `alekseyv-qwen3-v071-vmem73728` kept the v068 frontier fixed and changed only `--xla_tpu_scoped_vmem_limit_kib=81920` to `73728`. It completed cleanly with normal loss, reaching worker0 50,174 tok/s / 36.0% MFU and worker1 50,155 tok/s / 36.0% MFU. XProf run `2026-06-04-qwen3-jax-v071-shard-acts-s8k-bs3-tokamax-splash-vmem73728/2026_06_04_02_47_45` shows MXU 56.0%, custom-call 23.3%, DKV 12.3%, forward residuals 10.9%, loop fusion 17.4%, and peak HBM 28.31 GiB. The VMEM sweep is bracketed: 65536, 73728, 90112, and 98304 all lose to 81920. Move next to custom-call scheduler behavior at the v068 frontier point.

## [2026-06-04] loop-iteration | v070-shard-acts-s8k-bs3-tokamax-splash-vmem90112 on 8B/v6e-8: refuted (50,034 tok/s, 35.9% MFU; upper VMEM bracket slower)

Workload `alekseyv-qwen3-v070-vmem90112` kept the v068 frontier fixed and changed only `--xla_tpu_scoped_vmem_limit_kib=81920` to `90112`. It completed cleanly with normal loss, but reached only worker0 50,015 tok/s / 35.9% MFU and worker1 50,034 tok/s / 35.9% MFU. XProf run `2026-06-04-qwen3-jax-v070-shard-acts-s8k-bs3-tokamax-splash-vmem90112/2026_06_04_02_37_40` shows MXU 55.7%, custom-call 23.3%, DKV 12.3%, forward residuals 11.0%, loop fusion 18.6%, and peak HBM 28.32 GiB. The upper-side bracket regresses; keep VMEM 81920 as the current frontier and finish with one lower-side midpoint before moving back to structural attention work.

## [2026-06-04] loop-iteration | v069-shard-acts-s8k-bs3-tokamax-splash-vmem65536 on 8B/v6e-8: refuted (49,785 tok/s, 35.7% MFU; VMEM cap too low)

Workload `alekseyv-qwen3-v069-vmem65536` kept the v068 frontier fixed and changed only `--xla_tpu_scoped_vmem_limit_kib=81920` to `65536`. It completed cleanly with normal loss, but regressed to worker0 49,785 tok/s / 35.7% MFU and worker1 49,761 tok/s / 35.7% MFU. XProf run `2026-06-04-qwen3-jax-v069-shard-acts-s8k-bs3-tokamax-splash-vmem65536/2026_06_04_02_25_54` shows MXU 55.6%, custom-call 23.0%, DKV 12.2%, forward residuals 10.8%, loop fusion 17.2%, and peak HBM 28.30 GiB. The lower cap increases wall-clock despite valid loss; keep VMEM 81920 as the current frontier and test one upper-side bracket point before leaving the VMEM axis.

## [2026-06-04] loop-iteration | v068-shard-acts-s8k-bs3-tokamax-splash-vmem81920 on 8B/v6e-8: supported (50,790 tok/s, 36.4% MFU; new VMEM frontier)

Workload `alekseyv-qwen3-v068-vmem81920` kept the v056 frontier fixed and changed only `--xla_tpu_scoped_vmem_limit_kib=98304` to `81920`, while keeping Tokamax Splash, `TOKAMAX_MAX_LOGIT_CONST=30.0`, scheduler rerun3, and AG backward pipelining enabled. It completed cleanly with normal loss, improving worker0 to 50,790 tok/s / 36.4% MFU and worker1 to 50,763 tok/s / 36.4% MFU. XProf run `2026-06-04-qwen3-jax-v068-shard-acts-s8k-bs3-tokamax-splash-vmem81920/2026_06_04_02_14_42` shows MXU 56.5%, custom-call 23.7%, DKV 12.5%, forward residuals 11.2%, loop fusion 18.0%, data formatting 4.1%, and peak HBM 28.32 GiB. Keep VMEM 81920 as the current frontier and follow with the lower VMEM sweep point, 65536 KiB.

## [2026-06-03] loop-iteration | v067-shard-acts-s8k-bs3-tokamax-splash-no-ag-bwd-pipeline on 8B/v6e-8: refuted (50,367 tok/s, 36.1% MFU; AG backward pipelining helps or is neutral)

Workload `alekseyv-qwen3-v067` kept the valid v056 frontier fixed and changed only `--xla_tpu_enable_ag_backward_pipelining=true` to `false`, testing whether the inherited scheduler bundle is adding overhead on the qseq1/FSDP8 shape. It completed cleanly with normal loss, but worker0 reached only 50,367 tok/s / 36.1% MFU and worker1 50,343 tok/s / 36.1% MFU, below v056. XProf run `2026-06-03-qwen3-jax-v067-shard-acts-s8k-bs3-tokamax-splash-no-ag-bwd-pipeline/2026_06_03_21_40_57` shows the same profile shape as v056: MXU 55.9%, custom-call 23.4%, DKV 12.3%, forward residuals 11.1%, data formatting 4.0%, and HBM 28.33 GiB. Keep AG backward pipelining enabled.

## [2026-06-03] loop-iteration | v066-shard-acts-s8k-bs3-tokamax-splash-save-residuals-unwrap on 8B/v6e-8: invalid (save_residuals unwrap breaks Tokamax backward tuple handling)

Workload `alekseyv-qwen3-v066` retried the v065 save-residuals hypothesis with the wrapper API fixed: when Tokamax returns `(output, stats)`, return only `output` through shard_map. Runtime kept v056 fixed with `TOKAMAX_MAX_LOGIT_CONST=30.0`, scheduler rerun3, qseq1, and `TOKAMAX_SAVE_RESIDUALS=1`. It failed before step 0 with `AttributeError: 'tuple' object has no attribute 'astype'` inside Tokamax `_splash_attention_bwd`, so unwrapping the output is not sufficient. No profile or HLO was produced. Do not continue save-residuals without a deeper custom-VJP wrapper.

## [2026-06-03] loop-iteration | v065-shard-acts-s8k-bs3-tokamax-splash-save-residuals on 8B/v6e-8: invalid (Tokamax save_residuals returns stats tuple; shard_map out_specs mismatch)

Workload `alekseyv-qwen3-v065` tested whether Tokamax `save_residuals=True` can reduce the valid qseq1 frontier's forward-residual custom-call bucket. The image added a `TOKAMAX_SAVE_RESIDUALS` env gate and passed it to `make_splash_mha`; runtime kept v056 fixed with `TOKAMAX_MAX_LOGIT_CONST=30.0`, scheduler rerun3, and qseq1. It failed before step 0 on both workers with a shard_map `out_specs` mismatch because `save_residuals=True` returns `(output, stats)` and the wrapper still returned the full tuple through a 4D output spec. No profile or HLO was produced. Follow with a wrapper fix that unwraps the attention output before returning from shard_map.

## [2026-06-03] loop-iteration | v064-shard-acts-s8k-bs3-tokamax-splash-qseq2-sharded-mask on 8B/v6e-8: invalid (54,795 tok/s, 39.3% nominal MFU; qseq2 still NaNs)

Workload `alekseyv-qwen3-v064` tested whether v050/v051's invalid but fast `TOKAMAX_Q_SEQ_SHARDS=2` path failed because the wrapper passed `q_seq_shards=2` to Tokamax but left `CausalMask(..., shard_count=1)`. The image changed only `/app/trainer/splash_attn.py` to construct the Tokamax causal mask with `shard_count=_q_seq_shards`; runtime kept the v056 stack plus `TOKAMAX_Q_SEQ_SHARDS=2 TOKAMAX_MAX_LOGIT_CONST=30.0`. It completed at worker0 54,795 tok/s / 39.3% nominal MFU and worker1 54,778 tok/s / 39.3%, but loss became NaN at step 1 on both workers. XProf run `2026-06-03-qwen3-jax-v064-shard-acts-s8k-bs3-tokamax-splash-qseq2-sharded-mask/2026_06_03_21_22_46` shows the invalid fast profile: MXU 54.1%, custom-call 17.2%, DKV 13.3%, forward residuals about 3.8%, HBM 28.33 GiB. The mask shard count is not the missing correctness piece; do not continue qseq2 runtime retries without a deeper correctness fix.

## [2026-06-03] loop-iteration | v063-shard-acts-s8k-bs3-tokamax-splash-maxlogit32p5 on 8B/v6e-8: refuted (50,383 tok/s, 36.1% MFU; tie, no frontier move)

Workload `alekseyv-qwen3-v063` kept the v056 frontier fixed and changed only `TOKAMAX_MAX_LOGIT_CONST` from 30.0 to 32.5, probing the narrow max-logit interval after 35.0 nearly tied and 40.0 regressed. It completed cleanly with normal loss and reached worker0 50,383 tok/s / 36.1% MFU and worker1 50,361 tok/s / 36.1% MFU. XProf run `2026-06-03-qwen3-jax-v063-shard-acts-s8k-bs3-tokamax-splash-maxlogit32p5/2026_06_03_21_12_06` shows the same profile as v056/v060: MXU 56.0%, custom-call 23.4%, DKV 12.3%, forward residuals 11.1%, and peak HBM 28.33 GiB. This ties the displayed worker0 frontier but does not beat it; pivot away from narrow max-logit sweeps.

## [2026-06-03] loop-iteration | v062-shard-acts-s8k-bs3-tokamax-splash-maxlogit30-dq2 on 8B/v6e-8: invalid (Tokamax only supports dq_reduction_steps=3 or None)

Workload `alekseyv-qwen3-v062` kept the v056 frontier fixed and set only `TOKAMAX_DQ_REDUCTION_STEPS=2`, testing whether a less aggressive DQ reduction than v054's `3` could reduce DKV without slowing wall-clock below the frontier. It failed before train-step compile on both workers with `ValueError: Invalid dq_reduction_steps: 2, only 3 or None are supported.` There is no DQ2 middle point to test.

## [2026-06-03] loop-iteration | v061-shard-acts-s8k-bs3-tokamax-splash-maxlogit30-bkvdkv4096 on 8B/v6e-8: refuted (41,587 tok/s, 29.8% MFU; DKV explodes)

Workload `alekseyv-qwen3-v061` kept the v056 frontier fixed and changed only DKV KV tiling (`SPLASH_BKV_DKV=4096 SPLASH_BKV_DKV_COMPUTE=4096`, leaving `SPLASH_BQ_DKV=2048`). It completed cleanly with normal loss but regressed hard: worker0 41,572 tok/s / 29.8% MFU and worker1 41,587 tok/s / 29.8% MFU. XProf run `2026-06-03-qwen3-jax-v061-shard-acts-s8k-bs3-tokamax-splash-maxlogit30-bkvdkv4096/2026_06_03_21_01_25` shows MXU 48.0%, custom-call 37.1%, and DKV alone 27.9%. Larger DKV KV blocks are the wrong direction; keep DKV block sizes at defaults.

## [2026-06-03] loop-iteration | v060-shard-acts-s8k-bs3-tokamax-splash-maxlogit35 on 8B/v6e-8: refuted (50,381 tok/s, 36.1% MFU; ties but does not beat v056)

Workload `alekseyv-qwen3-v060` kept the v056 frontier fixed and changed only `TOKAMAX_MAX_LOGIT_CONST` from 30.0 to 35.0, a midpoint after v059 showed 40.0 regresses and v053 showed 20.0 ties/slightly under 30.0. It completed cleanly with normal loss, but landed just below v056: worker0 50,381 tok/s / 36.1% MFU and worker1 50,358 tok/s / 36.1% MFU. XProf run `2026-06-03-qwen3-jax-v060-shard-acts-s8k-bs3-tokamax-splash-maxlogit35/2026_06_03_20_55_19` shows MXU 56.0%, custom-call 23.4%, DKV 12.3%, forward residuals 11.1%, and peak HBM 28.33 GiB. Treat 30.0 as the best max-logit bound.

## [2026-06-03] loop-iteration | v059-shard-acts-s8k-bs3-tokamax-splash-maxlogit40 on 8B/v6e-8: refuted (50,374 tok/s, 36.1% MFU; upper max-logit bound does not beat v056)

Workload `alekseyv-qwen3-v059` kept the v056 frontier fixed and changed only `TOKAMAX_MAX_LOGIT_CONST` from 30.0 to 40.0. It completed cleanly with normal loss, but landed below v056: worker0 50,350 tok/s / 36.1% MFU and worker1 50,374 tok/s / 36.1% MFU. XProf run `2026-06-03-qwen3-jax-v059-shard-acts-s8k-bs3-tokamax-splash-maxlogit40/2026_06_03_20_49_31` shows MXU 55.9%, custom-call 23.4%, DKV 12.3%, forward residuals 11.1%, and peak HBM 28.33 GiB. The upper-side max-logit probe does not beat v056.

## [2026-06-03] loop-iteration | v058-shard-acts-s8k-bs3-tokamax-splash-maxlogit30-cost-est on 8B/v6e-8: refuted (50,379 tok/s, 36.1% MFU; cost hints do not move device time)

Workload `alekseyv-qwen3-v058` kept the v056 frontier fixed and added only explicit Tokamax/Pallas cost hints (`TOKAMAX_COST_FWD=3.3e12 TOKAMAX_COST_BWD=5.2e12`) inferred from v057 XProf per-call custom-call FLOPs. It completed cleanly with normal loss, but landed below v056: worker0 50,357 tok/s / 36.1% MFU and worker1 50,379 tok/s / 36.1% MFU. XProf run `2026-06-03-qwen3-jax-v058-shard-acts-s8k-bs3-tokamax-splash-maxlogit30-cost-est/2026_06_03_20_39_11` shows MXU 56.0%, custom-call 23.4%, DKV 12.3%, forward residuals 11.1%, and peak HBM 28.33 GiB. The hints change reported custom-call FLOPs but not observed device time.

## [2026-06-03] loop-iteration | v057-shard-acts-s8k-bs3-tokamax-splash-maxlogit30-rerun4 on 8B/v6e-8: refuted (50,369 tok/s, 36.1% MFU; rerun axis saturated)

Workload `alekseyv-qwen3-v057` kept the v056 max-logit target-sequence frontier fixed and changed only the XLA latency-hiding scheduler rerun count from 3 to 4. It completed cleanly with normal loss, but worker0 landed at 50,369 tok/s / 36.1% MFU and worker1 at 50,354 tok/s / 36.1% MFU, below v056's 50,383 tok/s. XProf run `2026-06-03-qwen3-jax-v057-shard-acts-s8k-bs3-tokamax-splash-maxlogit30-rerun4/2026_06_03_20_29_04` shows the same profile as v056: MXU 55.9%, custom-call 23.4%, DKV 12.3%, forward residuals 11.1%, peak HBM 28.33 GiB. Treat scheduler reruns as saturated and pivot back to attention custom-call scheduling/shape work.

## [2026-06-03] loop-iteration | v056-shard-acts-s8k-bs3-tokamax-splash-maxlogit30-rerun3 on 8B/v6e-8: supported (50,383 tok/s, 36.1% MFU; marginal throughput frontier)

Workload `alekseyv-qwen3-v056` kept the v052 max-logit frontier fixed and changed only the XLA latency-hiding scheduler rerun count from 2 to 3. It completed cleanly with normal loss: worker0 50,383 tok/s / 36.1% MFU and worker1 50,356 tok/s / 36.1% MFU. XProf run `2026-06-03-qwen3-jax-v056-shard-acts-s8k-bs3-tokamax-splash-maxlogit30-rerun3/2026_06_03_04_48_42` shows MXU 56.0%, peak HBM 28.33 GiB, custom-call 23.4%, and essentially the same profile as v052. This is a marginal throughput high-water mark; follow with rerun4 before treating scheduler reruns as saturated.

## [2026-06-03] loop-iteration | v055-shard-acts-s8k-bs3-tokamax-splash-maxlogit30-bqdkv4096 on 8B/v6e-8: refuted (41,599 tok/s, 29.8% MFU; DKV explodes)

Workload `alekseyv-qwen3-v055` kept the v052 max-logit frontier fixed and set only `SPLASH_BQ_DKV=4096`, testing whether a larger DKV query block reduces the remaining backward Splash bucket. It completed cleanly but regressed hard: worker0 41,599 tok/s / 29.8% MFU and worker1 41,581 tok/s / 29.8% MFU. XProf run `2026-06-03-qwen3-jax-v055-shard-acts-s8k-bs3-tokamax-splash-maxlogit30-bqdkv4096/2026_06_03_04_39_08` shows custom-call rising to 36.8% and DKV alone to 27.6%, so larger DKV query blocks are the wrong direction.

## [2026-06-03] loop-iteration | v054-shard-acts-s8k-bs3-tokamax-splash-maxlogit30-dq3 on 8B/v6e-8: refuted (50,163 tok/s, 36.0% MFU; DQ3 lowers DKV but slows wall-clock)

Workload `alekseyv-qwen3-v054` kept the v052 `TOKAMAX_MAX_LOGIT_CONST=30.0` frontier fixed and added `TOKAMAX_DQ_REDUCTION_STEPS=3`. It completed cleanly but landed below v052: worker0 50,163 tok/s / 36.0% MFU and worker1 50,136 tok/s / 35.9% MFU. XProf run `2026-06-03-qwen3-jax-v054-shard-acts-s8k-bs3-tokamax-splash-maxlogit30-dq3/2026_06_03_04_33_12` shows custom-call falling to 22.9% and DKV to 11.9%, but the wall-clock loss means DQ3 remains refuted.

## [2026-06-03] loop-iteration | v053-shard-acts-s8k-bs3-tokamax-splash-maxlogit20 on 8B/v6e-8: refuted (50,375 tok/s, 36.1% MFU; tied/slightly below v052)

Workload `alekseyv-qwen3-v053` kept the v052 frontier fixed and tightened only `TOKAMAX_MAX_LOGIT_CONST` from `30.0` to `20.0`. It completed cleanly with normal loss, but landed at worker0 50,375 tok/s / 36.1% MFU and worker1 50,356 tok/s / 36.1% MFU, essentially tied with but slightly below v052. XProf run `2026-06-03-qwen3-jax-v053-shard-acts-s8k-bs3-tokamax-splash-maxlogit20/2026_06_03_04_27_42` shows the same 23.4% custom-call split as v052, so keep `TOKAMAX_MAX_LOGIT_CONST=30.0`.

## [2026-06-03] loop-iteration | v052-shard-acts-s8k-bs3-tokamax-splash-maxlogit30 on 8B/v6e-8: supported (50,376 tok/s, 36.1% MFU; new frontier)

Workload `alekseyv-qwen3-v052` kept the v045 target shape and valid `q_seq_shards=1` tokamax Splash path fixed, setting only `TOKAMAX_MAX_LOGIT_CONST=30.0`. It completed cleanly and becomes the new target-sequence frontier: worker0 50,376 tok/s / 36.1% MFU, worker1 50,345 tok/s / 36.1% MFU, with normal loss ending at 12.0508. XProf run `2026-06-03-qwen3-jax-v052-shard-acts-s8k-bs3-tokamax-splash-maxlogit30/2026_06_03_04_21_56` shows MXU 55.9%, peak HBM 28.33 GiB, and custom-call dropping to 23.4%, mainly forward residuals falling to 11.1%.

## [2026-06-03] loop-iteration | v051-shard-acts-s8k-bs3-tokamax-splash-qseq2-maxlogit30 on 8B/v6e-8: invalid (max-logit bound does not fix qseq2 NaNs)

Workload `alekseyv-qwen3-v051` followed up v050's fast-but-invalid q-sequence split by adding `TOKAMAX_MAX_LOGIT_CONST=30.0` to test whether the NaN loss was a softmax stability issue. It reached worker0 54,956 tok/s / 39.4% MFU and worker1 54,930 tok/s / 39.4% MFU, but loss still became NaN at step 1 on both workers. XProf run `2026-06-03-qwen3-jax-v051-shard-acts-s8k-bs3-tokamax-splash-qseq2-maxlogit30/2026_06_03_04_16_31` shows custom-call at 17.2%, but the result is invalid.

## [2026-06-03] loop-iteration | v050-shard-acts-s8k-bs3-tokamax-splash-qseq2 on 8B/v6e-8: invalid (NaN loss from step 1; apparent 54,537 tok/s unusable)

Workload `alekseyv-qwen3-v050` kept the v045 target-sequence frontier fixed and set only `TOKAMAX_Q_SEQ_SHARDS=2` to test whether splitting tokamax Splash query-sequence work improves attention custom-call scheduling. It ran much faster on paper, worker0 54,513 tok/s / 39.1% MFU and worker1 54,537 tok/s / 39.1% MFU, but loss became NaN at step 1 on both workers. XProf run `2026-06-03-qwen3-jax-v050-shard-acts-s8k-bs3-tokamax-splash-qseq2/2026_06_03_04_10_35` shows custom-call falling to 18.0%, mainly by cutting forward residuals, but the result is invalid.

## [2026-06-03] loop-iteration | v049-shard-acts-s8k-bs3-tokamax-splash-dq3 on 8B/v6e-8: refuted (48,855 tok/s, 35.0% MFU; lower custom-call share but slower)

Workload `alekseyv-qwen3-v049` kept the v045 target-sequence frontier fixed and set only `TOKAMAX_DQ_REDUCTION_STEPS=3` to test whether tokamax Splash can reduce backward attention custom-call time. It completed cleanly at worker0 48,855 tok/s / 35.0% MFU and worker1 48,834 tok/s / 35.0% MFU, below v045's 49,069 tok/s / 35.2% MFU. XProf run `2026-06-03-qwen3-jax-v049-shard-acts-s8k-bs3-tokamax-splash-dq3/2026_06_03_04_05_09` shows custom-call falling to 25.0% with DKV at 11.6%, but the wall-clock loss means the default `dq_reduction_steps=None` remains better.

## [2026-06-03] loop-iteration | v048-shard-acts-s8k-bs3-tokamax-splash-no-fuse-recip on 8B/v6e-8: refuted (48,626 tok/s, 34.9% MFU; fused reciprocal helps)

Workload `alekseyv-qwen3-v048` kept v045 fixed and set only `TOKAMAX_FUSE_RECIPROCAL=0` to test tokamax Splash's fused reciprocal path. It completed cleanly but landed below v045: worker0 48,626 tok/s / 34.9% MFU, worker1 48,606 tok/s / 34.9% MFU. Keep `TOKAMAX_FUSE_RECIPROCAL=1`.

## [2026-06-03] loop-iteration | v047-shard-acts-s8k-bs3-tokamax-splash-no-base2 on 8B/v6e-8: refuted (48,853 tok/s, 35.0% MFU; base2 exp helps)

Workload `alekseyv-qwen3-v047` kept v045 fixed and set only `TOKAMAX_USE_BASE2_EXP=0` to test tokamax Splash's base2-exp path. It completed cleanly but landed below v045: worker0 48,831 tok/s / 35.0% MFU, worker1 48,853 tok/s / 35.0% MFU. Keep `TOKAMAX_USE_BASE2_EXP=1`.

## [2026-06-03] loop-iteration | v046-shard-acts-s8k-bs3-tokamax-splash-exp-sched on 8B/v6e-8: refuted (48,202 tok/s, 34.6% MFU; exp scheduler regresses)

Workload `alekseyv-qwen3-v046` kept v045 fixed and set only `TOKAMAX_USE_EXP_SCHED=1` to test tokamax Splash's experimental scheduler. It completed cleanly but regressed to worker0 48,202 tok/s / 34.6% MFU and worker1 48,177 tok/s / 34.5% MFU, below v045's 49,069 tok/s / 35.2% MFU. Keep `TOKAMAX_USE_EXP_SCHED=0`.

## [2026-06-03] loop-iteration | v045-shard-acts-s8k-bs3-tokamax-splash on 8B/v6e-8: supported (49,069 tok/s, 35.2% MFU; new frontier)

Workload `alekseyv-qwen3-v045` kept v041's target shape and MaxText CE stack fixed, but set `USE_TOKAMAX_SPLASH=1` to swap only the Splash attention implementation. It completed cleanly and becomes the new target-sequence frontier: worker0 49,069 tok/s / 35.2% MFU, worker1 49,048 tok/s / 35.2% MFU, versus v041's 48,312 tok/s / 34.6% MFU. XProf run `2026-06-03-qwen3-jax-v045-shard-acts-s8k-bs3-tokamax-splash/2026_06_03_03_38_35` shows peak HBM unchanged at 28.33 GiB and custom-call dropping to 25.5% from v041's 26.7%, mainly by reducing forward Splash residuals to 13.5%; DKV remains about 12.0%.

## [2026-06-03] loop-iteration | v044-shard-acts-s8k-bs3-label-ce-internal-cast on 8B/v6e-8: invalid (compile-time HBM OOM; f32 logit dot persists)

Workload `alekseyv-qwen3-v044` retried label-CE after v043's invalid f32-logit-dot OOM, but kept the custom VJP primal logits in bf16 and cast inside the CE body. Compile failed with the same signature: used 34.88 GiB of 31.25 GiB HBM, exceeded by 3.64 GiB, with a 13.91 GiB `f32[3,8192,151936]` lm-head `dot_general` temp. The label-CE custom VJP branch is invalid in this form.

## [2026-06-03] loop-iteration | v043-shard-acts-s8k-bs3-label-ce on 8B/v6e-8: invalid (compile-time HBM OOM; f32 logit dot)

Workload `alekseyv-qwen3-v043` layered a single `train.py` change onto the v041 image and replaced MaxText one-hot CE with integer-label custom VJP CE at the same `fsdp=8,tp=1`, seq8192, per-chip batch 3/global batch 24 target shape. Compile failed with `RESOURCE_EXHAUSTED`: used 34.88 GiB of 31.25 GiB HBM, exceeded by 3.64 GiB. The largest allocation was a new 13.91 GiB `f32[3,8192,151936]` lm-head `dot_general` temp, caused by passing `logits.astype(jnp.float32)` into the custom VJP. This implementation is invalid; follow up by keeping the custom VJP primal logits in bf16 and casting inside the CE body.

## [2026-06-03] loop-iteration | v042-shard-acts-s8k-bs3-noscan on 8B/v6e-8: invalid (compile-time HBM OOM; 34.11G > 31.25G)

Workload `alekseyv-qwen3-v042` kept v041's supported target-sequence shape fixed (`fsdp=8,tp=1`, seq8192, per-chip batch 3/global batch 24), but disabled scan (`--use_scan=False`) to test whether unrolling can reduce the loop-fusion/scan overhead without crossing the HBM limit. Compile failed with `RESOURCE_EXHAUSTED`: used 34.11 GiB of 31.25 GiB HBM, exceeded by 2.86 GiB. The largest temps were a 6.96 GiB `bf16[3,8192,151936]` full-vocab logit tensor, a 768 MiB Splash DKV temp, and repeated 576 MiB MLP remat temps; scan is required at v041's target shape.

## [2026-06-03] loop-iteration | v034-shard-acts-s8k-gb24-tp2-noscan on 8B/v6e-8: invalid (compile-time HBM OOM; 34.66G > 31.25G)

Workload `alekseyv-qwen3-v034` retried v033's `fsdp=4,tp=2`, global batch 24 target shape but disabled scan (`--use_scan=False`) to test whether removing the stacked-layer scan layout avoids the 13.50 GiB `[36,6,8192,4096]` broadcast OOM. Compile still failed with `RESOURCE_EXHAUSTED`: used 34.66 GiB of 31.25 GiB HBM, exceeded by 3.42 GiB. The largest temps became two 6.96 GiB `bf16[6,8192,75968]` half-vocab logit tensors plus repeated 384 MiB unrolled-layer temps, so TP2 at v041's global-batch-24 target shape is invalid both with and without scan.

## [2026-06-03] loop-iteration | v033-shard-acts-s8k-gb24-tp2 on 8B/v6e-8: invalid (compile-time HBM OOM; 33.78G > 31.25G)

Workload `alekseyv-qwen3-v033` kept v041's global batch 24 target-sequence work fixed, but changed the mesh from `fsdp=8,tp=1` to `fsdp=4,tp=2` using `--batch_size=6 --tp_parallelism=2`. Compile failed with `RESOURCE_EXHAUSTED`: used 33.78 GiB of 31.25 GiB HBM, exceeded by 2.53 GiB. The largest temp is a 13.50 GiB `bf16[36,6,8192,4096]` stacked-layer broadcast, so TP2 plus scan is worse than v041's TP1 layout at this shape.

## [2026-06-03] loop-iteration | v032-shard-acts-s8k-bs4-maxtext-offload on 8B/v6e-8: refuted (45,953 tok/s, 32.9% MFU; offload fits but slows)

Workload `alekseyv-qwen3-v032` retried v026's MaxText-CE batch-4/global-32 target shape, adding only `--offload_remat=True`. It completed cleanly and peak HBM was 28.30 GiB with 2.95 GiB free, but throughput stayed below v041 at 45,953 tok/s / 32.9% MFU. XProf shows custom-call at 25.6% but data formatting rising to 8.2%; offload makes the shape fit but costs too much runtime.

## [2026-06-03] loop-iteration | v031-shard-acts-s8k-bs4-tokamax-ce on 8B/v6e-8: refuted (45,461 tok/s, 32.6% MFU; tokamax CE batch 4 fits but remains slow)

Workload `alekseyv-qwen3-v031` kept v030's streamed tokamax CE stack and increased only per-chip batch from 3 to 4 (global batch 32). It completed cleanly and peak HBM stayed at 24.75 GiB with 6.50 GiB free, but throughput remained below the frontier at 45,461 tok/s / 32.6% MFU vs v041's 48,312 tok/s / 34.6% MFU. XProf shows custom-call at 34.3%, including 7.3% tokamax CE backward; tokamax CE solves memory but not speed.

## [2026-06-03] loop-iteration | v030-shard-acts-s8k-bs3-tokamax-ce on 8B/v6e-8: refuted (45,103 tok/s, 32.3% MFU; HBM drops to 20.77 GiB)

Workload `alekseyv-qwen3-v030` kept the v041 activation-sharded target-sequence frontier shape (`fsdp=8,tp=1`, seq8192, per-chip batch 3/global batch 24) and replaced only MaxText CE with streamed tokamax CE (`--use_tokamax_ce=True --tokamax_ce_impl=mosaic_tpu`). It completed cleanly but regressed to 45,103 tok/s / 32.3% MFU vs v041's 48,312 tok/s / 34.6% MFU. XProf shows custom-call rising to 34.0%, including 7.2% tokamax CE backward, but peak HBM drops from 28.33 GiB to 20.77 GiB with 10.47 GiB free; follow up with batch 4/global 32.

## [2026-06-03] loop-iteration | v029-shard-acts-s8k-bs3-splash-unfused-bwd on 8B/v6e-8: refuted (46,103 tok/s, 33.1% MFU; custom-call worsened)

Workload `alekseyv-qwen3-v029` kept the v041 activation-sharded target-sequence frontier shape (`fsdp=8,tp=1`, seq8192, per-chip batch 3/global batch 24) and set only `SPLASH_FUSED_BWD=0` to test whether the fused Splash backward path is responsible for the DKV custom-call bucket. It completed cleanly but regressed to 46,103 tok/s / 33.1% MFU vs v041's 48,312 tok/s / 34.6% MFU. XProf shows custom-call rising to 30.9%, split across forward residuals (14.3%), DKV (8.8%), and DQ (7.8%); keep fused backward enabled.

## [2026-06-03] loop-iteration | v028-shard-acts-s8k-bs3-splash-bkv-dkv1024 on 8B/v6e-8: refuted (47,563 tok/s, 34.1% MFU; DKV custom-call worsened)

Workload `alekseyv-qwen3-v028` kept the v041 activation-sharded frontier shape and forward Splash defaults, but set `SPLASH_BKV_DKV=1024 SPLASH_BKV_DKV_COMPUTE=1024` to test the backward DKV block-size path. It completed cleanly but regressed to 47,563 tok/s / 34.1% MFU vs v041's 48,312 tok/s / 34.6% MFU. XProf shows custom-call at 27.0% vs v041's 26.7%, with DKV backward rising to 12.2%; keep the default DKV block size.

## [2026-06-03] loop-iteration | v027-shard-acts-s8k-bs3-splash-bkv512 on 8B/v6e-8: refuted (47,128 tok/s, 33.8% MFU; custom-call worsened)

Workload `alekseyv-qwen3-v027` kept the v041 activation-sharded frontier shape (`fsdp=8,tp=1`, seq8192, per-chip batch 3, global batch 24) and set only `SPLASH_BKV=512 SPLASH_BKV_COMPUTE=512`. It completed cleanly but regressed to 47,128 tok/s / 33.8% MFU. XProf shows custom-call increasing from v041's 26.7% to 28.5%, with forward residual custom calls rising to 17.1%; keep the forward KV block at the default.

## [2026-06-03] loop-iteration | v026-shard-acts-s8k-bs4 on 8B/v6e-8: invalid (compile-time HBM OOM; 33.53G > 31.25G)

Workload `alekseyv-qwen3-v026` reused the adopted v041 activation-sharded image and flags, kept seq8192/TP1/FSDP8, and increased per-chip batch from 3 to 4 (global batch 32). Compile failed with `RESOURCE_EXHAUSTED`: used 33.53G of 31.25G HBM, exceeded by 2.29G. The OOM is now dominated by two 9.27 GiB `bf16[4,8192,151936]` vocab/logit tensors, so v041 remains the activation-sharded batch frontier.

## [2026-06-03] loop-iteration | v041-shard-acts-s8k-bs3 on 8B/v6e-8: supported (48,312 tok/s, 34.6% MFU; peak HBM 28.33 GiB)

Adopted completed workload `alekseyv-qwen3-cc-jax-v041-shardact`. The `v041-shard-acts` image uses scan + remat + Splash + MaxText CE + activation sharding at `fsdp=8,tp=1`, seq8192, per-chip batch 3, global batch 24. It completed cleanly and becomes the new overall and target-sequence frontier: worker0 reported 48,312 tok/s / 34.6% MFU, XProf shows 53.7% MXU, peak HBM 28.33 GiB, and 2.916 GiB free. Top costs are convolution fusion 46.1%, Splash custom-call 26.7%, and loop fusion 17.9%.

## [2026-06-03] loop-iteration | v025-splash-tp4-tokamax-remat-fused-s8k-bs4 on 8B/v6e-8: invalid (compile-time HBM OOM; 35.46G > 31.25G)

Workload `alekseyv-qwen3-v025` used the v015 image and target seq8192, but switched from `fsdp=4,tp=2` to `fsdp=2,tp=4` and tried per-chip batch 4 (global batch 8). Compile failed with `RESOURCE_EXHAUSTED`: used 35.46G of 31.25G HBM, exceeded by 4.21G. TP4 made the tokamax/fused-RoPE target-sequence batch test worse than v024, so the next path is activation sharding / CE memory work, not more TP.

## [2026-06-03] loop-iteration | v024-splash-tp2-tokamax-remat-fused-s8k-bs2 on 8B/v6e-8: invalid (compile-time HBM OOM; 33.99G > 31.25G)

Workload `alekseyv-qwen3-v024` kept target seq8192 and increased per-chip batch from v023's 1 to 2 (global batch 8) using the same v015 image and TP2 flag stack. Compile failed with `RESOURCE_EXHAUSTED`: used 33.99G of 31.25G HBM, exceeded by 2.75G. The largest allocation was a 2.32 GiB tokamax backward dot with labels shape `i32[65536]`; v023 remains the simple TP2 batch-scaling frontier at target sequence length.

## [2026-06-03] loop-iteration | v023-splash-tp2-tokamax-remat-fused-s8k-bs1 on 8B/v6e-8: supported (37,148 tok/s, 26.6% MFU; peak HBM 26.57 GiB)

Workload `alekseyv-qwen3-v023` completed cleanly at seq8192, per-chip batch 1, global batch 4, using the v015 image and TP2 flag stack. This establishes the first supported target-sequence point in this lane: worker1 reported 37,148 tok/s / 26.6% MFU, XProf shows 34.0% MXU, peak HBM 26.57 GiB, and 4.67 GiB free. Custom-call rises to 20.4% self-time, with all-reduce-scatter 14.2% and all-gather 10.4%.

## [2026-06-03] loop-iteration | v022-splash-tp2-tokamax-remat-fused-s4k-bs2 on 8B/v6e-8: supported (41,592 tok/s, 27.8% MFU; peak HBM 25.90 GiB)

Workload `alekseyv-qwen3-v022` completed cleanly at seq4096, per-chip batch 2, global batch 8, using the v015 image and TP2 flag stack. This is the first supported seq4096 point in the current stack: worker1 reported 41,592 tok/s / 27.8% MFU, XProf shows 32.3% MXU, peak HBM 25.90 GiB, and 5.35 GiB free. Top costs shift toward collectives/custom-call: all-reduce-scatter 15.4%, custom-call 13.9%, all-gather 11.8%.

## [2026-06-03] loop-iteration | v021-splash-tp2-tokamax-remat-fused-s4k-bs4 on 8B/v6e-8: invalid (compile-time HBM OOM; 33.99G > 31.25G)

Planned workload `alekseyv-qwen3-v021`: reused the v015 image and successful TP2 flag stack, moved from seq2048 to seq4096, and used even per-chip batch 4 (global batch 16) to avoid the odd-batch layout OOM seen in v020. Compile failed with `RESOURCE_EXHAUSTED`: used 33.99G of 31.25G HBM, exceeded by 2.74G. The tokamax CE shape is `labels=i32[65536]`, but this seq4096 graph does not fit like v015's seq2048/global32 graph.

## [2026-06-03] loop-iteration | v020-splash-tp2-tokamax-remat-fused-bs7 on 8B/v6e-8: invalid (compile-time HBM OOM; odd batch layout)

Planned workload `alekseyv-qwen3-v020` to map the memory/perf curve between v018 batch 6 and v015 batch 8. It reused the v015 image and successful flag stack with `fsdp=4,tp=2`, per-chip batch 7, global batch 28, seq_len 2048. First launch exited 143 before train-step compile and left only tiny RNG HLO dumps; retry `alekseyv-qwen3-v020b` reached train-step compile and failed with `RESOURCE_EXHAUSTED`: used 35.61G of 31.25G HBM, exceeded by 4.37G. Odd per-chip batch 7 is worse than both batch 6 and batch 8; continue with even batch shapes.

## [2026-06-03] loop-iteration | v019-phase2-bundle-cost-model-off on 8B/v6e-8: invalid (compile-time HBM OOM; 34.00G > 31.25G)

Adopted live workload `alekseyv-qwen3-v019`. It reused the v015 image and frontier shape (`fsdp=4,tp=2`, per-chip batch 8, global batch 32, seq_len 2048) with `--xla_tpu_use_bundle_aware_cost_model_for_fusions=false` in `LIBTPU_INIT_ARGS`. Compile failed on both workers with `RESOURCE_EXHAUSTED`: used 34.00G of 31.25G HBM, exceeded by 2.75G. OOM artifacts were dumped under the run's `hlo/` prefix; keep the bundle-aware fusion cost model enabled.

## [2026-06-03] loop-iteration | v018-splash-tp2-tokamax-remat-fused-bs6 on 8B/v6e-8: supported (46,526 tok/s, 29.9% MFU; peak HBM 28.32 GiB)

Cluster idle after v017. Reused the v015 image with `fsdp=4,tp=2`, per-chip batch 6, global batch 24, seq_len 2048. Clean run; not a new MFU frontier, but it recovered substantial HBM margin: peak HBM 28.32 GiB with 2.92 GiB free versus v015/v017 at 31.23 GiB and effectively no margin.

## [2026-06-03] loop-iteration | v017-splash-tp2-tokamax-remat-scan-bs8 on 8B/v6e-8: refuted (47,864 tok/s, 30.8% MFU; no memory/profile win)

Active `alekseyv-qwen3-v017` workload used the v015 image with `fsdp=4,tp=2`, per-chip batch 8, global batch 32, seq_len 2048, and run path `2026-06-03-qwen3-jax-v017-splash-tp2-tokamax-remat-scan-bs8`. Earlier TP4 attempt under the same name saw SIGTERM/143 before first train step; the live workload was recreated with this TP2 scan command, and the durable stub was updated to match the actual pod spec. Clean run, but XProf shows no meaningful improvement over v015: MXU 35.4% vs 35.5%, peak HBM unchanged at 31.23 GiB, loop fusion slightly worse at 12.9%.

## [2026-06-03] loop-iteration | v016-splash-tp1-tokamax-remat-fused-bs4 on 8B/v6e-8: invalid (compile-time HBM OOM; 31.96G > 31.25G)

Adopted workload `alekseyv-qwen3-v016` with a durable stub before completion. It tested `fsdp=8,tp=1`, global batch 32, and the v015 Splash/tokamax/remat/fused image to remove TP communication. Compile failed before any step completed: used 31.96G of 31.25G HBM. OOM dump points to HLO temp fragmentation and rematted hidden-state temps rather than CE logits.

## [2026-06-03] loop-iteration | v015-splash-tp2-tokamax-remat-fused-rope-bs8 on 8B/v6e-8: supported (47,846 tok/s, 30.8% MFU)

Loop protocol resumed with the never-stop hook disabled. Backfilled completed unstubbed workloads v013 and v014 as `inconclusive` observations, then adopted workload `alekseyv-qwen3-v015` with a durable stub before completion. v015 completed cleanly and moved the formal frontier from v012's 39,438 tok/s / 25.4% MFU to 47,846 tok/s / 30.8% MFU. XProf run `2026-06-03-qwen3-jax-v015-splash-tp2-tokamax-remat-fused-rope-bs8/2026_06_03_00_51_58` shows 35.5% MXU and peak HBM 31.23 GiB.

## [2026-06-03] loop-iteration | v012-splash-tp4-bs8 on 8B/v6e-8: supported (39,438 tok/s, 25.4% MFU)

Adopted live workload `alekseyv-qwen3-v012` as the active `/start-experiment` track. It completed cleanly on `<your-cluster>` with `fsdp=2,tp=4`, Splash enabled, global batch 16, and seq_len 2048. XProf run `2026-06-02-qwen3-jax-v012-splash-tp4-bs8/2026_06_03_00_24_23` shows MXU 26.9%, peak HBM 27.75 GiB, and `jit_train_step` 97.8% of device time. This moves the native-JAX frontier from the baseline 31,955 tok/s / 20.5% MFU to 39,438 tok/s / 25.4% MFU.

## [2026-06-03] loop-iteration | v011-splash-tp2-remat-dots-bs8 on 8B/v6e-8: invalid (compile-time HBM OOM; 35.21G > 31.25G)

## [2026-06-03] start | /start-experiment session begin

**Op**: start
**Cluster pool**: <your-cluster>
**Parallelism**: 1
**First-pick hypothesis**: adopt live `alekseyv-qwen3-v011` track (Qwen3 JAX splash + TP=2 + per-chip batch 8) and backfill/analyze before dispatching another duplicate.
**Notes**: session opened via Codex start-experiment; Codex never-stop marker active at `.codex/.loop_active.json`.
## [2026-06-04] loop-iteration | v167-replicate-rmsnorm-vmem100864 on 8B/v6e-8: running

Workload `alekseyv-qwen3-v167-repnorm-vmem100864` tests whether the
replicated-RMSNorm HLO family from v165/v166 has a shifted scoped-VMEM optimum.
It keeps the v165 image and confirmed v154 runtime stack fixed except for
raising `--xla_tpu_scoped_vmem_limit_kib` from 100352 to 100864.

## [2026-06-04] loop-iteration | v166-replicate-rmsnorm-params-rerun on 8B/v6e-8: refuted/tie (57,735 tok/s best-worker avg, 41.4% MFU; below v153/v154)

Workload `alekseyv-qwen3-v166-repnormr` exactly reruns v165's replicated
RMSNorm sharding image and confirmed v154 runtime stack. This is required
because v165 produced a real smaller HLO and all-gather-count reduction but
landed just below the confirmed throughput average. It completed cleanly with
normal loss ending at 12.0461 and `EXIT_CODE=0`, reaching worker0 57,735 tok/s /
7,217 tok/s/chip / 41.4% MFU and worker1 57,709 tok/s / 7,214 tok/s/chip /
41.4% MFU. Best observed post-warmup step was 57,846 tok/s. No fresh HLO dump
appeared under v166, consistent with a persistent-cache replay of v165's
778,215-byte HLO
(`f928230a6d0bf402ad387404cace7c42dd989d5cc630b5f2d190f9cdc1eeff56`). The
rerun confirms replicated RMSNorm params as a near-frontier tie/refutation: do
not carry.

## [2026-06-04] loop-iteration | v165-replicate-rmsnorm-params on 8B/v6e-8: refuted/tie (57,730 tok/s best-worker avg, 41.4% MFU; HLO smaller and all-gather count lower)

Workload `alekseyv-qwen3-v165-repnorm` tests a small-collective sharding
hypothesis after the CE/lm-head boundary constraints refuted. The image is
layered from the confirmed `v041-shard-acts` base and changes only the
sharding specs for decoder input RMSNorm, post-attention RMSNorm, and final
RMSNorm weights from `("fsdp",)` to replicated `()`. Large matmul, embedding,
lm-head, CE, attention, optimizer, and runtime flags are fixed to the confirmed
v154 stack. It completed cleanly with normal loss ending at 12.0461 and
`EXIT_CODE=0`, reaching worker0 57,730 tok/s / 7,216 tok/s/chip / 41.4% MFU
and worker1 57,707 tok/s / 7,213 tok/s/chip / 41.4% MFU. Best observed
post-warmup step was 57,889 tok/s. Fresh train-step HLO shrank to 778,215
bytes with SHA256
`f928230a6d0bf402ad387404cace7c42dd989d5cc630b5f2d190f9cdc1eeff56`; quick
opcode counts show all-gather occurrences dropped from v153's 211 to 177 while
all-reduce, convolution, custom-call, and copy counts stayed unchanged. Treat
as a useful HLO observation but do not carry after v166 exact rerun stayed below
the confirmed v153/v154 average.

## [2026-06-04] loop-iteration | v164-final-hidden-fsdp-constraint on 8B/v6e-8: refuted (56,380 tok/s best-worker avg, 40.4% MFU; distinct HLO but slower)

Workload `alekseyv-qwen3-v164-finalhidden` tests the next CE/lm-head layout
boundary after v163's full-logits constraint refutation. The image is layered
from the confirmed `v041-shard-acts` base and changes only
`Qwen3ForCausalLM.__call__`, applying the existing `_sac(hidden)` activation
constraint immediately after the final model output and before `lm_head`.
Runtime flags and shape match the confirmed v154 stack, including batch
4/global 32, tokamax Splash, MaxText CE, collective-matmul modes disabled, and
scoped VMEM 100352 KiB. It completed cleanly with normal loss ending at 12.0461
and `EXIT_CODE=0`, but reached only worker0 56,380 tok/s / 7,047 tok/s/chip /
40.4% MFU and worker1 56,365 tok/s / 7,046 tok/s/chip / 40.4% MFU. The first
step took 37.8 s and a fresh optimized train-step HLO was emitted: 784,126
bytes, SHA256
`59fb6287422baec57b38443c7e58063e531ba1d98205de3aca1017ce5c37aa01`.
Profile was written under
`2026-06-04-qwen3-jax-v164-final-hidden-fsdp-constraint/plugins/profile/2026_06_04_18_57_15/`.
Do not carry the final-hidden constraint.

## [2026-06-04] loop-iteration | v163-maxtext-ce-logits-fsdp-constraint on 8B/v6e-8: refuted (56,408 tok/s best-worker avg, 40.4% MFU; distinct HLO but slower)

Workload `alekseyv-qwen3-v163-ce-logits-fsdp` pivots after the v162 refutation
to a narrower CE/logits layout hypothesis. The image is layered from the
confirmed `v041-shard-acts` base and changes only `_ce_maxtext` by constraining
the full logits tensor to `NamedSharding(mesh, P("fsdp", None, None))` before
the MaxText/T5X custom VJP CE. Runtime flags and shape match the confirmed
v154 stack, including batch 4/global 32, tokamax Splash, MaxText CE,
collective-matmul modes disabled, and scoped VMEM 100352 KiB. It completed
cleanly with normal loss ending at 12.0461 and `EXIT_CODE=0`, but reached only
worker0 56,393 tok/s / 7,049 tok/s/chip / 40.4% MFU and worker1 56,408 tok/s /
7,051 tok/s/chip / 40.4% MFU, below the confirmed v153/v154 frontier. The first
step took 38.5 s, and a fresh optimized train-step HLO was emitted:
784,080 bytes, SHA256
`9f7762782921bf93af7dddc93c2c93d04207b6f73ab0ec2ceb205709d93c903f`.
Profile was written under
`2026-06-04-qwen3-jax-v163-maxtext-ce-logits-fsdp-constraint/plugins/profile/2026_06_04_18_50_07/`.
Do not carry the logits constraint.

## [2026-06-04] loop-iteration | v170-no-mlp-intermediate-sac-rerun on 8B/v6e-8: supported (57,935 tok/s best-worker avg, 41.5% MFU; confirms v169 frontier)

Workload `alekseyv-qwen3-v170-mlpsacr` exactly reran v169's image and runtime
stack. It completed cleanly with normal loss ending at 12.0461 and `EXIT_CODE=0`,
reaching worker0 57,913 tok/s / 7,239 tok/s/chip / 41.5% MFU and worker1
57,935 tok/s / 7,242 tok/s/chip / 41.5% MFU. Best observed post-warmup step was
58,050 tok/s. Profile was written under
`2026-06-04-qwen3-jax-v170-no-mlp-intermediate-sac-rerun/plugins/profile/2026_06_04_19_39_56/`.
No fresh HLO dump appeared under the v170 prefix, consistent with persistent
cache replay of v169's 783,769-byte train-step HLO
(`443b5f34d11a802519fca1f96187c541566bdb9d532c86177249d46b11e3d644`). Carry
the no-MLP-intermediate-`_sac` change as the current frontier.

## [2026-06-04] loop-iteration | v171-no-mlp-intermediate-sac-vmem100864 on 8B/v6e-8: refuted/tie (57,941 tok/s best-worker avg, 41.5% MFU; same HLO as v169)

Workload `alekseyv-qwen3-v171-vmemhi` tested whether the v169/v170
no-MLP-intermediate-`_sac` HLO family had a shifted scoped-VMEM optimum by
changing only `--xla_tpu_scoped_vmem_limit_kib=100352` to `100864`. It
completed cleanly with normal loss ending at 12.0461 and `EXIT_CODE=0`,
reaching worker0 57,925 tok/s / 7,241 tok/s/chip / 41.5% MFU and worker1
57,941 tok/s / 7,243 tok/s/chip / 41.5% MFU. Best observed post-warmup step was
58,050 tok/s. Profile was written under
`2026-06-04-qwen3-jax-v171-no-mlp-sac-vmem100864/plugins/profile/2026_06_04_20_00_11/`.
Fresh train-step HLO is 783,769 bytes with SHA256
`443b5f34d11a802519fca1f96187c541566bdb9d532c86177249d46b11e3d644`, byte-identical
to v169. Do not carry 100864 KiB; keep v169/v170 as the confirmed frontier and
pivot away from nearby scoped-VMEM probes in this HLO family.

## [2026-06-04] loop-iteration | v172-no-attn-output-sac on 8B/v6e-8: refuted (57,770 tok/s best-worker avg, 41.4% MFU; attention boundary constraint is useful)

Workload `alekseyv-qwen3-v172-noattnsac` kept the confirmed v169
no-MLP-intermediate-`_sac` image fixed and removed only the `_sac` around the
attention output before the residual add. CPU equivalence passed before TPU
launch (`==== EQUIVALENCE PASS ====`, logits max delta `2.682e-07`, all 25
gradients OK). The TPU run completed cleanly with normal loss ending at 12.0461
and `EXIT_CODE=0`, but reached only worker0 57,770 tok/s / 7,221 tok/s/chip /
41.4% MFU and worker1 57,749 tok/s / 7,219 tok/s/chip / 41.4% MFU. Best
observed post-warmup steps were 57,915 and 57,922 tok/s, below the confirmed
v169/v170 band. Profile was written under
`2026-06-04-qwen3-jax-v172-no-attn-output-sac/plugins/profile/2026_06_04_20_08_58/`.
Fresh train-step HLO is 783,590 bytes with SHA256
`fdcf65285f8c251968d5fe4279d8914e6b8e50978e3f43f236aa5d52c54afcb5`; major op
counts match v169/v170 but schedule/layout details changed. Do not carry this:
restore the attention-output `_sac` and keep v169/v170 as the confirmed
frontier.

## [2026-06-04] loop-iteration | v173-no-pre-attn-norm-sac on 8B/v6e-8: refuted (57,766 tok/s best-worker avg, 41.4% MFU; pre-attention norm constraint is useful)

Workload `alekseyv-qwen3-v173-noinnormsac` kept the confirmed v169
no-MLP-intermediate-`_sac` image fixed and removed only the `_sac` around
`input_layernorm(hidden_states)` before attention. CPU equivalence passed before
TPU launch (`==== EQUIVALENCE PASS ====`, logits max delta `2.682e-07`, all 25
gradients OK). The TPU run completed cleanly with normal loss ending at 12.0461
and `EXIT_CODE=0`, but reached only worker0 57,766 tok/s / 7,221 tok/s/chip /
41.4% MFU and worker1 57,740 tok/s / 7,218 tok/s/chip / 41.4% MFU. Best
observed post-warmup step was 57,884 tok/s on both workers. Profile was written
under
`2026-06-04-qwen3-jax-v173-no-pre-attn-norm-sac/plugins/profile/2026_06_04_20_17_57/`.
Fresh train-step HLO is 783,587 bytes with SHA256
`06b6bbab8ca9dc93be1d6984eed9673ab7501b082b3137f439f929f0819ee49f`; major op
counts match v169/v172 but schedule/layout details changed. Do not carry this:
restore the pre-attention norm-output `_sac` and keep v169/v170 as the
confirmed frontier.

## [2026-06-04] loop-iteration | v174-no-pre-mlp-norm-sac on 8B/v6e-8: refuted (57,759 tok/s best-worker avg, 41.4% MFU; pre-MLP norm constraint is useful)

Workload `alekseyv-qwen3-v174-nomlpnormsac` kept the confirmed v169
no-MLP-intermediate-`_sac` image fixed and removed only the `_sac` around
`post_attention_layernorm(hidden_states)` before the MLP. CPU equivalence passed
before TPU launch (`==== EQUIVALENCE PASS ====`, logits max delta `2.682e-07`,
all 25 gradients OK). The TPU run completed cleanly with normal loss ending at
12.0461 and `EXIT_CODE=0`, but reached only worker0 57,759 tok/s / 7,220
tok/s/chip / 41.4% MFU and worker1 57,737 tok/s / 7,217 tok/s/chip / 41.4%
MFU. Best observed post-warmup steps were 57,879 and 57,883 tok/s. Profile was
written under
`2026-06-04-qwen3-jax-v174-no-pre-mlp-norm-sac/plugins/profile/2026_06_04_20_24_26/`.
Fresh train-step HLO is 783,590 bytes with SHA256
`14ca3242bea4cc41c2897f6c5f5f3d7da24597980944a1863c85b98c495f905e`; major op
counts match v169/v172/v173 but schedule/layout details changed. Do not carry
this: restore the pre-MLP norm-output `_sac` and keep v169/v170 as the
confirmed frontier.

## [2026-06-04] loop-iteration | v175-no-layer-input-sac on 8B/v6e-8: refuted (57,751 tok/s best-worker avg, 41.4% MFU; layer input constraint is useful)

Workload `alekseyv-qwen3-v175-nolayersac` kept the confirmed v169
no-MLP-intermediate-`_sac` image fixed and removed only the decoder layer input
`_sac(hidden_states)`. CPU equivalence passed before TPU launch
(`==== EQUIVALENCE PASS ====`, logits max delta `2.682e-07`, all 25 gradients
OK). The TPU run completed cleanly with normal loss ending at 12.0461 and
`EXIT_CODE=0`, but reached only worker0 57,751 tok/s / 7,219 tok/s/chip /
41.4% MFU and worker1 57,738 tok/s / 7,217 tok/s/chip / 41.4% MFU. Best
observed post-warmup steps were 57,858 and 57,866 tok/s. Profiles were written
under
`2026-06-04-qwen3-jax-v175-no-layer-input-sac/plugins/profile/2026_06_04_20_31_17/`
and `2026_06_04_20_31_18/`. Fresh train-step HLO is 783,587 bytes with SHA256
`76fc341db3f31dbef527b130e601dce97d3100c2dbd259ebca185f5e2b1d27f1`; major op
counts match v169/v172-v174 but schedule/layout details changed. Do not carry
this: restore the layer-input `_sac`. The supported v169 gain is isolated to
removing the large MLP intermediate `_sac`.

## [2026-06-04] loop-iteration | v169-no-mlp-intermediate-sac on 8B/v6e-8: provisional supported (57,947 tok/s best-worker avg, 41.5% MFU; rerun before carry)

Workload `alekseyv-qwen3-v169-mlpsac` removed only the `_sac` activation
sharding constraint around the large MLP gate/up product before `down_proj`,
leaving all layer-boundary residual-stream constraints enabled. It completed
cleanly with normal loss ending at 12.0461 and `EXIT_CODE=0`, reaching worker0
57,947 tok/s / 7,243 tok/s/chip / 41.5% MFU and worker1 57,919 tok/s / 7,240
tok/s/chip / 41.5% MFU. Best observed post-warmup step was 58,051 tok/s. Profile
was written under
`2026-06-04-qwen3-jax-v169-no-mlp-intermediate-sac/plugins/profile/2026_06_04_19_35_11/`.
Fresh train-step HLO is 783,769 bytes with SHA256
`443b5f34d11a802519fca1f96187c541566bdb9d532c86177249d46b11e3d644`; compiled
memory remains 26.86 GiB. This is above the v153/v154 confirmed average band, so
run an exact rerun before carrying.

## [2026-06-04] loop-iteration | v168-lmhead-replicated on 8B/v6e-8: refuted (56,464 tok/s best-worker avg, 40.5% MFU; distinct HLO but much slower)

Workload `alekseyv-qwen3-v168-lmrepb` tested a structural CE/lm-head sharding
rewrite by fully replicating only `lm_head.weight` (`"lm_head.weight": ()`) on
top of the confirmed v154 runtime stack. A first inert JobSet
`alekseyv-qwen3-v168-lmrepl` was deleted before it ran because it retained
queue-owned metadata from a completed template; the clean retry ran normally.
The experiment completed with normal loss ending at 12.0461 and `EXIT_CODE=0`,
but throughput regressed to worker0 56,464 tok/s / 7,058 tok/s/chip / 40.5% MFU
and worker1 56,445 tok/s / 7,056 tok/s/chip / 40.5% MFU. Profile was written
under
`2026-06-04-qwen3-jax-v168-lmhead-replicated/plugins/profile/2026_06_04_19_28_50/`.
Fresh train-step HLO is 780,820 bytes with SHA256
`49953ae38096108b72b65ed59d492d32348ebada51efa9070bd396e100b67c68`; the change
reduces textual all-gather count but explodes copy/fusion count and loses wall
time. Do not carry full lm-head replication.

## [2026-06-04] loop-iteration | v167-replicate-rmsnorm-vmem100864 on 8B/v6e-8: refuted (57,724 tok/s best-worker avg, 41.4% MFU; same HLO as v165 and slower)

Workload `alekseyv-qwen3-v167-repnorm-vmem100864` tested whether the v165
replicated-RMSNorm HLO family had a shifted scoped-VMEM optimum by changing only
`--xla_tpu_scoped_vmem_limit_kib=100352` to `100864`. It completed cleanly with
normal loss ending at 12.0461 and `EXIT_CODE=0`, but reached only worker0
57,724 tok/s / 7,216 tok/s/chip / 41.4% MFU and worker1 57,697 tok/s / 7,212
tok/s/chip / 41.4% MFU. Best observed post-warmup step was 57,854 tok/s.
Profile was written under
`2026-06-04-qwen3-jax-v167-replicate-rmsnorm-vmem100864/plugins/profile/2026_06_04_19_16_42/`.
Fresh train-step HLO is 778,215 bytes with SHA256
`f928230a6d0bf402ad387404cace7c42dd989d5cc630b5f2d190f9cdc1eeff56`, byte-identical
to v165. Do not carry replicated RMSNorm params or 100864 KiB; v153/v154 remain
the confirmed frontier.

## [2026-06-05] loop-iteration | v254-no-scan-sparsecore-rs-ar-rs2-ar2-qknorm-after-transpose on 8B/v6e-8: invalid (missing Tokamax env)

Workload `alekseyv-q3-v254-qknorm-xpose` moved Q/K RMSNorm after the q/k
transpose so QK-norm and RoPE operate in the same `(B,H,T,D)` layout. This is
mathematically equivalent because RMSNorm reduces over the last `D` dimension,
and the loss curve remained normal. The run completed cleanly with worker0
**56,438 tok/s / 40.5% MFU** and worker1 **56,417 tok/s / 40.5% MFU**, but the
launch omitted `USE_TOKAMAX_SPLASH=1` and `TOKAMAX_MAX_LOGIT_CONST=30.0`; the
optimized HLO has `tokamax` count 0. This is not comparable to the v230/v236
**59,750 tok/s / 42.9% MFU** frontier. XProf
`2026-06-05-qwen3-jax-v254-no-scan-sparsecore-rs-ar-rs2-ar2-qknorm-after-transpose/2026_06_05_12_20_35`
shows **4656.9 ms**, **64.3% MXU**, and **30.87 GiB** peak HBM; top buckets are
convolution fusion **51.6%**, custom-call **31.4%**, and loop fusion **9.9%**.
HLO SHA256 `a5ce1755308298feb3ef35f7ae7d74b34ab2efe6d307e7101e68ed48f0946d84`,
size 19,013,836 bytes, 113,720 lines; compiled memory **27.67 GiB**. Rerun as
v257 before judging the QK-norm-after-transpose layout change.

## [2026-06-05] loop-iteration | v253-no-scan-sparsecore-rs-ar-rs2-ar2-fused-mlp-gate-up-vmem98304 on 8B/v6e-8: invalid/refuted (load-time HBM OOM repeats)

Workload `alekseyv-q3-v253-fusedmlp-vmem98` reused the v252 fused MLP gate/up
image and changed only `--xla_tpu_scoped_vmem_limit_kib` from 100352 to 98304.
It failed before step 0 with the same load-time resource exhaustion:
`Attempting to allocate 1.81G. That was not possible. There are 1.54G free.`
Fresh HLO was emitted with SHA256
`61dd6f8b385914d01a62d398a8d1e9378091ecf6a856b5c3d7de490422a9eb5d`, size
17,406,954 bytes, 103,597 lines, and compiled memory **28.56 GiB**. Lower VMEM
does not recover the fused gate/up projection. Close this direction.

## [2026-06-05] loop-iteration | v252-no-scan-sparsecore-rs-ar-rs2-ar2-fused-mlp-gate-up on 8B/v6e-8: invalid/refuted (combined MLP projection load-time HBM OOM)

Workload `alekseyv-q3-v252-fusedmlp` patched only `Qwen3MLP.__call__` in a thin
image derived from `qwen3-8b-jax:v169-no-mlp-sac`: it concatenated the existing
`gate_proj.weight` and `up_proj.weight`, ran one larger `dot_general`, split the
result, and kept the same `silu(gate) * up` math. The program compiled but failed
before step 0 with `Attempting to allocate 1.82G. That was not possible. There
are 1.54G free.` HLO SHA256
`efe69c0f710972418836d958ed55c446c61a43691461c9fa8e85bf5a532e299b`, size
17,454,915 bytes, 103,987 lines; compiled memory **28.57 GiB**. Run v253 as a
single bounded VMEM-relief follow-up before closing the direction.

## [2026-06-05] loop-iteration | v251-no-scan-sparsecore-rs-ar-rs2-ar2-levanter-xla-ce on 8B/v6e-8: refuted (54,073 tok/s, 38.8% MFU; Levanter XLA CE slower)

Workload `alekseyv-q3-v251-levxla-ce` added a minimal `--levanter_ce_impl`
switch to the v248 Levanter CE image and ran the fused CE call with
`implementation="xla"` instead of `pallas_tpu`. It preserved the v230/v236
runtime stack: no-scan, remat, Tokamax Splash, activation sharding, SparseCore
RS/AR offload plus aggregator, RS latency multiplier 2, AR latency multiplier
2, rerun3, and scoped VMEM 100352.

The run completed cleanly on both workers with finite loss ending at 12.0487,
but regressed to worker0 **54,073 tok/s / 38.8% MFU** and worker1 **54,063
tok/s / 38.8% MFU**. XProf
`2026-06-05-qwen3-jax-v251-no-scan-sparsecore-rs-ar-rs2-ar2-levanter-xla-ce/2026_06_05_11_44_54`
shows **4861.4 ms** average step, **61.8% MXU**, and **29.23 GiB** peak HBM.
The optimized train-step HLO was dumped successfully with SHA256
`b2fe8cf7500da79d467b0df5360d76f9788367a6afec3d2e84fa22f825ba75da`, size
19,041,778 bytes, 114,165 lines. Compiled memory is only **26.00 GiB**, so the
failure is device work/scheduling rather than capacity.

Close Levanter CE as the next MaxText-CE replacement path. v249 remains the
best Levanter point, and the native-JAX frontier remains v230/v236 at about
**59,750 tok/s / 42.9% MFU**.

## [2026-06-04] loop-iteration | v162-shard-acts-s8k-bs4-lmhead-vocab-fsdp-rerun on 8B/v6e-8: refuted (56,494 tok/s best-worker avg, 40.5% MFU; v161 gain did not rerun)

Workload `alekseyv-qwen3-v162-bs4-lmheadvocabr` exactly reruns v161's
lm-head vocab-axis FSDP sharding image and confirmed v154 runtime stack. This
was the required confirmation run before carrying the v161 structural sharding
change. It completed cleanly with normal loss ending at 12.0458 and `EXIT_CODE=0`,
but reached only worker0 56,494 tok/s / 7,062 tok/s/chip / 40.5% MFU and
worker1 56,482 tok/s / 7,060 tok/s/chip / 40.5% MFU. Best observed post-warmup
step was 56,914 tok/s, below the confirmed v153/v154 frontier's 57,888 best
observed step and below v161's 57,908 provisional best. The profile was written
under
`2026-06-04-qwen3-jax-v162-shard-acts-s8k-bs4-lmhead-vocab-fsdp-rerun/plugins/profile/2026_06_04_18_42_10/`.
No new HLO dump was emitted under the v162 prefix, consistent with a
persistent-cache replay of v161's 785,947-byte train-step HLO
(`3648b17f7b1fdb754e9b9330400997ed2e06c3f25543cb549bb90559a8b7a348`). Do not
carry lm-head vocab-axis FSDP; treat v161 as run noise unless future profile
analysis identifies a controllable runtime difference.

## [2026-06-04] loop-iteration | v161-shard-acts-s8k-bs4-lmhead-vocab-fsdp on 8B/v6e-8: provisional supported (57,765 tok/s best-worker avg, 41.4% MFU; distinct HLO and lower compiled memory)

Workload `alekseyv-qwen3-v161-bs4-lmheadvocab` pivots from the closed local
VMEM bracket to the retrospective's structural CE/lm-head direction. It uses a
layered image from `qwen3-8b-jax:v041-shard-acts` with only
`lm_head.weight` sharding changed from `("tp", "fsdp")` to `("fsdp", None)`.
The runtime shape and flags otherwise match the confirmed v154 stack, including
scoped VMEM 100352 KiB. It completed cleanly with normal loss ending at 12.0458,
reaching worker0 57,765 tok/s / 7,221 tok/s/chip / 41.4% MFU and worker1
57,744 tok/s / 7,218 tok/s/chip / 41.4% MFU. Best observed steady steps were
57,908 and 57,904 tok/s. Fresh train-step HLO changed to 785,947 bytes with
SHA256 `3648b17f7b1fdb754e9b9330400997ed2e06c3f25543cb549bb90559a8b7a348`,
and compiled memory dropped from the v153/v154 26.86 GiB family to 26.76 GiB.
Treat as provisional supported and run an exact rerun before carrying.

## [2026-06-04] loop-iteration | v160-shard-acts-s8k-bs4-tokamax-splash-collective-matmul-none-vmem100864 on 8B/v6e-8: refuted/tie (57,762 tok/s best-worker avg, 41.4% MFU; byte-identical HLO)

Workload `alekseyv-qwen3-v160-bs4-vmem100864` tests the high-side midpoint
between confirmed 100352 and refuted 101376. It changes only
`--xla_tpu_scoped_vmem_limit_kib=100864` from the confirmed v154 stack. It
completed cleanly with normal loss ending at 12.0461, reaching worker0 57,762
tok/s / 7,220 tok/s/chip / 41.4% MFU and worker1 57,736 tok/s / 7,217
tok/s/chip / 41.4% MFU. The best observed steady step was 57,854 tok/s on both
workers, below v153's 57,888. Fresh optimized train-step HLO is 783,949 bytes
with SHA256 `3734fae066fc35c4b9c6c4cce9b0015413ab4513c34a09ba33a0e1862bff3a3d`,
byte-identical to v153/v158, and compiled memory remains 26.86 GiB. Do not
carry 100864; the apparent +5 tok/s worker0 average is same-HLO noise. Keep
100352 KiB and pivot out of local VMEM sweeps.

## [2026-06-04] loop-iteration | v159-shard-acts-s8k-bs4-tokamax-splash-collective-matmul-none-vmem99328 on 8B/v6e-8: refuted (57,519 tok/s, 41.2% MFU; lower midpoint slower)

Workload `alekseyv-qwen3-v159-bs4-vmem99328` tests the lower-side scoped-VMEM
midpoint between the confirmed 98304 and 100352 schedules. It changes only
`--xla_tpu_scoped_vmem_limit_kib=99328` from the confirmed v154 stack. It
completed cleanly with worker0 57,519 tok/s and worker1 57,494 tok/s, both
41.2% MFU. Fresh HLO changed to 784,575 bytes
(`fd192aaa99e632c9a2affba51d0309a58ff2449fac37c8663517df6d42850f49`) but the
schedule is slower, so keep 100352 KiB.

## [2026-06-04] loop-iteration | v158-shard-acts-s8k-bs4-tokamax-splash-collective-matmul-none-vmem100352-layeragg-pair on 8B/v6e-8: refuted/tie (57,755 tok/s, 41.4% MFU; byte-identical HLO)

Workload `alekseyv-qwen3-v158-bs4-layeragg` tests only the explicit
layer-scheduler plus sparse-core collective-aggregator pair on top of the
confirmed v154 `--xla_tpu_scoped_vmem_limit_kib=100352` frontier. All other
flags, mesh, batch, Splash settings, and profile window remain fixed. It
completed cleanly with worker0 57,730 tok/s and worker1 57,755 tok/s, both
41.4% MFU. Fresh train-step HLO is byte-identical to v153
(`3734fae066fc35c4b9c6c4cce9b0015413ab4513c34a09ba33a0e1862bff3a3d`,
783,949 bytes), with the true/true pair confirmed in the flagfile, so the pair
is a no-gain no-op on this frontier.

## [2026-06-04] loop-iteration | v157-shard-acts-s8k-bs4-tokamax-splash-collective-matmul-none-vmem100352-bkvdkv1024 on 8B/v6e-8: refuted (55,258 tok/s, 39.6% MFU; DKV tiling worsened)

Workload `alekseyv-qwen3-v157-bs4-bkvdkv1024` retested
`SPLASH_BKV_DKV=1024 SPLASH_BKV_DKV_COMPUTE=1024` on the confirmed
batch-4/vmem100352 frontier. It completed cleanly with normal loss but
regressed to worker0 55,258 tok/s and worker1 55,237 tok/s, both 39.6% MFU.
Fresh HLO was emitted (`783,982` bytes,
`5be74e908e49455d08150bed79701690304fc72bae18460845d1ecba43d994eb`), and
metadata confirms DKV `block_kv_dkv=1024`. The DKV custom-call scratch includes
a 4.12 GiB preallocated temp, so keep default DKV tiling.
## [2026-06-13] loop-iteration | v109-2k-batch16-splash2048-async-rs-cap1 on 8B/v6e-8: refuted/inert (70,963 tok/s, 45.6% MFU; byte-identical HLO)

Workload `alekseyv-qwen3-jax-v109-2ksp2048rs1` forced
`--xla_max_concurrent_async_reduce_scatters=1` on top of the confirmed
v095/v096 BKV2048 frontier. It completed cleanly with worker0 70,946 tok/s and
worker1 70,963 tok/s, both 45.6% MFU. XProf
`2026-06-13-qwen3-jax-v109-2k-bs16-vmem98304-splash2048-rs1/2026_06_13_06_27_55`
reported 3724.2 ms step time, 67.8% MXU utilization, and 0.2% idle. The fresh
optimized train-step HLO is byte-identical to v095/v096/v105/v107/v108:
`a8b0c4b3833fb22bb72597d9e68cdf28fd70e18ba0a695e247b2ebf56ed69bd2`,
19,375,297 bytes / 115,424 lines, with the same collective and fusion counts.
Do not carry the explicit reduce-scatter cap; `auto` is already equivalent for
this frontier.
## [2026-06-13] loop-iteration | v110-2k-batch16-splash2048-layerlhs on 8B/v6e-8: hard refuted (compile OOM; 78.08 GiB HBM requirement)

Workload `alekseyv-qwen3-jax-v110-2ksp2048layerlhs` forced
`--xla_tpu_enable_latency_hiding_layer_scheduler=true` on top of the confirmed
v095/v096 BKV2048 frontier. It failed during train-step compilation with
`RESOURCE_EXHAUSTED`, requiring 78.08 GiB HBM on a 31.25 GiB v6e device. The
fresh optimized train-step HLO changed to
`a93471451b4a2868f507f4eb56fea29378d5e5742242c338bfa3d9934abe21d9`,
15,893,236 bytes / 94,891 lines, with `all-gather=4928`, `async-start=253`,
`custom-call=2220`, and `copy=22493`. The memory report shows 83.76 GiB total
bytes used and 78.01 GiB preallocated temp; OOM artifacts identify repeated
768 MiB remat dot/fusion temps fed by all-gather producers. Do not carry the
explicit layer scheduler.
## [2026-06-13] loop-iteration | v111-2k-batch16-splash2048-async-ar-cap1 on 8B/v6e-8: refuted/inert (70,941 tok/s, 45.6% MFU; byte-identical HLO)

Workload `alekseyv-qwen3-jax-v111-2ksp2048ar1` forced
`--xla_max_concurrent_async_all_reduces=1` on top of the confirmed v095/v096
BKV2048 frontier. It completed cleanly with worker0 70,935 tok/s and worker1
70,941 tok/s, both 45.6% MFU. XProf run `2026_06_13_06_56_24` reported
3724.4 ms step time, 67.8% MXU utilization, and 0.2% idle. The optimized
train-step HLO is byte-identical to v095/v096/v105/v107/v108/v109:
`a8b0c4b3833fb22bb72597d9e68cdf28fd70e18ba0a695e247b2ebf56ed69bd2`,
19,375,297 bytes / 115,424 lines, with the same collective and fusion counts
and 27.88 GiB compiled memory. Do not carry the all-reduce cap.
## [2026-06-13] loop-iteration | v112-2k-batch16-splash2048-no-allreduce-offload on 8B/v6e-8: refuted (70,804 tok/s, 45.5% MFU; changed HLO but slower)

Workload `alekseyv-qwen3-jax-v112-2ksp2048noaroff` disabled
SparseCore all-reduce offload on top of the confirmed v095/v096 BKV2048
frontier. It completed cleanly with worker0 70,795 tok/s and worker1
70,804 tok/s, both 45.5% MFU. XProf run `2026_06_13_07_14_38` reported
3730.0 ms step time, 67.5% MXU utilization, and 0.2% idle. Runtime memory
improved to 30.60 GiB peak with 0.6493 GiB free, but throughput regressed. The
optimized train-step HLO changed to
`557078060d617d27e02f09218d4a1c240d9ac4e16ff60824bb3919a5877b55ff`,
19,386,544 bytes / 115,563 lines, with fewer all-reduces but more copies
(`copy=17906` vs frontier `copy=17231`) and higher compiled memory
28.49 GiB. Keep SparseCore all-reduce offload enabled.
## [2026-06-13] loop-iteration | v113-2k-batch16-splash2048-rs-lat2 on 8B/v6e-8: refuted (70,934 tok/s, 45.6% MFU; lower temp but slower)

Workload `alekseyv-qwen3-jax-v113-2ksp2048rslat2` lowered
`--xla_tpu_sparse_core_reduce_scatter_latency_multiplier` from 3 to 2 on the
confirmed v095/v096 BKV2048 frontier. It completed cleanly with worker0
70,934 tok/s and worker1 70,921 tok/s, both 45.6% MFU. XProf
`2026_06_13_07_30_10` reported 3728.0 ms step time, 67.7% MXU utilization, and
0.2% idle. The HLO changed to
`9a9241ee079ef0e18347ced2a070cb4d77a6fd8826e22f42d49b9243665bc4e5`, with
compiled memory down to 27.65 GiB and preallocated temp 21.90 GiB, but the
profile window slowed. Do not carry reduce-scatter latency multiplier 2.
## [2026-06-13] loop-iteration | v114-2k-batch16-splash2048-rs-lat4 on 8B/v6e-8: tentative frontier (71,037 tok/s, 45.7% MFU; scheduler-only change)

Workload `alekseyv-qwen3-jax-v114-2ksp2048rslat4` raised
`--xla_tpu_sparse_core_reduce_scatter_latency_multiplier` from 3 to 4 on the
confirmed v095/v096 BKV2048 frontier. It completed cleanly with worker0
71,035 tok/s and worker1 71,037 tok/s, both 45.7% MFU. XProf
`2026_06_13_07_44_15` reported 3721.2 ms step time, 68.0% MXU utilization, and
0.2% idle. The HLO changed to
`b03d411afe24c9ff0ed55829979aa76f3b9d3b8de4932139e8d398f29a54eff3`,
19,351,131 bytes / 115,235 lines, with compiled memory unchanged at 27.88 GiB
and 22.12 GiB preallocated temp. The gain is small and XProf step time is
effectively tied with v095, but both workers agree and the scheduler change is
real, so promote v114 tentatively and continue the latency multiplier bracket
with multiplier 5.
## [2026-06-13] loop-iteration | v115-2k-batch16-splash2048-rs-lat5 on 8B/v6e-8: hard refuted (runtime load OOM; 28.25 GiB compiled memory)

Workload `alekseyv-qwen3-jax-v115-2ksp2048rslat5` tests
`--xla_tpu_sparse_core_reduce_scatter_latency_multiplier=5` on top of v114. All
other flags, mesh, batch, Splash settings, profile window, and compile cache
policy are fixed. It failed before step 0 with `RESOURCE_EXHAUSTED` while
loading `jit_train_step`: a 2.12 GiB allocation could not be satisfied with
1.86 GiB free. The optimized train-step HLO changed to
`9a8061f2a54e1bb193b8ed78b21a5818cc65184cae7600e21d81c6053b39b350`,
19,345,298 bytes / 115,222 lines, with compiled memory up to 28.25 GiB and
22.50 GiB preallocated temp. This crosses the runtime headroom boundary. Do not
carry reduce-scatter latency multiplier 5; keep v114's multiplier 4.
## [2026-06-13] loop-iteration | v116-2k-batch16-splash2048-rs-lat4-ar-lat1 on 8B/v6e-8: refuted (70,960 tok/s, 45.6% MFU; slower profile)

Workload `alekseyv-qwen3-v116-2krsl4ar1` tests
`--xla_tpu_sparse_core_all_reduce_latency_multiplier=1` on the tentative v114
frontier with reduce-scatter latency multiplier 4. All other flags, mesh,
batch, Splash settings, and profile window remain fixed. It completed cleanly
with worker0 70,960 tok/s and worker1 70,952 tok/s, both 45.6% MFU. XProf
`2026_06_13_08_09_47` reported 3726.8 ms step time, 67.7% MXU utilization, and
0.2% idle. HLO changed to
`ce008beb94648c98bd92a7bacb2f70a5ce8d00a633f1e4540be0a4a4b180d1a8`,
19,354,349 bytes / 115,268 lines, with memory unchanged at 27.88 GiB total and
22.12 GiB preallocated temp. Do not carry all-reduce latency multiplier 1.
## [2026-06-13] loop-iteration | v117-2k-batch16-splash2048-rs-lat4-ar-lat3 on 8B/v6e-8: not promoted (70,989 tok/s, 45.6% MFU; better memory headroom)

Workload `alekseyv-qwen3-v117-2krsl4ar3` tests
`--xla_tpu_sparse_core_all_reduce_latency_multiplier=3` on the tentative v114
frontier with reduce-scatter latency multiplier 4. All other flags, mesh,
batch, Splash settings, and profile window remain fixed. It completed cleanly
with worker0 70,978 tok/s and worker1 70,989 tok/s, both 45.6% MFU. XProf
`2026_06_13_08_23_46` reported 3717.8 ms step time, 67.9% MXU utilization, and
0.2% idle. HLO changed to
`50046481b1b469d9ac0e642de7648eb4bdea368996f787f4224ba64a90fd9ab3`,
19,353,411 bytes / 115,256 lines, with compiled memory down to 27.65 GiB and
21.90 GiB preallocated temp. Do not promote over v114 by harness throughput,
but use the extra memory headroom to retry reduce-scatter latency multiplier 5
combined with all-reduce latency multiplier 3.
## [2026-06-13] loop-iteration | v118-2k-batch16-splash2048-rs-lat5-ar-lat3 on 8B/v6e-8: hard refuted (runtime load OOM; RS5 memory shape persists)

Workload `alekseyv-qwen3-v118-2krsl5ar3` combines
`--xla_tpu_sparse_core_reduce_scatter_latency_multiplier=5` with
`--xla_tpu_sparse_core_all_reduce_latency_multiplier=3`. This tests whether
v117's memory relief can make the v115 reduce-scatter multiplier 5 schedule fit
and improve over v114. All other flags, mesh, batch, Splash settings, and
profile window remain fixed. It failed before step 0 with `RESOURCE_EXHAUSTED`
while loading `jit_train_step`: a 2.12 GiB allocation could not be satisfied
with 1.86 GiB free. The optimized train-step HLO changed to
`d4897181619cde937837186d2856d72e08a9af32e11930f41e5824f5412c8e53`,
19,345,317 bytes / 115,222 lines, with compiled memory 28.25 GiB and
22.50 GiB preallocated temp. This matches the v115 bad RS5 memory shape; close
the reduce-scatter latency multiplier 5 path.
## [2026-06-13] loop-iteration | v119-2k-batch16-splash2048-rs-lat4-rerun4 on 8B/v6e-8: measurement frontier (71,053 tok/s, 45.7% MFU; byte-identical HLO)

Workload `alekseyv-qwen3-v119-2krsl4rer4` tests
`--xla_latency_hiding_scheduler_rerun=4` on the tentative v114 frontier. The
collective latency multipliers remain at v114 values: reduce-scatter 4 and
all-reduce 2. All other flags, mesh, batch, Splash settings, and profile window
remain fixed. It completed cleanly with worker0 71,047 tok/s and worker1
71,053 tok/s, both 45.7% MFU. XProf `2026_06_13_08_48_33` reported 3718.9 ms
step time, 67.9% MXU utilization, and 0.2% idle. The optimized train-step HLO
is byte-identical to v114:
`b03d411afe24c9ff0ed55829979aa76f3b9d3b8de4932139e8d398f29a54eff3`,
19,351,131 bytes / 115,235 lines, with the same 27.88 GiB compiled memory and
22.12 GiB preallocated temp. Promote as the current measurement best, but do
not treat scheduler rerun 4 as a distinct graph improvement.
## [2026-06-13] loop-iteration | v120-2k-batch16-splash2048-rs-lat4-agoff on 8B/v6e-8: hard refuted (load OOM; async/copy explosion)

Workload `alekseyv-qwen3-v120-2krsl4agoff` tests
`--xla_tpu_enable_sparse_core_collective_offload_all_gather=true` on the
v114/v119 graph while keeping the SparseCore collective aggregator enabled. All
other flags, mesh, batch, Splash settings, and profile window remain fixed. It
failed before step 0 with `RESOURCE_EXHAUSTED` while loading `jit_train_step`:
a 2.23 GiB allocation could not be satisfied with 1.60 GiB free. The optimized
train-step HLO changed to
`0283d250de511c60f43462edf47712f5996549a9130b9d562cd3108b729d2a01`,
16,475,320 bytes / 94,949 lines, with `async-start=1426`, `copy=22253`, and
compiled memory 28.50 GiB / 22.75 GiB preallocated temp. Keep SparseCore
all-gather offload disabled.
## [2026-06-13] loop-iteration | v121-2k-batch16-splash2048-rs-lat4-noagg on 8B/v6e-8: started

Workload `alekseyv-qwen3-v121-2krsl4noagg` tests
`--xla_tpu_enable_sparse_core_collective_aggregator=false` while keeping
SparseCore all-gather offload disabled. All other v114/v119 graph settings,
mesh, batch, Splash settings, and profile window remain fixed.

## [2026-06-13] loop-iteration | v121-2k-batch16-splash2048-rs-lat4-noagg on 8B/v6e-8: refuted (71,044 tok/s, byte-identical HLO)

Workload `alekseyv-qwen3-v121-2krsl4noagg` completed cleanly with worker0
71,035 tok/s and worker1 71,044 tok/s, both about 45.7% MFU. XProf
`2026_06_13_09_14_12` reported 3720.7 ms step time, 67.9% MXU utilization,
and 0.2% idle. The optimized train-step HLO was byte-identical to v114/v119:
`b03d411afe24c9ff0ed55829979aa76f3b9d3b8de4932139e8d398f29a54eff3`,
19,351,131 bytes / 115,235 lines, with the same 27.88 GiB compiled memory and
22.12 GiB preallocated temp. Treat the SparseCore aggregator flag as inert for
this graph and keep the v114/v119 settings.

## [2026-06-13] loop-iteration | v122-2k-batch16-splash2048-rs-lat4-shmem99 on 8B/v6e-8: started

Workload `alekseyv-qwen3-v122-2krsl4sh99` tests a close shared-memory
scheduler bracket on the v114/v119 graph:
`--xla_tpu_scheduler_percent_shared_memory_limit=99` instead of 100. All other
frontier settings remain fixed, including batch 16, sequence 2048, Splash
BKV/BKV_COMPUTE 2048, scoped VMEM 98304 KiB, reduce-scatter latency multiplier
4, all-reduce latency multiplier 2, SparseCore all-gather offload disabled, and
the SparseCore collective aggregator enabled.

## [2026-06-13] loop-iteration | v122-2k-batch16-splash2048-rs-lat4-shmem99 on 8B/v6e-8: refuted/tie (71,054 tok/s, byte-identical HLO)

Workload `alekseyv-qwen3-v122-2krsl4sh99` completed cleanly with worker0
71,043 tok/s and worker1 71,054 tok/s, both about 45.7% MFU. XProf
`2026_06_13_09_29_38` reported 3721.5 ms step time, 68.0% MXU utilization,
and 0.2% idle. The optimized train-step HLO remained byte-identical to
v114/v119/v121:
`b03d411afe24c9ff0ed55829979aa76f3b9d3b8de4932139e8d398f29a54eff3`,
19,351,131 bytes / 115,235 lines, with 27.88 GiB compiled memory and 22.12
GiB preallocated temp. Treat shmem99 as a measurement tie/no-op rather than a
new frontier.

## [2026-06-13] loop-iteration | v123-2k-batch16-splash2048-rs-lat4-vmem98816 on 8B/v6e-8: started

Workload `alekseyv-qwen3-v123-2krsl4vm98816` tests a close high-side scoped
VMEM midpoint, `--xla_tpu_scoped_vmem_limit_kib=98816`, after v098 showed the
farther high-side VMEM100352 point selected a distinct but slower schedule.
All other v114/v119 settings remain fixed, including shmem100, reduce-scatter
latency multiplier 4, all-reduce latency multiplier 2, SparseCore all-gather
offload disabled, SparseCore collective aggregator enabled, batch 16, sequence
2048, and Splash BKV/BKV_COMPUTE 2048.

## [2026-06-13] loop-iteration | v123-2k-batch16-splash2048-rs-lat4-vmem98816 on 8B/v6e-8: refuted (70,851 tok/s, copy-heavy HLO)

Workload `alekseyv-qwen3-v123-2krsl4vm98816` completed cleanly with both
workers at 70,851 tok/s, about 45.6% MFU. XProf `2026_06_13_09_44_15`
reported 3731.5 ms step time, 67.8% MXU utilization, and 0.2% idle. Runtime
memory improved to 31.11/31.25 GiB with 0.1312 GiB free, but the optimized
train-step HLO changed to
`f2b99e5a7776ed4888130702cfee553e065b85fd62d518083c6fbcecd1efedb3`,
19,427,455 bytes / 115,867 lines, with `copy=19248` and `all-gather=6990`.
This is a copy-heavy high-VMEM schedule and is slower; close the immediate
high-side VMEM branch and keep 98304 KiB.

## [2026-06-13] loop-iteration | v124-2k-batch16-splash-bq1024-bkv2048-rs-lat4 on 8B/v6e-8: started

Workload `alekseyv-qwen3-v124-2kbq1024` keeps the v114/v119 collectives stack
and successful forward `SPLASH_BKV=2048`, but sets `SPLASH_BQ=1024`. This tests
whether the smaller forward query tile favored in related Llama Splash kernel
tuning helps the Qwen3 2k custom-call bucket without reverting to the slower
BKV1024 graph.

## [2026-06-13] loop-iteration | v124-2k-batch16-splash-bq1024-bkv2048-rs-lat4 on 8B/v6e-8: refuted (70,821 tok/s, copy-heavy Splash HLO)

Workload `alekseyv-qwen3-v124-2kbq1024` completed cleanly with worker0
70,821 tok/s and worker1 70,817 tok/s, about 45.5% MFU. XProf
`2026_06_13_09_58_46` reported 3731.0 ms step time, 67.8% MXU utilization,
and 0.2% idle. Runtime memory improved to 31.11/31.25 GiB with 0.1355 GiB
free, but the optimized train-step HLO changed to
`4345f4d975386a13c760a2f10c5500db2f482fd82c455815d79abf92bbcf05af`,
19,686,144 bytes / 117,638 lines, with `copy=20234`, `fusion=30791`, and
`tokamax=2`. Smaller forward query blocks are worse here; keep
`SPLASH_BQ=2048 SPLASH_BKV=2048`.

## [2026-06-13] loop-iteration | v125-2k-batch16-splash2048-rs-lat4-noacfg on 8B/v6e-8: started

Workload `alekseyv-qwen3-v125-2knoacfg` reuses the v114/v119 frontier graph and
changes only `--xla_tpu_enable_async_collective_fusion_fuse_all_gather=false`.
This isolates all-gather ACF on the current 2k BKV2048/RS-lat4 graph while
leaving base ACF, reduce-scatter ACF, multiple-step ACF, SparseCore offload
policy, VMEM, shmem, batch, sequence length, and Splash tiles fixed.

## [2026-06-13] loop-iteration | v125-2k-batch16-splash2048-rs-lat4-noacfg on 8B/v6e-8: refuted (69,485 tok/s, memory-only win)

Workload `alekseyv-qwen3-v125-2knoacfg` completed cleanly with worker0
69,485 tok/s and worker1 69,482 tok/s, about 44.7% MFU. XProf
`2026_06_13_10_12_32` reported 3798.8 ms step time, 66.1% MXU utilization, and
0.2% idle. Memory improved materially to 30.55/31.25 GiB with 0.6985 GiB free,
and HLO shrank to
`081ebafd20724172c5037e15c8977080cf21b68d680098b342240d8a0e95982b`,
13,005,762 bytes / 76,725 lines, compiled memory 27.32 GiB. The speed loss is
too large; keep all-gather ACF enabled.

## [2026-06-13] loop-iteration | v126-2k-batch16-splash2048-rs-lat4-noacfms on 8B/v6e-8: started

Workload `alekseyv-qwen3-v126-2knoacfms` reuses the v114/v119 frontier graph
and changes only `--xla_tpu_enable_async_collective_fusion_multiple_steps=false`.
This preserves all-gather and reduce-scatter ACF while testing whether the
multiple-step subtype is adding harmful schedule pressure on the current 2k
BKV2048/RS-lat4 graph.

## [2026-06-13] loop-iteration | v126-2k-batch16-splash2048-rs-lat4-noacfms on 8B/v6e-8: refuted (70,858 tok/s, lower profile quality)

Workload `alekseyv-qwen3-v126-2knoacfms` completed cleanly but stayed below the
v119 frontier. Worker0 reached **70,838 tok/s / 8,855 tok/s/chip / 45.5% MFU**;
worker1 reached **70,858 tok/s / 8,857 tok/s/chip / 45.6% MFU**. XProf run
`2026_06_13_10_25_56` reported **3727.6 ms** average step time, **67.8% MXU**,
**0.3%** idle in op profile, and **31.05 GiB / 31.25 GiB** peak HBM.
Fresh train-step HLO hash
`4c515807306666ee46948268f8732ca0362e6a729e2d96fea81424de6ee33518` is
13,345,898 bytes / 76,638 lines with compile memory 27.85 GiB. Keep ACF
multiple-steps enabled.

## [2026-06-13] loop-iteration | v127-2k-batch16-splash2048-rs-lat4-noacf on 8B/v6e-8: started

Launching workload `alekseyv-qwen3-v127-2knoacf` from the v119 named frontier,
changing only `--xla_tpu_enable_async_collective_fusion=false`. The run keeps
batch16, sequence 2048, Splash BKV/BKV_COMPUTE 2048, VMEM 98304, shmem100,
scheduler rerun 4, RS latency multiplier 4, AR latency multiplier 2, and the
SparseCore offload/aggregator settings fixed. This closes the current ACF sweep
after v125 and v126 both regressed.

## [2026-06-13] loop-iteration | v127-2k-batch16-splash2048-rs-lat4-noacf on 8B/v6e-8: refuted (69,472 tok/s, same HLO as no all-gather ACF)

Workload `alekseyv-qwen3-v127-2knoacf` completed cleanly but regressed hard:
worker0 reached **69,472 tok/s / 8,684 tok/s/chip / 44.7% MFU**, and worker1
reached **69,470 tok/s / 8,684 tok/s/chip / 44.7% MFU**. The optimized HLO hash
is `081ebafd20724172c5037e15c8977080cf21b68d680098b342240d8a0e95982b`, exactly
matching v125's no-all-gather-ACF program: 13,005,762 bytes / 76,725 lines,
compile memory 27.32 GiB, all-gather=2694, reduce-scatter=2774, copy=16336.
This confirms the base ACF gate is required and collapses to the same slower
schedule family as disabling all-gather ACF.

## [2026-06-13] loop-iteration | v128-2k-batch16-splash2048-rs-lat4-nomemtrack on 8B/v6e-8: started

Launching workload `alekseyv-qwen3-v128-2knomem` from the v119 named frontier,
adding only `--xla_tpu_enable_scheduler_memory_pressure_tracking=false`. This
tests whether TPU scheduler memory-pressure tracking is costing schedule
quality or runtime overhead on the HBM-tight 2k batch16 graph while all known
frontier-critical ACF and SparseCore settings remain enabled.

## [2026-06-13] loop-iteration | v128-2k-batch16-splash2048-rs-lat4-nomemtrack on 8B/v6e-8: refuted/tie (70,985 tok/s, byte-identical HLO)

Workload `alekseyv-qwen3-v128-2knomem` completed cleanly with worker0
**70,985 tok/s / 8,873 tok/s/chip / 45.6% MFU** and worker1 **70,983 tok/s /
8,873 tok/s/chip / 45.6% MFU**. XProf run `2026_06_13_10_53_48` reported
**3723.8 ms** average step time, **67.9% MXU**, **0.2%** idle, and **31.14 GiB
/ 31.25 GiB** peak HBM with **0.1016 GiB** free. The optimized HLO remained
byte-identical to v114/v119/v121/v122:
`b03d411afe24c9ff0ed55829979aa76f3b9d3b8de4932139e8d398f29a54eff3`.
Disabling TPU scheduler memory-pressure tracking does not improve the 2k
frontier; keep the default.

## [2026-06-13] loop-iteration | v129-2k-batch16-splash2048-rs-lat4-dkvc512 on 8B/v6e-8: started

Launching workload `alekseyv-qwen3-v129-2kdkvc512` from the v119 named
frontier, changing only `SPLASH_BKV_DKV_COMPUTE=512`. This is a targeted
Tokamax Splash backward-tile probe against the remaining DKV custom-call
profile bucket, while keeping batch16, sequence 2048, forward Splash
BQ/BKV/BKV_COMPUTE 2048, backward BQ_DKV/BKV_DKV 2048, VMEM 98304, shmem100,
scheduler rerun 4, RS latency multiplier 4, AR latency multiplier 2, and the
frontier ACF/SparseCore settings fixed.

## [2026-06-13] loop-iteration | v129-2k-batch16-splash2048-rs-lat4-dkvc512 on 8B/v6e-8: refuted (70,986 tok/s, changed HLO but no gain)

Workload `alekseyv-qwen3-v129-2kdkvc512` completed cleanly but did not improve
the frontier. Worker0 reached **70,986 tok/s / 8,873 tok/s/chip / 45.6% MFU**;
worker1 reached **70,973 tok/s / 8,872 tok/s/chip / 45.6% MFU**. XProf run
`2026_06_13_11_10_06` reported **3722.4 ms** average step time, **67.6% MXU**,
**0.2%** idle, and **31.14 GiB / 31.25 GiB** peak HBM with **0.1011 GiB** free.
The optimized HLO changed to
`a75fe20ce6532c3f2758037ee0ca96090351cc2fc6d6a2583475c3465d7d8043` but kept the
same high-level op-token counts as v119/v128. Do not carry
`SPLASH_BKV_DKV_COMPUTE=512`.

## [2026-06-13] loop-iteration | v130-2k-batch16-splash2048-rs-lat4-seqminor on 8B/v6e-8: started

Launching workload `alekseyv-qwen3-v130-2kseqminor` from the v119 named
frontier using thin image `qwen3-8b-jax:v130-splash-layout-env`, which only
exposes Tokamax Splash Q/K/V layout selection through env vars. The run sets
`TOKAMAX_Q_LAYOUT=SEQ_MINOR TOKAMAX_K_LAYOUT=SEQ_MINOR TOKAMAX_V_LAYOUT=SEQ_MINOR`
to probe whether the attention data-formatting copy/reshape bucket can be
reduced while leaving batch, sequence length, Splash block sizes, VMEM, shmem,
collective flags, and SparseCore settings fixed.

## [2026-06-13] loop-iteration | v130-2k-batch16-splash2048-rs-lat4-seqminor on 8B/v6e-8: supported (73,582 tok/s, 47.3% MFU)

Workload `alekseyv-qwen3-v130-2kseqminor` completed cleanly and established a
new 2k frontier. Worker0 reached **73,576 tok/s / 9,197 tok/s/chip / 47.3%
MFU**; worker1 reached **73,582 tok/s / 9,198 tok/s/chip / 47.3% MFU**. The
patched image confirmed `layouts=(SEQ_MINOR,SEQ_MINOR,SEQ_MINOR)` in the
Tokamax Splash config. Optimized HLO hash
`66462ca89d8bad5075122542dab49ce88a47ad9923e87107e81746cc4d23d831` is
19,377,734 bytes / 115,099 lines. HLO copy-token count dropped from
17,285 in v119/v128 to **15,956** while collective counts stayed aligned with
the frontier family. Carry the v130 layout-env image and all-`SEQ_MINOR`
Tokamax Splash layout as the current frontier.

## [2026-06-13] loop-iteration | v131-2k-batch16-splash2048-rs-lat4-qseqminor on 8B/v6e-8: started

Launching workload `alekseyv-qwen3-v131-2kqseq` to isolate v130's gain. It uses
the same v130 layout-env image and v119 frontier settings, but sets only
`TOKAMAX_Q_LAYOUT=SEQ_MINOR`; K/V remain at the default `HEAD_DIM_MINOR`.

## [2026-06-13] loop-iteration | v131-2k-batch16-splash2048-rs-lat4-qseqminor on 8B/v6e-8: refuted versus v130 (72,890 tok/s, 46.9% MFU)

Workload `alekseyv-qwen3-v131-2kqseq` completed cleanly with worker0
**72,883 tok/s / 9,110 tok/s/chip / 46.9% MFU** and worker1 **72,890 tok/s /
9,111 tok/s/chip / 46.9% MFU**. This beats the old v119 frontier but falls
short of v130's 73,582 tok/s. HLO hash
`8445702d8654166be7694a400e695e916eaabc75774b189b0d81eef82514c570` has
17,861 copy tokens, worse than both v119 and v130. Q-only `SEQ_MINOR` is not
the right carry setting.

## [2026-06-13] loop-iteration | v132-2k-batch16-splash2048-rs-lat4-kvseqminor on 8B/v6e-8: started

Launching workload `alekseyv-qwen3-v132-2kkvseq` as the complementary isolation
probe: K/V use `SEQ_MINOR`, Q remains at the default `HEAD_DIM_MINOR`.

## [2026-06-13] loop-iteration | v132-2k-batch16-splash2048-rs-lat4-kvseqminor on 8B/v6e-8: refuted versus v130 (71,601 tok/s, 46.0% MFU)

Workload `alekseyv-qwen3-v132-2kkvseq` completed cleanly with both workers at
**71,601 tok/s / 8,950 tok/s/chip / 46.0% MFU**. The run confirmed
`layouts=(HEAD_DIM_MINOR,SEQ_MINOR,SEQ_MINOR)` but did not reproduce v130's
all-`SEQ_MINOR` improvement. HLO hash
`3028daac1b565bb7860390679ec90ff208b2ba9670bb7f3a32a57166b97c2338` has
17,107 copy tokens, still far above v130's 15,956. Keep v130 all-Q/K/V
`SEQ_MINOR` as the frontier; K/V-only is not useful.

## [2026-06-13] loop-iteration | v133-2k-batch16-splash2048-rs-lat4-kseqminor on 8B/v6e-8: started

Launching workload `alekseyv-qwen3-v133-2kkseq` to isolate the K operand after
v131 and v132. It uses the same v130 layout-env image and frontier settings,
but sets only `TOKAMAX_K_LAYOUT=SEQ_MINOR`; Q/V remain at the default
`HEAD_DIM_MINOR`.

## [2026-06-13] loop-iteration | v133-2k-batch16-splash2048-rs-lat4-kseqminor on 8B/v6e-8: refuted versus v130 (71,531 tok/s, 46.0% MFU)

Workload `alekseyv-qwen3-v133-2kkseq` completed cleanly with worker0
**71,531 tok/s / 8,941 tok/s/chip / 46.0% MFU** and worker1 **71,529 tok/s /
8,941 tok/s/chip / 46.0% MFU**. The run confirmed
`layouts=(HEAD_DIM_MINOR,SEQ_MINOR,HEAD_DIM_MINOR)` but stayed well below v130.
HLO hash `ac1cd1d2831f6b34fd97710d35a3a87a275586960475b5d7d638c59271560757`
has 17,215 copy tokens, so K-only does not reproduce the v130 formatting
improvement.

## [2026-06-13] loop-iteration | v134-2k-batch16-splash2048-rs-lat4-vseqminor on 8B/v6e-8: started

Launching workload `alekseyv-qwen3-v134-2kvseq` as the final single-operand
layout isolation. It uses the same v130 layout-env image and frontier settings,
but sets only `TOKAMAX_V_LAYOUT=SEQ_MINOR`; Q/K remain at the default
`HEAD_DIM_MINOR`.

## [2026-06-13] loop-iteration | v134-2k-batch16-splash2048-rs-lat4-vseqminor on 8B/v6e-8: refuted versus v130 (71,129 tok/s, 45.7% MFU)

Workload `alekseyv-qwen3-v134-2kvseq` completed cleanly with worker0
**71,119 tok/s / 8,890 tok/s/chip / 45.7% MFU** and worker1 **71,129 tok/s /
8,891 tok/s/chip / 45.7% MFU**. The run confirmed
`layouts=(HEAD_DIM_MINOR,HEAD_DIM_MINOR,SEQ_MINOR)` but stayed below even the
old v119 frontier. HLO hash
`4bdade7017b817c11c17ca3cd95c66ae51d327e60b10ae1efe1cbfbf52b0d9b4` has
17,165 copy tokens. The single-operand/subset layout sweep is closed: carry
v130's all-Q/K/V `SEQ_MINOR` layout.

## [2026-06-13] loop-iteration | v135-2k-batch17-splash2048-rs-lat4-seqminor-shmem90 on 8B/v6e-8: started

Launching workload `alekseyv-qwen3-v135-2kbs17seqsh90` to test whether v130's
all-Q/K/V `SEQ_MINOR` graph combines with the previously memory-fitting
batch17/shmem90 branch. This changes only `--batch_size=17` and
`--xla_tpu_scheduler_percent_shared_memory_limit=90` relative to v130; all
layout, Splash, VMEM, scheduler rerun, RS latency, ACF, and SparseCore settings
remain on the v130 frontier.

## [2026-06-13] loop-iteration | v135-2k-batch17-splash2048-rs-lat4-seqminor-shmem90 on 8B/v6e-8: refuted/tie versus v130 (73,579 tok/s, 47.3% MFU)

Workload `alekseyv-qwen3-v135-2kbs17seqsh90` completed cleanly and proved the
batch17 all-`SEQ_MINOR` shape fits at shmem90. Worker0 reached **73,579 tok/s /
9,197 tok/s/chip / 47.3% MFU**; worker1 reached **73,577 tok/s / 9,197
tok/s/chip / 47.3% MFU**. This nearly ties but does not beat v130's **73,582
tok/s**. HLO hash
`bede1203771f2043f4fd05ce4850c14dedd2e6c9fa5e588866a10c19c8ebbd8a` has 16,174
copy tokens and 28.77 GiB compiled memory. The next useful bracket is batch17
all-`SEQ_MINOR` with shmem95, because v135 appears schedule-limited by shmem90
while direct batch17/shmem100 was previously over the memory edge.

## [2026-06-13] loop-iteration | v136-2k-batch17-splash2048-rs-lat4-seqminor-shmem95 on 8B/v6e-8: started

Launching workload `alekseyv-qwen3-v136-2kbs17seqsh95` from v135, changing only
`--xla_tpu_scheduler_percent_shared_memory_limit=95`. The goal is to recover
schedule quality versus shmem90 while checking whether the all-`SEQ_MINOR`
batch17 graph still fits below the runtime memory edge.

## [2026-06-13] loop-iteration | v136-2k-batch17-splash2048-rs-lat4-seqminor-shmem95 on 8B/v6e-8: provisionally supported (73,587 tok/s, 47.3% MFU)

Workload `alekseyv-qwen3-v136-2kbs17seqsh95` completed cleanly with worker0
**73,587 tok/s / 9,198 tok/s/chip / 47.3% MFU** and worker1 **73,580 tok/s /
9,198 tok/s/chip / 47.3% MFU**. This is only **+5 tok/s** over v130 but is the
current raw-throughput high-water mark. HLO hash
`b78961298603163ac732723bbbd718731d73864426456267aa9a0df5db37f8be` has 16,157
copy tokens and 29.08 GiB compiled memory. Treat v136 as a provisional frontier
candidate and bracket the shmem limit upward before trusting the small gain.

## [2026-06-13] loop-iteration | v137-2k-batch17-splash2048-rs-lat4-seqminor-shmem97 on 8B/v6e-8: started

Launching workload `alekseyv-qwen3-v137-2kbs17seqsh97` from v136, changing only
`--xla_tpu_scheduler_percent_shared_memory_limit=97`. This tests whether the
batch17 all-`SEQ_MINOR` branch has a real schedule-quality optimum above
shmem95 before it hits the known batch17/shmem100 memory boundary.
