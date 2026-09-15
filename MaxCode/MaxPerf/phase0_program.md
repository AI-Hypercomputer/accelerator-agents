<!-- disableFinding(LINK_RELATIVE_G3DOC) -->
<!-- disableFinding(LINE_OVER_80) -->
<!-- disableFinding(LIST_NO_LINE) -->
<!-- disableFinding(SNIPPET_EM_DASH) -->
<!-- disableFinding(WHITESPACE_LINES) -->
<!-- disableFinding(SPACES) -->
<!-- disableFinding(HTML_OPEN) -->
<!-- disableFinding(HTML_BROKEN) -->

# phase0_program.md — vLLM Flag-Level Tuning for Qwen2.5-Coder-3B

This file governs **Phase 0** (Level 0 tuning) for the `Qwen/Qwen2.5-Coder-3B-Instruct` model on Cloud TPUs. Phase 0 focuses exclusively on finding the optimal combination of vLLM serving flags to maximize steady-state throughput before passing the best configuration to the MaxPerf phase for graph and kernel optimization.

---

## Target

| Field | Value |
|-------|-------|
| **Model** | `Qwen/Qwen2.5-Coder-3B-Instruct` |
| **Stack** | vLLM TPU backend (V1 engine) |
| **Hardware** | TPU v6e-16 (Assume 16 chips, adjustable based on availability) |
| **Scenario** | Serving inference |

## Metrics

**Primary Metric:** Steady-state throughput measured in **Tokens/Second/Chip**.

**Secondary Diagnostics:**
- **p50/p99 TTFT** (Time to First Token)
- **p50/p99 TPOT** (Time Per Output Token)
- **HBM peak per chip** (Memory high-water mark)

---

## Tuning Knobs (Search Space)

The following vLLM flags are the primary levers for optimization. Each experiment should vary these to find the sweet spot between memory capacity and compute efficiency.

| Flag | Description | Candidate Values |
|------|-------------|------------------|
| `--block-size` | Paged KV cache block size | `[128, 256, 512]` |
| `--kv-cache-dtype` | Precision of stored KV states | `[fp8, bfloat16]` |
| `--tensor-parallel-size` | Sharding axis for dense projections | `[8, 16]` |
| `--enable-expert-parallel` | Distribute MoE experts across chips | `[True, False]` |
| `--max-model-len` | Maximum context length supported | `[8192, 9216]` |
| `--max-num-seqs` | Concurrency limit | `[32, 64, 128, 512]` |
| `--async-scheduling` | Enable overlapping scheduling with execution | `[True, False]` |

---

## Fixed Workload Bindings

To ensure comparable results across experiments, benchmarks will be run against a fixed set of synthetic workloads simulating different traffic regimes (based on `tpu_run_model.sh`):

| Regime | Input Length | Output Length | Request Rate | Prompts |
|--------|--------------|---------------|--------------|---------|
| **Prefill Bound** | 8192 | 1024 | `inf` | 320 |
| **Decode Bound** | 1024 | 8192 | `inf` | 320 |
| **Balanced** | 1024 | 1024 | `inf` | 320 |

*All runs must use `--max-concurrency=64` as the default traffic density.*

---

## Search Protocol

Tuning will follow an iterative isolation protocol to avoid combinatorial explosion:

1. **Step 1: Baseline Capture**
   - Run the configuration defined in `tpu_run_model.sh`.
   - Record initial TPS, TTFT, and TPOT.

2. **Step 2: Independent Sweeps**
   - Sweep `--block-size` values while keeping others fixed.
   - Test `--kv-cache-dtype=bfloat16` to measure the memory/speed tradeoff.
   - Vary `--max-num-seqs` to find the peak concurrency before latency degrades.

3. **Step 3: Candidate Fusion**
   - Combine the winning values from Step 2.
   - Verify stability and verify that gains compound.

4. **Step 4: Hand-off**
   - Document the best configuration in `runs/phase0-best.sh`.
   - Populate the `Baseline` section in `program.md` with the resulting metrics.

---

## Hardware Connection & Execution

To execute benchmarks on the TPU v6e-16 VM, connect using the connection helper script:

```bash
# 1. Start an interactive SSH session
./runs/ssh_tpu.sh

# 2. Run a specific command on the TPU VM
./runs/ssh_tpu.sh "your-command-here"
```

The SSH parameters are preserved in [runs/ssh_tpu.sh](file:///Users/gvanica/dev/MaxPerf/max-perf-g/MaxPerf/runs/ssh_tpu.sh):
- **VM Name**: `maxperf-v6e-1`
- **Zone**: `asia-northeast1-b`
- **Project**: `tpu-prod-env-multipod`


---

## Baseline Ledger (RESULTS_PHASE0.tsv)

A separate ledger `RESULTS_PHASE0.tsv` should be created to track these runs.

| Exp ID | Block Size | KV Dtype | TP Size | Max Seqs | TPS/Chip | p50 TTFT | p50 TPOT | Status |
|--------|------------|----------|---------|----------|----------|----------|----------|--------|
| `000`  | 256        | fp8      | 8       | 512      | `<TBD>`  | `<TBD>`  | `<TBD>`  | Baseline |
