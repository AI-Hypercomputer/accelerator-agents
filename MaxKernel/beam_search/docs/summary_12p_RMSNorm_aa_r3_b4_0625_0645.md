# Dynamic Beam Search Run Summary Report (AutoAgent)

**Command Line:**
```bash
python3 run_beam_search.py --use_beam_worker false --rounds 3 --beam_size 4 --task_id 12p_RMSNorm
```

This document summarizes the execution and performance analysis of the local Dynamic Beam Search orchestrator run optimizing the **`12p_RMSNorm`** JAX/Pallas kernel using **`auto_agent`** mode.

---

## 1. Run Metadata
*   **Target Kernel**: `12p_RMSNorm` (RMS Normalization block)
*   **Search Configuration**:
    *   Rounds: `3`
    *   Beam Size ($B$): `4`
    *   Keep Factor: `1.0` (Regression Budget Gate enabled)
    *   Worker Mode: `auto_agent` (using `AutonomousPipelineAgent` with compilation validation, autotuning, and profiling)
*   **Orchestrator Mode**: Mock compiler evaluation mode (`mock_mode=True`) using `FakeKernelEvaluator` for fallbacks, with autotune results bypass enabled.
*   **Baseline Latency**: **`11.400 ms`**

---

## 2. Search Progression Timeline

### Round 1
*   **Baseline Code**: Reference RMSNorm kernel ($11.400\text{ ms}$).
*   **Worker Execution**:
    *   **`Round_1_Worker_0`** (`beam_r1_w0`): Succeeded. Autotuned latency parsed: **`8.030 ms`** (Bypassed evaluator).
    *   **`Round_1_Worker_1`** (`beam_r1_w1`): Succeeded (fallback to LLM selection, skipped autotuning). Evaluator fallback latency: **`10.830 ms`**.
    *   **`Round_1_Worker_2`** (`beam_r1_w2`): Succeeded. Autotuned latency parsed: **`8.000 ms`** (Bypassed evaluator).
    *   **`Round_1_Worker_3`** (`beam_r1_w3`): Failed correctness tests in Mode 1. Exited.
*   **Incumbent Preservation**: Because `Round_1_Worker_3` failed correctness, the original baseline candidate was preserved as Rank 4.

#### Round 1 Beam State:
```
==================================================
            ROUND 1 BEAM STATE
==================================================
  Rank 1: Latency=8.000 ms  | Path=beam_search/output/12p_RMSNorm_aa_r3_b4_0625_0645/beam_r1_w2_1782369908/optimized_kernel.py
  Rank 2: Latency=8.030 ms  | Path=beam_search/output/12p_RMSNorm_aa_r3_b4_0625_0645/beam_r1_w0_1782369908/optimized_kernel.py
  Rank 3: Latency=10.830 ms | Path=beam_search/output/12p_RMSNorm_aa_r3_b4_0625_0645/beam_r1_w1_1782369908/optimized_kernel.py
  Rank 4: Latency=11.400 ms | Path=beam_search/output/12p_RMSNorm/baseline.py (Incumbent Preserved)
==================================================
```

---

### Round 2
*   **Worker Launch Paths**:
    *   `Round_2_Worker_0` spawned from Round 1 Rank 1 baseline ($8.000\text{ ms}$).
    *   `Round_2_Worker_1` spawned from Round 1 Rank 2 baseline ($8.030\text{ ms}$).
    *   `Round_2_Worker_2` spawned from Round 1 Rank 3 baseline ($10.830\text{ ms}$).
    *   `Round_2_Worker_3` spawned from Round 1 Rank 4 baseline ($11.400\text{ ms}$).
*   **Worker Execution**:
    *   **`Round_2_Worker_0`** (`beam_r2_w0`): Succeeded. Autotuned latency parsed: **`8.030 ms`** (Pruned since it didn't beat the parent limit).
    *   **`Round_2_Worker_1`** (`beam_r2_w1`): Succeeded. Autotuned latency parsed: **`6.100 ms`** (Rank 1).
    *   **`Round_2_Worker_2`** (`beam_r2_w2`): Succeeded. Autotuned latency parsed: **`8.020 ms`** (Rank 3).
    *   **`Round_2_Worker_3`** (`beam_r2_w3`): Succeeded after self-healing import naming errors in tests. Evaluator fallback latency: **`10.288 ms`** (Pruned).
*   **Incumbent Preservation**: Round 2 Worker 0 and Worker 3 were pruned. The orchestrator preserved Rank 2 (`beam_r1_w2` @ 8.0 ms) and Rank 4 (`beam_r1_w0` @ 8.030 ms) from Round 1.

#### Round 2 Beam State:
```
==================================================
            ROUND 2 BEAM STATE
==================================================
  Rank 1: Latency=6.100 ms | Path=beam_search/output/12p_RMSNorm_aa_r3_b4_0625_0645/beam_r2_w1_1782371027/optimized_kernel.py
  Rank 2: Latency=8.000 ms | Path=beam_search/output/12p_RMSNorm_aa_r3_b4_0625_0645/beam_r1_w2_1782369908/optimized_kernel.py (Incumbent Preserved)
  Rank 3: Latency=8.020 ms | Path=beam_search/output/12p_RMSNorm_aa_r3_b4_0625_0645/beam_r2_w2_1782371027/optimized_kernel.py
  Rank 4: Latency=8.030 ms | Path=beam_search/output/12p_RMSNorm_aa_r3_b4_0625_0645/beam_r1_w0_1782369908/optimized_kernel.py (Incumbent Preserved)
==================================================
```

---

### Round 3
*   **Worker Launch Paths**:
    *   `Round_3_Worker_0` spawned from Round 2 Rank 1 baseline ($6.100\text{ ms}$).
    *   `Round_3_Worker_1` spawned from Round 2 Rank 2 baseline ($8.000\text{ ms}$).
    *   `Round_3_Worker_2` spawned from Round 2 Rank 3 baseline ($8.020\text{ ms}$).
    *   `Round_3_Worker_3` spawned from Round 2 Rank 4 baseline ($8.030\text{ ms}$).
*   **Worker Execution**:
    *   **`Round_3_Worker_0`** (`beam_r3_w0`): Succeeded. Autotuned latency parsed: **`6.010 ms`** (Rank 1).
    *   **`Round_3_Worker_1`** (`beam_r3_w1`): Succeeded. Evaluator fallback latency: **`9.774 ms`** (Pruned).
    *   **`Round_3_Worker_2`** (`beam_r3_w2`): Succeeded. Autotuned latency parsed: **`6.200 ms`** (Rank 4).
    *   **`Round_3_Worker_3`** (`beam_r3_w3`): Succeeded. Autotuned latency parsed: **`6.075 ms`** (Rank 2).
*   **Incumbent Preservation**: Round 3 Worker 1 was pruned. The orchestrator preserved the best candidate from Round 2 (`beam_r2_w1` @ 6.100 ms) as Rank 3, which beat `beam_r3_w2` (6.200 ms).

#### Round 3 Beam State (Final State):
```
==================================================
            ROUND 3 BEAM STATE
==================================================
  Rank 1: Latency=6.010 ms | Path=beam_search/output/12p_RMSNorm_aa_r3_b4_0625_0645/beam_r3_w0_1782373360/optimized_kernel.py
  Rank 2: Latency=6.075 ms | Path=beam_search/output/12p_RMSNorm_aa_r3_b4_0625_0645/beam_r3_w3_1782373360/optimized_kernel.py
  Rank 3: Latency=6.100 ms | Path=beam_search/output/12p_RMSNorm_aa_r3_b4_0625_0645/beam_r2_w1_1782371027/optimized_kernel.py (Incumbent Preserved)
  Rank 4: Latency=6.200 ms | Path=beam_search/output/12p_RMSNorm_aa_r3_b4_0625_0645/beam_r3_w2_1782373360/optimized_kernel.py
==================================================
```

---

## 3. Final Search Summary & Performance

*   **Best Kernel Candidate**: Produced by `Round_3_Worker_0` at:
    [optimized_kernel.py](file:///usr/local/google/home/ligh/github/accelerator-agents/MaxKernel/beam_search/output/12p_RMSNorm_aa_r3_b4_0625_0645/beam_r3_w0_1782373360/optimized_kernel.py)
*   **Baseline Latency**: `11.400 ms`
*   **Best Optimized Latency**: **`6.010 ms`**
*   **Speedup**: **`1.90x` speedup** (a **`47.3%` latency reduction**).

---

## 4. Evaluator Bypassing and Orchestrator Mechanics Verified

1.  **Direct Metrics Extraction (Bypass)**: Successfully verified the core change: the orchestrator now parses the worker's `autotune_results.json` directly to retrieve `best_time_ms` instead of re-running the mock compiler profile process. This prevents the timing degradation that occurred when running sequential profiling steps.
2.  **Robust Fallback Mechanics**: In cases where `autotune_results.json` was missing (e.g., `beam_r1_w1` and `beam_r3_w1` where autotuning was skipped), the orchestrator successfully fell back to evaluating the candidate using `FakeKernelEvaluator`.
3.  **Cross-Round Preservations**: Verified that the incumbent preservation rules dynamically handle a mix of autotuned and evaluated baseline candidates. The preservation of `beam_r2_w1` (6.10 ms) in Rank 3 of Round 3 demonstrates correct comparative logic across candidate types.
