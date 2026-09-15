<!-- disableFinding(LINK_RELATIVE_G3DOC) -->
<!-- disableFinding(LINE_OVER_80) -->
<!-- disableFinding(LIST_NO_LINE) -->
<!-- disableFinding(SNIPPET_EM_DASH) -->
<!-- disableFinding(WHITESPACE_LINES) -->
<!-- disableFinding(SPACES) -->
<!-- disableFinding(HTML_OPEN) -->
<!-- disableFinding(HTML_BROKEN) -->

# MaxPerf for Gemini — Qwen3-Coder-480B on TPU v6e-16

MaxPerf is a multi-agent optimization system for pushing inference throughput
of `Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8` on TPU v6e beyond what
configuration tuning (Phase 0) achieves. It operates at the JAX graph, HLO,
Pallas kernel, and ISA-modeling layers.

This directory contains the operating contract and sub-agent prompts formatted
for Gemini. A single Gemini instance plays all roles by loading the appropriate
agent prompt from `agents/` when the orchestrator routes a hypothesis.

## Prerequisite

Phase 0 (vLLM flag-level tuning) must complete first. MaxPerf inherits the
Phase 0 best configuration as its baseline. Do not start MaxPerf experiments
until Phase 0 has settled and the baseline section of `program.md` is filled in.

## How to use

1. Populate `raw/` prerequisites (see Prerequisites checklist in `program.md`):
   ISA docs, latency tables, numeric-equiv corpus, Phase 0 config.
2. Provide `program.md` as system context to Gemini.
3. On each loop iteration, load the relevant agent prompt from `agents/` as
   role context (the orchestrator prompt tells you which one).
4. Follow the 12-step loop protocol defined in `program.md`.
5. Record results in `RESULTS.tsv` and experiment artifacts in `experiments/`.

## Directory structure

```
max_perf_g/
├── program.md                  # Operating contract
├── README.md                   # This file
├── RESULTS.tsv                 # Cumulative experiment ledger
├── agents/                     # Sub-agent system prompts
├── experiments/                # Per-experiment artifact directories
├── runs/                       # Phase 0 best config scripts
└── raw/                        # External artifacts (gitignored)
    ├── profiles/               # xprof traces
    ├── hlo/                    # HLO dumps (pre/post optimization)
    ├── xla_logs/               # XLA compiler logs
    ├── isa/                    # TPU ISA docs + latency tables
    └── numeric_ref/            # 100-prompt corpus + reference outputs
```

## Agents

| File | Role |
|------|------|
| `agents/maxperf_orchestrator.md` | Orchestrator: routes hypotheses, manages the loop |
| `agents/tpu_diagnose.md` | Produces diagnostic vectors from xprof traces |
| `agents/deep_research.md` | Symbolic reasoning: commutativity proofs, ISA models |
| `agents/max_shard.md` | Implements graph-rewrite hypotheses |
| `agents/max_kernel.md` | Writes Pallas custom kernels |
| `agents/max_tile.md` | Tile-size autotuning via algebraic derivation |
| `agents/auto_refactor.md` | HLO-barrier-removal code refactors |
| `agents/execution_config_agent.md` | Ingests configuration parameters and outputs optimization bounds |
| `agents/max_inference.md` | Sampling Agent: PD disaggregation, multi-host KV cache, and sharding |
| `agents/max_sync.md` | Transport Agent: Asynchronous weight synchronization and Raiden integration |
| `agents/max_flow.md` | Pipeline & Sandbox Agent: Zero-padding and local expert dispatching |
| `agents/max_align.md` | Convergence Agent: Numeric alignment, JAX-to-PyTorch FP8 loss convergence |

## Status

Template ready. Awaiting Phase 0 baseline before any experiments can run.
