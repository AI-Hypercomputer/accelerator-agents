<!-- disableFinding(LINK_RELATIVE_G3DOC) -->
<!-- disableFinding(LINE_OVER_80) -->
<!-- disableFinding(LIST_NO_LINE) -->
<!-- disableFinding(SNIPPET_EM_DASH) -->
<!-- disableFinding(WHITESPACE_LINES) -->
<!-- disableFinding(SPACES) -->
<!-- disableFinding(HTML_OPEN) -->
<!-- disableFinding(HTML_BROKEN) -->

# Checkpoint: Section 2 — Compiler Pipeline

Test your understanding of concepts 6-10 with these scenario-based questions.

---

## Question 1: Tracing Impact

A developer writes attention with a Python for-loop: `for h in range(32): scores[h] = q[h] @ k[h].T`. Under `jax.jit`, this traces to 32 separate dot operations in HLO.

**A)** Why does XLA struggle to fuse these 32 dots compared to a single batched einsum?

**B)** If each dot is [2048, 128] × [128, 2048] and produces a [2048, 2048] intermediate, what is the total HBM traffic if they remain unfused (assuming each writes and reads its output)?

**C)** What is the fix at the JAX level, and which concept page explains why?

---

## Question 2: HLO Reading

Given this HLO fragment:
```
%param.0 = bf16[4096,8192] parameter(0)
%param.1 = bf16[8192,4096] parameter(1)
%dot.1 = bf16[4096,4096] dot(%param.0, %param.1), lhs_cd={1}, rhs_cd={0}
%bias.1 = bf16[4096] parameter(2)
%broadcast.1 = bf16[4096,4096] broadcast(%bias.1), dimensions={1}
%add.1 = bf16[4096,4096] add(%dot.1, %broadcast.1)
```

**A)** What mathematical operation does this represent?

**B)** What is the arithmetic intensity of the dot operation alone (FLOPs / bytes loaded)?

**C)** Are the `broadcast` and `add` likely to be fused with the `dot` by XLA? Why or why not?

---

## Question 3: Pass Ordering Effects

AutoRefactor discovers that enabling `--xla_tpu_aggressive_fusion` before layout assignment produces a 15% speedup, but enabling it after layout assignment produces only 3%.

**A)** Why might pass ordering matter for fusion decisions?

**B)** What information does the layout assignment pass add that could constrain fusion?

**C)** Which method (M1-M9) does systematic pass flag exploration correspond to?

---

## Question 4: Fusion Boundaries

A model's HLO shows that XLA split a residual connection into two fusions:
- Fusion A: matmul + bias + activation (output: [4096, 8192] bf16)
- Fusion B: add(output_A, residual) + dropout

**A)** What is the HBM traffic cost of the boundary between Fusion A and Fusion B?

**B)** Why might XLA have split these (what heuristic concern)?

**C)** What are two approaches to eliminate this boundary (one via AutoRefactor, one via MaxKernel)?

---

## Question 5: Numeric Equivalence

An optimization proposes replacing `reduce_sum(x, axis=1)` on a [8192, 32768] fp32 tensor with a two-pass approach: chunk into 8 slices of 4096, reduce each, then sum the 8 results.

**A)** Why does this potentially change the numeric result?

**B)** For a training workload in bf16, would you accept or reject this if max absolute difference is 4.7e-4?

**C)** For a deterministic inference service requiring bitwise reproducibility, would you accept or reject?

---

## Answers guidance

Verify your reasoning against:
- [06 - JAX Tracing Model](../concepts/06-jax-tracing-model.md)
- [07 - HLO Intermediate Representation](../concepts/07-hlo-intermediate-representation.md)
- [08 - XLA Compilation Passes](../concepts/08-xla-compilation-passes.md)
- [09 - XLA Fusion & Custom-Call Boundaries](../concepts/09-xla-fusion-and-custom-call-boundaries.md)
- [10 - Numeric Equivalence](../concepts/10-numeric-equivalence.md)
