<!-- disableFinding(LINK_RELATIVE_G3DOC) -->
<!-- disableFinding(LINE_OVER_80) -->
<!-- disableFinding(LIST_NO_LINE) -->
<!-- disableFinding(SNIPPET_EM_DASH) -->
<!-- disableFinding(WHITESPACE_LINES) -->
<!-- disableFinding(SPACES) -->
<!-- disableFinding(HTML_OPEN) -->
<!-- disableFinding(HTML_BROKEN) -->

# JAX Tracing Model

## Prerequisites

None — this is a starting concept for the compiler section.

## Leads to

- [HLO Intermediate Representation](07-hlo-intermediate-representation.md)

## Used by agents

- [AutoRefactor](../../agents/auto_refactor.md) — understands trace-time decisions that affect HLO structure
- [MaxKernel](../../agents/max_kernel.md) — writes Pallas kernels that integrate with JAX tracing

## What it is

JAX uses a tracing-based compilation model. When you call a `jax.jit`-decorated function, JAX doesn't execute Python eagerly. Instead it traces the function: it passes abstract "tracer" objects through the Python code, recording every JAX operation into a computation graph (a jaxpr). This jaxpr is then lowered to HLO and compiled by XLA.

Key implications: (1) Python control flow during tracing becomes fixed in the graph — `if` statements evaluated at trace time are baked in, (2) shapes must be known at trace time (static shapes), (3) the traced graph is the unit of compilation, so function boundaries in Python become potential optimization barriers in the compiler.

The tracing model means that MaxPerf optimizations target the HLO graph that results from tracing, not the Python source. Two semantically identical Python programs can produce very different HLO graphs depending on how they're traced.

## Why it matters for MaxPerf

When AutoRefactor identifies suboptimal HLO structure, the root cause sometimes traces back to how the Python code was traced. For example, a Python loop that gets unrolled during tracing produces a flat HLO graph with no loop structure, preventing XLA's loop optimizations. Understanding the tracing model helps diagnose whether a performance issue belongs at the JAX level (restructure the Python to trace differently) or the XLA level (improve compilation of the existing HLO).

## Worked example

A user writes attention with a Python for-loop over heads: `for h in range(num_heads): out[h] = attention(q[h], k[h], v[h])`. Under `jax.jit`, this unrolls into `num_heads` independent attention subgraphs in HLO. XLA may fail to fuse across these boundaries. The fix: rewrite as a single vectorized operation `jnp.einsum('bhsd,bhsd->bhs', q, k)` which traces to a single fused HLO op. The tracing model determines whether we get one efficient op or N separate ones.

## See also

- [HLO Intermediate Representation](07-hlo-intermediate-representation.md) — the output of tracing/lowering
- [XLA Compilation Passes](08-xla-compilation-passes.md) — what happens to the traced graph next
- [Numeric Equivalence](10-numeric-equivalence.md) — ensuring rewrites preserve traced semantics
