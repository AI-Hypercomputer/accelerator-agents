<!-- disableFinding(LINK_RELATIVE_G3DOC) -->
<!-- disableFinding(LINE_OVER_80) -->
<!-- disableFinding(LIST_NO_LINE) -->
<!-- disableFinding(SNIPPET_EM_DASH) -->
<!-- disableFinding(WHITESPACE_LINES) -->
<!-- disableFinding(SPACES) -->
<!-- disableFinding(HTML_OPEN) -->
<!-- disableFinding(HTML_BROKEN) -->

# Checkpoint: Section 3 — Distributed Systems

Test your understanding of concepts 11-13 with these scenario-based questions.

---

## Question 1: Collective Cost Analysis

A model uses tensor parallelism across 8 TPU chips. Each forward pass requires:
- 2× all-gather of weight shards: [8192, 4096] bf16 each (full weight is [8192, 32768])
- 1× reduce-scatter of activation gradient: [4096, 32768] bf16

Effective ICI bandwidth per collective: 400 GB/s.

**A)** What is the total collective payload per forward + backward?

**B)** What is the total collective time at 400 GB/s?

**C)** If the total step time is 12 ms, what fraction is spent in collectives?

---

## Question 2: Commutativity Application

The backward pass has this structure:
```
gradient [4096, 8192] → multiply by learning_rate (scalar) → all-reduce → update weights
```

**A)** Can the scalar multiplication be commuted past the all-reduce? Why?

**B)** Does this commutation reduce payload? If not, what benefit (if any) does it provide?

**C)** Now consider: `gradient [4096, 8192] → layer_norm → all-reduce`. Can layer_norm commute past the all-reduce? Why or why not?

---

## Question 3: Payload Reduction Strategy

A model sums gradients from 4 independent loss heads before all-reduce:
```
g1 [4096, 8192] + g2 [4096, 8192] + g3 [4096, 8192] + g4 [4096, 8192] → all-reduce [4096, 8192]
```

Alternative: all-reduce each gradient separately, then sum locally.

**A)** Which approach has less total collective traffic and by how much?

**B)** Under what circumstance would the alternative (4 separate all-reduces) be preferable despite more traffic?

**C)** What if g1-g4 are produced at different times during the backward pass — does that change the analysis?

---

## Question 4: Routing Decision

A diagnostic vector shows:
```
step_time_ms: 850
collective_time_ms: 310
hbm_bw_utilization: 0.45
ici_bw_utilization: 0.82
bottleneck_class: communication
```

**A)** Which agent(s) should the orchestrator route this to?

**B)** Which method (M1-M9) is most likely to help?

**C)** If the dominant collective is `all-reduce` of shape [8192, 32768] and it's immediately followed by a matmul with a [32768, 4096] weight, what specific optimization would you propose?

---

## Question 5: Collective Fusion

A model has 6 parameter groups, each with its own all-reduce:
- Group 1: 128 MB
- Group 2: 64 MB
- Group 3: 64 MB
- Group 4: 32 MB
- Group 5: 16 MB
- Group 6: 8 MB

Each all-reduce has 0.1 ms fixed synchronization overhead plus time proportional to payload.

**A)** What is the total synchronization overhead with 6 separate all-reduces?

**B)** If fused into one 312 MB all-reduce, what is the overhead?

**C)** What constraint prevents us from always fusing all collectives into one?

---

## Answers guidance

Verify your reasoning against:
- [11 - Collective Operations](../concepts/11-collective-operations.md)
- [12 - Commutativity & Graph Commutation](../concepts/12-commutativity-and-graph-commutation.md)
- [13 - Payload Minimization](../concepts/13-payload-minimization.md)
