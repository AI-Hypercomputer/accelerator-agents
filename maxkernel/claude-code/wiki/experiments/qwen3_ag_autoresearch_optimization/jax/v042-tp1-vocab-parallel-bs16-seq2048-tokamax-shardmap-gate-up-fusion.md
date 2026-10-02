---
variant: 8B/v6e-8
hypothesis: "Fusing gate and up projections in the SwiGLU MLP reduces kernel launch overhead and improves compute density"
commit: <to-be-filled>
status: in_progress
verdict: pending
---

# v042: Gate+Up FFN fusion

## Hypothesis under test

**Hypothesis**: By combining the `gate_proj` and `up_proj` linear layers into a single `gate_up_proj` (returning `2 * intermediate_size`), we can reduce the number of matmul operations from 3 to 2 in the MLP block.
**Mechanism**: Structural rewrite replacing `self.gate_proj` and `self.up_proj` with `self.gate_up_proj` and splitting the output along the feature dimension before the SiLU activation.
**Predicted signal**: MXU utilization rises due to larger matrix multiplication dimensions, and overall MFU increases.

## Implementation

1. Cloned from `v041`.
2. Edited `modeling_qwen3.py` to replace `gate_proj` and `up_proj` with `gate_up_proj`.
3. Updated `sharding.py` to map `gate_up_proj.weight` to `("tp", "fsdp")`.
4. Run at `bs=16`, `seqlen=2048` with splash size `2048`.

## Results

(Pending)

## Verdict

(Pending)
