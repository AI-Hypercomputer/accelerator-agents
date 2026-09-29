---
status: filed
verdict: refuted
model: qwen3
lane: jax
type: experiment
---
# Experiment: v013-fused-glu

**Hypothesis**: By replacing the standard SwiGLU MLP implementation with a fused Pallas GLU kernel, we can reduce HBM bandwidth pressure in the MLP blocks, increasing overall MFU beyond the v009 baseline of 32.5%.

**Mechanism**: We implemented a Pallas `fused_glu` kernel for TPU that loads `X`, `W_up`, and `W_gate` tiles into SRAM, computes the two dot products, applies `silu`, and writes out the final result. The backward pass was written using standard JAX ops since writing a full Pallas backward kernel for GLU is complex and XLA fuses the standard backward pass reasonably well.

**Implementation**:
- Added `model/fused_glu.py` containing the `fused_glu` Pallas kernel with `jax.custom_vjp`.
- Modified `Qwen3MLP` in `model/modeling_qwen3.py` to call `fused_glu` when `config.fused_glu` is True.
- Base code is from `v009-scan-over-layers`.

**Command**:
```bash
JAX_ATTENTION_IMPL=splash XLA_FLAGS="--xla_dump_to=gs://tpu-pytorch-alekseyv-us-central2/autoresearch/qwen3_ag/qwen3-ag-jax-v013-fused-glu/hlo --xla_dump_hlo_as_text" python -m train --train_steps 20 --batch_size 1 --seqlen 8192 --profile_dir gs://tpu-pytorch-alekseyv-us-central2/autoresearch/qwen3_ag/qwen3-ag-jax-v013-fused-glu/plugins/profile --profile_start_step 10 --profile_steps 3
```

**Results**:
- Ran 20 steps successfully.
- `approx MFU`: 30.6% (baseline was 32.5%)
- `avg throughput`: 42625 tok/s (5328/chip)
- The Fused GLU kernel caused an MFU regression relative to the v009 baseline.

## Verdict
**Refuted.** The MFU regression is caused by redundant recomputation and excessive HBM traffic introduced by the custom VJP backward pass:

1. **Redundant Matmuls (3x instead of 2x)**: Because `fused_glu` is wrapped in a `jax.custom_vjp` that does not output the intermediate pre-activations (`u` and `gate`), the backward pass `fused_glu_bwd` explicitly recomputes them using `jnp.dot`. Furthermore, because the entire layer is wrapped in `jax.remat`, XLA re-evaluates the forward Pallas kernel (`shard_map.509`) during the backward pass to produce `out` for downstream gradients (e.g., `dw_down`). This results in the heavy `up` and `gate` projections being computed **three times** per step (Forward Pallas, Remat Pallas, Backward JAX), whereas the baseline computes them only twice (Forward, Remat).
2. **HBM Materialization in Backward Pass**: The backward pass uses standard `jnp.dot`, which forces the allocation and materialization of the massive `8192 x 12288` tensors (`u`, `gate`, `du`, `dgate`) into HBM. This completely negates any memory bandwidth savings achieved by fusing them in the forward pass.
3. **Performance**: The Pallas kernel itself (`shard_map.510`) takes ~2.93ms per call (~70 TFLOPs/s per chip), which is heavily bottlenecked by HBM memory transfers of the inputs. Standard XLA `dot_general` is highly optimized and may pipeline TMA loads better than this custom loop over `K`.

**Next Steps**: To realize MFU gains from a fused GLU, a custom Pallas *backward* kernel is strictly required to compute `dx`, `dw_up`, and `dw_gate` on the fly without materializing `u` and `gate` into HBM, or we should rely on standard XLA fusions for elementwise epilogues.
