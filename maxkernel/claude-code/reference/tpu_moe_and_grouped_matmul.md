---
name: tpu-moe-and-grouped-matmul
description: >-
  Reference for Mixture-of-Experts layers, grouped matmul and data-dependent routing on TPU.
  Covers static-shape routing (sort-by-expert vs capacity), the correct ragged_dot baseline,
  the megablox Pallas gmm/tgmm kernels (scalar prefetch, group metadata, boundary masking),
  what to fuse, backward pass, roofline formulas and pitfalls.
---

# MoE and Grouped Matmul on TPU

An MoE FFN is: router linear -> softmax / top-k -> gather tokens by expert -> per-expert up-proj (often fused SwiGLU) -> per-expert down-proj -> router-weighted scatter-sum back to tokens. On TPU the core primitive is a **grouped matmul** (gmm): one GEMM whose rows are split into contiguous, variable-size groups, each multiplied by a different weight matrix.

**Never** compute every expert for every token and then mask. That does E/K times the needed FLOPs, and if it ends up in `base.py` every reported speedup is inflated by roughly that factor.

Shorthand used below: T tokens, H hidden, I expert intermediate size, E experts, K top-k, M = T*K routed rows, M_e = M/E average rows per expert.

Everything here was checked against jax 0.11.0 (`jax/experimental/pallas/ops/tpu/megablox/` in site-packages).

---

## 1. Static Shapes vs Data-Dependent Routing

TPU programs are compiled for static shapes, but how many tokens go to each expert depends on the data. There are two static-shape formulations:

*   **Sort-by-expert (dropless, preferred)**: flatten the `[T, K]` expert ids to `[M]`, `argsort` them so rows are expert-contiguous, and get `group_sizes = jnp.bincount(flat_e, length=E)`. The total is always exactly M, so every array has a static shape. Only `group_sizes` (a small int32 `[E]`) is dynamic. This matches PyTorch/CUDA reference semantics exactly.
*   **Capacity-factor dispatch**: each expert gets a fixed `C = ceil(cf * M / E)` slots, giving dense `[E, C, H]` buffers. Tokens beyond capacity are **dropped**, and that **changes the output**. Only use this if the source model does it too (it has a `capacity_factor` and drops tokens). Otherwise it fails correctness, or passes only because the test inputs happen to be balanced.
*   `bincount` needs `length=E`. Without it the output shape depends on the data and will not trace under `jit`.

---

## 2. The Correct Baseline (`base.py`)

Idiomatic JAX uses argsort + gather + `jax.lax.ragged_dot` + scatter-add. This sketch was verified on CPU against a per-token NumPy loop (T=37, H=16, I=24, E=8, K=2, float32): max abs error 2.4e-7. `jax.grad` through it also works.

```python
def moe_ffn(x, w_router, w_gate, w_up, w_down, top_k):
    # x:[T,H]  w_router:[H,E]  w_gate,w_up:[E,H,I]  w_down:[E,I,H]
    T, E = x.shape[0], w_router.shape[1]
    probs = jax.nn.softmax((x @ w_router).astype(jnp.float32), axis=-1)
    topk_w, topk_e = jax.lax.top_k(probs, top_k)             # [T,K], [T,K]
    flat_e = topk_e.reshape(-1)                               # [M]
    order = jnp.argsort(flat_e, stable=True)                  # expert-contiguous
    tok = order // top_k                                      # source token per sorted row
    group_sizes = jnp.bincount(flat_e, length=E).astype(jnp.int32)
    xs = x[tok]                                               # [M,H] gathered copy
    g = jax.lax.ragged_dot(xs, w_gate, group_sizes, preferred_element_type=jnp.float32)
    u = jax.lax.ragged_dot(xs, w_up, group_sizes, preferred_element_type=jnp.float32)
    h = (jax.nn.silu(g) * u).astype(x.dtype)                  # [M,I]
    y = jax.lax.ragged_dot(h, w_down, group_sizes, preferred_element_type=jnp.float32)
    y = y * topk_w.reshape(-1)[order][:, None]                # router weight per row
    out = jax.ops.segment_sum(y, tok, num_segments=T)         # or zeros.at[tok].add(y)
    return out.astype(x.dtype)
```

*   `jax.lax.ragged_dot(lhs, rhs, group_sizes, precision=None, preferred_element_type=None, group_offset=None, out_sharding=None)` takes lhs `[m,k]`, rhs `[g,k,n]` and int32 `group_sizes` `[g]`. Rows past `sum(group_sizes)` come out as 0 (checked on CPU). `lax.ragged_dot_general` with `lax.RaggedDotDimensionNumbers` covers other layouts.
*   Match the source's renormalization: some models divide `topk_w` by its row sum, some do top-k on logits before softmax. Port whatever the source does.
*   **Precision trap when checking f32**: on TPU the default matmul precision for f32 is reduced. The same sketch on a TPU v6e gave a 5.8e-3 max error against the float32 NumPy loop, but 3.6e-7 under `with jax.default_matmul_precision("highest"):`. Compare on CPU or with highest precision before blaming the routing.
*   On TPU v6e, XLA lowers `ragged_dot` to a fused convolution+select with an iteration skipper over group starts. It does **not** run the dense FLOPs, but `compiled.cost_analysis()` still reports `2*E*m*k*n` FLOPs. Don't feed that number into `pl.CostEstimate` or roofline math.

---

## 3. Pallas Grouped Matmul: megablox

### A. What is installed
*   `jax.experimental.pallas.ops.tpu.megablox` exports only `gmm`, which is the `jax.custom_vjp`-wrapped version from `megablox/ops.py`.
*   The backend module `megablox/gmm.py` defines `gmm`, `tgmm`, `make_group_metadata`. Import it with `importlib.import_module("jax.experimental.pallas.ops.tpu.megablox.gmm")`: the package attribute `megablox.gmm` is the function, not the module.
*   Signature: `gmm(lhs[m,k], rhs[num_groups,k,n], group_sizes[int32], preferred_element_type=jnp.float32, tiling=(128,128,128), group_offset=None, existing_out=None, transpose_rhs=False, interpret=False) -> [m,n]`. `tiling` is `(tm, tk, tn)` or a callable `(m,k,n) -> tuple | None` (a lookup table).
*   Constraints from the source: lhs and rhs must be bf16 or f32, `group_sizes` must be int32, and **`m % tm == 0`** (otherwise it raises a ValueError). k and n may be ragged: the k remainder is masked in-kernel.
*   `interpret=True` runs on CPU (gmm and tgmm both matched loop references). Use it to unit-test custom variants.

### B. How it works
1.  **Metadata (computed in XLA, outside the kernel)**: `make_group_metadata` returns `(group_offsets[E+1], group_ids[tiles_m+E-1], m_tile_ids[tiles_m+E-1])` and `num_tiles`. Grid step i works on m-tile `m_tile_ids[i]` with expert `group_ids[i]`. A tile that straddles a group boundary is visited **once per group it touches**, and those visits are consecutive, so the output block stays resident. Example: `group_sizes=[100,0,60,96]`, m=256, tm=128 gives `group_ids=[0,2,2,3]` and `m_tile_ids=[0,0,1,1]`. That is 4 tile visits where a dense GEMM would need 2. Empty groups are skipped in gmm (`visit_empty_groups=False`) and visited in tgmm, which still has to zero their dW.
2.  **Scalar prefetch**: `pltpu.PrefetchScalarGridSpec(num_scalar_prefetch=2, grid=(tiles_n, num_tiles, tiles_k), in_specs=..., out_specs=..., scratch_shapes=[pltpu.VMEM((tm, tn), jnp.float32)])`. The metadata tuple and `group_offset` are placed in SMEM and passed to every index_map. The lhs index_map returns `(m_tile_ids[i], k)`. The rhs index_map returns `(group_ids[i] - group_offset, k, n)` with block `(None, tk, tn)`, so only the active expert's weight tile is DMA'd.
3.  **Dynamic grid**: `num_tiles` is a traced scalar used as a grid dimension. Pallas TPU accepts this, so no grid steps are spent on padding tiles.
4.  **Boundary masking**: accumulate in the f32 VMEM scratch over k. On the last k step, `_get_store_mask` builds a row iota and selects `acc` only for rows in `[group_start, group_end)`. Other rows keep the current `out` contents. MXU work on the masked rows is wasted.
5.  `dimension_semantics=("parallel", "arbitrary", "arbitrary")`: only n is parallel, because consecutive m-visits of the same tile must stay sequential.
6.  Rows `>= sum(group_sizes)` are never stored by the source. Treat them as undefined and slice them off.

### C. Use megablox gmm directly vs write a custom kernel
*   **Direct**: a quick, correct Pallas step for the plain GEMMs. Tune `tiling`: it is very sensitive. On v6e, bf16 m=4096 k=1024 n=2048 E=8 took 2168 us at (128,128,128), 108 us at (512,1024,1024) and 99.5 us at (512,1024,2048). `ragged_dot` took **63 us** on the same shape. Plain megablox does not automatically beat the XLA baseline, so the win has to come from fusion.
*   **Custom kernel** (copy the metadata and index_map pattern) when fusing any of:
    *   **Token gather into the up-proj**: avoid materializing `x[tok]` (`M x H`). Scalar-prefetch `tok`, keep `x` in HBM (`memory_space=pl.ANY`), and DMA the tm rows of each tile with `pltpu.make_async_copy(src_ref, dst_ref, sem)` into a VMEM buffer, double-buffered. This saves `2*M*H*bytes` of HBM traffic, which matters mainly in the memory-bound regime.
    *   **SwiGLU epilogue**: compute gate and up tiles for the same n-range (two rhs specs, or one rhs with a `2*tn` block if the layout allows) and write `silu(g)*u` in bf16. The `M x 2I` f32 intermediate never touches HBM.
    *   **Router weighting in the down-proj epilogue**: scalar-prefetch or block-load the per-row weight and multiply before the store.
    *   **Scatter-sum**: hard to fuse, because the output rows of a tile go to arbitrary tokens. With K small, a separate `segment_sum` over `[M,H]` is usually fine. Alternatively write `[M,H]` in sorted order and do the un-permute plus sum over K in XLA: `out = y[jnp.argsort(order)].reshape(T, K, H).sum(1)`.
*   The up+down FFN can be fused into one kernel per m-tile (the h tile stays in VMEM) only if `tm x I` fits in VMEM. Otherwise keep two gmm calls.

---

## 4. Backward Pass

`megablox/ops.py` defines the VJP. With `Y = gmm(X, W)` (X sorted by expert):
*   **dX** = `gmm(dY, W, group_sizes, transpose_rhs=not transpose_rhs)`. This is the same kernel reading W transposed in its BlockSpec, with no materialized transpose.
*   **dW** = `tgmm(X.swapaxes(0,1), dY, group_sizes, ..., num_actual_groups=E)` -> `[E, k, n]`, where `dW[e] = X_e^T @ dY_e`. tgmm iterates m-tiles innermost (grid `(tiles_n, tiles_k, num_tiles)`), accumulates while `group_ids` stays constant, and stores when the group changes. Straddling tiles are masked on **both** operands before the dot. tgmm lives only in `megablox/gmm.py` and is not re-exported. Verified: `jax.grad` of megablox gmm matches `jax.grad` of `ragged_dot` exactly on v6e (bf16).
*   The rest of the chain: the transpose of the gather `x[tok]` is a scatter-add into T rows. The transpose of `segment_sum` is a gather. `argsort`, `bincount` and top-k indices are integer outputs with no gradient. The gradient reaches the router only through `topk_w`: `top_k` values backprop to the selected positions (verified: grad is 1 at the chosen indices, 0 elsewhere), then through the softmax to `w_router`.
*   The activation backward needs g and u (or recomputation). Save `g, u` in bf16, or recompute them from `xs` to trade FLOPs for memory.

---

## 5. Roofline

Let b be the bytes per element (2 for bf16), GLU = gate+up.
*   **FLOPs**: router `2*T*H*E`. GLU FFN `2*M*H*(2I) + 2*M*I*H = 6*M*H*I`. Non-GLU FFN `4*M*H*I`. Backward is about 2x forward.
*   **Weight bytes**: GLU `3*E*H*I*b`, non-GLU `2*E*H*I*b`. All experts are read if `M >> E`. The expected number touched is `E*(1-(1-1/E)^M)`.
*   **Activation bytes (unfused)**: read x `T*H*b`. Gathered copy written and read `2*M*H*b`. GLU intermediate written and read `2*M*2I*4` (f32) or `2*M*I*b` after a fused activation. Down-proj out `M*H*4` written and read for the scatter. Output `T*H*b`.
*   **Key ratio**: per expert, the weight tile is reused across its M_e rows, so with weights dominating bytes the **arithmetic intensity is about M_e = T*K/E FLOPs/byte** (bf16). Compare it with the ridge in `tpu_memory_overlapping.md` (v5e ~240, v6e ~560, v7x ~312). Example: T=4096, K=8, E=128 gives M_e=256, which is memory-bound on v6e. Decode and small-batch MoE are always weight-bandwidth-bound: optimize weight streaming (big tk/tn, deep buffering), not MXU utilization.
*   **Padding waste**: each group costs `ceil`-to-tm tiles plus up to one straddle visit. Effective rows are at most `M + (E-1)*tm` (gmm) and `M + E*tm` (tgmm, empty groups visited). When M_e is about tm or smaller, this can double the FLOPs. Smaller tm cuts waste but lowers MXU efficiency, so the planner must pick tm against M_e.

---

## 6. Pitfalls

*   **Padding M to a multiple of tm**: megablox requires `m % tm == 0`, so pad the sorted rows to `round_up(M, tm)` without changing `group_sizes` (the trailing rows belong to no group) and drop them afterwards. Padding each group to tm in the *baseline* changes the FLOP count and the semantics. Don't.
*   **argsort cost**: sorting `M` int32 keys is not free on TPU. For large M, profile it. Alternatives: one stable `argsort` on a combined key, or compute positions as `cumsum` of one-hot counts (`[M,E]`, only acceptable for small E). Reuse `order` and `group_sizes` across up-proj, down-proj and backward. Compute them once.
*   **Top-k**: `jax.lax.top_k(operand, k, *, axis=-1, is_stable=True)` returns `(values, indices)`. `jax.lax.approx_max_k(operand, k, ..., recall_target=0.95)` is faster on TPU but approximate, so it changes routing. Don't use it unless the source does. Tie-breaking order affects which expert is picked. `is_stable=True` picks the lower index.
*   **Gate/up weight layout for SwiGLU**: check whether the source stores `[gate | up]` concatenated along the output dim (split at I) or **interleaved** (gate in even columns, up in odd, `w[..., 0::2]` / `w[..., 1::2]`), and which half is the gate. A wrong split still runs and gives plausible-looking wrong numbers. Pre-permute weights outside the kernel to the layout the epilogue wants, so each n-tile holds matching gate and up columns.
*   **Precision**: bf16 inputs, f32 accumulation (`preferred_element_type=jnp.float32` in `ragged_dot` / `lax.dot_general`; megablox already accumulates in f32 scratch). Do softmax and top-k in f32. Cast h back to bf16 before the down-proj only if the source does.
*   **Integer side outputs and aux loss**: if the source returns expert counts (`group_sizes` / tokens per expert) or a load-balancing loss (for example `E * sum(f_e * P_e)` with f_e = fraction of tokens routed to e and P_e = mean router prob), those are part of the output contract. Compute them from the same `topk_e` / `probs` and test them too (exact integer match for counts).
*   **Benchmark inputs**: random-normal routers give nearly uniform routing. Also test a skewed distribution (one hot expert, some empty experts): that exercises empty groups, straddles and worst-case padding.
