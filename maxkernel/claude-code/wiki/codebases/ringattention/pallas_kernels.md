---
title: "ringattention — Pallas kernels"
type: concept
tags: [pallas, kernels, tpu, ringattention]
created: 2026-07-04
updated: 2026-07-04
---

The `ringattention` repo (Liu/Zaharia/Abbeel, [arXiv:2310.01889](https://arxiv.org/abs/2310.01889))
defines **one novel TPU Pallas op — ring flash attention — realized as 3 `mosaic_tpu` `pl.pallas_call`
sites**: one forward kernel and two backward kernels (`dK,dV` and `dQ` are separate because they
contract over different grid axes). All three live in the single file
`ringattention/ringattention_pallas_tpu.py`; the surrounding `lax.scan` + `lax.ppermute` ring loop
and `jax.custom_vjp` wiring are pure JAX. This is the **canonical from-scratch ring-attention Pallas
kernel** (not a Splash wrapper), novel to this repo. Straight unidirectional ppermute ring with
`below_or_on_diag` causal skip — **no zig-zag / striped load-balancing variant**.

## Kernels

All backend `mosaic_tpu` (`jax.experimental.pallas.tpu`, `dimension_semantics=("parallel","parallel","parallel","arbitrary")`), stability research (paper companion). All novel-to-this-repo.

| Kernel | Backend | What it does | Source | Stability | Novel/Vendored |
|--------|---------|--------------|--------|-----------|----------------|
| `_flash_attention_kernel` (fwd, dispatches `_flash_attention_kernel_single_batch`) | mosaic_tpu | Forward online-softmax core for one ring step: `s=q·kᵀ`, running-max/sum update, `p·v` accumulation into VMEM scratch (`m_scratch`/`l_scratch`/`acc_scratch`), causal block skip via `below_or_on_diag` (DMA elided by redirecting `kv_index_map` to block 0, compute elided by `pl.when`). Two-level blocking `block_k_major`==`block_k` at fwd. Runs once per ring `lax.scan` step. | [ringattention_pallas_tpu.py#L434](../../../raw/code/ringattention/ringattention/ringattention_pallas_tpu.py#L434) (call [#L824](../../../raw/code/ringattention/ringattention/ringattention_pallas_tpu.py#L824)) / [catalog](catalog/ringattention/ringattention_pallas_tpu.md#_flash_attention_kernel_single_batch) | research | novel |
| `_flash_attention_dkv_kernel` (bwd dK,dV) | mosaic_tpu | Backward for `dK`/`dV`: recomputes `p` from saved `l`,`m` (no stored attention matrix), grid ordered `kv` outer / `q` inner so `dk_scratch`/`dv_scratch` accumulate across the inner q loop (reset at `q_seq_index==0`). `block_k_major` may differ from `block_k` here. | [ringattention_pallas_tpu.py#L854](../../../raw/code/ringattention/ringattention/ringattention_pallas_tpu.py#L854) (call [#L1236](../../../raw/code/ringattention/ringattention/ringattention_pallas_tpu.py#L1236)) / [catalog](catalog/ringattention/ringattention_pallas_tpu.md#_flash_attention_bwd_dkv) | research | novel |
| `_flash_attention_dq_kernel` (bwd dQ) | mosaic_tpu | Mirror-image backward for `dQ`: grid `q` outer / `kv` inner, `dq_scratch` accumulates over kv and flushes at the last kv step. Separate kernel from dK,dV so each accumulator sits on the correct grid axis; higher-order AD is refused (`NotImplementedError`). | [ringattention_pallas_tpu.py#L1265](../../../raw/code/ringattention/ringattention/ringattention_pallas_tpu.py#L1265) (call [#L1617](../../../raw/code/ringattention/ringattention/ringattention_pallas_tpu.py#L1617)) / [catalog](catalog/ringattention/ringattention_pallas_tpu.md#_flash_attention_bwd_dq) | research | novel |

Public differentiable entry point wrapping all three: [`ring_flash_attention_tpu`](catalog/ringattention/ringattention_pallas_tpu.md#ring_flash_attention_tpu) — `jax.custom_vjp`, `nondiff_argnums=[6,7,8]`, called inside a `shard_map` over the sequence-sharded axis ([source L190](../../../raw/code/ringattention/ringattention/ringattention_pallas_tpu.py#L190)).

## Notes

- **Reconciliation with the 2026-04-23 directory.** §6.4 of the [community directory](../../analyses/pallas-kernel-directory/06-community-research.md) lists a single row `ring_flash_attention_tpu` and flags it "canonical public Pallas Ring Attention, mosaic_tpu, no zig-zag variant." Reading the source confirms that verdict and refines the count: the one public op is **three `pallas_call` kernels** (1 fwd + 2 bwd) plus the JAX ring/`custom_vjp` scaffolding — the directory's one-line entry is correct but coarse. Source has **not** materially diverged (still one file, still unidirectional ring, still no striped/zig-zag load-balancing per Brandon et al.).
- **Novel, not vendored.** These are the reference paper's own hand-written Pallas kernels — the from-scratch flash-attention math, not a wrapper over jax/tokamax Splash. Downstream forks (LargeWorldModel family, and EasyDeL/ejkernel's *separate* Splash-wrapped ring) vendor or re-implement this idea; this file is the origin.
- **`PatchBlockSpec` compat shim.** Every `BlockSpec` is built through `PatchBlockSpec` (subclasses `pl.BlockSpec` only to swap positional-arg order) — evidence the kernel targets an older Pallas `BlockSpec` signature; a reuse gotcha if upstreaming.
- **Autotune / block surface.** `BlockSizes` dataclass exposes `block_q`, `block_k`, `block_k_major`, `block_b` plus `*_dkv`/`*_dq` variants; `__post_init__` requires each "major" size divide its "minor". `MIN_BLOCK_SIZE = NUM_LANES = 128` (TPU VPU lane width) — scalar-per-row softmax stats are broadcast to 128 lanes before entering the kernel.
- **Hardware.** No in-source TPU-generation check; targets TPU generically via `pltpu`. README claims "tens of millions of tokens of context without adding any communication or computation overhead" — **no measured numbers**; the compute/`ppermute` overlap is design intent, not a profiled fact (no tests in-repo).
- **Full mechanism/design writeup** already exists as the wikify concept page below — reuse it rather than duplicating.

## See also
- [ringattention — overview](overview.md) — the repo's grounded wiki front door.
- [Ring Flash Attention — TPU Pallas kernel](concepts/ringattention-ringattention_pallas_tpu.md) — the deep wikify concept page: full mechanism, design rationale (two-level blocking, causal skip at index-map level, separate dK,dV vs dQ kernels, custom_vjp), and edge cases.
- [Ring Attention — pure-JAX reference](concepts/ringattention-ringattention_jax.md) — the non-Pallas twin (einsum under `lax.scan`/`jax.checkpoint`); shares `below_or_on_diag`.
- [Pallas kernel directory (catalog)](../../analyses/2026-04-23-pallas-kernel-directory.md) — the cross-repo Pallas catalog; this repo appears in [subpage 06 §6.4](../../analyses/pallas-kernel-directory/06-community-research.md).

## Sources
- [raw/code/ringattention/ringattention/ringattention_pallas_tpu.py](../../../raw/code/ringattention/ringattention/ringattention_pallas_tpu.py)
