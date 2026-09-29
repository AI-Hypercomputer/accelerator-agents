---
title: 'Module: tpu_inference/kernels/gdn/v3/vmem_ldst.py'
type: catalog
provenance: extracted
module: tpu_inference/kernels/gdn/v3/vmem_ldst.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.kernels.gdn.v3.vmem_ldst`/load_
symbols:
  load_as_qkv_large: as_qkv_large().
  load_as_qkv_compact: as_qkv_compact().
  load_compact_to_large: compact_to_large().
  load_and_select_states: and_select_states().
  load_activation_as_compact: activation_as_compact().
  load_activation_as_large: activation_as_large().
---
# Module: [`tpu_inference/kernels/gdn/v3/vmem_ldst.py`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v3/vmem_ldst.py)

## Functions
- `load_activation_as_compact(qkv_vreg: jax.Array, qkv_vmem_ref: jax.Ref, b_vmem_ref: jax.Ref, a_vmem_ref: jax.Ref, cfgs: config.GDNConfig)` — [`L214`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v3/vmem_ldst.py#L214) — Load activations from VMEM as a compact layout.
- `load_activation_as_large(qkv_vreg: jax.Array, qkv_vmem_ref: jax.Ref, b_vmem_ref: jax.Ref, a_vmem_ref: jax.Ref, cfgs: config.GDNConfig)` — [`L230`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v3/vmem_ldst.py#L230) — Load activations from VMEM as a large layout.
- `load_and_select_states(metadata_ref: memory_ref.MetadataRef, p_id: jax.Array, conv_state_slot_ref: jax.Ref, recurrent_slot_ref: jax.Ref, carry_conv_scratch_ref: jax.Ref | None, carry_recurrent_scratch_ref: jax.Ref | None, cfg: config.GDNConfig)` — [`L137`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v3/vmem_ldst.py#L137) — Load correct states from HBM or prior tile, and masks invalid states.
- `load_as_qkv_compact(qkv_vmem_ref: jax.Ref, cfg: config.GDNConfig)` — [`L75`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v3/vmem_ldst.py#L75) — Split qkv and transpose by performing 1 load per head for compact layout.
- `load_as_qkv_large(qkv_vmem_ref: jax.Ref, cfgs: config.GDNConfig)` — [`L22`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v3/vmem_ldst.py#L22) — Split qkv and transpose by performing 1 load per chunk for large layout.
- `load_compact_to_large(vmem_ref: jax.Ref)` — [`L117`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v3/vmem_ldst.py#L117) — Use strided load to convert compact to large layout without transpose.

