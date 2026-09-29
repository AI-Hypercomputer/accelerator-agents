---
title: 'Module: tpu_inference/kernels/gdn/v2/recurrent_scan_impl.py'
type: catalog
provenance: extracted
module: tpu_inference/kernels/gdn/v2/recurrent_scan_impl.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.kernels.gdn.v2.recurrent_scan_impl`/
symbols:
  BranchRefs: BranchRefs#
  BranchRefs.qkv: BranchRefs#qkv.
  BranchRefs.a_raw: BranchRefs#a_raw.
  BranchRefs.b_raw: BranchRefs#b_raw.
  BranchRefs.output: BranchRefs#output.
  DMAHelper: DMAHelper#
  COL_PREFILL_VALID: COL_PREFILL_VALID.
  COL_PREFILL_OFFSET: COL_PREFILL_OFFSET.
  COL_DECODE_VALID: COL_DECODE_VALID.
  COL_DECODE_OFFSET: COL_DECODE_OFFSET.
  SharedRefs: SharedRefs#
  SharedRefs.a_log: SharedRefs#a_log.
  SharedRefs.dt_bias: SharedRefs#dt_bias.
  SharedRefs.recurrent_state_in: SharedRefs#recurrent_state_in.
  SharedRefs.recurrent_state_out: SharedRefs#recurrent_state_out.
  PrefillScratchRefs: PrefillScratchRefs#
  PrefillScratchRefs.scratch: PrefillScratchRefs#scratch.
  PrefillScratchRefs.semaphore: PrefillScratchRefs#semaphore.
  DecodeScratchRefs: DecodeScratchRefs#
  DecodeScratchRefs.state: DecodeScratchRefs#state.
  DecodeScratchRefs.load: DecodeScratchRefs#load.
  DecodeScratchRefs.store: DecodeScratchRefs#store.
  DecodeScratchRefs.output: DecodeScratchRefs#output.
  DecodeScratchRefs.read_semaphores: DecodeScratchRefs#read_semaphores.
  DecodeScratchRefs.write_semaphore: DecodeScratchRefs#write_semaphore.
  ModelDims: ModelDims#
  ModelDims.n_kq: ModelDims#n_kq.
  ModelDims.n_v: ModelDims#n_v.
  ModelDims.d_k: ModelDims#d_k.
  ModelDims.d_v: ModelDims#d_v.
  TilingConfig: TilingConfig#
  TilingConfig.C: TilingConfig#C.
  TilingConfig.BT: TilingConfig#BT.
  TilingConfig.sublanesize: TilingConfig#sublanesize.
  ScanConfig: ScanConfig#
  ScanConfig.model: ScanConfig#model.
  ScanConfig.tiling: ScanConfig#tiling.
  ScanConfig.use_qk_norm_in_gdn: ScanConfig#use_qk_norm_in_gdn.
  ScanConfig.decode_tokens: ScanConfig#decode_tokens.
  COL_IS_TRANSITION: COL_IS_TRANSITION.
  ScheduleStep: ScheduleStep#
  PrefillProcessor: PrefillProcessor#
  PrefillProcessor.process: PrefillProcessor#process().
  DecodeProcessor: DecodeProcessor#
  DecodeProcessor.process: DecodeProcessor#process().
  l2_normalize: l2_normalize().
  ModelDims.key_dim: ModelDims#key_dim().
  ModelDims.repeat_factor: ModelDims#repeat_factor().
  DMAHelper.__init__: DMAHelper#__init__().
  DMAHelper.build_copy_in: DMAHelper#build_copy_in().
  DMAHelper.commit_in: DMAHelper#commit_in().
  DMAHelper.copy_out: DMAHelper#copy_out().
  DMAHelper.wait_out: DMAHelper#wait_out().
  COL_PREFILL_REQ_ID: COL_PREFILL_REQ_ID.
  COL_PREFILL_COUNT: COL_PREFILL_COUNT.
  COL_DECODE_REQ_ID: COL_DECODE_REQ_ID.
  COL_DECODE_COUNT: COL_DECODE_COUNT.
  COL_IS_LAST_CHUNK: COL_IS_LAST_CHUNK.
  COL_IS_FIRST_CHUNK: COL_IS_FIRST_CHUNK.
  COL_SUBLANE_REQ_IDS: COL_SUBLANE_REQ_IDS.
  ScheduleStep.__init__: ScheduleStep#__init__().
  ScanProcessor: ScanProcessor#
  ScanProcessor.__init__: ScanProcessor#__init__().
  invert_triangular_matrix: invert_triangular_matrix().
  PrefillProcessor.__init__: PrefillProcessor#__init__().
  PrefillProcessor._process_regular_prefill: PrefillProcessor#_process_regular_prefill().
  PrefillProcessor._process_transition_prefill: PrefillProcessor#_process_transition_prefill().
  DecodeProcessor.__init__: DecodeProcessor#__init__().
  DecodeProcessor.get_target_idx: DecodeProcessor#get_target_idx().
---
# Module: [`tpu_inference/kernels/gdn/v2/recurrent_scan_impl.py`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v2/recurrent_scan_impl.py)

## Classes
### `BranchRefs`
- def: [`tpu_inference/kernels/gdn/v2/recurrent_scan_impl.py:38`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v2/recurrent_scan_impl.py#L38)
- doc: Inputs/Outputs specific to a single execution branch (prefill or decode).
- signature: `class BranchRefs:`
- members:
  - `a_raw` — [`L42`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v2/recurrent_scan_impl.py#L42)
  - `b_raw` — [`L43`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v2/recurrent_scan_impl.py#L43)
  - `output` — [`L44`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v2/recurrent_scan_impl.py#L44)
  - `qkv` — [`L41`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v2/recurrent_scan_impl.py#L41)
- used by: [`inner_kernel`](recurrent_scan_v2.md#inner_kernel)

### `DMAHelper`
- def: [`tpu_inference/kernels/gdn/v2/recurrent_scan_impl.py:123`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v2/recurrent_scan_impl.py#L123)
- doc: Manages asynchronous state copies and double-buffering semaphores.
- signature: `class DMAHelper:`
- members:
  - `build_copy_in(self, slot: int, state_idx: int)` — [`L134`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v2/recurrent_scan_impl.py#L134)
  - `commit_in(self, copy_op, slot: int, dst_ref, dst_slot: int)` — [`L142`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v2/recurrent_scan_impl.py#L142)
  - `copy_out(self, slot: int, state_idx: int, src_scratch)` — [`L148`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v2/recurrent_scan_impl.py#L148)
  - `wait_out(self, slot: int, state_idx: int)` — [`L160`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v2/recurrent_scan_impl.py#L160)
- protocol/private: `__init__`[`L126`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v2/recurrent_scan_impl.py#L126)
- used by: [`inner_kernel`](recurrent_scan_v2.md#inner_kernel)

### `DecodeProcessor`
- def: [`tpu_inference/kernels/gdn/v2/recurrent_scan_impl.py:663`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v2/recurrent_scan_impl.py#L663)
- doc: Handles batch decode step processing using double-buffering logic.
- signature: `class DecodeProcessor(ScanProcessor):`
- members:
  - `get_target_idx(self, b)` — [`L683`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v2/recurrent_scan_impl.py#L683)
  - `process(self)` — [`L688`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v2/recurrent_scan_impl.py#L688) — Processes decode steps in blocks.
- protocol/private: `__init__`[`L666`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v2/recurrent_scan_impl.py#L666)
- used by: [`inner_kernel`](recurrent_scan_v2.md#inner_kernel), [`decode_wrapper`](recurrent_scan_v2.md#inner_kernel.decode_wrapper)

### `DecodeScratchRefs`
- def: [`tpu_inference/kernels/gdn/v2/recurrent_scan_impl.py:69`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v2/recurrent_scan_impl.py#L69)
- doc: Scratch VMEM and semaphores allocated for decode.
- signature: `class DecodeScratchRefs:`
- members:
  - `load` — [`L73`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v2/recurrent_scan_impl.py#L73)
  - `output` — [`L75`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v2/recurrent_scan_impl.py#L75)
  - `read_semaphores` — [`L76`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v2/recurrent_scan_impl.py#L76)
  - `state` — [`L72`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v2/recurrent_scan_impl.py#L72)
  - `store` — [`L74`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v2/recurrent_scan_impl.py#L74)
  - `write_semaphore` — [`L77`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v2/recurrent_scan_impl.py#L77)
- used by: [`inner_kernel`](recurrent_scan_v2.md#inner_kernel)

### `ModelDims`
- def: [`tpu_inference/kernels/gdn/v2/recurrent_scan_impl.py:84`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v2/recurrent_scan_impl.py#L84)
- doc: Dimensions of the Recurrent Scan model configuration.
- signature: `class ModelDims:`
- members:
  - `key_dim(self)` — [`L93`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v2/recurrent_scan_impl.py#L93)
  - `repeat_factor(self)` — [`L97`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v2/recurrent_scan_impl.py#L97)
  - `d_k` — [`L89`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v2/recurrent_scan_impl.py#L89)
  - `d_v` — [`L90`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v2/recurrent_scan_impl.py#L90)
  - `n_kq` — [`L87`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v2/recurrent_scan_impl.py#L87)
  - `n_v` — [`L88`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v2/recurrent_scan_impl.py#L88)
- used by: [`inner_kernel`](recurrent_scan_v2.md#inner_kernel)

### `PrefillProcessor`
- def: [`tpu_inference/kernels/gdn/v2/recurrent_scan_impl.py:275`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v2/recurrent_scan_impl.py#L275)
- doc: Handles prefill step processing.
- signature: `class PrefillProcessor(ScanProcessor):`
- members:
  - `_process_regular_prefill(self)` — [`L304`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v2/recurrent_scan_impl.py#L304) — Processes a regular prefill step without transition boundary overlaps.
  - `_process_transition_prefill(self)` — [`L498`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v2/recurrent_scan_impl.py#L498) — Processes a transition prefill step with sublane stitching.
  - `process(self)` — [`L295`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v2/recurrent_scan_impl.py#L295)
- protocol/private: `__init__`[`L278`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v2/recurrent_scan_impl.py#L278)
- used by: [`inner_kernel`](recurrent_scan_v2.md#inner_kernel), [`process_prefill`](recurrent_scan_v2.md#inner_kernel.process_prefill)

### `PrefillScratchRefs`
- def: [`tpu_inference/kernels/gdn/v2/recurrent_scan_impl.py:60`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v2/recurrent_scan_impl.py#L60)
- doc: Scratch VMEM and semaphores allocated for prefill.
- signature: `class PrefillScratchRefs:`
- members:
  - `scratch` — [`L63`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v2/recurrent_scan_impl.py#L63)
  - `semaphore` — [`L64`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v2/recurrent_scan_impl.py#L64)
- used by: [`inner_kernel`](recurrent_scan_v2.md#inner_kernel)

### `ScanConfig`
- def: [`tpu_inference/kernels/gdn/v2/recurrent_scan_impl.py:111`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v2/recurrent_scan_impl.py#L111)
- doc: Configuration holding model dimensions and tiling options.
- signature: `class ScanConfig:`
- members:
  - `decode_tokens` — [`L117`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v2/recurrent_scan_impl.py#L117)
  - `model` — [`L114`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v2/recurrent_scan_impl.py#L114)
  - `tiling` — [`L115`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v2/recurrent_scan_impl.py#L115)
  - `use_qk_norm_in_gdn` — [`L116`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v2/recurrent_scan_impl.py#L116)
- used by: [`inner_kernel`](recurrent_scan_v2.md#inner_kernel)

### `ScanProcessor`
- def: [`tpu_inference/kernels/gdn/v2/recurrent_scan_impl.py:209`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v2/recurrent_scan_impl.py#L209)
- doc: Base class for executing step calculations.
- signature: `class ScanProcessor:`
- protocol/private: `__init__`[`L212`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v2/recurrent_scan_impl.py#L212)

### `ScheduleStep`
- def: [`tpu_inference/kernels/gdn/v2/recurrent_scan_impl.py:185`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v2/recurrent_scan_impl.py#L185)
- doc: Unpacks and holds the scheduling metadata for the current step.
- signature: `class ScheduleStep:`
- protocol/private: `__init__`[`L188`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v2/recurrent_scan_impl.py#L188)
- used by: [`inner_kernel`](recurrent_scan_v2.md#inner_kernel)

### `SharedRefs`
- def: [`tpu_inference/kernels/gdn/v2/recurrent_scan_impl.py:49`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v2/recurrent_scan_impl.py#L49)
- doc: Inputs/Outputs refs shared between both branches(prefill and decode).
- signature: `class SharedRefs:`
- members:
  - `a_log` — [`L52`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v2/recurrent_scan_impl.py#L52)
  - `dt_bias` — [`L53`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v2/recurrent_scan_impl.py#L53)
  - `recurrent_state_in` — [`L54`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v2/recurrent_scan_impl.py#L54)
  - `recurrent_state_out` — [`L55`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v2/recurrent_scan_impl.py#L55)
- used by: [`inner_kernel`](recurrent_scan_v2.md#inner_kernel)

### `TilingConfig`
- def: [`tpu_inference/kernels/gdn/v2/recurrent_scan_impl.py:102`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v2/recurrent_scan_impl.py#L102)
- doc: Tiling dimensions for memory copy blocks.
- signature: `class TilingConfig:`
- members:
  - `BT` — [`L106`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v2/recurrent_scan_impl.py#L106)
  - `C` — [`L105`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v2/recurrent_scan_impl.py#L105)
  - `sublanesize` — [`L107`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v2/recurrent_scan_impl.py#L107)
- used by: [`inner_kernel`](recurrent_scan_v2.md#inner_kernel)

## Functions
- `invert_triangular_matrix(A, block_size=16)` — [`L225`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v2/recurrent_scan_impl.py#L225) — Inverts a unit lower triangular matrix A block-wise.
- `l2_normalize(x, eps=1e-06)` — [`L27`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v2/recurrent_scan_impl.py#L27)

## Module values
- `COL_DECODE_COUNT` — [`L178`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v2/recurrent_scan_impl.py#L178)
- `COL_DECODE_OFFSET` — [`L176`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v2/recurrent_scan_impl.py#L176)
- `COL_DECODE_REQ_ID` — [`L177`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v2/recurrent_scan_impl.py#L177)
- `COL_DECODE_VALID` — [`L175`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v2/recurrent_scan_impl.py#L175)
- `COL_IS_FIRST_CHUNK` — [`L180`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v2/recurrent_scan_impl.py#L180)
- `COL_IS_LAST_CHUNK` — [`L179`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v2/recurrent_scan_impl.py#L179)
- `COL_IS_TRANSITION` — [`L181`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v2/recurrent_scan_impl.py#L181)
- `COL_PREFILL_COUNT` — [`L174`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v2/recurrent_scan_impl.py#L174)
- `COL_PREFILL_OFFSET` — [`L172`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v2/recurrent_scan_impl.py#L172)
- `COL_PREFILL_REQ_ID` — [`L173`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v2/recurrent_scan_impl.py#L173)
- `COL_PREFILL_VALID` — [`L171`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v2/recurrent_scan_impl.py#L171)
- `COL_SUBLANE_REQ_IDS` — [`L182`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v2/recurrent_scan_impl.py#L182)

