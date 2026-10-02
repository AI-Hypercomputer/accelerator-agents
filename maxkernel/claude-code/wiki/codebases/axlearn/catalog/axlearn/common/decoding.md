---
title: 'Module: axlearn/common/decoding.py'
type: catalog
provenance: extracted
module: axlearn/common/decoding.py
status: fresh
symbol_base: scip-python python axlearn 0.0.0 `axlearn.common.decoding`/
symbols:
  beam_search_decode.beam_search_loop_body_fn: beam_search_decode().beam_search_loop_body_fn().
  beam_search_decode: beam_search_decode().
  _beam_init: _beam_init().
  sample_decode: sample_decode().
  _decode_init: _decode_init().
  sample_decode.sample_decode_loop_body_fn: sample_decode().sample_decode_loop_body_fn().
  StopDecodingCondition: StopDecodingCondition#
  _gather_beams: _gather_beams().
  beam_search_decode.beam_search_loop_cond_fn: beam_search_decode().beam_search_loop_cond_fn().
  DecodingState.stop_decoding: DecodingState#stop_decoding.
  NEG_INF: NEG_INF.
  StopOnSubsequence.__call__: StopOnSubsequence#__call__().
  _BeamState.live_seqs: _BeamState#live_seqs.
  DecodingState.sequences: DecodingState#sequences.
  sample_decode.sample_decode_loop_cond_fn: sample_decode().sample_decode_loop_cond_fn().
  _gather_topk_beams: _gather_topk_beams().
  _BeamState.live_scores: _BeamState#live_scores.
  _BeamState.finished_scores: _BeamState#finished_scores.
  _BeamState.cur_index: _BeamState#cur_index.
  _BeamState.finished_seqs: _BeamState#finished_seqs.
  flatten_decoding_dim: flatten_decoding_dim().
  unflatten_decoding_dim: unflatten_decoding_dim().
  beam_search_decode.beam_search_loop_body_fn.mask_out_of_sequence: beam_search_decode().beam_search_loop_body_fn().mask_out_of_sequence().
  DecodingState.cur_index: DecodingState#cur_index.
  DecodingState.token_scores: DecodingState#token_scores.
  brevity_penalty_fn: brevity_penalty_fn().
  _merge_prefixes: _merge_prefixes().
  PrefixMerger.init_state: PrefixMerger#init_state().
  PrefixMerger.compute: PrefixMerger#compute().
  PrefixMerger.update: PrefixMerger#update().
  _BeamState: _BeamState#
  DecodingState: DecodingState#
  _BeamState.cache: _BeamState#cache.
  _BeamState.prefix_merger: _BeamState#prefix_merger.
  DecodingState.cache: DecodingState#cache.
  DecodingState.prng_key: DecodingState#prng_key.
  infer_initial_time_step: infer_initial_time_step().
  StopOnMaxLength.__call__: StopOnMaxLength#__call__().
  CompositeDecodingCondition.__call__: CompositeDecodingCondition#__call__().
  add_decoding_dim: add_decoding_dim().
  StopOnSubsequence: StopOnSubsequence#
  StopOnSubsequence.targets: StopOnSubsequence#targets.
  BrevityPenaltyFn: BrevityPenaltyFn#
  BeamSearchOutputs: BeamSearchOutputs#
  SampleOutputs: SampleOutputs#
  _top_k_two_stage: _top_k_two_stage().
  brevity_penalty_fn.fn: brevity_penalty_fn().fn().
  BeamSearchOutputs.sequences: BeamSearchOutputs#sequences.
  BeamSearchOutputs.scores: BeamSearchOutputs#scores.
  BeamSearchOutputs.live_sequences: BeamSearchOutputs#live_sequences.
  BeamSearchOutputs.live_scores: BeamSearchOutputs#live_scores.
  SampleOutputs.sequences: SampleOutputs#sequences.
  SampleOutputs.token_scores: SampleOutputs#token_scores.
  StopOnSubsequence.longest: StopOnSubsequence#longest.
  BrevityPenaltyFn.__call__: BrevityPenaltyFn#__call__().
  compute_merge_matrix_by_prefix_ids: compute_merge_matrix_by_prefix_ids().
  StopDecodingCondition.__call__: StopDecodingCondition#__call__().
  StopOnMaxLength: StopOnMaxLength#
  CompositeDecodingCondition: CompositeDecodingCondition#
  CompositeDecodingCondition.__init__: CompositeDecodingCondition#__init__().
  _gather_beams.gather_fn: _gather_beams().gather_fn().
  PrefixMerger: PrefixMerger#
  StopOnSubsequence.pad_value: StopOnSubsequence#pad_value.
  StopOnMaxLength.max_num_decodes: StopOnMaxLength#max_num_decodes.
  CompositeDecodingCondition.conditions: CompositeDecodingCondition#conditions.
  StopOnSubsequence.__init__: StopOnSubsequence#__init__().
  StopOnMaxLength.__init__: StopOnMaxLength#__init__().
---
# Module: [`axlearn/common/decoding.py`](../../../../../../raw/code/axlearn/axlearn/common/decoding.py)

## Classes
### `BeamSearchOutputs`  ·  implements/extends PyTreeNode
- def: [`axlearn/common/decoding.py:357`](../../../../../../raw/code/axlearn/axlearn/common/decoding.py#L357)
- doc: Output values after performing beam search decoding.
- signature: `class BeamSearchOutputs(flax_struct.PyTreeNode):`
- members:
  - `live_scores` — [`L370`](../../../../../../raw/code/axlearn/axlearn/common/decoding.py#L370)
  - `live_sequences` — [`L367`](../../../../../../raw/code/axlearn/axlearn/common/decoding.py#L367)
  - `scores` — [`L364`](../../../../../../raw/code/axlearn/axlearn/common/decoding.py#L364)
  - `sequences` — [`L361`](../../../../../../raw/code/axlearn/axlearn/common/decoding.py#L361)
- uses (calls/refs, reference-scoped): [`Tensor`](utils.md#Tensor)
- used by: [`beam_search_decode`](decoding.md#beam_search_decode), [`beam_search_decode`](decoder.md#DecodingLayer.beam_search_decode)

### `BrevityPenaltyFn`  ·  implements/extends Protocol
- def: [`axlearn/common/decoding.py:190`](../../../../../../raw/code/axlearn/axlearn/common/decoding.py#L190)
- signature: `class BrevityPenaltyFn(Protocol):`
- members:
  - `__call__(self, *, length: Tensor, raw_scores: Tensor)` — [`L191`](../../../../../../raw/code/axlearn/axlearn/common/decoding.py#L191) — Compute the brevity penalty based on the length of a decoding and its raw scores.
- uses (calls/refs, reference-scoped): [`Tensor`](utils.md#Tensor)
- used by: [`beam_search_decode`](decoding.md#beam_search_decode), [`beam_search_decode`](decoder.md#DecodingLayer.beam_search_decode), [`brevity_penalty_fn`](decoding.md#brevity_penalty_fn)

### `CompositeDecodingCondition`  ·  implements/extends StopDecodingCondition
- def: [`axlearn/common/decoding.py:939`](../../../../../../raw/code/axlearn/axlearn/common/decoding.py#L939)
- doc: Combines multiple stopping conditions with OR logic.
- signature: `class CompositeDecodingCondition(StopDecodingCondition):`
- members:
  - `__call__(self, *, index: Tensor, sequences: Tensor, prefix_len: Tensor)` — [`L955`](../../../../../../raw/code/axlearn/axlearn/common/decoding.py#L955) — Returns True when ANY condition is met.
  - `__init__(self, conditions: Sequence[StopDecodingCondition])` — [`L942`](../../../../../../raw/code/axlearn/axlearn/common/decoding.py#L942) — Stops decoding when ANY of the conditions is met.
  - `conditions` — [`L953`](../../../../../../raw/code/axlearn/axlearn/common/decoding.py#L953)
- uses (calls/refs, reference-scoped): [`Tensor`](utils.md#Tensor), [`StopDecodingCondition`](decoding.md#StopDecodingCondition)
- used by: [`StopDecodingCondition`](decoding.md#StopDecodingCondition)

### `DecodingState`  ·  implements/extends NamedTuple
- def: [`axlearn/common/decoding.py:731`](../../../../../../raw/code/axlearn/axlearn/common/decoding.py#L731)
- doc: Holds sample decoding state data.
- signature: `class DecodingState(NamedTuple):`
- members:
  - `cache` — [`L743`](../../../../../../raw/code/axlearn/axlearn/common/decoding.py#L743)
  - `cur_index` — [`L735`](../../../../../../raw/code/axlearn/axlearn/common/decoding.py#L735)
  - `prng_key` — [`L745`](../../../../../../raw/code/axlearn/axlearn/common/decoding.py#L745)
  - `sequences` — [`L737`](../../../../../../raw/code/axlearn/axlearn/common/decoding.py#L737)
  - `stop_decoding` — [`L741`](../../../../../../raw/code/axlearn/axlearn/common/decoding.py#L741)
  - `token_scores` — [`L739`](../../../../../../raw/code/axlearn/axlearn/common/decoding.py#L739)
- uses (calls/refs, reference-scoped): [`Tensor`](utils.md#Tensor), [`NestedTensor`](utils.md#NestedTensor)
- used by: [`sample_decode`](decoding.md#sample_decode), [`_decode_init`](decoding.md#_decode_init), [`sample_decode_loop_body_fn`](decoding.md#sample_decode.sample_decode_loop_body_fn), [`sample_decode_loop_cond_fn`](decoding.md#sample_decode.sample_decode_loop_cond_fn)

### `PrefixMerger`
- def: [`axlearn/common/decoding.py:286`](../../../../../../raw/code/axlearn/axlearn/common/decoding.py#L286)
- doc: A PrefixMerger computes merge matrix and merges prefixes accordingly.
- signature: `class PrefixMerger:`
- members:
  - `compute(self, state: NestedTensor)` — [`L313`](../../../../../../raw/code/axlearn/axlearn/common/decoding.py#L313) — Computes the merge matrix.
  - `init_state(self, *, tokens: Tensor)` — [`L300`](../../../../../../raw/code/axlearn/axlearn/common/decoding.py#L300) — Initializes prefix merger state.
  - `update(self, *, tokens: Tensor, state: NestedTensor)` — [`L325`](../../../../../../raw/code/axlearn/axlearn/common/decoding.py#L325) — Updates state.
- uses (calls/refs, reference-scoped): [`Tensor`](utils.md#Tensor), [`NestedTensor`](utils.md#NestedTensor)
- used by: [`beam_search_loop_body_fn`](decoding.md#beam_search_decode.beam_search_loop_body_fn), [`beam_search_decode`](decoding.md#beam_search_decode), [`_beam_init`](decoding.md#_beam_init)

### `SampleOutputs`  ·  implements/extends PyTreeNode
- def: [`axlearn/common/decoding.py:817`](../../../../../../raw/code/axlearn/axlearn/common/decoding.py#L817)
- doc: Output values after performing sample decoding.
- signature: `class SampleOutputs(flax_struct.PyTreeNode):`
- members:
  - `sequences` — [`L821`](../../../../../../raw/code/axlearn/axlearn/common/decoding.py#L821)
  - `token_scores` — [`L823`](../../../../../../raw/code/axlearn/axlearn/common/decoding.py#L823)
- uses (calls/refs, reference-scoped): [`Tensor`](utils.md#Tensor)
- used by: [`sample_decode`](decoder.md#DecodingLayer.sample_decode), [`sample_decode`](decoding.md#sample_decode)

### `StopDecodingCondition`  ·  implements/extends Protocol
- def: [`axlearn/common/decoding.py:826`](../../../../../../raw/code/axlearn/axlearn/common/decoding.py#L826)
- doc: Callable which, given index, sequences and prefix length, returns
- signature: `class StopDecodingCondition(Protocol):`
- members:
  - `__call__(self, *, index: Tensor, sequences: Tensor, prefix_len: Tensor)` — [`L830`](../../../../../../raw/code/axlearn/axlearn/common/decoding.py#L830) — Given the current index and sequences, return a (batch x decodes) boolean
- uses (calls/refs, reference-scoped): [`Tensor`](utils.md#Tensor), [`StopOnSubsequence`](decoding.md#StopOnSubsequence), [`CompositeDecodingCondition`](decoding.md#CompositeDecodingCondition), [`StopOnMaxLength`](decoding.md#StopOnMaxLength)
- used by: [`sample_decode`](decoder.md#DecodingLayer.sample_decode), [`sample_decode`](decoding.md#sample_decode), [`StopOnSubsequence`](decoding.md#StopOnSubsequence), [`CompositeDecodingCondition`](decoding.md#CompositeDecodingCondition), [`StopOnMaxLength`](decoding.md#StopOnMaxLength), [`__init__`](decoding.md#CompositeDecodingCondition.__init__)

### `StopOnMaxLength`  ·  implements/extends StopDecodingCondition
- def: [`axlearn/common/decoding.py:908`](../../../../../../raw/code/axlearn/axlearn/common/decoding.py#L908)
- doc: Early stopping when reaching maximum generation length.
- signature: `class StopOnMaxLength(StopDecodingCondition):`
- members:
  - `__call__(self, *, index: Tensor, sequences: Tensor, prefix_len: Tensor)` — [`L919`](../../../../../../raw/code/axlearn/axlearn/common/decoding.py#L919) — Returns True when generation length >= max_num_decodes.
  - `__init__(self, max_num_decodes: int)` — [`L911`](../../../../../../raw/code/axlearn/axlearn/common/decoding.py#L911) — Stops decoding when generation reaches max_num_decodes.
  - `max_num_decodes` — [`L917`](../../../../../../raw/code/axlearn/axlearn/common/decoding.py#L917)
- uses (calls/refs, reference-scoped): [`Tensor`](utils.md#Tensor), [`StopDecodingCondition`](decoding.md#StopDecodingCondition)
- used by: [`StopDecodingCondition`](decoding.md#StopDecodingCondition)

### `StopOnSubsequence`  ·  implements/extends StopDecodingCondition
- def: [`axlearn/common/decoding.py:847`](../../../../../../raw/code/axlearn/axlearn/common/decoding.py#L847)
- doc: Early stopping on suffix-matches.
- signature: `class StopOnSubsequence(StopDecodingCondition):`
- members:
  - `__init__(self, stopping_seqs: Union[int, Sequence[int], Sequence[Sequence[int]]], pad_value=-1)` — [`L850`](../../../../../../raw/code/axlearn/axlearn/common/decoding.py#L850) — Stops decoding when a sequence suffix-matches one of `stopping_seqs`.
  - `longest` — [`L879`](../../../../../../raw/code/axlearn/axlearn/common/decoding.py#L879)
  - `pad_value` — [`L865`](../../../../../../raw/code/axlearn/axlearn/common/decoding.py#L865)
  - `targets` — [`L880`](../../../../../../raw/code/axlearn/axlearn/common/decoding.py#L880)
- protocol/private: `__call__`[`L887`](../../../../../../raw/code/axlearn/axlearn/common/decoding.py#L887)
- uses (calls/refs, reference-scoped): [`Tensor`](utils.md#Tensor), [`StopDecodingCondition`](decoding.md#StopDecodingCondition)
- used by: [`sample_decode`](decoder.md#DecodingLayer.sample_decode), [`StopDecodingCondition`](decoding.md#StopDecodingCondition)

### `_BeamState`  ·  implements/extends NamedTuple
- def: [`axlearn/common/decoding.py:340`](../../../../../../raw/code/axlearn/axlearn/common/decoding.py#L340)
- doc: Holds beam search state data.
- signature: `class _BeamState(NamedTuple):`
- members:
  - `cache` — [`L352`](../../../../../../raw/code/axlearn/axlearn/common/decoding.py#L352)
  - `cur_index` — [`L344`](../../../../../../raw/code/axlearn/axlearn/common/decoding.py#L344)
  - `finished_scores` — [`L347`](../../../../../../raw/code/axlearn/axlearn/common/decoding.py#L347)
  - `finished_seqs` — [`L350`](../../../../../../raw/code/axlearn/axlearn/common/decoding.py#L350)
  - `live_scores` — [`L346`](../../../../../../raw/code/axlearn/axlearn/common/decoding.py#L346)
  - `live_seqs` — [`L349`](../../../../../../raw/code/axlearn/axlearn/common/decoding.py#L349)
  - `prefix_merger` — [`L354`](../../../../../../raw/code/axlearn/axlearn/common/decoding.py#L354)
- uses (calls/refs, reference-scoped): [`Tensor`](utils.md#Tensor), [`NestedTensor`](utils.md#NestedTensor)
- used by: [`beam_search_loop_body_fn`](decoding.md#beam_search_decode.beam_search_loop_body_fn), [`beam_search_decode`](decoding.md#beam_search_decode), [`_beam_init`](decoding.md#_beam_init), [`beam_search_loop_cond_fn`](decoding.md#beam_search_decode.beam_search_loop_cond_fn)

## Functions
- `_beam_init(*, inputs: Tensor, time_step: Tensor, beam_size: int, max_decode_len: int, cache: NestedTensor, pad_id: int, prefix_merger: Optional[PrefixMerger] = None)` — [`L373`](../../../../../../raw/code/axlearn/axlearn/common/decoding.py#L373) — Initializes the beam search state data structure.
- `_decode_init(*, inputs: Tensor, time_step: Tensor, num_decodes: int, max_decode_len: int, cache: NestedTensor, prng_key: Tensor, pad_id: int, token_scores: Optional[Tensor] = None)` — [`L748`](../../../../../../raw/code/axlearn/axlearn/common/decoding.py#L748) — Initializes the sample decode state data structure.
- `_gather_beams(nested: NestedTensor, beam_indices: Tensor, batch_size: int, old_beam_size: int, new_beam_size: int, one_hot: bool = True)` — [`L78`](../../../../../../raw/code/axlearn/axlearn/common/decoding.py#L78) — Gathers the beam slices indexed by beam_indices into new beam array.
- `_gather_topk_beams(nested: NestedTensor, score_or_log_prob: Tensor, batch_size: int, new_beam_size: int)` — [`L168`](../../../../../../raw/code/axlearn/axlearn/common/decoding.py#L168) — Gathers the top-k beam slices given by score_or_log_prob array.
- `_merge_prefixes(merge_matrix: Tensor, *, log_probs: Tensor)` — [`L262`](../../../../../../raw/code/axlearn/axlearn/common/decoding.py#L262) — Merges prefixes according to `merge_matrix`.
- `_top_k_two_stage(x: Tensor, k: int)` — [`L123`](../../../../../../raw/code/axlearn/axlearn/common/decoding.py#L123) — Wrapper around lax.top_k with low-batch optimization.
- `add_decoding_dim(x: Tensor, num_decodes: int)` — [`L44`](../../../../../../raw/code/axlearn/axlearn/common/decoding.py#L44) — Creates num_decodes as second dimension in non-scalar array x and tiles into it.
- `beam_search_decode(*, inputs: Tensor, time_step: Tensor, cache: NestedTensor, tokens_to_scores: Callable[[Tensor, NestedTensor], tuple[Tensor, NestedTensor]], eos_id: int, num_decodes: int, max_decode_len: Optional[int] = None, loop: Literal["lax", "python"] = "lax", brevity_penalty: Optional[BrevityPenaltyFn] = None, pad_id: int = 0, prefix_merger: Optional[PrefixMerger] = None)` — [`L433`](../../../../../../raw/code/axlearn/axlearn/common/decoding.py#L433) — Performs beam search decoding.
- `beam_search_loop_body_fn(state: _BeamState)` — [`L541`](../../../../../../raw/code/axlearn/axlearn/common/decoding.py#L541) — Beam search loop state update function.
- `beam_search_loop_cond_fn(state: _BeamState)` — [`L518`](../../../../../../raw/code/axlearn/axlearn/common/decoding.py#L518) — Beam search loop termination condition.
- `brevity_penalty_fn(*, alpha: float = 0, bp_type: Literal["t5", "hf"] = "t5")` — [`L203`](../../../../../../raw/code/axlearn/axlearn/common/decoding.py#L203) — Brevity penalty function to do length normalization during beam search.
- `compute_merge_matrix_by_prefix_ids(prefix_ids: Tensor)` — [`L240`](../../../../../../raw/code/axlearn/axlearn/common/decoding.py#L240) — Computes a merge matrix by comparing prefixes to merge equivalent prefixes.
- `flatten_decoding_dim(x: Tensor)` — [`L57`](../../../../../../raw/code/axlearn/axlearn/common/decoding.py#L57) — Flattens the first two dimensions of a non-scalar array.
- `fn(*, length: Tensor, raw_scores: Tensor)` — [`L226`](../../../../../../raw/code/axlearn/axlearn/common/decoding.py#L226)
- `gather_fn(x)` — [`L108`](../../../../../../raw/code/axlearn/axlearn/common/decoding.py#L108)
- `infer_initial_time_step(prefix: Tensor, *, pad_id: int)` — [`L1154`](../../../../../../raw/code/axlearn/axlearn/common/decoding.py#L1154) — Computes initial time step based on prefix.
- `mask_out_of_sequence(old: Tensor, new: Tensor)` — [`L690`](../../../../../../raw/code/axlearn/axlearn/common/decoding.py#L690)
- `sample_decode(*, inputs: Tensor, time_step: Tensor, cache: NestedTensor, tokens_to_scores: Callable[[Tensor, NestedTensor], tuple[Tensor, NestedTensor]], stop_decoding_condition: StopDecodingCondition, num_decodes: int, prng_key: Tensor, max_decode_len: Optional[int] = None, loop: Literal["lax", "python"] = "lax", pad_id: int = 0, input_token_scores: Optional[Tensor] = None)` — [`L978`](../../../../../../raw/code/axlearn/axlearn/common/decoding.py#L978) — Performs sampling decoding.
- `sample_decode_loop_body_fn(state: DecodingState)` — [`L1055`](../../../../../../raw/code/axlearn/axlearn/common/decoding.py#L1055) — Sample decode loop state update function.
- `sample_decode_loop_cond_fn(state: DecodingState)` — [`L1048`](../../../../../../raw/code/axlearn/axlearn/common/decoding.py#L1048) — Sample decode loop termination condition.
- `unflatten_decoding_dim(x: Tensor, batch_size: int, num_decodes: int)` — [`L67`](../../../../../../raw/code/axlearn/axlearn/common/decoding.py#L67) — Unflattens the first, flat batch*decoding dimension of a non-scalar array.

## Module values
- `NEG_INF` — [`L38`](../../../../../../raw/code/axlearn/axlearn/common/decoding.py#L38)

