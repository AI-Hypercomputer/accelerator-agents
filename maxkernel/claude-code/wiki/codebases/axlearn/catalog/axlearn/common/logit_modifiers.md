---
title: 'Module: axlearn/common/logit_modifiers.py'
type: catalog
provenance: extracted
module: axlearn/common/logit_modifiers.py
status: fresh
symbol_base: scip-python python axlearn 0.0.0 `axlearn.common.logit_modifiers`/
symbols:
  _float32_binary_search: _float32_binary_search().
  LogitsToLogitsFn: LogitsToLogitsFn.
  top_p_logits.fn: top_p_logits().fn().
  top_k_logits.fn: top_k_logits().fn().
  top_p_logits: top_p_logits().
  top_k_logits: top_k_logits().
  chain.fn: chain().fn().
  top_k_logits.smallest_index_fn: top_k_logits().smallest_index_fn().
  _int32_binary_search: _int32_binary_search().
  chain: chain().
  scale_by: scale_by().
  _monotonic_int32_to_float32_bit_mask: _monotonic_int32_to_float32_bit_mask().
  scale_by.fn: scale_by().fn().
  top_p_logits.fn.predicate: top_p_logits().fn().predicate().
  top_k_logits.fn.predicate: top_k_logits().fn().predicate().
  _int32_binary_search.loop_body: _int32_binary_search().loop_body().
  _float32_binary_search.int32_predicate: _float32_binary_search().int32_predicate().
---
# Module: [`axlearn/common/logit_modifiers.py`](../../../../../../raw/code/axlearn/axlearn/common/logit_modifiers.py)

## Functions
- `_float32_binary_search(batched_shape: tuple[int], *, predicate: Callable[[Tensor], Tensor])` — [`L254`](../../../../../../raw/code/axlearn/axlearn/common/logit_modifiers.py#L254) — Binary search to find the largest finite float32 value for which predicate is False.
- `_int32_binary_search(batched_shape: tuple[int], *, predicate: Callable[[Tensor], Tensor])` — [`L220`](../../../../../../raw/code/axlearn/axlearn/common/logit_modifiers.py#L220) — Binary search to find the largest finite int32 value for which the predicate is False.
- `_monotonic_int32_to_float32_bit_mask(x: Tensor)` — [`L198`](../../../../../../raw/code/axlearn/axlearn/common/logit_modifiers.py#L198) — Converts an int32 value to an int32 representing a float32 bit mask.
- `chain(*args)` — [`L33`](../../../../../../raw/code/axlearn/axlearn/common/logit_modifiers.py#L33) — Returns a function to apply multiple logits-to-logits functions/configs in series.
- `fn(logits: Tensor)` — [`L36`](../../../../../../raw/code/axlearn/axlearn/common/logit_modifiers.py#L36)
- `fn(logits: Tensor)` — [`L63`](../../../../../../raw/code/axlearn/axlearn/common/logit_modifiers.py#L63)
- `fn(logits: Tensor)` — [`L94`](../../../../../../raw/code/axlearn/axlearn/common/logit_modifiers.py#L94)
- `fn(logits: Tensor)` — [`L151`](../../../../../../raw/code/axlearn/axlearn/common/logit_modifiers.py#L151)
- `int32_predicate(x)` — [`L276`](../../../../../../raw/code/axlearn/axlearn/common/logit_modifiers.py#L276)
- `loop_body(i: int, solution: Tensor)` — [`L243`](../../../../../../raw/code/axlearn/axlearn/common/logit_modifiers.py#L243)
- `predicate(float32_query: Tensor, top_p: Union[float, Tensor])` — [`L106`](../../../../../../raw/code/axlearn/axlearn/common/logit_modifiers.py#L106)
- `predicate(float32_query: Tensor)` — [`L162`](../../../../../../raw/code/axlearn/axlearn/common/logit_modifiers.py#L162)
- `scale_by(temperature: float, *, min_temperature: float = 0.0001)` — [`L45`](../../../../../../raw/code/axlearn/axlearn/common/logit_modifiers.py#L45) — Build a function that returns logits scaled for temperature sampling.
- `smallest_index_fn(logits: Tensor)` — [`L177`](../../../../../../raw/code/axlearn/axlearn/common/logit_modifiers.py#L177)
- `top_k_logits(k: int, *, break_ties: Literal["all", "smallest_index"] = "all")` — [`L124`](../../../../../../raw/code/axlearn/axlearn/common/logit_modifiers.py#L124) — Build a function that returns logits suitably normalized for top-k sampling.
- `top_p_logits(p: Union[float, Tensor])` — [`L69`](../../../../../../raw/code/axlearn/axlearn/common/logit_modifiers.py#L69) — Build a function that returns logits suitably normalized for top-p sampling.

## Module values
- `LogitsToLogitsFn` — [`L30`](../../../../../../raw/code/axlearn/axlearn/common/logit_modifiers.py#L30)

