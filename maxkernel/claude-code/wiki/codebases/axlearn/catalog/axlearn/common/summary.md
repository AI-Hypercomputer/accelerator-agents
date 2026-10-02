---
title: 'Module: axlearn/common/summary.py'
type: catalog
provenance: extracted
module: axlearn/common/summary.py
status: fresh
symbol_base: scip-python python axlearn 0.0.0 `axlearn.common.summary`/
symbols:
  Summary: Summary#
  Summary.value: Summary#value().
  Summary.accumulate: Summary#accumulate().
  CallbackSummary.value: CallbackSummary#value().
  Summary.validate: Summary#validate().
  ImageSummary.value: ImageSummary#value().
  AudioSummary.value: AudioSummary#value().
  ImageSummary: ImageSummary#
  ImageSummary._value: ImageSummary#_value.
  AudioSummary: AudioSummary#
  AudioSummary._value: AudioSummary#_value.
  ImageSummary.validate: ImageSummary#validate().
  ImageSummary.accumulate: ImageSummary#accumulate().
  AudioSummary.validate: AudioSummary#validate().
  AudioSummary.accumulate: AudioSummary#accumulate().
  CallbackSummary: CallbackSummary#
  CallbackSummary.accumulate: CallbackSummary#accumulate().
  CallbackSummary.fn: CallbackSummary#fn.
  CallbackSummary.args: CallbackSummary#args.
  CallbackSummary.kwargs: CallbackSummary#kwargs.
  AudioSummary.sample_rate: AudioSummary#sample_rate.
  CallbackSummary.__init__: CallbackSummary#__init__().
---
# Module: [`axlearn/common/summary.py`](../../../../../../raw/code/axlearn/axlearn/common/summary.py)

## Classes
### `AudioSummary`  ·  implements/extends Summary
- def: [`axlearn/common/summary.py:80`](../../../../../../raw/code/axlearn/axlearn/common/summary.py#L80)
- doc: Audio summary.
- signature: `class AudioSummary(Summary):`
- members:
  - `accumulate(self, other: Summary)` — [`L123`](../../../../../../raw/code/axlearn/axlearn/common/summary.py#L123)
  - `validate(self)` — [`L104`](../../../../../../raw/code/axlearn/axlearn/common/summary.py#L104)
  - `value(self)` — [`L111`](../../../../../../raw/code/axlearn/axlearn/common/summary.py#L111) — Returns the audio tensor in shape [t, c], with floating-point values in [-1.0, 1.0]
  - `sample_rate` — [`L102`](../../../../../../raw/code/axlearn/axlearn/common/summary.py#L102)
- protocol/private: `_value`[`L101`](../../../../../../raw/code/axlearn/axlearn/common/summary.py#L101)
- uses (calls/refs, reference-scoped): [`Tensor`](utils.md#Tensor), [`Summary`](summary.md#Summary)
- used by: [`Summary`](summary.md#Summary), [`_match_summary_type`](summary_writer.md#_match_summary_type), [`value`](summary.md#Summary.value), [`accumulate`](summary.md#Summary.accumulate), [`validate`](summary.md#Summary.validate)

### `CallbackSummary`  ·  implements/extends Summary
- def: [`axlearn/common/summary.py:127`](../../../../../../raw/code/axlearn/axlearn/common/summary.py#L127)
- doc: A summary defined using a callback that is only called outside of JIT. The arguments
- signature: `class CallbackSummary(Summary):`
- members:
  - `__init__(self, fn: Callable, *args, **kwargs)` — [`L152`](../../../../../../raw/code/axlearn/axlearn/common/summary.py#L152) — Initializes the class.
  - `accumulate(self, other: Summary)` — [`L185`](../../../../../../raw/code/axlearn/axlearn/common/summary.py#L185)
  - `value(self)` — [`L180`](../../../../../../raw/code/axlearn/axlearn/common/summary.py#L180)
  - `args` — [`L149`](../../../../../../raw/code/axlearn/axlearn/common/summary.py#L149)
  - `fn` — [`L148`](../../../../../../raw/code/axlearn/axlearn/common/summary.py#L148)
  - `kwargs` — [`L150`](../../../../../../raw/code/axlearn/axlearn/common/summary.py#L150)
- uses (calls/refs, reference-scoped): [`Tensor`](utils.md#Tensor), [`Summary`](summary.md#Summary)
- used by: [`Summary`](summary.md#Summary), [`value`](summary.md#Summary.value), [`accumulate`](summary.md#Summary.accumulate)

### `ImageSummary`  ·  implements/extends Summary
- def: [`axlearn/common/summary.py:43`](../../../../../../raw/code/axlearn/axlearn/common/summary.py#L43)
- doc: A summary that should be logged as a batch of images.
- signature: `class ImageSummary(Summary):`
- members:
  - `accumulate(self, other: Summary)` — [`L76`](../../../../../../raw/code/axlearn/axlearn/common/summary.py#L76)
  - `validate(self)` — [`L56`](../../../../../../raw/code/axlearn/axlearn/common/summary.py#L56)
  - `value(self)` — [`L69`](../../../../../../raw/code/axlearn/axlearn/common/summary.py#L69)
- protocol/private: `_value`[`L54`](../../../../../../raw/code/axlearn/axlearn/common/summary.py#L54)
- uses (calls/refs, reference-scoped): [`Tensor`](utils.md#Tensor), [`Summary`](summary.md#Summary)
- used by: [`Summary`](summary.md#Summary), [`_match_summary_type`](summary_writer.md#_match_summary_type), [`value`](summary.md#Summary.value), [`accumulate`](summary.md#Summary.accumulate), [`validate`](summary.md#Summary.validate)

### `Summary`  ·  implements/extends PyTreeNode
- def: [`axlearn/common/summary.py:14`](../../../../../../raw/code/axlearn/axlearn/common/summary.py#L14)
- doc: Base class for a summary value.
- signature: `class Summary(flax_struct.PyTreeNode):`
- members:
  - `accumulate(self, other: Summary)` — [`L31`](../../../../../../raw/code/axlearn/axlearn/common/summary.py#L31) — The default way this summary should be accumulated.
  - `validate(self)` — [`L24`](../../../../../../raw/code/axlearn/axlearn/common/summary.py#L24) — Validates that the summary was constructed with valid data.
  - `value(self)` — [`L20`](../../../../../../raw/code/axlearn/axlearn/common/summary.py#L20) — Returns a value for logging.
- uses (calls/refs, reference-scoped): [`NestedTensor`](utils.md#NestedTensor), [`MetricSummary`](metrics.md#MetricSummary), [`value`](metrics.md#_ReducerSummary.value), [`value`](summary.md#CallbackSummary.value), [`accumulate`](metrics.md#_ReducerSummary.accumulate), [`accumulate`](metrics.md#WeightedSummary.accumulate), [`validate`](metrics.md#_ReducerSummary.validate), [`value`](metrics.md#WeightedValue.value), [`value`](summary.md#AudioSummary.value), [`value`](summary.md#ImageSummary.value), [`AudioSummary`](summary.md#AudioSummary), [`ImageSummary`](summary.md#ImageSummary), [`CallbackSummary`](summary.md#CallbackSummary), [`accumulate`](summary.md#AudioSummary.accumulate), [`accumulate`](summary.md#CallbackSummary.accumulate), [`accumulate`](summary.md#ImageSummary.accumulate), [`validate`](summary.md#AudioSummary.validate), [`validate`](summary.md#ImageSummary.validate)
- used by: [`add_summary`](module.md#Module.add_summary), [`add_summary`](module.md#InvocationContext.add_summary), [`_match_summary_type`](summary_writer.md#_match_summary_type), [`write`](summary_writer.md#SummaryWriter.write), [`MetricSummary`](metrics.md#MetricSummary), [`convert`](summary_writer.md#WandBWriter.convert), [`update`](metrics.md#MetricAccumulator.update), [`validate`](module.md#InvocationContext.validate), [`accumulate`](metrics.md#WeightedSummary.accumulate), [`AudioSummary`](summary.md#AudioSummary), [`ImageSummary`](summary.md#ImageSummary), [`is_leaf`](summary_writer.md#SummaryWriter.is_leaf), [`is_leaf`](summary_writer.md#WandBWriter.is_leaf), [`_accumulate`](module.md#propagate_repeated_output_collections._accumulate), [`is_leaf`](metrics.md#MetricAccumulator.is_leaf), [`CallbackSummary`](summary.md#CallbackSummary), [`accumulate`](summary.md#AudioSummary.accumulate), [`accumulate`](summary.md#CallbackSummary.accumulate), [`accumulate`](summary.md#ImageSummary.accumulate)

