---
title: 'Module: axlearn/common/inference_output.py'
type: catalog
provenance: extracted
module: axlearn/common/inference_output.py
status: fresh
symbol_base: scip-python python axlearn 0.0.0 `axlearn.common.inference_output`/
symbols:
  TfExampleRecordSink.__init__: TfExampleRecordSink#__init__().
  JsonlExampleRecordSink.__init__: JsonlExampleRecordSink#__init__().
  OutputRecordWriter.__init__: OutputRecordWriter#__init__().
  OutputRecordWriter.Config.sink: OutputRecordWriter#Config#sink.
  BaseRecordSink: BaseRecordSink#
  TfExampleRecordSink.write: TfExampleRecordSink#write().
  JsonlExampleRecordSink.write: JsonlExampleRecordSink#write().
  BaseOutputWriter.Config: BaseOutputWriter#Config#
  BaseOutputWriter: BaseOutputWriter#
  TfExampleRecordSink.Config: TfExampleRecordSink#Config#
  JsonlExampleRecordSink.Config: JsonlExampleRecordSink#Config#
  OutputRecordWriter.Config: OutputRecordWriter#Config#
  BaseOutputWriter.Config.batch_partition_spec: BaseOutputWriter#Config#batch_partition_spec.
  BaseRecordSink.write: BaseRecordSink#write().
  TfExampleRecordSink.Config.output_path: TfExampleRecordSink#Config#output_path.
  JsonlExampleRecordSink.Config.output_path: JsonlExampleRecordSink#Config#output_path.
  OutputRecordWriter: OutputRecordWriter#
  OutputRecordWriter._build_record: OutputRecordWriter#_build_record().
  BaseOutputWriter.write: BaseOutputWriter#write().
  BaseRecordSink.flush: BaseRecordSink#flush().
  OutputRecordWriter.write: OutputRecordWriter#write().
  _tf_feature: _tf_feature().
  _json_feature: _json_feature().
  TfExampleRecordSink: TfExampleRecordSink#
  BaseOutputWriter.flush: BaseOutputWriter#flush().
  TfExampleRecordSink.flush: TfExampleRecordSink#flush().
  JsonlExampleRecordSink: JsonlExampleRecordSink#
  JsonlExampleRecordSink.flush: JsonlExampleRecordSink#flush().
  InputOutputRecordWriter: InputOutputRecordWriter#
  InputOutputRecordWriter._build_record: InputOutputRecordWriter#_build_record().
  TfExampleRecordSink._writer: TfExampleRecordSink#_writer.
  JsonlExampleRecordSink._writer: JsonlExampleRecordSink#_writer.
  OutputRecordWriter.flush: OutputRecordWriter#flush().
---
# Module: [`axlearn/common/inference_output.py`](../../../../../../raw/code/axlearn/axlearn/common/inference_output.py)

## Classes
### `BaseOutputWriter`  ·  implements/extends Module
- def: [`axlearn/common/inference_output.py:26`](../../../../../../raw/code/axlearn/axlearn/common/inference_output.py#L26)
- doc: Base class for OutputWriter, which writes records for inference outputs.
- signature: `class BaseOutputWriter(Module):`
- members:
  - `flush(self)` — [`L43`](../../../../../../raw/code/axlearn/axlearn/common/inference_output.py#L43) — Flushes the written records.
  - `write(self, *, input_batch: NestedTensor, output_batch: NestedTensor)` — [`L34`](../../../../../../raw/code/axlearn/axlearn/common/inference_output.py#L34) — Writes records extracted from the given input/output batch.
- uses (calls/refs, reference-scoped): [`NestedTensor`](utils.md#NestedTensor), [`Module`](module.md#Module), [`OutputRecordWriter`](inference_output.md#OutputRecordWriter), [`write`](inference_output.md#OutputRecordWriter.write), [`flush`](inference_output.md#OutputRecordWriter.flush)
- used by: [`Module`](module.md#Module), [`Config`](inference_output.md#OutputRecordWriter.Config), [`output_writer`](evaler.md#SpmdEvaler.Config.output_writer), [`OutputRecordWriter`](inference_output.md#OutputRecordWriter)

### `BaseRecordSink`  ·  implements/extends Module
- def: [`axlearn/common/inference_output.py:48`](../../../../../../raw/code/axlearn/axlearn/common/inference_output.py#L48)
- signature: `class BaseRecordSink(Module):`
- members:
  - `flush(self)` — [`L53`](../../../../../../raw/code/axlearn/axlearn/common/inference_output.py#L53) — Flushes the written records.
  - `write(self, record: NestedTensor)` — [`L49`](../../../../../../raw/code/axlearn/axlearn/common/inference_output.py#L49) — Writes `record` to the sink.
- uses (calls/refs, reference-scoped): [`NestedTensor`](utils.md#NestedTensor), [`Module`](module.md#Module), [`write`](inference_output.md#JsonlExampleRecordSink.write), [`write`](inference_output.md#TfExampleRecordSink.write), [`TfExampleRecordSink`](inference_output.md#TfExampleRecordSink), [`JsonlExampleRecordSink`](inference_output.md#JsonlExampleRecordSink), [`flush`](inference_output.md#JsonlExampleRecordSink.flush), [`flush`](inference_output.md#TfExampleRecordSink.flush)
- used by: [`Module`](module.md#Module), [`sink`](inference_output.md#OutputRecordWriter.Config.sink), [`TfExampleRecordSink`](inference_output.md#TfExampleRecordSink), [`JsonlExampleRecordSink`](inference_output.md#JsonlExampleRecordSink)

### `Config`  ·  implements/extends Config
- def: [`axlearn/common/inference_output.py:181`](../../../../../../raw/code/axlearn/axlearn/common/inference_output.py#L181)
- signature: `class Config(BaseOutputWriter.Config):`
- members:
  - `batch_partition_spec` — [`L32`](../../../../../../raw/code/axlearn/axlearn/common/inference_output.py#L32)
  - `output_path` — [`L102`](../../../../../../raw/code/axlearn/axlearn/common/inference_output.py#L102)
  - `output_path` — [`L144`](../../../../../../raw/code/axlearn/axlearn/common/inference_output.py#L144)
  - `sink` — [`L182`](../../../../../../raw/code/axlearn/axlearn/common/inference_output.py#L182)
- uses (calls/refs, reference-scoped): [`default_config`](config.md#Configurable.default_config), [`config_class`](config.md#config_class), [`REQUIRED`](config.md#REQUIRED), [`Required`](config.md#Required), [`Config`](module.md#Module.Config), [`DataPartitionType`](utils.md#DataPartitionType), [`BaseRecordSink`](inference_output.md#BaseRecordSink), [`Config`](inference_output.md#BaseOutputWriter.Config), [`BaseOutputWriter`](inference_output.md#BaseOutputWriter), [`TfExampleRecordSink`](inference_output.md#TfExampleRecordSink)
- used by: [`__init__`](inference_output.md#JsonlExampleRecordSink.__init__), [`__init__`](inference_output.md#OutputRecordWriter.__init__), [`__init__`](inference_output.md#TfExampleRecordSink.__init__)

### `InputOutputRecordWriter`  ·  implements/extends OutputRecordWriter
- def: [`axlearn/common/inference_output.py:222`](../../../../../../raw/code/axlearn/axlearn/common/inference_output.py#L222)
- doc: An output writer that writes each example's input/output as a separate record to a sink.
- signature: `class InputOutputRecordWriter(OutputRecordWriter):`
- members:
  - `_build_record(self, *, input_example: NestedTensor, output_example: NestedTensor)` — [`L235`](../../../../../../raw/code/axlearn/axlearn/common/inference_output.py#L235) — Writes both the input and output for each example.
- uses (calls/refs, reference-scoped): [`NestedTensor`](utils.md#NestedTensor), [`OutputRecordWriter`](inference_output.md#OutputRecordWriter)
- used by: [`OutputRecordWriter`](inference_output.md#OutputRecordWriter), [`_build_record`](inference_output.md#OutputRecordWriter._build_record)

### `JsonlExampleRecordSink`  ·  implements/extends BaseRecordSink
- def: [`axlearn/common/inference_output.py:131`](../../../../../../raw/code/axlearn/axlearn/common/inference_output.py#L131)
- doc: A sink that writes each example as a record to a JSON Lines file.
- signature: `class JsonlExampleRecordSink(BaseRecordSink):`
- members:
  - `flush(self)` — [`L166`](../../../../../../raw/code/axlearn/axlearn/common/inference_output.py#L166)
  - `write(self, record: NestedTensor)` — [`L162`](../../../../../../raw/code/axlearn/axlearn/common/inference_output.py#L162)
- protocol/private: `__init__`[`L146`](../../../../../../raw/code/axlearn/axlearn/common/inference_output.py#L146), `_writer`[`L160`](../../../../../../raw/code/axlearn/axlearn/common/inference_output.py#L160)
- uses (calls/refs, reference-scoped): [`config`](config.md#Configurable.config), [`NestedTensor`](utils.md#NestedTensor), [`Module`](module.md#Module), [`flatten_items`](utils.md#flatten_items), [`__init__`](module.md#Module.__init__), [`get_data_dir`](utils.md#get_data_dir), [`BaseRecordSink`](inference_output.md#BaseRecordSink), [`Config`](inference_output.md#JsonlExampleRecordSink.Config), [`output_path`](inference_output.md#JsonlExampleRecordSink.Config.output_path), [`_json_feature`](inference_output.md#_json_feature)
- used by: [`BaseRecordSink`](inference_output.md#BaseRecordSink), [`write`](inference_output.md#BaseRecordSink.write), [`flush`](inference_output.md#BaseRecordSink.flush)

### `OutputRecordWriter`  ·  implements/extends BaseOutputWriter
- def: [`axlearn/common/inference_output.py:170`](../../../../../../raw/code/axlearn/axlearn/common/inference_output.py#L170)
- doc: An output writer that writes each output example as a separate record to a given sink.
- signature: `class OutputRecordWriter(BaseOutputWriter):`
- members:
  - `_build_record(self, *, input_example: NestedTensor, output_example: NestedTensor)` — [`L211`](../../../../../../raw/code/axlearn/axlearn/common/inference_output.py#L211) — Writes only the output example by default.
  - `flush(self)` — [`L218`](../../../../../../raw/code/axlearn/axlearn/common/inference_output.py#L218)
  - `write(self, *, input_batch: NestedTensor, output_batch: NestedTensor)` — [`L194`](../../../../../../raw/code/axlearn/axlearn/common/inference_output.py#L194) — Writes records extracted from the given input/output batch pair.
- protocol/private: `__init__`[`L184`](../../../../../../raw/code/axlearn/axlearn/common/inference_output.py#L184)
- uses (calls/refs, reference-scoped): [`config`](config.md#Configurable.config), [`NestedTensor`](utils.md#NestedTensor), [`Module`](module.md#Module), [`_add_child`](module.md#Module._add_child), [`__init__`](module.md#Module.__init__), [`sink`](inference_output.md#OutputRecordWriter.Config.sink), [`BaseOutputWriter`](inference_output.md#BaseOutputWriter), [`Config`](inference_output.md#OutputRecordWriter.Config), [`InputOutputRecordWriter`](inference_output.md#InputOutputRecordWriter), [`_build_record`](inference_output.md#InputOutputRecordWriter._build_record)
- used by: [`BaseOutputWriter`](inference_output.md#BaseOutputWriter), [`write`](inference_output.md#BaseOutputWriter.write), [`InputOutputRecordWriter`](inference_output.md#InputOutputRecordWriter), [`flush`](inference_output.md#BaseOutputWriter.flush)

### `TfExampleRecordSink`  ·  implements/extends BaseRecordSink
- def: [`axlearn/common/inference_output.py:90`](../../../../../../raw/code/axlearn/axlearn/common/inference_output.py#L90)
- doc: A sink that writes each example as a record to a TF record file.
- signature: `class TfExampleRecordSink(BaseRecordSink):`
- members:
  - `flush(self)` — [`L127`](../../../../../../raw/code/axlearn/axlearn/common/inference_output.py#L127)
  - `write(self, record: NestedTensor)` — [`L121`](../../../../../../raw/code/axlearn/axlearn/common/inference_output.py#L121)
- protocol/private: `__init__`[`L104`](../../../../../../raw/code/axlearn/axlearn/common/inference_output.py#L104), `_writer`[`L119`](../../../../../../raw/code/axlearn/axlearn/common/inference_output.py#L119)
- uses (calls/refs, reference-scoped): [`config`](config.md#Configurable.config), [`NestedTensor`](utils.md#NestedTensor), [`Module`](module.md#Module), [`flatten_items`](utils.md#flatten_items), [`__init__`](module.md#Module.__init__), [`get_data_dir`](utils.md#get_data_dir), [`BaseRecordSink`](inference_output.md#BaseRecordSink), [`Config`](inference_output.md#TfExampleRecordSink.Config), [`output_path`](inference_output.md#TfExampleRecordSink.Config.output_path), [`_tf_feature`](inference_output.md#_tf_feature)
- used by: [`sink`](inference_output.md#OutputRecordWriter.Config.sink), [`BaseRecordSink`](inference_output.md#BaseRecordSink), [`write`](inference_output.md#BaseRecordSink.write), [`flush`](inference_output.md#BaseRecordSink.flush)

## Functions
- `_json_feature(value: Union[Tensor, tf.Tensor])` — [`L75`](../../../../../../raw/code/axlearn/axlearn/common/inference_output.py#L75)
- `_tf_feature(value: Union[Tensor, tf.Tensor])` — [`L58`](../../../../../../raw/code/axlearn/axlearn/common/inference_output.py#L58)

