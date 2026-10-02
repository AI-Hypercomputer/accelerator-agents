---
title: 'Module: axlearn/common/rnn.py'
type: catalog
provenance: extracted
module: axlearn/common/rnn.py
status: fresh
symbol_base: scip-python python axlearn 0.0.0 `axlearn.common.rnn`/
symbols:
  BaseRNNCell: BaseRNNCell#
  BaseRNNCell.Config: BaseRNNCell#Config#
  BaseRNNCell.Config.input_dim: BaseRNNCell#Config#input_dim.
  BaseRNNCell.Config.output_dim: BaseRNNCell#Config#output_dim.
  BaseRNNCell.output_dim: BaseRNNCell#output_dim().
  BaseRNNCell.init_states: BaseRNNCell#init_states().
  BaseRNNCell.extend_step: BaseRNNCell#extend_step().
  BaseRNNCell.forward: BaseRNNCell#forward().
  BaseRNNCell._batch_size: BaseRNNCell#_batch_size().
  BaseRNNCell._seq_len: BaseRNNCell#_seq_len().
  LSTMCell: LSTMCell#
  LSTMCell.Config: LSTMCell#Config#
  LSTMCell.Config.hidden_dim: LSTMCell#Config#hidden_dim.
  LSTMCell.Config.input_proj: LSTMCell#Config#input_proj.
  LSTMCell.Config.output_proj: LSTMCell#Config#output_proj.
  LSTMCell.Config.norm: LSTMCell#Config#norm.
  LSTMCell.Config.max_cell_value: LSTMCell#Config#max_cell_value.
  LSTMCell.hidden_dim: LSTMCell#hidden_dim().
  LSTMCell.__init__: LSTMCell#__init__().
  LSTMCell.init_states: LSTMCell#init_states().
  LSTMCell.extend_step: LSTMCell#extend_step().
  LSTMCell._batch_size: LSTMCell#_batch_size().
  LSTMCell._seq_len: LSTMCell#_seq_len().
  StackedRNNLayer: StackedRNNLayer#
  StackedRNNLayer.Config: StackedRNNLayer#Config#
  StackedRNNLayer.Config.layers: StackedRNNLayer#Config#layers.
  StackedRNNLayer.__init__: StackedRNNLayer#__init__().
  StackedRNNLayer.initialize_parameters_recursively: StackedRNNLayer#initialize_parameters_recursively().
  StackedRNNLayer.init_states: StackedRNNLayer#init_states().
  StackedRNNLayer.extend_step: StackedRNNLayer#extend_step().
  StackedRNNLayer._batch_size: StackedRNNLayer#_batch_size().
  StackedRNNLayer._seq_len: StackedRNNLayer#_seq_len().
  StackedRNNLayer.output_dim: StackedRNNLayer#output_dim().
  _RNNRepeat: _RNNRepeat#
  _RNNRepeat.init_states: _RNNRepeat#init_states().
  _RNNRepeat.extend_step: _RNNRepeat#extend_step().
  RepeatedRNNLayer: RepeatedRNNLayer#
  RepeatedRNNLayer.Config: RepeatedRNNLayer#Config#
  RepeatedRNNLayer.Config.num_layers: RepeatedRNNLayer#Config#num_layers.
  RepeatedRNNLayer.Config.layer: RepeatedRNNLayer#Config#layer.
  RepeatedRNNLayer.__init__: RepeatedRNNLayer#__init__().
  RepeatedRNNLayer._batch_size: RepeatedRNNLayer#_batch_size().
  RepeatedRNNLayer._seq_len: RepeatedRNNLayer#_seq_len().
  RepeatedRNNLayer.initialize_parameters_recursively: RepeatedRNNLayer#initialize_parameters_recursively().
  RepeatedRNNLayer.init_states: RepeatedRNNLayer#init_states().
  RepeatedRNNLayer.extend_step: RepeatedRNNLayer#extend_step().
  IdentityCell: IdentityCell#
  IdentityCell.__init__: IdentityCell#__init__().
  IdentityCell.init_states: IdentityCell#init_states().
  IdentityCell.extend_step: IdentityCell#extend_step().
  IdentityCell._batch_size: IdentityCell#_batch_size().
  IdentityCell._seq_len: IdentityCell#_seq_len().
---
# Module: [`axlearn/common/rnn.py`](../../../../../../raw/code/axlearn/axlearn/common/rnn.py)

## Classes
### `BaseRNNCell`
- def: [`axlearn/common/rnn.py:30`](../../../../../../raw/code/axlearn/axlearn/common/rnn.py#L30)
- doc: An abstract class to define the common interface of all RNN cell layers, including:
- signature: `class BaseRNNCell(BaseLayer):`
- members:
  - `_batch_size(self, inputs: Tensor)` — [`L116`](../../../../../../raw/code/axlearn/axlearn/common/rnn.py#L116) — Infers batch size from `inputs`.
  - `_seq_len(self, inputs: Tensor)` — [`L121`](../../../../../../raw/code/axlearn/axlearn/common/rnn.py#L121) — Infers sequence length from `inputs`.
  - `extend_step(self, *, cached_states: Nested[Tensor], data: Tensor)` — [`L53`](../../../../../../raw/code/axlearn/axlearn/common/rnn.py#L53) — Computes the outputs and state updates for one step.
  - `forward(self, time_major_inputs: Tensor)` — [`L72`](../../../../../../raw/code/axlearn/axlearn/common/rnn.py#L72) — Computes RNN outputs given full-sequence inputs.
  - `init_states(self, *, batch_size: int)` — [`L49`](../../../../../../raw/code/axlearn/axlearn/common/rnn.py#L49) — Returns the initial states, to be used by `extend_step`.
  - `output_dim(self)` — [`L44`](../../../../../../raw/code/axlearn/axlearn/common/rnn.py#L44)

### `Config`
- def: [`axlearn/common/rnn.py:387`](../../../../../../raw/code/axlearn/axlearn/common/rnn.py#L387)
- doc: Configures RepeatedRNNLayer.
- signature: `class Config(BaseRNNCell.Config):`
- members:
  - `hidden_dim` — [`L147`](../../../../../../raw/code/axlearn/axlearn/common/rnn.py#L147)
  - `input_dim` — [`L40`](../../../../../../raw/code/axlearn/axlearn/common/rnn.py#L40)
  - `input_proj` — [`L148`](../../../../../../raw/code/axlearn/axlearn/common/rnn.py#L148)
  - `layer` — [`L393`](../../../../../../raw/code/axlearn/axlearn/common/rnn.py#L393)
  - `layers` — [`L245`](../../../../../../raw/code/axlearn/axlearn/common/rnn.py#L245)
  - `max_cell_value` — [`L151`](../../../../../../raw/code/axlearn/axlearn/common/rnn.py#L151)
  - `norm` — [`L150`](../../../../../../raw/code/axlearn/axlearn/common/rnn.py#L150)
  - `num_layers` — [`L391`](../../../../../../raw/code/axlearn/axlearn/common/rnn.py#L391)
  - `output_dim` — [`L41`](../../../../../../raw/code/axlearn/axlearn/common/rnn.py#L41)
  - `output_proj` — [`L149`](../../../../../../raw/code/axlearn/axlearn/common/rnn.py#L149)

### `IdentityCell`
- def: [`axlearn/common/rnn.py:443`](../../../../../../raw/code/axlearn/axlearn/common/rnn.py#L443)
- doc: Identity RNN cell.
- signature: `class IdentityCell(BaseRNNCell):`
- members:
  - `extend_step(self, *, cached_states: Nested[Tensor], data: Tensor)` — [`L460`](../../../../../../raw/code/axlearn/axlearn/common/rnn.py#L460)
  - `init_states(self, *, batch_size: int)` — [`L456`](../../../../../../raw/code/axlearn/axlearn/common/rnn.py#L456) — Returns the initial states, to be used by `extend_step`.
- protocol/private: `__init__`[`L446`](../../../../../../raw/code/axlearn/axlearn/common/rnn.py#L446), `_batch_size`[`L470`](../../../../../../raw/code/axlearn/axlearn/common/rnn.py#L470), `_seq_len`[`L475`](../../../../../../raw/code/axlearn/axlearn/common/rnn.py#L475)

### `LSTMCell`
- def: [`axlearn/common/rnn.py:126`](../../../../../../raw/code/axlearn/axlearn/common/rnn.py#L126)
- doc: Implements a variant of LSTM that supports normalization and output projection.
- signature: `class LSTMCell(BaseRNNCell):`
- members:
  - `extend_step(self, *, cached_states: Nested[Tensor], data: Tensor)` — [`L192`](../../../../../../raw/code/axlearn/axlearn/common/rnn.py#L192)
  - `hidden_dim(self)` — [`L154`](../../../../../../raw/code/axlearn/axlearn/common/rnn.py#L154)
  - `init_states(self, *, batch_size: int)` — [`L182`](../../../../../../raw/code/axlearn/axlearn/common/rnn.py#L182)
- protocol/private: `__init__`[`L158`](../../../../../../raw/code/axlearn/axlearn/common/rnn.py#L158), `_batch_size`[`L225`](../../../../../../raw/code/axlearn/axlearn/common/rnn.py#L225), `_seq_len`[`L231`](../../../../../../raw/code/axlearn/axlearn/common/rnn.py#L231)

### `RepeatedRNNLayer`
- def: [`axlearn/common/rnn.py:383`](../../../../../../raw/code/axlearn/axlearn/common/rnn.py#L383)
- doc: Repeated RNN layer.
- signature: `class RepeatedRNNLayer(BaseRNNCell):`
- members:
  - `extend_step(self, *, cached_states: Nested[Tensor], data: Tensor)` — [`L434`](../../../../../../raw/code/axlearn/axlearn/common/rnn.py#L434)
  - `init_states(self, *, batch_size: int)` — [`L431`](../../../../../../raw/code/axlearn/axlearn/common/rnn.py#L431)
  - `initialize_parameters_recursively(self, prng_key: Tensor, *, prebuilt: Optional[Nested[Tensor]]=None)` — [`L416`](../../../../../../raw/code/axlearn/axlearn/common/rnn.py#L416)
- protocol/private: `__init__`[`L395`](../../../../../../raw/code/axlearn/axlearn/common/rnn.py#L395), `_batch_size`[`L408`](../../../../../../raw/code/axlearn/axlearn/common/rnn.py#L408), `_seq_len`[`L412`](../../../../../../raw/code/axlearn/axlearn/common/rnn.py#L412)

### `StackedRNNLayer`
- def: [`axlearn/common/rnn.py:237`](../../../../../../raw/code/axlearn/axlearn/common/rnn.py#L237)
- doc: Stacked RNN layer.
- signature: `class StackedRNNLayer(BaseRNNCell):`
- members:
  - `extend_step(self, *, cached_states: list[Nested[Tensor]], data: Tensor)` — [`L297`](../../../../../../raw/code/axlearn/axlearn/common/rnn.py#L297) — Computes the outputs and all layers state updates for one step.
  - `init_states(self, *, batch_size: int)` — [`L292`](../../../../../../raw/code/axlearn/axlearn/common/rnn.py#L292) — Returns a list of initial step states from all layers.
  - `initialize_parameters_recursively(self, prng_key: Tensor, *, prebuilt: Optional[Nested[Tensor]]=None)` — [`L275`](../../../../../../raw/code/axlearn/axlearn/common/rnn.py#L275)
  - `output_dim(self)` — [`L331`](../../../../../../raw/code/axlearn/axlearn/common/rnn.py#L331)
- protocol/private: `__init__`[`L247`](../../../../../../raw/code/axlearn/axlearn/common/rnn.py#L247), `_batch_size`[`L322`](../../../../../../raw/code/axlearn/axlearn/common/rnn.py#L322), `_seq_len`[`L326`](../../../../../../raw/code/axlearn/axlearn/common/rnn.py#L326)

### `_RNNRepeat`
- def: [`axlearn/common/rnn.py:343`](../../../../../../raw/code/axlearn/axlearn/common/rnn.py#L343)
- doc: A Repeat layer with layer = children class of BaseRNNCell.
- signature: `class _RNNRepeat(Repeat):`
- members:
  - `extend_step(self, *, cached_states: Nested[Tensor], data: Tensor)` — [`L356`](../../../../../../raw/code/axlearn/axlearn/common/rnn.py#L356) — Computes the outputs and state updates for one step for all layers.
  - `init_states(self, *, batch_size: int)` — [`L347`](../../../../../../raw/code/axlearn/axlearn/common/rnn.py#L347) — Returns the initial states of all layers.

