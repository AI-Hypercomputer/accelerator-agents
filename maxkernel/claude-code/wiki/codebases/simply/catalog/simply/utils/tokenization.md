---
title: 'Module: simply/utils/tokenization.py'
type: catalog
provenance: extracted
module: simply/utils/tokenization.py
status: fresh
symbol_base: scip-python python simply 0.0.0 `simply.utils.tokenization`/
symbols:
  SimplyVocab: SimplyVocab#
  TestVocab: TestVocab#
  TokenizerRegistry: TokenizerRegistry#
  SimplyVocab.encode: SimplyVocab#encode().
  SimplyVocab.decode: SimplyVocab#decode().
  SimplySentencePieceVocab.__init__: SimplySentencePieceVocab#__init__().
  HuggingFaceVocab.get_token_id: HuggingFaceVocab#get_token_id().
  TestVocab.__init__: TestVocab#__init__().
  TestVocab.encode: TestVocab#encode().
  SimplySentencePieceVocab._sp: SimplySentencePieceVocab#_sp.
  HuggingFaceVocab.tokenizer: HuggingFaceVocab#tokenizer().
  SimplyVocab.bos_id: SimplyVocab#bos_id.
  TestVocab._rev_vocab_dict: TestVocab#_rev_vocab_dict.
  SimplySentencePieceVocab: SimplySentencePieceVocab#
  HuggingFaceVocab: HuggingFaceVocab#
  HuggingFaceVocab.tokenizer_config: HuggingFaceVocab#tokenizer_config().
  SimplyVocab.pad_id: SimplyVocab#pad_id.
  SimplyVocab.eos_id: SimplyVocab#eos_id.
  TestVocab.decode: TestVocab#decode().
  SimplySentencePieceVocab.encode: SimplySentencePieceVocab#encode().
  SimplySentencePieceVocab.decode: SimplySentencePieceVocab#decode().
  HuggingFaceVocab.bos_id: HuggingFaceVocab#bos_id().
  HuggingFaceVocab.eos_id: HuggingFaceVocab#eos_id().
  HuggingFaceVocab.pad_id: HuggingFaceVocab#pad_id().
  HuggingFaceVocab.encode: HuggingFaceVocab#encode().
  HuggingFaceVocab.decode: HuggingFaceVocab#decode().
  TestVocab._vocab_dict: TestVocab#_vocab_dict.
  HuggingFaceVocab.vocab_path: HuggingFaceVocab#vocab_path.
  TestVocab.unk_id: TestVocab#unk_id.
  TokenizerRegistry.namespace: TokenizerRegistry#namespace.
  HuggingFaceVocab.__init__: HuggingFaceVocab#__init__().
---
# Module: [`simply/utils/tokenization.py`](../../../../../../raw/code/simply/simply/utils/tokenization.py)

## Classes
### `HuggingFaceVocab`  ·  implements/extends SimplyVocab
- def: [`simply/utils/tokenization.py:87`](../../../../../../raw/code/simply/simply/utils/tokenization.py#L87) — documented in [simply-utils-tokenization](../../../concepts/simply-utils-tokenization.md)
- doc: Generic class for HuggingFace vocab.
- signature: `class HuggingFaceVocab(SimplyVocab[str]):`
- members:
  - `bos_id(self)` — [`L117`](../../../../../../raw/code/simply/simply/utils/tokenization.py#L117)
  - `decode(self, token_ids: list[int])` — [`L132`](../../../../../../raw/code/simply/simply/utils/tokenization.py#L132)
  - `encode(self, text: str)` — [`L128`](../../../../../../raw/code/simply/simply/utils/tokenization.py#L128) — documented in [simply-utils-tokenization](../../../concepts/simply-utils-tokenization.md)
  - `eos_id(self)` — [`L121`](../../../../../../raw/code/simply/simply/utils/tokenization.py#L121)
  - `get_token_id(self, name: str)` — [`L106`](../../../../../../raw/code/simply/simply/utils/tokenization.py#L106)
  - `pad_id(self)` — [`L125`](../../../../../../raw/code/simply/simply/utils/tokenization.py#L125)
  - `tokenizer(self)` — [`L94`](../../../../../../raw/code/simply/simply/utils/tokenization.py#L94) — documented in [simply-utils-tokenization](../../../concepts/simply-utils-tokenization.md)
  - `tokenizer_config(self)` — [`L101`](../../../../../../raw/code/simply/simply/utils/tokenization.py#L101)
  - `vocab_path` — [`L91`](../../../../../../raw/code/simply/simply/utils/tokenization.py#L91)
- protocol/private: `__init__`[`L90`](../../../../../../raw/code/simply/simply/utils/tokenization.py#L90)
- uses (calls/refs, reference-scoped): [`SimplyVocab`](tokenization.md#SimplyVocab)
- used by: [`SimplyVocab`](tokenization.md#SimplyVocab), [`decode`](tokenization.md#SimplyVocab.decode), [`encode`](tokenization.md#SimplyVocab.encode), [`register_hf_vocabs`](../data_lib.md#register_hf_vocabs)

### `SimplySentencePieceVocab`  ·  implements/extends SimplyVocab
- def: [`simply/utils/tokenization.py:68`](../../../../../../raw/code/simply/simply/utils/tokenization.py#L68) — documented in [simply-utils-tokenization](../../../concepts/simply-utils-tokenization.md)
- doc: Wrapper around sentencepiece.SentencePieceProcessor.
- signature: `class SimplySentencePieceVocab(SimplyVocab[str]):`
- members:
  - `decode(self, token_ids: list[int])` — [`L83`](../../../../../../raw/code/simply/simply/utils/tokenization.py#L83)
  - `encode(self, text: str)` — [`L80`](../../../../../../raw/code/simply/simply/utils/tokenization.py#L80) — documented in [simply-utils-tokenization](../../../concepts/simply-utils-tokenization.md)
- protocol/private: `__init__`[`L71`](../../../../../../raw/code/simply/simply/utils/tokenization.py#L71), `_sp`[`L72`](../../../../../../raw/code/simply/simply/utils/tokenization.py#L72)
- uses (calls/refs, reference-scoped): [`SimplyVocab`](tokenization.md#SimplyVocab), [`bos_id`](tokenization.md#SimplyVocab.bos_id), [`eos_id`](tokenization.md#SimplyVocab.eos_id), [`pad_id`](tokenization.md#SimplyVocab.pad_id)
- used by: [`register_spm_vocabs`](../data_lib.md#register_spm_vocabs), [`SimplyVocab`](tokenization.md#SimplyVocab), [`decode`](tokenization.md#SimplyVocab.decode), [`encode`](tokenization.md#SimplyVocab.encode)

### `SimplyVocab`  ·  implements/extends Generic, Protocol
- def: [`simply/utils/tokenization.py:35`](../../../../../../raw/code/simply/simply/utils/tokenization.py#L35) — documented in [simply-utils-tokenization](../../../concepts/simply-utils-tokenization.md)
- signature: `class SimplyVocab(Protocol, Generic[common.RawT]):`
- members:
  - `decode(self, token_ids: list[int])` — [`L43`](../../../../../../raw/code/simply/simply/utils/tokenization.py#L43)
  - `encode(self, text: common.RawT)` — [`L40`](../../../../../../raw/code/simply/simply/utils/tokenization.py#L40) — documented in [simply-utils-tokenization](../../../concepts/simply-utils-tokenization.md)
  - `bos_id` — [`L37`](../../../../../../raw/code/simply/simply/utils/tokenization.py#L37)
  - `eos_id` — [`L38`](../../../../../../raw/code/simply/simply/utils/tokenization.py#L38)
  - `pad_id` — [`L36`](../../../../../../raw/code/simply/simply/utils/tokenization.py#L36)
- uses (calls/refs, reference-scoped): [`TestVocab`](tokenization.md#TestVocab), [`encode`](../model_lib_test.md#TestNpArrayQuantizer.encode), [`encode`](tokenization.md#TestVocab.encode), [`RawT`](common.md#RawT), [`decode`](../model_lib_test.md#TestNpArrayQuantizer.decode), [`HuggingFaceVocab`](tokenization.md#HuggingFaceVocab), [`SimplySentencePieceVocab`](tokenization.md#SimplySentencePieceVocab), [`TestNpArrayQuantizer`](../model_lib_test.md#TestNpArrayQuantizer), [`decode`](tokenization.md#HuggingFaceVocab.decode), [`decode`](tokenization.md#SimplySentencePieceVocab.decode), [`decode`](tokenization.md#TestVocab.decode), [`encode`](tokenization.md#HuggingFaceVocab.encode), [`encode`](tokenization.md#SimplySentencePieceVocab.encode)
- used by: [`test_lm_interface_generate_with_local_state`](../model_lib_test.md#ModelLibTest.test_lm_interface_generate_with_local_state), [`test_lm_interface_score_tokens`](../model_lib_test.md#ModelLibTest.test_lm_interface_score_tokens), [`TestVocab`](tokenization.md#TestVocab), [`__init__`](../model_lib.md#LMInterface.__init__), [`__init__`](tokenization.md#SimplySentencePieceVocab.__init__), [`__init__`](../model_lib_test.md#TestNpArrayQuantizer.__init__), [`__init__`](tokenization.md#TestVocab.__init__), [`HuggingFaceVocab`](tokenization.md#HuggingFaceVocab), [`SimplySentencePieceVocab`](tokenization.md#SimplySentencePieceVocab), [`TestNpArrayQuantizer`](../model_lib_test.md#TestNpArrayQuantizer)

### `TestVocab`  ·  implements/extends SimplyVocab
- def: [`simply/utils/tokenization.py:47`](../../../../../../raw/code/simply/simply/utils/tokenization.py#L47) — documented in [simply-utils-tokenization](../../../concepts/simply-utils-tokenization.md)
- doc: Test vocab.
- signature: `class TestVocab(SimplyVocab[str]):`
- members:
  - `decode(self, token_ids: list[int])` — [`L64`](../../../../../../raw/code/simply/simply/utils/tokenization.py#L64)
  - `encode(self, text: str)` — [`L61`](../../../../../../raw/code/simply/simply/utils/tokenization.py#L61) — documented in [simply-utils-tokenization](../../../concepts/simply-utils-tokenization.md)
  - `unk_id` — [`L54`](../../../../../../raw/code/simply/simply/utils/tokenization.py#L54)
- protocol/private: `__init__`[`L50`](../../../../../../raw/code/simply/simply/utils/tokenization.py#L50), `_rev_vocab_dict`[`L59`](../../../../../../raw/code/simply/simply/utils/tokenization.py#L59), `_vocab_dict`[`L56`](../../../../../../raw/code/simply/simply/utils/tokenization.py#L56)
- uses (calls/refs, reference-scoped): [`SimplyVocab`](tokenization.md#SimplyVocab), [`bos_id`](tokenization.md#SimplyVocab.bos_id), [`eos_id`](tokenization.md#SimplyVocab.eos_id), [`pad_id`](tokenization.md#SimplyVocab.pad_id)
- used by: [`test_batch_sampling_with_max_seq_len`](../model_lib_test.md#ModelLibTest.test_batch_sampling_with_max_seq_len), [`test_sampling_token_scores`](../model_lib_test.md#ModelLibTest.test_sampling_token_scores), [`test_batch_sampling`](../model_lib_test.md#ModelLibTest.test_batch_sampling), [`test_lm_interface_generate_with_local_state`](../model_lib_test.md#ModelLibTest.test_lm_interface_generate_with_local_state), [`test_continue_decoding`](../model_lib_test.md#ModelLibTest.test_continue_decoding), [`test_lm_interface_generate_without_scoring`](../model_lib_test.md#ModelLibTest.test_lm_interface_generate_without_scoring), [`test_sampling_max_decode_steps_equals_prefill_size`](../model_lib_test.md#ModelLibTest.test_sampling_max_decode_steps_equals_prefill_size), [`test_sampling_output_logprobs`](../model_lib_test.md#ModelLibTest.test_sampling_output_logprobs), [`test_lm_interface_batch`](../model_lib_test.md#ModelLibTest.test_lm_interface_batch), [`test_lm_interface_generate`](../model_lib_test.md#ModelLibTest.test_lm_interface_generate), [`test_lm_interface_score`](../model_lib_test.md#ModelLibTest.test_lm_interface_score), [`test_lm_interface_score_tokens`](../model_lib_test.md#ModelLibTest.test_lm_interface_score_tokens), [`SimplyVocab`](tokenization.md#SimplyVocab), [`decode`](tokenization.md#SimplyVocab.decode), [`encode`](tokenization.md#SimplyVocab.encode)

### `TokenizerRegistry`  ·  implements/extends RootRegistry
- def: [`simply/utils/tokenization.py:29`](../../../../../../raw/code/simply/simply/utils/tokenization.py#L29) — documented in [simply-utils-tokenization](../../../concepts/simply-utils-tokenization.md)
- doc: Tokenizer registry.
- signature: `class TokenizerRegistry(registry.RootRegistry):`
- members:
  - `namespace` — [`L32`](../../../../../../raw/code/simply/simply/utils/tokenization.py#L32)
- uses (calls/refs, reference-scoped): [`RootRegistry`](registry.md#RootRegistry)
- used by: [`run_experiment`](../rl_lib.md#run_experiment), [`main`](../eval/decode_eval.md#main), [`RootRegistry`](registry.md#RootRegistry), [`input_processor`](../serving/page_batcher.md#Batcher.input_processor), [`input_processor`](../serving/vanilla_server.md#Batcher.input_processor), [`register_spm_vocabs`](../data_lib.md#register_spm_vocabs), [`_mock_vocab`](../rl_lib_test.md#RunExperimentTest._mock_vocab), [`register_hf_vocabs`](../data_lib.md#register_hf_vocabs), [`_get_tokenizer`](../data_lib.md#_get_tokenizer), [`tearDown`](../rl_lib_test.md#RunExperimentTest.tearDown)

