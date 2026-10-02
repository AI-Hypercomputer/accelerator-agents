---
title: 'Module: simply/rl_lib.py'
type: catalog
provenance: extracted
module: simply/rl_lib.py
status: fresh
symbol_base: scip-python python simply 0.0.0 `simply.rl_lib`/
symbols:
  run_experiment: run_experiment().
  create_train_batch: create_train_batch().
  compute_ppo_loss: compute_ppo_loss().
  compute_stats: compute_stats().
  Array: Array.
  RLTrainingExampleBatch.pad_sequences: RLTrainingExampleBatch#pad_sequences().
  RLTrainingExampleBatch.normalize_reward: RLTrainingExampleBatch#normalize_reward().
  np_safe_mean: np_safe_mean().
  compute_logprobs._compute_logprobs: compute_logprobs()._compute_logprobs().
  compute_return: compute_return().
  np_safe_weighted_mean: np_safe_weighted_mean().
  RewardNormalizer.Global: RewardNormalizer#Global#
  RewardNormalizer.ByGroup: RewardNormalizer#ByGroup#
  compute_logprobs: compute_logprobs().
  RLTrainingExampleBatch.input_tokens: RLTrainingExampleBatch#input_tokens.
  RLTrainingExampleBatch.is_valid_for_training: RLTrainingExampleBatch#is_valid_for_training.
  RewardedSample: RewardedSample#
  RewardedSample.sampling_input: RewardedSample#sampling_input.
  RewardedSample.sampling_output: RewardedSample#sampling_output.
  RewardNormalizer.ByGroup.normalize_by_group: RewardNormalizer#ByGroup#normalize_by_group().
  run_experiment.train_one_step_fn: run_experiment().train_one_step_fn().
  PyTree: PyTree.
  RewardNormalizerRegistry: RewardNormalizerRegistry#
  RLTrainingExampleBatch.target_tokens: RLTrainingExampleBatch#target_tokens.
  RLTrainingExampleBatch.logprobs: RLTrainingExampleBatch#logprobs.
  RLTrainingExampleBatch.target_mask: RLTrainingExampleBatch#target_mask.
  RLTrainingExampleBatch.answer_mask: RLTrainingExampleBatch#answer_mask.
  RLTrainingExampleBatch.reward: RLTrainingExampleBatch#reward.
  np_safe_std: np_safe_std().
  RewardNormalizer.Global.normalize: RewardNormalizer#Global#normalize().
  RewardedSample.reward: RewardedSample#reward.
  prepare_params_for_decoding: prepare_params_for_decoding().
  RLTrainingExampleBatch.in_batch_example_id: RLTrainingExampleBatch#in_batch_example_id.
  RLTrainingExampleBatch.ref_logprobs: RLTrainingExampleBatch#ref_logprobs.
  RLTrainingExampleBatch.extra_inputs: RLTrainingExampleBatch#extra_inputs.
  RLTrainingExampleBatch: RLTrainingExampleBatch#
  RewardedSample.in_batch_example_index: RewardedSample#in_batch_example_index.
  ExperimentHelper: ExperimentHelper.
  RLTrainingExampleBatch.is_correct: RLTrainingExampleBatch#is_correct.
  RLTrainingExampleBatch.default_pytree_shape: RLTrainingExampleBatch#default_pytree_shape().
  compute_return._update_fn: compute_return()._update_fn().
  RewardNormalizer: RewardNormalizer#
  RewardNormalizer.Base: RewardNormalizer#Base#
  RewardedSample.is_valid_for_training: RewardedSample#is_valid_for_training.
  RewardedSample.correct: RewardedSample#correct.
  Batch: Batch.
  TrainLoopRegistry: TrainLoopRegistry.
  RewardNormalizer.ByGroup.normalize: RewardNormalizer#ByGroup#normalize().
  RLTrainingExampleBatch.batch_size: RLTrainingExampleBatch#batch_size().
  decoding_mesh_context: decoding_mesh_context().
  mesh_in_params: mesh_in_params().
  tree_convert_dtype: tree_convert_dtype().
  RLTrainingExampleBatch.tree_structure: RLTrainingExampleBatch#tree_structure().
  RewardedSample.raw_example: RewardedSample#raw_example.
  RewardedSample.step: RewardedSample#step.
  RewardNormalizer.Base.normalize: RewardNormalizer#Base#normalize().
  RLTrainingExampleBatch.assert_no_nan: RLTrainingExampleBatch#assert_no_nan().
  RewardedSample.reward_result: RewardedSample#reward_result.
  RewardedSample.reward_types: RewardedSample#reward_types.
  RewardedSample.update_with_evaluation_result: RewardedSample#update_with_evaluation_result().
  RewardNormalizerRegistry.namespace: RewardNormalizerRegistry#namespace.
  RewardedSample.raw_evaluation_result: RewardedSample#raw_evaluation_result.
---
# Module: [`simply/rl_lib.py`](../../../../../raw/code/simply/simply/rl_lib.py)

## Classes
### `Base`  ·  implements/extends ABC
- def: [`simply/rl_lib.py:82`](../../../../../raw/code/simply/simply/rl_lib.py#L82)
- signature: `class Base(abc.ABC):`
- members:
  - `normalize(self, rewards: np.ndarray, example_ids: np.ndarray, masks: np.ndarray)` — [`L85`](../../../../../raw/code/simply/simply/rl_lib.py#L85) — Normalizes the rewards given they are grouped by example_ids.
- used by: [`normalize_reward`](rl_lib.md#RLTrainingExampleBatch.normalize_reward), [`ByGroup`](rl_lib.md#RewardNormalizer.ByGroup), [`Global`](rl_lib.md#RewardNormalizer.Global)

### `ByGroup`
- def: [`simply/rl_lib.py:112`](../../../../../raw/code/simply/simply/rl_lib.py#L112)
- signature: `class ByGroup(Base):`
- members:
  - `normalize(self, rewards: np.ndarray, example_ids: np.ndarray, masks: np.ndarray)` — [`L143`](../../../../../raw/code/simply/simply/rl_lib.py#L143)
  - `normalize_by_group(self, rewards: np.ndarray, example_ids: np.ndarray, masks: np.ndarray, std: np.ndarray | None = None)` — [`L114`](../../../../../raw/code/simply/simply/rl_lib.py#L114)
- uses (calls/refs, reference-scoped): [`register`](utils/registry.md#RootRegistry.register), [`np_safe_mean`](rl_lib.md#np_safe_mean), [`RewardNormalizerRegistry`](rl_lib.md#RewardNormalizerRegistry), [`np_safe_std`](rl_lib.md#np_safe_std), [`Base`](rl_lib.md#RewardNormalizer.Base)
- used by: [`test_by_group`](rl_lib_test.md#RewardNormalizerTest.test_by_group)

### `Global`
- def: [`simply/rl_lib.py:102`](../../../../../raw/code/simply/simply/rl_lib.py#L102)
- signature: `class Global(Base):`
- members:
  - `normalize(self, rewards: np.ndarray, example_ids: np.ndarray, masks: np.ndarray)` — [`L104`](../../../../../raw/code/simply/simply/rl_lib.py#L104)
- uses (calls/refs, reference-scoped): [`register`](utils/registry.md#RootRegistry.register), [`np_safe_mean`](rl_lib.md#np_safe_mean), [`RewardNormalizerRegistry`](rl_lib.md#RewardNormalizerRegistry), [`np_safe_std`](rl_lib.md#np_safe_std), [`Base`](rl_lib.md#RewardNormalizer.Base)
- used by: [`test_global`](rl_lib_test.md#RewardNormalizerTest.test_global)

### `RLTrainingExampleBatch`
- def: [`simply/rl_lib.py:151`](../../../../../raw/code/simply/simply/rl_lib.py#L151)
- doc: Batch of examples used in training step of RL.
- signature: `class RLTrainingExampleBatch:`
- members:
  - `assert_no_nan(self)` — [`L196`](../../../../../raw/code/simply/simply/rl_lib.py#L196)
  - `batch_size(self)` — [`L203`](../../../../../raw/code/simply/simply/rl_lib.py#L203)
  - `default_pytree_shape(cls, max_seq_len: int)` — [`L174`](../../../../../raw/code/simply/simply/rl_lib.py#L174) — Returns the default tree shape of a batch example.
  - `normalize_reward(self, normalizer: RewardNormalizer.Base)` — [`L225`](../../../../../raw/code/simply/simply/rl_lib.py#L225)
  - `pad_sequences(self, to_length)` — [`L207`](../../../../../raw/code/simply/simply/rl_lib.py#L207) — documented in [simply-rl_lib](../../concepts/simply-rl_lib.md)
  - `tree_structure(self)` — [`L191`](../../../../../raw/code/simply/simply/rl_lib.py#L191)
  - `answer_mask` — [`L165`](../../../../../raw/code/simply/simply/rl_lib.py#L165)
  - `extra_inputs` — [`L171`](../../../../../raw/code/simply/simply/rl_lib.py#L171)
  - `in_batch_example_id` — [`L166`](../../../../../raw/code/simply/simply/rl_lib.py#L166)
  - `input_tokens` — [`L158`](../../../../../raw/code/simply/simply/rl_lib.py#L158)
  - `is_correct` — [`L168`](../../../../../raw/code/simply/simply/rl_lib.py#L168)
  - `is_valid_for_training` — [`L169`](../../../../../raw/code/simply/simply/rl_lib.py#L169)
  - `logprobs` — [`L160`](../../../../../raw/code/simply/simply/rl_lib.py#L160)
  - `ref_logprobs` — [`L170`](../../../../../raw/code/simply/simply/rl_lib.py#L170)
  - `reward` — [`L167`](../../../../../raw/code/simply/simply/rl_lib.py#L167)
  - `target_mask` — [`L163`](../../../../../raw/code/simply/simply/rl_lib.py#L163)
  - `target_tokens` — [`L159`](../../../../../raw/code/simply/simply/rl_lib.py#L159)
- uses (calls/refs, reference-scoped): [`Array`](rl_lib.md#Array), [`pad_to_along_axis`](model_lib.md#pad_to_along_axis), [`PyTree`](rl_lib.md#PyTree), [`Base`](rl_lib.md#RewardNormalizer.Base), [`RewardNormalizer`](rl_lib.md#RewardNormalizer), [`normalize`](rl_lib.md#RewardNormalizer.Base.normalize)
- used by: [`run_experiment`](rl_lib.md#run_experiment), [`create_train_batch`](rl_lib.md#create_train_batch), [`compute_ppo_loss`](rl_lib.md#compute_ppo_loss)

### `RewardNormalizer`
- def: [`simply/rl_lib.py:80`](../../../../../raw/code/simply/simply/rl_lib.py#L80)
- signature: `class RewardNormalizer:`
- used by: [`normalize_reward`](rl_lib.md#RLTrainingExampleBatch.normalize_reward), [`test_by_group`](rl_lib_test.md#RewardNormalizerTest.test_by_group), [`test_global`](rl_lib_test.md#RewardNormalizerTest.test_global)

### `RewardNormalizerRegistry`  ·  implements/extends RootRegistry
- def: [`simply/rl_lib.py:76`](../../../../../raw/code/simply/simply/rl_lib.py#L76)
- signature: `class RewardNormalizerRegistry(registry.RootRegistry):`
- members:
  - `namespace` — [`L77`](../../../../../raw/code/simply/simply/rl_lib.py#L77)
- uses (calls/refs, reference-scoped): [`RootRegistry`](utils/registry.md#RootRegistry)
- used by: [`RootRegistry`](utils/registry.md#RootRegistry), [`create_train_batch`](rl_lib.md#create_train_batch), [`ByGroup`](rl_lib.md#RewardNormalizer.ByGroup), [`Global`](rl_lib.md#RewardNormalizer.Global)

### `RewardedSample`
- def: [`simply/rl_lib.py:235`](../../../../../raw/code/simply/simply/rl_lib.py#L235)
- doc: Example with sample output and reward information.
- signature: `class RewardedSample:`
- members:
  - `update_with_evaluation_result(self, eval_result)` — [`L253`](../../../../../raw/code/simply/simply/rl_lib.py#L253)
  - `correct` — [`L247`](../../../../../raw/code/simply/simply/rl_lib.py#L247)
  - `in_batch_example_index` — [`L240`](../../../../../raw/code/simply/simply/rl_lib.py#L240)
  - `is_valid_for_training` — [`L244`](../../../../../raw/code/simply/simply/rl_lib.py#L244)
  - `raw_evaluation_result` — [`L246`](../../../../../raw/code/simply/simply/rl_lib.py#L246)
  - `raw_example` — [`L238`](../../../../../raw/code/simply/simply/rl_lib.py#L238)
  - `reward` — [`L248`](../../../../../raw/code/simply/simply/rl_lib.py#L248)
  - `reward_result` — [`L250`](../../../../../raw/code/simply/simply/rl_lib.py#L250)
  - `reward_types` — [`L251`](../../../../../raw/code/simply/simply/rl_lib.py#L251)
  - `sampling_input` — [`L241`](../../../../../raw/code/simply/simply/rl_lib.py#L241)
  - `sampling_output` — [`L243`](../../../../../raw/code/simply/simply/rl_lib.py#L243)
  - `step` — [`L239`](../../../../../raw/code/simply/simply/rl_lib.py#L239)
- uses (calls/refs, reference-scoped): [`SamplingOutput`](model_lib.md#SamplingOutput), [`SamplingInput`](utils/sampling_lib.md#SamplingInput)
- used by: [`run_experiment`](rl_lib.md#run_experiment), [`create_train_batch`](rl_lib.md#create_train_batch), [`compute_stats`](rl_lib.md#compute_stats)

## Functions
- `_compute_logprobs(microbatch: dict[str, Array])` — [`L272`](../../../../../raw/code/simply/simply/rl_lib.py#L272)
- `_update_fn(g: Array, r: Array)` — [`L613`](../../../../../raw/code/simply/simply/rl_lib.py#L613)
- `compute_logprobs(model, params: common.PyTree, batch: dict[str, Array], microbatch_size: int | None = None)` — [`L264`](../../../../../raw/code/simply/simply/rl_lib.py#L264) — Computes the logprobs of the decoder tokens.
- `compute_ppo_loss(model, params: common.PyTree, batch: RLTrainingExampleBatch, gamma: float = 1, kl_coeff: float = 0.001, use_grpo: bool = False, ppo_clip_eps_high: float = 0.2, ppo_clip_eps_low: float = 0.2, policy_ratio_cap: float | None = 10, normalize_advantage: bool = True, max_abs_advantage: float | None = 10, use_policy_logp_as_sampler_logp: bool = False)` — [`L628`](../../../../../raw/code/simply/simply/rl_lib.py#L628) — Compute PPO loss. — documented in [simply-rl_lib](../../concepts/simply-rl_lib.md)
- `compute_return(reward: Array, mask: Array, gamma: float = 1)` — [`L606`](../../../../../raw/code/simply/simply/rl_lib.py#L606) — Computes the discounted return.
- `compute_stats(rewarded_completed_batch: Mapping[int, Sequence[RewardedSample]], evaluation: eval_lib.Evaluation)` — [`L337`](../../../../../raw/code/simply/simply/rl_lib.py#L337)
- `create_train_batch(rewarded_batch: Mapping[int, Sequence[RewardedSample]], num_valid_samples: np.ndarray, train_batch_size: int, max_seq_len: int = 1024, normalize_reward_method: str = '', ref_params: PyTree | None = None, compute_logprobs_fn: Callable[..., Array] | None = None)` — [`L482`](../../../../../raw/code/simply/simply/rl_lib.py#L482) — Creates a batch of data for training. — documented in [simply-rl_lib](../../concepts/simply-rl_lib.md)
- `decoding_mesh_context(decoding_mesh_shape: Sequence[int] | None = None, dcn_mesh_shape: Sequence[int] | None = None, axis_names: Sequence[str] | None = None)` — [`L770`](../../../../../raw/code/simply/simply/rl_lib.py#L770)
- `mesh_in_params(params: common.PyTree)` — [`L783`](../../../../../raw/code/simply/simply/rl_lib.py#L783) — Returns the mesh in params.
- `np_safe_mean(x, where)` — [`L321`](../../../../../raw/code/simply/simply/rl_lib.py#L321)
- `np_safe_std(x, where)` — [`L330`](../../../../../raw/code/simply/simply/rl_lib.py#L330)
- `np_safe_weighted_mean(x, w)` — [`L325`](../../../../../raw/code/simply/simply/rl_lib.py#L325)
- `prepare_params_for_decoding(params: common.PyTree, abstract_decoding_params: common.PyTree = None, quant_scheme: str = 'bfloat16')` — [`L798`](../../../../../raw/code/simply/simply/rl_lib.py#L798) — Quantizes params and then reshards them to the current mesh.
- `run_experiment(config, experiment_dir='', mesh_shape=None, dcn_mesh_shape=None, decoding_mesh_shape=None, sharding_config=None, create_dataset=None)` — [`L820`](../../../../../raw/code/simply/simply/rl_lib.py#L820) — documented in [simply-rl_lib](../../concepts/simply-rl_lib.md)
- `train_one_step_fn(state, batch, lr, add_log_info=False)` — [`L909`](../../../../../raw/code/simply/simply/rl_lib.py#L909)
- `tree_convert_dtype(tree: PyTree, dtype: jax.typing.DTypeLike)` — [`L793`](../../../../../raw/code/simply/simply/rl_lib.py#L793)

## Module values
- `Array` — [`L69`](../../../../../raw/code/simply/simply/rl_lib.py#L69)
- `Batch` — [`L70`](../../../../../raw/code/simply/simply/rl_lib.py#L70)
- `ExperimentHelper` — [`L73`](../../../../../raw/code/simply/simply/rl_lib.py#L73)
- `PyTree` — [`L71`](../../../../../raw/code/simply/simply/rl_lib.py#L71)
- `TrainLoopRegistry` — [`L72`](../../../../../raw/code/simply/simply/rl_lib.py#L72)

