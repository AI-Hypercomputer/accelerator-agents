---
title: 'Module: simply/utils/optimizers.py'
type: catalog
provenance: extracted
module: simply/utils/optimizers.py
status: fresh
symbol_base: scip-python python simply 0.0.0 `simply.utils.optimizers`/
symbols:
  LinearWarmupCosineDecay.__call__: LinearWarmupCosineDecay#__call__().
  LinearWarmupConstant: LinearWarmupConstant#
  Muon._orthogonalize_via_newton_schulz: Muon#_orthogonalize_via_newton_schulz().
  Muon._param_update: Muon#_param_update().
  Adam: Adam#
  LinearWarmupCosineDecay: LinearWarmupCosineDecay#
  Array: Array.
  Muon.apply: Muon#apply().
  LinearWarmupConstant.__call__: LinearWarmupConstant#__call__().
  create_lr_schedule: create_lr_schedule().
  Lion.init: Lion#init().
  PyTree: PyTree.
  OptimizerRegistry: OptimizerRegistry#
  SGD: SGD#
  Muon.init_adam: Muon#init_adam().
  get_init_steps: get_init_steps().
  Lion: Lion#
  Muon.init_muon: Muon#init_muon().
  LinearWarmupCosineDecay._finalize: LinearWarmupCosineDecay#_finalize().
  Optimizer.__post_init__: Optimizer#__post_init__().
  Adam.init: Adam#init().
  Adam.apply: Adam#apply().
  Lion.apply: Lion#apply().
  Muon: Muon#
  Muon.init: Muon#init().
  Constant: Constant#
  EarlyStop.__post_init__: EarlyStop#__post_init__().
  Schedule: Schedule#
  replace_fraction: replace_fraction().
  LinearWarmupConstant._finalize: LinearWarmupConstant#_finalize().
  create_lr_schedule_v0: create_lr_schedule_v0().
  ThresholdDef: ThresholdDef.
  ScheduleRegistry: ScheduleRegistry#
  EarlyStopRegistry: EarlyStopRegistry#
  OptimizerRegistry.namespace: OptimizerRegistry#namespace.
  Optimizer: Optimizer#
  Optimizer.apply_updates: Optimizer#apply_updates().
  Constant.warmup_steps: Constant#warmup_steps.
  cosine_decay_schedule: cosine_decay_schedule().
  constant_schedule: constant_schedule().
  AnnotatedArray: AnnotatedArray.
  Muon._mu: Muon#_mu().
  Muon._adam_m: Muon#_adam_m().
  Muon._adam_v: Muon#_adam_v().
  Muon.merge_repeated_dims: Muon#merge_repeated_dims().
  SimpleEarlyStop: SimpleEarlyStop#
  SimpleEarlyStop.thresholds: SimpleEarlyStop#thresholds.
  Adam.beta1: Adam#beta1.
  Adam.beta2: Adam#beta2.
  Muon.adam_b1: Muon#adam_b1.
  Muon.adam_b2: Muon#adam_b2.
  Optimizer.init: Optimizer#init().
  Optimizer.apply: Optimizer#apply().
  SGD.init: SGD#init().
  Schedule.__call__: Schedule#__call__().
  EarlyStop.should_stop: EarlyStop#should_stop().
  SimpleEarlyStop.should_stop: SimpleEarlyStop#should_stop().
  Counter: Counter.
  Lion.beta1: Lion#beta1.
  Lion.beta2: Lion#beta2.
  Lion.momentum_dtype: Lion#momentum_dtype.
  Muon.beta: Muon#beta.
  Muon.eps: Muon#eps.
  Muon.dim_threshold: Muon#dim_threshold.
  EarlyStop: EarlyStop#
  Adam.epsilon: Adam#epsilon.
  Muon.muon_a: Muon#muon_a.
  Muon.muon_b: Muon#muon_b.
  Muon.muon_c: Muon#muon_c.
  Muon.ns_steps: Muon#ns_steps.
  Muon.nesterov: Muon#nesterov.
  Muon.reconstruct_from_merged: Muon#reconstruct_from_merged().
  LinearWarmupConstant.value: LinearWarmupConstant#value.
  LinearWarmupConstant.warmup_steps: LinearWarmupConstant#warmup_steps.
  LinearWarmupCosineDecay.value: LinearWarmupCosineDecay#value.
  LinearWarmupCosineDecay.start_value: LinearWarmupCosineDecay#start_value.
  LinearWarmupCosineDecay.end_decay: LinearWarmupCosineDecay#end_decay.
  LinearWarmupCosineDecay.decay_start: LinearWarmupCosineDecay#decay_start.
  LinearWarmupCosineDecay.decay_steps: LinearWarmupCosineDecay#decay_steps.
  LinearWarmupCosineDecay.warmup_steps: LinearWarmupCosineDecay#warmup_steps.
  SGD.apply: SGD#apply().
  Muon.learning_rate: Muon#learning_rate.
  ScheduleRegistry.namespace: ScheduleRegistry#namespace.
  LinearWarmupConstant.warmup_fraction: LinearWarmupConstant#warmup_fraction.
  Constant.value: Constant#value.
  LinearWarmupCosineDecay.steps_after_decay: LinearWarmupCosineDecay#steps_after_decay.
  LinearWarmupCosineDecay.decay_start_fraction: LinearWarmupCosineDecay#decay_start_fraction.
  LinearWarmupCosineDecay.decay_fraction: LinearWarmupCosineDecay#decay_fraction.
  LinearWarmupCosineDecay.fraction_after_decay: LinearWarmupCosineDecay#fraction_after_decay.
  LinearWarmupCosineDecay.warmup_fraction: LinearWarmupCosineDecay#warmup_fraction.
  EarlyStopRegistry.namespace: EarlyStopRegistry#namespace.
---
# Module: [`simply/utils/optimizers.py`](../../../../../../raw/code/simply/simply/utils/optimizers.py)

## Classes
### `Adam`
- def: [`simply/utils/optimizers.py:107`](../../../../../../raw/code/simply/simply/utils/optimizers.py#L107)
- doc: Adam Optimizer.
- signature: `class Adam(Optimizer):`
- members:
  - `apply(self, state, grad)` — [`L132`](../../../../../../raw/code/simply/simply/utils/optimizers.py#L132)
  - `init(self, params)` — [`L114`](../../../../../../raw/code/simply/simply/utils/optimizers.py#L114)
  - `beta1` — [`L110`](../../../../../../raw/code/simply/simply/utils/optimizers.py#L110)
  - `beta2` — [`L111`](../../../../../../raw/code/simply/simply/utils/optimizers.py#L111)
  - `epsilon` — [`L112`](../../../../../../raw/code/simply/simply/utils/optimizers.py#L112)
- uses (calls/refs, reference-scoped): [`register`](registry.md#RootRegistry.register), [`with_sharding_constraint`](sharding.md#with_sharding_constraint), [`OptimizerRegistry`](optimizers.md#OptimizerRegistry), [`get_init_steps`](optimizers.md#get_init_steps), [`get_array_sharding`](sharding.md#get_array_sharding), [`Optimizer`](optimizers.md#Optimizer)
- used by: [`deepseek_qwen2_1p5b_it_dsr40k_r1_distill_cot_0shot_rl`](../config_lib.md#deepseek_qwen2_1p5b_it_dsr40k_r1_distill_cot_0shot_rl), [`gemma3_4b_it_simple_qa_number_only_tool_use_rl`](../config_lib.md#gemma3_4b_it_simple_qa_number_only_tool_use_rl), [`apply_simple_rl`](../config_lib.md#apply_simple_rl), [`test_grad_accumulation`](../model_lib_test.md#ModelLibTest.test_grad_accumulation), [`test_backward_pass`](../model_lib_test.md#ModelLibTest.test_backward_pass), [`optimizer`](../config_lib.md#BaseExperimentConfig.optimizer), [`test_adam`](optimizers_test.md#OptimizerTest.test_adam)

### `Constant`
- def: [`simply/utils/optimizers.py:451`](../../../../../../raw/code/simply/simply/utils/optimizers.py#L451)
- signature: `class Constant(LinearWarmupConstant):`
- members:
  - `value` — [`L452`](../../../../../../raw/code/simply/simply/utils/optimizers.py#L452)
  - `warmup_steps` — [`L453`](../../../../../../raw/code/simply/simply/utils/optimizers.py#L453)
- uses (calls/refs, reference-scoped): [`register`](registry.md#RootRegistry.register), [`LinearWarmupConstant`](optimizers.md#LinearWarmupConstant), [`ScheduleRegistry`](optimizers.md#ScheduleRegistry)

### `EarlyStop`  ·  implements/extends ABC
- def: [`simply/utils/optimizers.py:610`](../../../../../../raw/code/simply/simply/utils/optimizers.py#L610)
- doc: Base class for early stopping.
- signature: `class EarlyStop(abc.ABC):`
- members:
  - `should_stop(self, step, metrics: Mapping[str, Any])` — [`L627`](../../../../../../raw/code/simply/simply/utils/optimizers.py#L627) — Returns whether the early stopping should be triggered.
- protocol/private: `__post_init__`[`L614`](../../../../../../raw/code/simply/simply/utils/optimizers.py#L614)
- uses (calls/refs, reference-scoped): [`get`](registry.md#RootRegistry.get), [`fullname`](registry.md#RootRegistry.fullname), [`EarlyStopRegistry`](optimizers.md#EarlyStopRegistry), [`should_stop`](optimizers.md#SimpleEarlyStop.should_stop)
- used by: [`SimpleEarlyStop`](optimizers.md#SimpleEarlyStop), [`early_stop`](../config_lib.md#BaseExperimentConfig.early_stop)

### `EarlyStopRegistry`
- def: [`simply/utils/optimizers.py:632`](../../../../../../raw/code/simply/simply/utils/optimizers.py#L632)
- members:
  - `namespace` — [`L635`](../../../../../../raw/code/simply/simply/utils/optimizers.py#L635)
- uses (calls/refs, reference-scoped): [`RootRegistry`](registry.md#RootRegistry)
- used by: [`__post_init__`](optimizers.md#EarlyStop.__post_init__), [`ThresholdDef`](optimizers.md#ThresholdDef)

### `LinearWarmupConstant`
- def: [`simply/utils/optimizers.py:421`](../../../../../../raw/code/simply/simply/utils/optimizers.py#L421) — documented in [simply-utils-optimizers](../../../concepts/simply-utils-optimizers.md)
- doc: Constant schedule.
- signature: `class LinearWarmupConstant(Schedule):`
- members:
  - `value` — [`L423`](../../../../../../raw/code/simply/simply/utils/optimizers.py#L423)
  - `warmup_fraction` — [`L427`](../../../../../../raw/code/simply/simply/utils/optimizers.py#L427)
  - `warmup_steps` — [`L425`](../../../../../../raw/code/simply/simply/utils/optimizers.py#L425)
- protocol/private: `__call__`[`L438`](../../../../../../raw/code/simply/simply/utils/optimizers.py#L438), `_finalize`[`L429`](../../../../../../raw/code/simply/simply/utils/optimizers.py#L429)
- uses (calls/refs, reference-scoped): [`register`](registry.md#RootRegistry.register), [`Array`](optimizers.md#Array), [`Schedule`](optimizers.md#Schedule), [`replace_fraction`](optimizers.md#replace_fraction), [`ScheduleRegistry`](optimizers.md#ScheduleRegistry), [`constant_schedule`](optimizers.md#constant_schedule)
- used by: [`deepseek_qwen2_1p5b_it_dsr40k_r1_distill_cot_0shot_rl`](../config_lib.md#deepseek_qwen2_1p5b_it_dsr40k_r1_distill_cot_0shot_rl), [`gemma3_4b_it_simple_qa_number_only_tool_use_rl`](../config_lib.md#gemma3_4b_it_simple_qa_number_only_tool_use_rl), [`lm_rl_test`](../config_lib.md#lm_rl_test), [`gemma2_2b_gsm8k_0shot_rl`](../config_lib.md#gemma2_2b_gsm8k_0shot_rl), [`deepseek_qwen2_1p5b_it_dsr40k_r1_distill_cot_0shot_rl_f32_v2`](../config_lib.md#deepseek_qwen2_1p5b_it_dsr40k_r1_distill_cot_0shot_rl_f32_v2), [`apply_simple_rl`](../config_lib.md#apply_simple_rl), [`gemma2_2b_it_gsm8k_0shot_no_ref_rl`](../config_lib.md#gemma2_2b_it_gsm8k_0shot_no_ref_rl), [`deepseek_qwen2_14b_it_dsr40k_r1_distill_cot_0shot_rl_v3`](../config_lib.md#deepseek_qwen2_14b_it_dsr40k_r1_distill_cot_0shot_rl_v3), [`deepseek_qwen2_1p5b_it_dsr40k_r1_distill_cot_0shot_rl_f32_v3`](../config_lib.md#deepseek_qwen2_1p5b_it_dsr40k_r1_distill_cot_0shot_rl_f32_v3), [`deepseek_qwen2_1p5b_it_dsr40k_r1_distill_cot_0shot_rl_f32_v4`](../config_lib.md#deepseek_qwen2_1p5b_it_dsr40k_r1_distill_cot_0shot_rl_f32_v4), [`deepseek_qwen2_32b_it_dsr40k_r1_distill_cot_0shot_rl_v3`](../config_lib.md#deepseek_qwen2_32b_it_dsr40k_r1_distill_cot_0shot_rl_v3), [`deepseek_qwen2_7b_it_dsr40k_r1_distill_cot_0shot_rl_f32_v3`](../config_lib.md#deepseek_qwen2_7b_it_dsr40k_r1_distill_cot_0shot_rl_f32_v3), [`Constant`](optimizers.md#Constant), [`MockConfigV1`](optimizers_test.md#OptimizerTest.test_schedule_backward_compatibility_constant.MockConfigV1)

### `LinearWarmupCosineDecay`
- def: [`simply/utils/optimizers.py:458`](../../../../../../raw/code/simply/simply/utils/optimizers.py#L458) — documented in [simply-utils-optimizers](../../../concepts/simply-utils-optimizers.md)
- doc: Constant schedule.
- signature: `class LinearWarmupCosineDecay(Schedule):`
- members:
  - `_finalize(self, num_train_steps: Array | int)` — [`L478`](../../../../../../raw/code/simply/simply/utils/optimizers.py#L478) — Creates a schedule with absolute steps. — documented in [simply-utils-optimizers](../../../concepts/simply-utils-optimizers.md)
  - `decay_fraction` — [`L470`](../../../../../../raw/code/simply/simply/utils/optimizers.py#L470)
  - `decay_start` — [`L464`](../../../../../../raw/code/simply/simply/utils/optimizers.py#L464)
  - `decay_start_fraction` — [`L469`](../../../../../../raw/code/simply/simply/utils/optimizers.py#L469)
  - `decay_steps` — [`L465`](../../../../../../raw/code/simply/simply/utils/optimizers.py#L465)
  - `end_decay` — [`L462`](../../../../../../raw/code/simply/simply/utils/optimizers.py#L462)
  - `fraction_after_decay` — [`L471`](../../../../../../raw/code/simply/simply/utils/optimizers.py#L471)
  - `start_value` — [`L461`](../../../../../../raw/code/simply/simply/utils/optimizers.py#L461)
  - `steps_after_decay` — [`L466`](../../../../../../raw/code/simply/simply/utils/optimizers.py#L466)
  - `value` — [`L460`](../../../../../../raw/code/simply/simply/utils/optimizers.py#L460)
  - `warmup_fraction` — [`L476`](../../../../../../raw/code/simply/simply/utils/optimizers.py#L476)
  - `warmup_steps` — [`L474`](../../../../../../raw/code/simply/simply/utils/optimizers.py#L474)
- protocol/private: `__call__`[`L505`](../../../../../../raw/code/simply/simply/utils/optimizers.py#L505)
- uses (calls/refs, reference-scoped): [`Array`](optimizers.md#Array), [`Schedule`](optimizers.md#Schedule), [`replace_fraction`](optimizers.md#replace_fraction), [`cosine_decay_schedule`](optimizers.md#cosine_decay_schedule)
- used by: [`flops6e20_tfm2b_c4_l2048`](../config_lib.md#flops6e20_tfm2b_c4_l2048), [`lm_test`](../config_lib.md#lm_test), [`lm_test`](../model_lib_test.md#lm_test), [`qwen3_4b_gsm8k_sft`](../config_lib.md#qwen3_4b_gsm8k_sft), [`flops1e18_tfm111m_c4_l2048`](../config_lib.md#flops1e18_tfm111m_c4_l2048), [`flops1e19_tfm338m_c4_l2048`](../config_lib.md#flops1e19_tfm338m_c4_l2048), [`flops2e16_tfm15m_c4_l2048`](../config_lib.md#flops2e16_tfm15m_c4_l2048), [`flops2e17_tfm41m_c4_l2048`](../config_lib.md#flops2e17_tfm41m_c4_l2048), [`MockConfigV1`](optimizers_test.md#OptimizerTest.test_schedule_backward_compatibility_cosine_decay.MockConfigV1), [`MockConfigV1`](optimizers_test.md#OptimizerTest.test_schedule_backward_compatibility_cosine_decay_fraction.MockConfigV1), [`lr`](../config_lib.md#BaseExperimentConfig.lr)

### `Lion`
- def: [`simply/utils/optimizers.py:147`](../../../../../../raw/code/simply/simply/utils/optimizers.py#L147)
- doc: Lion Optimizer.
- signature: `class Lion(Optimizer):`
- members:
  - `apply(self, state, grad)` — [`L167`](../../../../../../raw/code/simply/simply/utils/optimizers.py#L167)
  - `init(self, params)` — [`L154`](../../../../../../raw/code/simply/simply/utils/optimizers.py#L154)
  - `beta1` — [`L150`](../../../../../../raw/code/simply/simply/utils/optimizers.py#L150)
  - `beta2` — [`L151`](../../../../../../raw/code/simply/simply/utils/optimizers.py#L151)
  - `momentum_dtype` — [`L152`](../../../../../../raw/code/simply/simply/utils/optimizers.py#L152)
- uses (calls/refs, reference-scoped): [`register`](registry.md#RootRegistry.register), [`with_sharding_constraint`](sharding.md#with_sharding_constraint), [`OptimizerRegistry`](optimizers.md#OptimizerRegistry), [`get_init_steps`](optimizers.md#get_init_steps), [`get_array_sharding`](sharding.md#get_array_sharding), [`Optimizer`](optimizers.md#Optimizer)
- used by: [`test_lion`](optimizers_test.md#OptimizerTest.test_lion)

### `Muon`
- def: [`simply/utils/optimizers.py:182`](../../../../../../raw/code/simply/simply/utils/optimizers.py#L182)
- doc: Implementation of the Muon Optimizer.
- signature: `class Muon(Optimizer):`
- members:
  - `_orthogonalize_via_newton_schulz(self, x, dim_annotation)` — [`L281`](../../../../../../raw/code/simply/simply/utils/optimizers.py#L281) — Newton-Schulz orthogonalization. — documented in [simply-utils-optimizers](../../../concepts/simply-utils-optimizers.md)
  - `apply(self, state, grad)` — [`L235`](../../../../../../raw/code/simply/simply/utils/optimizers.py#L235) — documented in [simply-utils-optimizers](../../../concepts/simply-utils-optimizers.md)
  - `init(self, params)` — [`L201`](../../../../../../raw/code/simply/simply/utils/optimizers.py#L201) — Initializes the optimizer state, using string indices as keys.
  - `init_adam(p)` — [`L218`](../../../../../../raw/code/simply/simply/utils/optimizers.py#L218)
  - `init_muon(p)` — [`L210`](../../../../../../raw/code/simply/simply/utils/optimizers.py#L210)
  - `merge_repeated_dims(self, tensor, dim_annotation)` — [`L312`](../../../../../../raw/code/simply/simply/utils/optimizers.py#L312) — Merges repeated dimensions in a tensor using einops. — documented in [simply-utils-optimizers](../../../concepts/simply-utils-optimizers.md)
  - `reconstruct_from_merged(self, merged_tensor, recipe)` — [`L390`](../../../../../../raw/code/simply/simply/utils/optimizers.py#L390) — Reshapes a merged tensor back to its original shape using a recipe.
  - `adam_b1` — [`L197`](../../../../../../raw/code/simply/simply/utils/optimizers.py#L197)
  - `adam_b2` — [`L198`](../../../../../../raw/code/simply/simply/utils/optimizers.py#L198)
  - `beta` — [`L194`](../../../../../../raw/code/simply/simply/utils/optimizers.py#L194)
  - `dim_threshold` — [`L199`](../../../../../../raw/code/simply/simply/utils/optimizers.py#L199)
  - `eps` — [`L195`](../../../../../../raw/code/simply/simply/utils/optimizers.py#L195)
  - `learning_rate` — [`L189`](../../../../../../raw/code/simply/simply/utils/optimizers.py#L189)
  - `muon_a` — [`L190`](../../../../../../raw/code/simply/simply/utils/optimizers.py#L190)
  - `muon_b` — [`L191`](../../../../../../raw/code/simply/simply/utils/optimizers.py#L191)
  - `muon_c` — [`L192`](../../../../../../raw/code/simply/simply/utils/optimizers.py#L192)
  - `nesterov` — [`L196`](../../../../../../raw/code/simply/simply/utils/optimizers.py#L196)
  - `ns_steps` — [`L193`](../../../../../../raw/code/simply/simply/utils/optimizers.py#L193)
- protocol/private: `_adam_m`[`L242`](../../../../../../raw/code/simply/simply/utils/optimizers.py#L242), `_adam_v`[`L247`](../../../../../../raw/code/simply/simply/utils/optimizers.py#L247), `_mu`[`L237`](../../../../../../raw/code/simply/simply/utils/optimizers.py#L237), `_param_update`[`L254`](../../../../../../raw/code/simply/simply/utils/optimizers.py#L254)
- uses (calls/refs, reference-scoped): [`register`](registry.md#RootRegistry.register), [`with_sharding_constraint`](sharding.md#with_sharding_constraint), [`OptimizerRegistry`](optimizers.md#OptimizerRegistry), [`get_init_steps`](optimizers.md#get_init_steps), [`get_array_sharding`](sharding.md#get_array_sharding), [`Optimizer`](optimizers.md#Optimizer), [`AnnotatedArray`](optimizers.md#AnnotatedArray), [`Counter`](optimizers.md#Counter)

### `Optimizer`
- def: [`simply/utils/optimizers.py:49`](../../../../../../raw/code/simply/simply/utils/optimizers.py#L49)
- doc: Base class for optimizers.
- signature: `class Optimizer(abc.ABC):`
- members:
  - `apply(self, state: PyTree, grad: PyTree)` — [`L70`](../../../../../../raw/code/simply/simply/utils/optimizers.py#L70) — Applies the update rule to the optimizer state and the gradient.
  - `apply_updates(self, state, updates)` — [`L73`](../../../../../../raw/code/simply/simply/utils/optimizers.py#L73) — Applies the update to the parameters.
  - `init(self, params: PyTree)` — [`L66`](../../../../../../raw/code/simply/simply/utils/optimizers.py#L66) — Initializes the state associated with the optimizer.
- protocol/private: `__post_init__`[`L53`](../../../../../../raw/code/simply/simply/utils/optimizers.py#L53)
- uses (calls/refs, reference-scoped): [`get`](registry.md#RootRegistry.get), [`fullname`](registry.md#RootRegistry.fullname), [`transfer_metadata`](common.md#transfer_metadata), [`OptimizerRegistry`](optimizers.md#OptimizerRegistry), [`PyTree`](optimizers.md#PyTree)
- used by: [`Adam`](optimizers.md#Adam), [`SGD`](optimizers.md#SGD), [`Lion`](optimizers.md#Lion), [`Muon`](optimizers.md#Muon), [`optimizer`](../config_lib.md#BaseExperimentConfig.optimizer)

### `OptimizerRegistry`  ·  implements/extends RootRegistry
- def: [`simply/utils/optimizers.py:41`](../../../../../../raw/code/simply/simply/utils/optimizers.py#L41) — documented in [simply-utils-optimizers](../../../concepts/simply-utils-optimizers.md)
- doc: Registry for optimizers.
- signature: `class OptimizerRegistry(registry.RootRegistry):`
- members:
  - `namespace` — [`L44`](../../../../../../raw/code/simply/simply/utils/optimizers.py#L44)
- uses (calls/refs, reference-scoped): [`register`](registry.md#RootRegistry.register), [`RootRegistry`](registry.md#RootRegistry)
- used by: [`RootRegistry`](registry.md#RootRegistry), [`Adam`](optimizers.md#Adam), [`SGD`](optimizers.md#SGD), [`Lion`](optimizers.md#Lion), [`Muon`](optimizers.md#Muon), [`__post_init__`](optimizers.md#Optimizer.__post_init__)

### `SGD`
- def: [`simply/utils/optimizers.py:92`](../../../../../../raw/code/simply/simply/utils/optimizers.py#L92)
- doc: Stochastic Gradient Descent Optimizer.
- signature: `class SGD(Optimizer):`
- members:
  - `apply(self, state, grad)` — [`L101`](../../../../../../raw/code/simply/simply/utils/optimizers.py#L101)
  - `init(self, params)` — [`L95`](../../../../../../raw/code/simply/simply/utils/optimizers.py#L95)
- uses (calls/refs, reference-scoped): [`register`](registry.md#RootRegistry.register), [`OptimizerRegistry`](optimizers.md#OptimizerRegistry), [`get_init_steps`](optimizers.md#get_init_steps), [`Optimizer`](optimizers.md#Optimizer)
- used by: [`test_dump`](optimizers_test.md#OptimizerTest.test_dump), [`test_sgd`](optimizers_test.md#OptimizerTest.test_sgd)

### `Schedule`  ·  implements/extends ABC
- def: [`simply/utils/optimizers.py:411`](../../../../../../raw/code/simply/simply/utils/optimizers.py#L411) — documented in [simply-utils-optimizers](../../../concepts/simply-utils-optimizers.md)
- doc: Base class for schedules.
- signature: `class Schedule(abc.ABC):`
- members:
  - `__call__(self, steps: Array | int)` — [`L414`](../../../../../../raw/code/simply/simply/utils/optimizers.py#L414) — Returns the value of the schedule at the given steps.
- uses (calls/refs, reference-scoped): [`Array`](optimizers.md#Array)
- used by: [`LinearWarmupConstant`](optimizers.md#LinearWarmupConstant), [`LinearWarmupCosineDecay`](optimizers.md#LinearWarmupCosineDecay), [`replace_fraction`](optimizers.md#replace_fraction), [`MockConfigV1`](optimizers_test.md#OptimizerTest.test_schedule_backward_compatibility_constant.MockConfigV1), [`MockConfigV1`](optimizers_test.md#OptimizerTest.test_schedule_backward_compatibility_cosine_decay.MockConfigV1), [`MockConfigV1`](optimizers_test.md#OptimizerTest.test_schedule_backward_compatibility_cosine_decay_fraction.MockConfigV1), [`lr`](../config_lib.md#BaseExperimentConfig.lr)

### `ScheduleRegistry`  ·  implements/extends RootRegistry
- def: [`simply/utils/optimizers.py:404`](../../../../../../raw/code/simply/simply/utils/optimizers.py#L404) — documented in [simply-utils-optimizers](../../../concepts/simply-utils-optimizers.md)
- doc: Registry for Schedule.
- signature: `class ScheduleRegistry(registry.RootRegistry):`
- members:
  - `namespace` — [`L407`](../../../../../../raw/code/simply/simply/utils/optimizers.py#L407)
- uses (calls/refs, reference-scoped): [`RootRegistry`](registry.md#RootRegistry)
- used by: [`RootRegistry`](registry.md#RootRegistry), [`LinearWarmupConstant`](optimizers.md#LinearWarmupConstant), [`Constant`](optimizers.md#Constant), [`warmup_steps`](optimizers.md#Constant.warmup_steps)

### `SimpleEarlyStop`
- def: [`simply/utils/optimizers.py:643`](../../../../../../raw/code/simply/simply/utils/optimizers.py#L643)
- doc: Early stopping based on metrics.
- signature: `class SimpleEarlyStop(EarlyStop):`
- members:
  - `should_stop(self, step, metrics: Mapping[str, Any])` — [`L648`](../../../../../../raw/code/simply/simply/utils/optimizers.py#L648) — Returns whether the early stopping should be triggered.
  - `thresholds` — [`L646`](../../../../../../raw/code/simply/simply/utils/optimizers.py#L646)
- uses (calls/refs, reference-scoped): [`ThresholdDef`](optimizers.md#ThresholdDef), [`EarlyStop`](optimizers.md#EarlyStop)
- used by: [`lm_rl_test`](../config_lib.md#lm_rl_test), [`should_stop`](optimizers.md#EarlyStop.should_stop)

## Functions
- `constant_schedule(steps, val, warmup_steps=None)` — [`L596`](../../../../../../raw/code/simply/simply/utils/optimizers.py#L596)
- `cosine_decay_schedule(steps, val, decay_steps, warmup_steps=1, end_decay=0.1, decay_start=None, start_value=0)` — [`L577`](../../../../../../raw/code/simply/simply/utils/optimizers.py#L577) — Linear warmup and cosine decay schedule. — documented in [simply-utils-optimizers](../../../concepts/simply-utils-optimizers.md)
- `create_lr_schedule(config)` — [`L538`](../../../../../../raw/code/simply/simply/utils/optimizers.py#L538)
- `create_lr_schedule_v0(config)` — [`L547`](../../../../../../raw/code/simply/simply/utils/optimizers.py#L547) — Creates a learning rate schedule from a experiment config.
- `get_init_steps()` — [`L86`](../../../../../../raw/code/simply/simply/utils/optimizers.py#L86)
- `replace_fraction(schedule: Schedule, num_train_steps: Array | int, fraction_field: str, steps_field: str)` — [`L519`](../../../../../../raw/code/simply/simply/utils/optimizers.py#L519) — documented in [simply-utils-optimizers](../../../concepts/simply-utils-optimizers.md)

## Module values
- `AnnotatedArray` — [`L35`](../../../../../../raw/code/simply/simply/utils/optimizers.py#L35)
- `Array` — [`L34`](../../../../../../raw/code/simply/simply/utils/optimizers.py#L34)
- `Counter` — [`L32`](../../../../../../raw/code/simply/simply/utils/optimizers.py#L32)
- `PyTree` — [`L33`](../../../../../../raw/code/simply/simply/utils/optimizers.py#L33)
- `ThresholdDef` — [`L638`](../../../../../../raw/code/simply/simply/utils/optimizers.py#L638)

