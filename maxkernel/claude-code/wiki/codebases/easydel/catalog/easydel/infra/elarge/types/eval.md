---
title: 'Module: easydel/infra/elarge/types/eval.py'
type: catalog
provenance: extracted
module: easydel/infra/elarge/types/eval.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.infra.elarge.types.eval`/
symbols:
  BenchmarkConfig: BenchmarkConfig#
  BenchmarkTask: BenchmarkTask.
  EvalKwargs: EvalKwargs#
  ResolvedBenchmarkConfig.tasks: ResolvedBenchmarkConfig#tasks.
  BenchmarkTasks: BenchmarkTasks.
  ResolvedBenchmarkConfig.name: ResolvedBenchmarkConfig#name.
  ResolvedBenchmarkConfig: ResolvedBenchmarkConfig#
  ResolvedBenchmarkConfig.eval_kwargs: ResolvedBenchmarkConfig#eval_kwargs.
  EvalKwargs.sampling_params: EvalKwargs#sampling_params.
  BenchmarkConfig.tasks: BenchmarkConfig#tasks.
  EvalKwargs.num_fewshot: EvalKwargs#num_fewshot.
  EvalKwargs.max_new_tokens: EvalKwargs#max_new_tokens.
  EvalKwargs.hard_max_new_tokens: EvalKwargs#hard_max_new_tokens.
  EvalKwargs.enable_thinking: EvalKwargs#enable_thinking.
  EvalKwargs.chat_template_args: EvalKwargs#chat_template_args.
  EvalKwargs.think_start_token: EvalKwargs#think_start_token.
  EvalKwargs.think_end_token: EvalKwargs#think_end_token.
  EvalKwargs.ignore_benchmark_eos_flags: EvalKwargs#ignore_benchmark_eos_flags.
  EvalKwargs.temperature: EvalKwargs#temperature.
  EvalKwargs.top_p: EvalKwargs#top_p.
  EvalKwargs.normalize_math_answers: EvalKwargs#normalize_math_answers.
  EvalKwargs.math_answer_task_hints: EvalKwargs#math_answer_task_hints.
  EvalKwargs.code_eval_num_workers: EvalKwargs#code_eval_num_workers.
  EvalKwargs.code_eval_timeout: EvalKwargs#code_eval_timeout.
  EvalKwargs.batch_size: EvalKwargs#batch_size.
  EvalKwargs.max_batch_size: EvalKwargs#max_batch_size.
  EvalKwargs.device: EvalKwargs#device.
  EvalKwargs.use_cache: EvalKwargs#use_cache.
  EvalKwargs.limit: EvalKwargs#limit.
  EvalKwargs.cache_requests: EvalKwargs#cache_requests.
  EvalKwargs.rewrite_requests_cache: EvalKwargs#rewrite_requests_cache.
  EvalKwargs.delete_requests_cache: EvalKwargs#delete_requests_cache.
  EvalKwargs.check_integrity: EvalKwargs#check_integrity.
  EvalKwargs.write_out: EvalKwargs#write_out.
  EvalKwargs.log_samples: EvalKwargs#log_samples.
  EvalKwargs.evaluation_tracker: EvalKwargs#evaluation_tracker.
  EvalKwargs.system_instruction: EvalKwargs#system_instruction.
  EvalKwargs.apply_chat_template: EvalKwargs#apply_chat_template.
  EvalKwargs.fewshot_as_multiturn: EvalKwargs#fewshot_as_multiturn.
  EvalKwargs.gen_kwargs: EvalKwargs#gen_kwargs.
  EvalKwargs.task_manager: EvalKwargs#task_manager.
  EvalKwargs.verbosity: EvalKwargs#verbosity.
  EvalKwargs.predict_only: EvalKwargs#predict_only.
  EvalKwargs.samples: EvalKwargs#samples.
  EvalKwargs.bootstrap_iters: EvalKwargs#bootstrap_iters.
  EvalKwargs.random_seed: EvalKwargs#random_seed.
  EvalKwargs.numpy_random_seed: EvalKwargs#numpy_random_seed.
  EvalKwargs.torch_random_seed: EvalKwargs#torch_random_seed.
  EvalKwargs.fewshot_random_seed: EvalKwargs#fewshot_random_seed.
  EvalKwargs.confirm_run_unsafe_code: EvalKwargs#confirm_run_unsafe_code.
  EvalKwargs.metadata: EvalKwargs#metadata.
  EvalKwargs.include_path: EvalKwargs#include_path.
  EvalKwargs.include_defaults: EvalKwargs#include_defaults.
  BenchmarkConfig.name: BenchmarkConfig#name.
---
# Module: [`easydel/infra/elarge/types/eval.py`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/eval.py)

## Classes
### `BenchmarkConfig`  ·  implements/extends EvalKwargs
- def: [`easydel/infra/elarge/types/eval.py:154`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/eval.py#L154)
- doc: Configuration for a single named benchmark suite.
- signature: `class BenchmarkConfig(EvalKwargs, total=False):`
- members:
  - `name` — [`L165`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/eval.py#L165)
  - `tasks` — [`L166`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/eval.py#L166)
- uses (calls/refs, reference-scoped): [`EvalKwargs`](eval.md#EvalKwargs), [`BenchmarkTasks`](eval.md#BenchmarkTasks)
- used by: [`_import_structure`](../../../__init__.md#_import_structure), [`run_benchmarks`](../model.md#eLargeModel.run_benchmarks), [`eval`](../model.md#eLargeModel.eval), [`normalize_benchmark_configs`](../benchmarking.md#normalize_benchmark_configs), [`__getattr__`](../../__init__.md#__getattr__), [`benchmarks`](../../../trainers/training_configurations.md#TrainingArguments.benchmarks), [`EvalKwargs`](eval.md#EvalKwargs), [`benchmarks`](training.md#BaseTrainerCfg.benchmarks)

### `EvalKwargs`  ·  implements/extends _TypedDict
- def: [`easydel/infra/elarge/types/eval.py:33`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/eval.py#L33)
- doc: Keyword arguments passed to lm-evaluation-harness's `simple_evaluate`
- signature: `class EvalKwargs(TypedDict, total=False):`
- members:
  - `apply_chat_template` — [`L136`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/eval.py#L136)
  - `batch_size` — [`L123`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/eval.py#L123)
  - `bootstrap_iters` — [`L143`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/eval.py#L143)
  - `cache_requests` — [`L128`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/eval.py#L128)
  - `chat_template_args` — [`L112`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/eval.py#L112)
  - `check_integrity` — [`L131`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/eval.py#L131)
  - `code_eval_num_workers` — [`L121`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/eval.py#L121)
  - `code_eval_timeout` — [`L122`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/eval.py#L122)
  - `confirm_run_unsafe_code` — [`L148`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/eval.py#L148)
  - `delete_requests_cache` — [`L130`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/eval.py#L130)
  - `device` — [`L125`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/eval.py#L125)
  - `enable_thinking` — [`L111`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/eval.py#L111)
  - `evaluation_tracker` — [`L134`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/eval.py#L134)
  - `fewshot_as_multiturn` — [`L137`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/eval.py#L137)
  - `fewshot_random_seed` — [`L147`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/eval.py#L147)
  - `gen_kwargs` — [`L138`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/eval.py#L138)
  - `hard_max_new_tokens` — [`L110`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/eval.py#L110)
  - `ignore_benchmark_eos_flags` — [`L115`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/eval.py#L115)
  - `include_defaults` — [`L151`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/eval.py#L151)
  - `include_path` — [`L150`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/eval.py#L150)
  - `limit` — [`L127`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/eval.py#L127)
  - `log_samples` — [`L133`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/eval.py#L133)
  - `math_answer_task_hints` — [`L120`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/eval.py#L120)
  - `max_batch_size` — [`L124`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/eval.py#L124)
  - `max_new_tokens` — [`L109`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/eval.py#L109)
  - `metadata` — [`L149`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/eval.py#L149)
  - `normalize_math_answers` — [`L119`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/eval.py#L119)
  - `num_fewshot` — [`L108`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/eval.py#L108)
  - `numpy_random_seed` — [`L145`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/eval.py#L145)
  - `predict_only` — [`L141`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/eval.py#L141)
  - `random_seed` — [`L144`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/eval.py#L144)
  - `rewrite_requests_cache` — [`L129`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/eval.py#L129)
  - `samples` — [`L142`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/eval.py#L142)
  - `sampling_params` — [`L118`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/eval.py#L118)
  - `system_instruction` — [`L135`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/eval.py#L135)
  - `task_manager` — [`L139`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/eval.py#L139)
  - `temperature` — [`L116`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/eval.py#L116)
  - `think_end_token` — [`L114`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/eval.py#L114)
  - `think_start_token` — [`L113`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/eval.py#L113)
  - `top_p` — [`L117`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/eval.py#L117)
  - `torch_random_seed` — [`L146`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/eval.py#L146)
  - `use_cache` — [`L126`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/eval.py#L126)
  - `verbosity` — [`L140`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/eval.py#L140)
  - `write_out` — [`L132`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/eval.py#L132)
- uses (calls/refs, reference-scoped): [`SamplingParams`](../../../inference/sampling_params.md#SamplingParams), [`BenchmarkConfig`](eval.md#BenchmarkConfig)
- used by: [`run_benchmarks`](../model.md#eLargeModel.run_benchmarks), [`eval`](../model.md#eLargeModel.eval), [`eLMConfig`](root.md#eLMConfig), [`BenchmarkConfig`](eval.md#BenchmarkConfig), [`set_eval`](../model.md#eLargeModel.set_eval)

### `ResolvedBenchmarkConfig`
- def: [`easydel/infra/elarge/types/eval.py:170`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/eval.py#L170)
- doc: Normalized, ready-to-run form of a `BenchmarkConfig` after validation.
- signature: `class ResolvedBenchmarkConfig:`
- members:
  - `eval_kwargs` — [`L181`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/eval.py#L181)
  - `name` — [`L179`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/eval.py#L179)
  - `tasks` — [`L180`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/eval.py#L180)
- uses (calls/refs, reference-scoped): [`BenchmarkTask`](eval.md#BenchmarkTask)
- used by: [`maybe_benchmark`](../../../trainers/base_trainer.md#BaseTrainer.maybe_benchmark), [`run_benchmarks`](../model.md#eLargeModel.run_benchmarks), [`normalize_benchmark_configs`](../benchmarking.md#normalize_benchmark_configs)

## Module values
- `BenchmarkTask` — [`L26`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/eval.py#L26)
- `BenchmarkTasks` — [`L29`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/eval.py#L29)

