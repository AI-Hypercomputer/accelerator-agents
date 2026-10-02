---
title: simply.config_lib — composable experiment configs via chained dataclasses.replace
type: concept
provenance: mixed
concept: simply-config_lib
updated: 2026-07-03
status: fresh
---
# simply.config_lib — composable experiment configs via chained dataclasses.replace

## Overview

This module is where every trainable/servable model variant in Simply is defined: one large
[`BaseExperimentConfig`](../catalog/simply/config_lib.md#BaseExperimentConfig) dataclass (~150 fields
covering architecture, data, optimizer, checkpointing, and sharding) plus a subclass
[`RLExperimentConfig`](../catalog/simply/config_lib.md#deepseek_qwen2_1p5b_it_dsr40k_r1_distill_cot_0shot_rl)
for RL-specific fields, both registered under
[`ExperimentConfigRegistry`](../catalog/simply/config_lib.md#ExperimentConfigRegistry), and then
**dozens of registered factory functions** (`gemma2_2b`, `deepseek_qwen2_1p5b`, `qwen3_0p6b`, ...) that
each build one named, immediately-usable config by chaining `dataclasses.replace` calls — either
starting from `BaseExperimentConfig()` directly or from an already-registered sibling preset (e.g.
[`deepseek_qwen2_7b`](../catalog/simply/config_lib.md#deepseek_qwen2_14b) is
`dataclasses.replace(deepseek_qwen2_1p5b(), model_dim=3584, ...)`). A parallel, smaller
[`ShardingConfig`](../catalog/simply/config_lib.md#gspmd_sharding) hierarchy defines the
per-architecture-shape partition annotations every layer reads (see
[simply-utils-sharding](simply-utils-sharding.md)).

## Diagram

```mermaid
flowchart TD
  BASE["BaseExperimentConfig()\n(~150 fields, all defaults)"]
  BASE -->|dataclasses.replace(arch fields)| SIZE1["deepseek_qwen2_1p5b()"]
  SIZE1 -->|dataclasses.replace(bigger dims)| SIZE2["deepseek_qwen2_7b()"]
  SIZE1 --> SIZE3["deepseek_qwen2_14b()"]
  RLBASE["RLExperimentConfig()"] -->|override_from(size_preset)| RLCOMPOSE["config"]
  RLCOMPOSE -->|apply_simple_rl(config)| RLFINAL["deepseek_qwen2_1p5b_it_..._rl()"]
  SHARDBASE["BaseSharding() (gspmd_sharding)"] -->|dataclasses.replace| MOESHARD["moe_sharding() / moe_sharding_v1()"]
  SHARDBASE -->|to_decoding_sharding()| DECODESHARD["decoding_sharding_config\n(drops seq-len sharding)"]
  REGISTRY["ExperimentConfigRegistry"] -.->|--experiment_config=<name>| MAINPY["main.py / servers"]
```

## Design rationale (why it's built this way)

**Every config is a frozen dataclass, and composition is always `dataclasses.replace`, never mutation
or inheritance-based field override.** [`BaseExperimentConfig`](../catalog/simply/config_lib.md#BaseExperimentConfig)
and every preset factory (`gemma2_2b`, `deepseek_qwen2_1p5b`, ...) build new instances via
`dataclasses.replace(config, **overrides)` — this is what lets
[`deepseek_qwen2_32b`](../catalog/simply/config_lib.md#deepseek_qwen2_32b) simply override
`{vocab_size, model_dim, ffn_expand_dim, n_layers, n_heads, n_kv_heads, position_encoding,
rms_norm_epsilon, init_ckpt_dir}` on top of `deepseek_qwen2_1p5b()`'s already-set fields (activation
function, bias flags, tied-embedding setting, etc.) rather than needing to restate every field or
define a parallel subclass per model size.

**`override_from` performs a cross-hierarchy field transplant, restricted to the *intersection* of
both dataclasses' field names — a safety net against silently dropping or misapplying fields.**
[`ExperimentConfig.override_from`](../catalog/simply/config_lib.md) computes `source_names & target_names`
and only copies those; if the source config has fields the target doesn't recognize, it raises
`ValueError` unless `error_on_extra=False` — this is exactly the mechanism
`RLExperimentConfig().override_from(deepseek_qwen2_1p5b())` uses to pull a model-architecture preset
(a `BaseExperimentConfig`) into an `RLExperimentConfig` instance: only the fields both classes share
(model dims, tokenizer, checkpoint paths, etc.) transfer, while `RLExperimentConfig`-only fields
(reward normalization, PPO clip epsilon, ...) keep their own defaults.

**`apply_simple_rl` is a reusable overlay function, not another preset — it's composed *after* the
architecture and the RL base config are already merged.** Every `*_rl` preset follows the same
two-step pattern: `config = RLExperimentConfig().override_from(<size_preset>())`, then `config =
`[`apply_simple_rl`](../catalog/simply/config_lib.md)`(config)` — separating "which model" from "which
RL recipe" as two independently composable transforms, so a new RL recipe variant (e.g. a different
`kl_coeff` or optimizer) can be defined as a second overlay function without duplicating every model
size's preset.

**MoE sharding configs progressively add a `'seq'` mesh axis, and one variant exists purely to move
which activation axis that axis attaches to.** [`moe_sharding`](../catalog/simply/config_lib.md#gspmd_sharding)
adds `'seq'` sharding to the sequence-length axis of every partition spec on top of
[`gspmd_sharding`](../catalog/simply/config_lib.md#gspmd_sharding); the comment on
[`moe_sharding_v1`](../catalog/simply/config_lib.md) explains its one change ("Allocate `'seq'` to
batch dimension instead of `seq_len` dimension to avoid extra all-gather on the attention layer") —
a documented, measured trade between two ways of sharding the same logical axis, kept as two separate
registered configs rather than one config with a flag, presumably because the two variants aren't
simply interchangeable at every model scale.

**`to_decoding_sharding` exists because decode-time activations have no meaningful sequence-length
axis to shard (`seq_len == 1` per step), so the training sharding config's `seq_len` partition
annotation must be dropped, not just left as-is.**
[`BaseSharding.to_decoding_sharding`](../catalog/simply/config_lib.md#gspmd_sharding) replaces
`activation_partition`'s second element with `None` — every serving entry point
([simply-serving-page_batcher](simply-serving-page_batcher.md),
[simply-serving-vanilla_server](simply-serving-vanilla_server.md)) either reads a config's own
`decoding_sharding_config` field or falls back to calling this method, since sharding a length-1 axis
provides no benefit and can trigger degenerate reshard behavior.

> [!inferred] `BaseExperimentConfig.sharding_config` defaults to `gspmd_sharding()` called eagerly at
> class-definition time (a mutable-looking default that's safe here only because
> `ShardingConfig`/`BaseSharding` are themselves frozen dataclasses) — this is a common
> frozen-dataclass idiom for "default is a constructed value, not a factory," relying on the nested
> config also being immutable to avoid the usual mutable-default-argument hazard.

## Entry points

- [`ExperimentConfigRegistry.get_config`](../catalog/simply/config_lib.md#ExperimentConfigRegistry)
  (`= get(name)()`) — resolves a `--experiment_config=<name>` CLI flag into a concrete config
  instance; called from `main.load_experiment_config` and every serving entry point.
- Every registered preset factory (e.g.
  [`gemma3_1b`](../catalog/simply/config_lib.md),
  [`deepseek_qwen2_1p5b`](../catalog/simply/config_lib.md#deepseek_qwen2_1p5b),
  [`qwen3_0p6b`](../catalog/simply/config_lib.md#qwen3_0p6b)) — the concrete, immediately-usable
  configs; each is one `@ExperimentConfigRegistry.register`-decorated zero-argument function.
- [`apply_simple_rl`](../catalog/simply/config_lib.md#apply_simple_rl) — the RL-recipe overlay every
  `*_rl` preset applies after merging in its base architecture config.

## Mechanism (step-by-step)

1. **A base architecture config is defined once, with every field defaulted.**
   [`BaseExperimentConfig`](../catalog/simply/config_lib.md#BaseExperimentConfig) itself is directly
   registered (as the `'BaseExperimentConfig'` name) so it's usable standalone for quick tests.
2. **Size-specific presets override only the fields that differ from their base.** A "root" preset
   (e.g. [`deepseek_qwen2_1p5b`](../catalog/simply/config_lib.md#deepseek_qwen2_1p5b),
   [`gemma3_1b`](../catalog/simply/config_lib.md)) starts from `BaseExperimentConfig()` and overrides
   the full architecture description; sibling sizes (`_7b`, `_14b`, `_32b`, ...) start from the root
   preset and override only what scales (dims, layer count, head count, checkpoint path).
3. **RL variants merge an architecture preset into `RLExperimentConfig` via `override_from`, then
   apply the shared RL recipe overlay.** `RLExperimentConfig().override_from(<size_preset>())` then
   [`apply_simple_rl(config)`](../catalog/simply/config_lib.md#apply_simple_rl); some presets further
   override RL-specific hyperparameters
   (`kl_coeff`, dataset, evaluation) after the overlay for a specific experiment's needs.
4. **At CLI/server startup, a config is resolved purely by string name, through
   [`ExperimentConfigRegistry`](../catalog/simply/config_lib.md#ExperimentConfigRegistry).**
   `config_lib.ExperimentConfigRegistry.get_instance(name)` (or `.get_config(name)`) is the only path
   any entry point uses — none of the training/serving code imports a specific preset function
   directly.
5. **Sharding configs compose the same way**, independently of experiment configs:
   [`gspmd_sharding`](../catalog/simply/config_lib.md#gspmd_sharding) →
   [`moe_sharding`](../catalog/simply/config_lib.md#gspmd_sharding) →
   [`moe_sharding_v1`](../catalog/simply/config_lib.md), each a `dataclasses.replace` over the
   previous, referenced by name or object from an experiment config's own `sharding_config` field.

## Key data structures

- **[`BaseExperimentConfig`](../catalog/simply/config_lib.md#BaseExperimentConfig)** — architecture
  (`model_dim`, `n_heads`, `n_layers`, MoE fields, position encoding, attention windowing), data
  (`dataset`, `batch_size`, `batch_mode`), optimizer (`optimizer`, `lr`, clipping thresholds),
  checkpoint (`init_ckpt_dir`/`init_ckpt_format`, `ckpt_interval`), and `sharding_config`, all as one
  flat dataclass.
- **[`RLExperimentConfig`](../catalog/simply/config_lib.md)** (subclasses `BaseExperimentConfig`) —
  adds `evaluation`, `lm_format_name`, sampling parameters (`sampling_temperature`,
  `sampling_max_decode_steps`, ...), and RL-algorithm fields (`gamma`, `kl_coeff`, `use_grpo`,
  `ppo_clip_eps`, `normalize_reward_method`).
- **[`ShardingConfig`](../catalog/simply/config_lib.md#gspmd_sharding)**/
  [`BaseSharding`](../catalog/simply/config_lib.md#gspmd_sharding)** — the partition-annotation set
  every layer's `EinsumLinear`/`Attention` reads (`ffn0_partition`, `attn_qkv_partition`,
  `activation_partition`, `mesh_axis_names`, ...).

## Dynamics (design intent)

Because every preset is a zero-argument function (not a class or a data literal), a preset can itself
contain arbitrary Python logic (as `apply_simple_rl` does) — this is what makes the RL-overlay pattern
possible at all; a purely declarative config format (e.g. YAML) would need a separate merge/templating
layer to express the same "architecture × recipe" composition.

## Edge cases

- [`override_from`](../catalog/simply/config_lib.md) raises on extra source fields by default
  (`error_on_extra=True`) — merging an `RLExperimentConfig`-shaped source into a plain
  `BaseExperimentConfig` target (the reverse direction from the common usage) would raise unless the
  caller explicitly opts out of the safety check.
- Several fields are explicitly marked deprecated in comments (`dataset_name`, `use_packing`,
  `use_validation_set`, `lr_schedule_name`/`lr_schedule_config`) but still present and defaulted —
  callers relying on the old fields still work, but new configs should prefer `dataset`/`lr`
  directly.

## Open questions

- Whether there's a lint/test enforcing that every new preset's `dataclasses.replace` call only
  touches fields that actually differ from its base (vs. accidentally restating an unchanged
  default) isn't visible in this packet's subgraph.

## See also
- [simply-utils-registry](simply-utils-registry.md) — `RootRegistry`, the base
  `ExperimentConfigRegistry`/`ShardingConfigRegistry` inherit from.
- [simply-utils-sharding](simply-utils-sharding.md) — the `PartitionAnnotation` type every
  `ShardingConfig` field uses.
- [simply-utils-optimizers](simply-utils-optimizers.md) — `Optimizer`/`Schedule`, referenced by
  `BaseExperimentConfig.optimizer`/`lr`.
- [simply-model_lib](simply-model_lib.md) — `TrainLoopRegistry`/`run_experiment`, the consumer of a
  resolved `BaseExperimentConfig`.
