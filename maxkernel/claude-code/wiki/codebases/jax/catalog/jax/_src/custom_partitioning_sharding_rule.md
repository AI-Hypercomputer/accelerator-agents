---
title: 'Module: jax/_src/custom_partitioning_sharding_rule.py'
type: catalog
provenance: extracted
module: jax/_src/custom_partitioning_sharding_rule.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax._src.custom_partitioning_sharding_rule`/
symbols:
  sdy_sharding_rule_to_mlir: sdy_sharding_rule_to_mlir().
  ArrayMapping: ArrayMapping#
  SdyShardingRule.__init__: SdyShardingRule#__init__().
  SdyShardingRule.__str__: SdyShardingRule#__str__().
  _parse_values: _parse_values().
  BATCHING.BATCHING: BATCHING.BATCHING.
  str_to_sdy_sharding_rule: str_to_sdy_sharding_rule().
  SdyShardingRule: SdyShardingRule#
  SdyShardingRule.operand_mappings: SdyShardingRule#operand_mappings.
  SdyShardingRule.result_mappings: SdyShardingRule#result_mappings.
  ArrayMapping.__init__: ArrayMapping#__init__().
  _is_batching: _is_batching().
  sdy_sharding_rule_to_mlir.add_batching_dim_factor: sdy_sharding_rule_to_mlir().add_batching_dim_factor().
  sdy_sharding_rule_to_mlir.build_dim_mapping_for_compound_factors: sdy_sharding_rule_to_mlir().build_dim_mapping_for_compound_factors().
  sdy_sharding_rule_to_mlir.add_factor: sdy_sharding_rule_to_mlir().add_factor().
  CompoundFactor: CompoundFactor#
  CompoundFactor.__init__: CompoundFactor#__init__().
  _get_batching_dim_factor_name: _get_batching_dim_factor_name().
  SdyShardingRule.factor_sizes: SdyShardingRule#factor_sizes.
  SdyShardingRule.reduction_factors: SdyShardingRule#reduction_factors.
  SdyShardingRule.need_replication_factors: SdyShardingRule#need_replication_factors.
  SdyShardingRule.permutation_factors: SdyShardingRule#permutation_factors.
  SdyShardingRule.check_special_factors: SdyShardingRule#check_special_factors().
  SdyShardingRule.to_str: SdyShardingRule#to_str().
  _parse_values.add_factor: _parse_values().add_factor().
  sdy_sharding_rule_to_mlir.get_size_for_value_dim: sdy_sharding_rule_to_mlir().get_size_for_value_dim().
  sdy_sharding_rule_to_mlir.factors_to_indices: sdy_sharding_rule_to_mlir().factors_to_indices().
  CompoundFactor.__new__: CompoundFactor#__new__().
  ArrayMapping.__new__: ArrayMapping#__new__().
  _BATCHING_DIM_FACTOR_PREFIX: _BATCHING_DIM_FACTOR_PREFIX.
  _check_factor: _check_factor().
  _get_batching_group: _get_batching_group().
  sdy_sharding_rule_to_mlir.get_message_for_value: sdy_sharding_rule_to_mlir().get_message_for_value().
  sdy_sharding_rule_to_mlir.get_rank_for_value: sdy_sharding_rule_to_mlir().get_rank_for_value().
---
# Module: [`jax/_src/custom_partitioning_sharding_rule.py`](../../../../../../raw/code/jax/jax/_src/custom_partitioning_sharding_rule.py)

## Classes
### `ArrayMapping`  ·  implements/extends tuple
- def: [`jax/_src/custom_partitioning_sharding_rule.py:78`](../../../../../../raw/code/jax/jax/_src/custom_partitioning_sharding_rule.py#L78)
- doc: Describes the factors for an operand or result.
- signature: `class ArrayMapping(tuple):`
- protocol/private: `__init__`[`L88`](../../../../../../raw/code/jax/jax/_src/custom_partitioning_sharding_rule.py#L88), `__new__`[`L101`](../../../../../../raw/code/jax/jax/_src/custom_partitioning_sharding_rule.py#L101)
- uses (calls/refs, reference-scoped): [`_is_batching`](custom_partitioning_sharding_rule.md#_is_batching), [`CompoundFactor`](custom_partitioning_sharding_rule.md#CompoundFactor), [`_check_factor`](custom_partitioning_sharding_rule.md#_check_factor)
- used by: [`__init__`](custom_partitioning_sharding_rule.md#SdyShardingRule.__init__), [`_fwd_shardy_rule`](cudnn/fused_attention_stablehlo.md#_fwd_shardy_rule), [`_parse_values`](custom_partitioning_sharding_rule.md#_parse_values), [`_bwd_shardy_rule`](cudnn/fused_attention_stablehlo.md#_bwd_shardy_rule), [`operand_mappings`](custom_partitioning_sharding_rule.md#SdyShardingRule.operand_mappings), [`result_mappings`](custom_partitioning_sharding_rule.md#SdyShardingRule.result_mappings)

### `CompoundFactor`  ·  implements/extends tuple
- def: [`jax/_src/custom_partitioning_sharding_rule.py:57`](../../../../../../raw/code/jax/jax/_src/custom_partitioning_sharding_rule.py#L57)
- doc: Describes the factors for a compound factor.
- signature: `class CompoundFactor(tuple):`
- protocol/private: `__init__`[`L63`](../../../../../../raw/code/jax/jax/_src/custom_partitioning_sharding_rule.py#L63), `__new__`[`L74`](../../../../../../raw/code/jax/jax/_src/custom_partitioning_sharding_rule.py#L74)
- uses (calls/refs, reference-scoped): [`_is_batching`](custom_partitioning_sharding_rule.md#_is_batching), [`_check_factor`](custom_partitioning_sharding_rule.md#_check_factor)
- used by: [`_fwd_shardy_rule`](cudnn/fused_attention_stablehlo.md#_fwd_shardy_rule), [`_parse_values`](custom_partitioning_sharding_rule.md#_parse_values), [`__init__`](custom_partitioning_sharding_rule.md#ArrayMapping.__init__)

### `SdyShardingRule`
- def: [`jax/_src/custom_partitioning_sharding_rule.py:105`](../../../../../../raw/code/jax/jax/_src/custom_partitioning_sharding_rule.py#L105)
- doc: Represents a Shardy sharding rule.
- signature: `class SdyShardingRule:`
- members:
  - `check_special_factors(kind, factors)` — [`L161`](../../../../../../raw/code/jax/jax/_src/custom_partitioning_sharding_rule.py#L161)
  - `to_str(kind, factors)` — [`L190`](../../../../../../raw/code/jax/jax/_src/custom_partitioning_sharding_rule.py#L190)
  - `factor_sizes` — [`L119`](../../../../../../raw/code/jax/jax/_src/custom_partitioning_sharding_rule.py#L119)
  - `need_replication_factors` — [`L121`](../../../../../../raw/code/jax/jax/_src/custom_partitioning_sharding_rule.py#L121)
  - `operand_mappings` — [`L117`](../../../../../../raw/code/jax/jax/_src/custom_partitioning_sharding_rule.py#L117)
  - `permutation_factors` — [`L122`](../../../../../../raw/code/jax/jax/_src/custom_partitioning_sharding_rule.py#L122)
  - `reduction_factors` — [`L120`](../../../../../../raw/code/jax/jax/_src/custom_partitioning_sharding_rule.py#L120)
  - `result_mappings` — [`L118`](../../../../../../raw/code/jax/jax/_src/custom_partitioning_sharding_rule.py#L118)
- protocol/private: `__init__`[`L124`](../../../../../../raw/code/jax/jax/_src/custom_partitioning_sharding_rule.py#L124), `__str__`[`L189`](../../../../../../raw/code/jax/jax/_src/custom_partitioning_sharding_rule.py#L189)
- uses (calls/refs, reference-scoped): [`ArrayMapping`](custom_partitioning_sharding_rule.md#ArrayMapping)
- used by: [`_custom_partitioning_lowering_rule`](custom_partitioning.md#_custom_partitioning_lowering_rule), [`sdy_sharding_rule_to_mlir`](custom_partitioning_sharding_rule.md#sdy_sharding_rule_to_mlir), [`_fwd_shardy_rule`](cudnn/fused_attention_stablehlo.md#_fwd_shardy_rule), [`str_to_sdy_sharding_rule`](custom_partitioning_sharding_rule.md#str_to_sdy_sharding_rule), [`_bwd_shardy_rule`](cudnn/fused_attention_stablehlo.md#_bwd_shardy_rule), [`decode_shardings`](custom_partitioning.md#custom_partitioning.decode_shardings)

## Functions
- `_check_factor(factor: str)` — [`L31`](../../../../../../raw/code/jax/jax/_src/custom_partitioning_sharding_rule.py#L31) — Validates a factor.
- `_get_batching_dim_factor_name(batch_group: str, batch_dim_order: int)` — [`L202`](../../../../../../raw/code/jax/jax/_src/custom_partitioning_sharding_rule.py#L202) — Constructs a factor name for a batching dimension.
- `_get_batching_group(factor: str)` — [`L53`](../../../../../../raw/code/jax/jax/_src/custom_partitioning_sharding_rule.py#L53) — Extracts the batching group from a factor for leading batching dimensions.
- `_is_batching(factor: str)` — [`L43`](../../../../../../raw/code/jax/jax/_src/custom_partitioning_sharding_rule.py#L43) — Checks if a factor is a representation for leading batching dimensions.
- `_parse_values(rule: str)` — [`L212`](../../../../../../raw/code/jax/jax/_src/custom_partitioning_sharding_rule.py#L212) — Parses the LHS or RHS of an Einsum notation like string.
- `add_batching_dim_factor(batch_grp, batch_dim_order, factor_size)` — [`L428`](../../../../../../raw/code/jax/jax/_src/custom_partitioning_sharding_rule.py#L428)
- `add_factor(x)` — [`L249`](../../../../../../raw/code/jax/jax/_src/custom_partitioning_sharding_rule.py#L249)
- `add_factor(factor, size)` — [`L402`](../../../../../../raw/code/jax/jax/_src/custom_partitioning_sharding_rule.py#L402) — Adds a factor to factors_to_indices_sizes.
- `build_dim_mapping_for_compound_factors(i, j, factors)` — [`L431`](../../../../../../raw/code/jax/jax/_src/custom_partitioning_sharding_rule.py#L431)
- `factors_to_indices(factors)` — [`L448`](../../../../../../raw/code/jax/jax/_src/custom_partitioning_sharding_rule.py#L448)
- `get_message_for_value(i)` — [`L390`](../../../../../../raw/code/jax/jax/_src/custom_partitioning_sharding_rule.py#L390)
- `get_rank_for_value(i)` — [`L396`](../../../../../../raw/code/jax/jax/_src/custom_partitioning_sharding_rule.py#L396)
- `get_size_for_value_dim(i, j)` — [`L399`](../../../../../../raw/code/jax/jax/_src/custom_partitioning_sharding_rule.py#L399)
- `sdy_sharding_rule_to_mlir(rule: SdyShardingRule, operand_types: list[ir.Type], result_types: list[ir.Type])` — [`L363`](../../../../../../raw/code/jax/jax/_src/custom_partitioning_sharding_rule.py#L363) — Builds the MLIR representation for the sharding rule.
- `str_to_sdy_sharding_rule(rule: str, *, reduction_factors: tuple[str, ...] = (), need_replication_factors: tuple[str, ...] = (), permutation_factors: tuple[str, ...] = (), **factor_sizes: int)` — [`L314`](../../../../../../raw/code/jax/jax/_src/custom_partitioning_sharding_rule.py#L314) — Constructs a SdyShardingRule object from the Einsum notation like string.

## Module values
- `BATCHING` — [`L24`](../../../../../../raw/code/jax/jax/_src/custom_partitioning_sharding_rule.py#L24)
- `_BATCHING_DIM_FACTOR_PREFIX` — [`L28`](../../../../../../raw/code/jax/jax/_src/custom_partitioning_sharding_rule.py#L28)

