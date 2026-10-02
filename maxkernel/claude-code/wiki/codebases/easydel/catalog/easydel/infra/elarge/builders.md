---
title: 'Module: easydel/infra/elarge/builders.py'
type: catalog
provenance: extracted
module: easydel/infra/elarge/builders.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.infra.elarge.builders`/
symbols:
  build_model: build_model().
  build_esurge: build_esurge().
  build_sharded_source: build_sharded_source().
  _create_source_from_inform: _create_source_from_inform().
  to_load_state_kwargs: to_load_state_kwargs().
  to_from_pretrained_kwargs: to_from_pretrained_kwargs().
  build_dataset: build_dataset().
  to_data_mixture_kwargs: to_data_mixture_kwargs().
  build_tokenized_dataset: build_tokenized_dataset().
  to_esurge_kwargs: to_esurge_kwargs().
  tokenize_dataset: tokenize_dataset().
  save_dataset: save_dataset().
  _extract_dataset_name: _extract_dataset_name().
  _extract_dataset_name._normalize_name: _extract_dataset_name()._normalize_name().
  tokenize_dataset.tokenize_fn: tokenize_dataset().tokenize_fn().
  logger: logger.
---
# Module: [`easydel/infra/elarge/builders.py`](../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/builders.py)

## Functions
- `_create_source_from_inform(inform_cfg: Mapping[str, Any], mixture_cfg: Mapping[str, Any])` — [`L1407`](../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/builders.py#L1407) — Create a ShardedDataSource from an inform configuration.
- `_extract_dataset_name(inform_cfg: Mapping[str, Any], fallback_index: int = 0)` — [`L1265`](../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/builders.py#L1265) — Extract a meaningful dataset name from inform configuration.
- `_normalize_name(raw: str)` — [`L1319`](../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/builders.py#L1319)
- `build_dataset(cfg_like: eLMConfig | Mapping[str, Any])` — [`L818`](../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/builders.py#L818) — Build a dataset from ELM configuration with data mixture.
- `build_esurge(cfg_like: eLMConfig | Mapping[str, Any], model: EasyDeLBaseModule | None = None)` — [`L554`](../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/builders.py#L554) — Build an eSurge inference engine from ELM configuration.
- `build_model(cfg_like: eLMConfig | Mapping[str, Any])` — [`L275`](../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/builders.py#L275) — Build an EasyDeL model from ELM configuration.
- `build_sharded_source(cfg_like: eLMConfig | Mapping[str, Any])` — [`L1568`](../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/builders.py#L1568) — Build a ShardedDataSource from ELM configuration.
- `build_tokenized_dataset(cfg_like: eLMConfig | Mapping[str, Any], save: bool = True)` — [`L1115`](../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/builders.py#L1115) — Build, tokenize, and optionally save a dataset from ELM configuration.
- `save_dataset(dataset: Dataset | IterableDataset, output_path: str, format: str = "parquet", num_shards: int | None = None, compression: str | None = "snappy", max_shard_size: str | int = "500MB", overwrite: bool = False, push_to_hub: bool = False, hub_repo_id: str | None = None, hub_private: bool = False, hub_token: str | None = None)` — [`L997`](../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/builders.py#L997) — Save a dataset to disk or HuggingFace Hub.
- `to_data_mixture_kwargs(cfg_like: eLMConfig | Mapping[str, Any])` — [`L649`](../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/builders.py#L649) — Convert ELM configuration to kwargs for DatasetMixture creation.
- `to_esurge_kwargs(cfg_like: eLMConfig | Mapping[str, Any])` — [`L342`](../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/builders.py#L342) — Convert ELM configuration to kwargs for eSurge initialization.
- `to_from_pretrained_kwargs(cfg_like: eLMConfig | Mapping[str, Any])` — [`L87`](../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/builders.py#L87) — Convert ELM configuration to kwargs for model.from_pretrained() calls.
- `to_load_state_kwargs(cfg_like: eLMConfig | Mapping[str, Any])` — [`L171`](../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/builders.py#L171) — Convert ELM configuration to kwargs for :meth:`EasyDeLState.load_state`.
- `tokenize_dataset(dataset: Dataset | IterableDataset, tokenizer: PreTrainedTokenizerBase, text_field: str = "text", output_field: str = "tokens", max_length: int = 2048, truncation: bool = True, padding: bool | str = False, add_special_tokens: bool = True, return_attention_mask: bool = True, num_proc: int | None = None, batched: bool = True, batch_size: int = 1000, remove_columns: list[str] | None = None, keep_in_memory: bool = False)` — [`L879`](../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/builders.py#L879) — Tokenize a dataset using the provided tokenizer.
- `tokenize_fn(examples)` — [`L951`](../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/builders.py#L951)

## Module values
- `logger` — [`L84`](../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/builders.py#L84)

