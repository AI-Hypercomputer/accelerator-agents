---
title: 'Module: src/maxdiffusion/utils/hub_utils.py'
type: catalog
provenance: extracted
module: src/maxdiffusion/utils/hub_utils.py
status: fresh
symbol_base: scip-python python maxdiffusion 0.0.0 `src.maxdiffusion.utils.hub_utils`/
symbols:
  http_user_agent: http_user_agent().
  _get_model_file: _get_model_file().
  PushToHubMixin: PushToHubMixin#
  trace: trace.
  old_cache_is_not_empty: old_cache_is_not_empty.
  _add_variant: _add_variant().
  move_cache: move_cache().
  cache_version_file: cache_version_file.
  f: f.
  logger: logger.
  HF_HUB_OFFLINE: HF_HUB_OFFLINE.
  cache_version: cache_version.
  PushToHubMixin._upload_folder: PushToHubMixin#_upload_folder().
  old_diffusers_cache: old_diffusers_cache.
  create_model_card: create_model_card().
  PushToHubMixin.push_to_hub: PushToHubMixin#push_to_hub().
  DISABLE_TELEMETRY: DISABLE_TELEMETRY.
  extract_commit_hash: extract_commit_hash().
  e: e.
  HUGGINGFACE_CO_TELEMETRY: HUGGINGFACE_CO_TELEMETRY.
  MODEL_CARD_TEMPLATE_PATH: MODEL_CARD_TEMPLATE_PATH.
  SESSION_ID: SESSION_ID.
  get_full_repo_name: get_full_repo_name().
  hf_cache_home: hf_cache_home.
---
# Module: [`src/maxdiffusion/utils/hub_utils.py`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/utils/hub_utils.py)

## Classes
### `PushToHubMixin`
- def: [`src/maxdiffusion/utils/hub_utils.py:359`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/utils/hub_utils.py#L359)
- doc: A Mixin to push a model, scheduler, or pipeline to the Hugging Face Hub.
- signature: `class PushToHubMixin:`
- members:
  - `_upload_folder(self, working_dir: Union[str, os.PathLike], repo_id: str, token: Optional[str] = None, commit_message: Optional[str] = None, create_pr: bool = False)` — [`L364`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/utils/hub_utils.py#L364) — Uploads all files in `working_dir` to `repo_id`.
  - `push_to_hub(self, repo_id: str, commit_message: Optional[str] = None, private: Optional[bool] = None, token: Optional[str] = None, create_pr: bool = False, safe_serialization: bool = True, variant: Optional[str] = None)` — [`L388`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/utils/hub_utils.py#L388) — Upload model, scheduler, or pipeline files to the 🤗 Hugging Face Hub.
- uses (calls/refs, reference-scoped): [`FlaxModelMixin`](../models/modeling_flax_utils.md#FlaxModelMixin), [`FlaxSchedulerMixin`](../schedulers/scheduling_utils_flax.md#FlaxSchedulerMixin), [`FlaxDiffusionPipeline`](../pipelines/pipeline_flax_utils.md#FlaxDiffusionPipeline), [`logger`](hub_utils.md#logger), `repo_id`, [`ModelMixin`](../models/modeling_utils.md#ModelMixin)
- used by: [`FlaxModelMixin`](../models/modeling_flax_utils.md#FlaxModelMixin), [`FlaxSchedulerMixin`](../schedulers/scheduling_utils_flax.md#FlaxSchedulerMixin), [`FlaxDiffusionPipeline`](../pipelines/pipeline_flax_utils.md#FlaxDiffusionPipeline), [`save_pretrained`](../models/modeling_utils.md#ModelMixin.save_pretrained), [`save_pretrained`](../pipelines/pipeline_flax_utils.md#FlaxDiffusionPipeline.save_pretrained), [`save_pretrained`](../models/modeling_flax_utils.md#FlaxModelMixin.save_pretrained), [`ModelMixin`](../models/modeling_utils.md#ModelMixin)

## Functions
- `_add_variant(weights_name: str, variant: Optional[str] = None)` — [`L238`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/utils/hub_utils.py#L238)
- `_get_model_file(pretrained_model_name_or_path, *, weights_name, subfolder, cache_dir, force_download, proxies, resume_download, local_files_only, use_auth_token, user_agent, revision, commit_hash=None)` — [`L247`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/utils/hub_utils.py#L247)
- `create_model_card(args, model_name)` — [`L103`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/utils/hub_utils.py#L103)
- `extract_commit_hash(resolved_file: Optional[str], commit_hash: Optional[str] = None)` — [`L152`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/utils/hub_utils.py#L152) — Extracts the commit hash from a resolved filename toward a cache file.
- `get_full_repo_name(model_id: str, organization: Optional[str] = None, token: Optional[str] = None)` — [`L93`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/utils/hub_utils.py#L93)
- `http_user_agent(user_agent: Union[Dict, str, None] = None)` — [`L69`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/utils/hub_utils.py#L69) — Formats a user-agent string with basic info about a request.
- `move_cache(old_cache_dir: Optional[str] = None, new_cache_dir: Optional[str] = None)` — [`L176`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/utils/hub_utils.py#L176)

## Module values
- `DISABLE_TELEMETRY` — [`L65`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/utils/hub_utils.py#L65)
- `HF_HUB_OFFLINE` — [`L64`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/utils/hub_utils.py#L64)
- `HUGGINGFACE_CO_TELEMETRY` — [`L66`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/utils/hub_utils.py#L66)
- `MODEL_CARD_TEMPLATE_PATH` — [`L62`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/utils/hub_utils.py#L62)
- `SESSION_ID` — [`L63`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/utils/hub_utils.py#L63)
- `cache_version` — [`L200`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/utils/hub_utils.py#L200)
- `cache_version_file` — [`L198`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/utils/hub_utils.py#L198)
- `e` — [`L218`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/utils/hub_utils.py#L218)
- `f` — [`L202`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/utils/hub_utils.py#L202)
- `hf_cache_home` — [`L170`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/utils/hub_utils.py#L170)
- `logger` — [`L59`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/utils/hub_utils.py#L59)
- `old_cache_is_not_empty` — [`L209`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/utils/hub_utils.py#L209)
- `old_diffusers_cache` — [`L173`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/utils/hub_utils.py#L173)
- `trace` — [`L219`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/utils/hub_utils.py#L219)

