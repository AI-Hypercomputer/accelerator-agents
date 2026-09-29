---
title: 'Module: easydel/utils/readme_generator.py'
type: catalog
provenance: extracted
module: easydel/utils/readme_generator.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.utils.readme_generator`/
symbols:
  test_model_info: test_model_info.
  ReadmeGenerator.generate_readme: ReadmeGenerator#generate_readme().
  ReadmeGenerator._get_template: ReadmeGenerator#_get_template().
  readme_content: readme_content.
  ModelInfo.task_display: ModelInfo#task_display().
  ReadmeGenerator.env: ReadmeGenerator#env.
  ModelInfo.attn_enum: ModelInfo#attn_enum().
  ReadmeGenerator._compiled_template: ReadmeGenerator#_compiled_template.
  ModelInfo: ModelInfo#
  ModelInfo.name: ModelInfo#name.
  ModelInfo.repo_id: ModelInfo#repo_id.
  ModelInfo.pipeline_tag: ModelInfo#pipeline_tag().
  ModelInfo.auto_class: ModelInfo#auto_class().
  ModelInfo.repo_badge_message: ModelInfo#repo_badge_message().
  ModelInfo.model_task: ModelInfo#model_task.
  ModelInfo.attn_mechanism: ModelInfo#attn_mechanism.
  ReadmeGenerator: ReadmeGenerator#
  ModelInfo.version: ModelInfo#version.
  ModelInfo.repo_is_local: ModelInfo#repo_is_local().
  generator: generator.
  ModelInfo.model_type: ModelInfo#model_type.
  ModelInfo.model_type_for_display: ModelInfo#model_type_for_display().
  logger: logger.
  ModelInfo.description: ModelInfo#description.
  ReadmeGenerator.template_name: ReadmeGenerator#template_name.
  DEFAULT_MODEL_CARD_TEMPLATE_NAME: DEFAULT_MODEL_CARD_TEMPLATE_NAME.
  _collapse_extra_blank_lines: _collapse_extra_blank_lines().
  _get_default_easydel_version: _get_default_easydel_version().
  _slugify: _slugify().
  _normalize_identifier: _normalize_identifier().
  EASYDEL_TRAINER_README_TEMPLATE: EASYDEL_TRAINER_README_TEMPLATE.
  JINJA_TEMPLATE: JINJA_TEMPLATE.
  ModelInfo.features: ModelInfo#features.
  ModelInfo.supported_tasks: ModelInfo#supported_tasks.
  ModelInfo.limitations: ModelInfo#limitations.
  ModelInfo.overview: ModelInfo#overview.
  _TASK_DISPLAY_BY_SLUG._TASK_DISPLAY_BY_SLUG: _TASK_DISPLAY_BY_SLUG._TASK_DISPLAY_BY_SLUG.
  _AUTO_CLASS_BY_TASK_DISPLAY._AUTO_CLASS_BY_TASK_DISPLAY: _AUTO_CLASS_BY_TASK_DISPLAY._AUTO_CLASS_BY_TASK_DISPLAY.
  _PIPELINE_TAG_BY_TASK_DISPLAY._PIPELINE_TAG_BY_TASK_DISPLAY: _PIPELINE_TAG_BY_TASK_DISPLAY._PIPELINE_TAG_BY_TASK_DISPLAY.
  _ATTN_ENUM_BY_MECHANISM_KEY._ATTN_ENUM_BY_MECHANISM_KEY: _ATTN_ENUM_BY_MECHANISM_KEY._ATTN_ENUM_BY_MECHANISM_KEY.
  ModelInfo.__post_init__: ModelInfo#__post_init__().
  ReadmeGenerator.__init__: ReadmeGenerator#__init__().
---
# Module: [`easydel/utils/readme_generator.py`](../../../../../../raw/code/EasyDeL/easydel/utils/readme_generator.py)

## Classes
### `ModelInfo`
- def: [`easydel/utils/readme_generator.py:573`](../../../../../../raw/code/EasyDeL/easydel/utils/readme_generator.py#L573)
- doc: Model information container. Used to pass data to the Jinja template.
- signature: `class ModelInfo:`
- members:
  - `attn_enum(self)` — [`L622`](../../../../../../raw/code/EasyDeL/easydel/utils/readme_generator.py#L622)
  - `auto_class(self)` — [`L618`](../../../../../../raw/code/EasyDeL/easydel/utils/readme_generator.py#L618)
  - `model_type_for_display(self)` — [`L604`](../../../../../../raw/code/EasyDeL/easydel/utils/readme_generator.py#L604)
  - `pipeline_tag(self)` — [`L614`](../../../../../../raw/code/EasyDeL/easydel/utils/readme_generator.py#L614)
  - `repo_badge_message(self)` — [`L641`](../../../../../../raw/code/EasyDeL/easydel/utils/readme_generator.py#L641)
  - `repo_is_local(self)` — [`L627`](../../../../../../raw/code/EasyDeL/easydel/utils/readme_generator.py#L627)
  - `task_display(self)` — [`L608`](../../../../../../raw/code/EasyDeL/easydel/utils/readme_generator.py#L608)
  - `attn_mechanism` — [`L587`](../../../../../../raw/code/EasyDeL/easydel/utils/readme_generator.py#L587)
  - `description` — [`L581`](../../../../../../raw/code/EasyDeL/easydel/utils/readme_generator.py#L581)
  - `features` — [`L591`](../../../../../../raw/code/EasyDeL/easydel/utils/readme_generator.py#L591)
  - `limitations` — [`L593`](../../../../../../raw/code/EasyDeL/easydel/utils/readme_generator.py#L593)
  - `model_task` — [`L583`](../../../../../../raw/code/EasyDeL/easydel/utils/readme_generator.py#L583)
  - `model_type` — [`L582`](../../../../../../raw/code/EasyDeL/easydel/utils/readme_generator.py#L582)
  - `name` — [`L578`](../../../../../../raw/code/EasyDeL/easydel/utils/readme_generator.py#L578)
  - `overview` — [`L595`](../../../../../../raw/code/EasyDeL/easydel/utils/readme_generator.py#L595)
  - `repo_id` — [`L580`](../../../../../../raw/code/EasyDeL/easydel/utils/readme_generator.py#L580)
  - `supported_tasks` — [`L592`](../../../../../../raw/code/EasyDeL/easydel/utils/readme_generator.py#L592)
  - `version` — [`L594`](../../../../../../raw/code/EasyDeL/easydel/utils/readme_generator.py#L594)
- protocol/private: `__post_init__`[`L597`](../../../../../../raw/code/EasyDeL/easydel/utils/readme_generator.py#L597)
- uses (calls/refs, reference-scoped): [`_ATTN_ENUM_BY_MECHANISM_KEY`](readme_generator.md#_ATTN_ENUM_BY_MECHANISM_KEY._ATTN_ENUM_BY_MECHANISM_KEY), [`_AUTO_CLASS_BY_TASK_DISPLAY`](readme_generator.md#_AUTO_CLASS_BY_TASK_DISPLAY._AUTO_CLASS_BY_TASK_DISPLAY), [`_PIPELINE_TAG_BY_TASK_DISPLAY`](readme_generator.md#_PIPELINE_TAG_BY_TASK_DISPLAY._PIPELINE_TAG_BY_TASK_DISPLAY), [`_TASK_DISPLAY_BY_SLUG`](readme_generator.md#_TASK_DISPLAY_BY_SLUG._TASK_DISPLAY_BY_SLUG), [`_get_default_easydel_version`](readme_generator.md#_get_default_easydel_version), [`_normalize_identifier`](readme_generator.md#_normalize_identifier), [`_slugify`](readme_generator.md#_slugify)
- used by: [`huggingface_to_easydel_sequential`](../infra/mixins/bridge.md#EasyBridgeMixin.huggingface_to_easydel_sequential), [`_model_card`](../infra/mixins/bridge.md#EasyBridgeMixin._model_card), [`test_model_info`](readme_generator.md#test_model_info), [`generate_readme`](readme_generator.md#ReadmeGenerator.generate_readme), [`readme_content`](readme_generator.md#readme_content)

### `ReadmeGenerator`
- def: [`easydel/utils/readme_generator.py:651`](../../../../../../raw/code/EasyDeL/easydel/utils/readme_generator.py#L651)
- doc: Generate README files for EasyDeL models.
- signature: `class ReadmeGenerator:`
- members:
  - `__init__(self, template_dir: str | None = None, *, template_name: str | None = None)` — [`L654`](../../../../../../raw/code/EasyDeL/easydel/utils/readme_generator.py#L654) — Initialize the README generator.
  - `generate_readme(self, model_info: ModelInfo, output_path: str | None = None)` — [`L697`](../../../../../../raw/code/EasyDeL/easydel/utils/readme_generator.py#L697) — Generate README content for a model.
  - `env` — [`L668`](../../../../../../raw/code/EasyDeL/easydel/utils/readme_generator.py#L668)
  - `template_name` — [`L665`](../../../../../../raw/code/EasyDeL/easydel/utils/readme_generator.py#L665)
- protocol/private: `_compiled_template`[`L676`](../../../../../../raw/code/EasyDeL/easydel/utils/readme_generator.py#L676), `_get_template`[`L678`](../../../../../../raw/code/EasyDeL/easydel/utils/readme_generator.py#L678)
- uses (calls/refs, reference-scoped): [`ModelInfo`](readme_generator.md#ModelInfo), [`name`](readme_generator.md#ModelInfo.name), [`logger`](readme_generator.md#logger), [`DEFAULT_MODEL_CARD_TEMPLATE_NAME`](readme_generator.md#DEFAULT_MODEL_CARD_TEMPLATE_NAME), [`JINJA_TEMPLATE`](readme_generator.md#JINJA_TEMPLATE), [`_collapse_extra_blank_lines`](readme_generator.md#_collapse_extra_blank_lines), `filters`
- used by: [`huggingface_to_easydel_sequential`](../infra/mixins/bridge.md#EasyBridgeMixin.huggingface_to_easydel_sequential), [`_model_card`](../infra/mixins/bridge.md#EasyBridgeMixin._model_card), [`readme_content`](readme_generator.md#readme_content), [`generator`](readme_generator.md#generator)

## Functions
- `_collapse_extra_blank_lines(text: str)` — [`L50`](../../../../../../raw/code/EasyDeL/easydel/utils/readme_generator.py#L50)
- `_get_default_easydel_version()` — [`L55`](../../../../../../raw/code/EasyDeL/easydel/utils/readme_generator.py#L55) — Best-effort EasyDeL version without importing the whole package.
- `_normalize_identifier(value: str)` — [`L96`](../../../../../../raw/code/EasyDeL/easydel/utils/readme_generator.py#L96)
- `_slugify(value: str)` — [`L89`](../../../../../../raw/code/EasyDeL/easydel/utils/readme_generator.py#L89)

## Module values
- `DEFAULT_MODEL_CARD_TEMPLATE_NAME` — [`L45`](../../../../../../raw/code/EasyDeL/easydel/utils/readme_generator.py#L45)
- `EASYDEL_TRAINER_README_TEMPLATE` — [`L175`](../../../../../../raw/code/EasyDeL/easydel/utils/readme_generator.py#L175)
- `JINJA_TEMPLATE` — [`L380`](../../../../../../raw/code/EasyDeL/easydel/utils/readme_generator.py#L380)
- `_ATTN_ENUM_BY_MECHANISM_KEY` — [`L142`](../../../../../../raw/code/EasyDeL/easydel/utils/readme_generator.py#L142)
- `_AUTO_CLASS_BY_TASK_DISPLAY` — [`L120`](../../../../../../raw/code/EasyDeL/easydel/utils/readme_generator.py#L120)
- `_PIPELINE_TAG_BY_TASK_DISPLAY` — [`L133`](../../../../../../raw/code/EasyDeL/easydel/utils/readme_generator.py#L133)
- `_TASK_DISPLAY_BY_SLUG` — [`L103`](../../../../../../raw/code/EasyDeL/easydel/utils/readme_generator.py#L103)
- `generator` — [`L747`](../../../../../../raw/code/EasyDeL/easydel/utils/readme_generator.py#L747)
- `logger` — [`L47`](../../../../../../raw/code/EasyDeL/easydel/utils/readme_generator.py#L47)
- `readme_content` — [`L748`](../../../../../../raw/code/EasyDeL/easydel/utils/readme_generator.py#L748)
- `test_model_info` — [`L733`](../../../../../../raw/code/EasyDeL/easydel/utils/readme_generator.py#L733)

