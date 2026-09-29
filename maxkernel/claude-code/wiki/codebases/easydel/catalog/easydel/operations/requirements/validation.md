---
title: 'Module: easydel/operations/requirements/validation.py'
type: catalog
provenance: extracted
module: easydel/operations/requirements/validation.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.operations.requirements.validation`/
symbols:
  validate_metadata_availability: validate_metadata_availability().
  validate_cache_compatibility: validate_cache_compatibility().
  RequirementsValidator.validate: RequirementsValidator#validate().
  RequirementsValidator.validate_all: RequirementsValidator#validate_all().
  ValidationResult: ValidationResult#
  RequirementsValidator.get_combined_metadata_requirements: RequirementsValidator#get_combined_metadata_requirements().
  RequirementsValidator.get_common_cache_types: RequirementsValidator#get_common_cache_types().
  RequirementsValidator.__init__: RequirementsValidator#__init__().
  get_metadata_field_names: get_metadata_field_names().
  ValidationResult.success: ValidationResult#success().
  ValidationResult.failure: ValidationResult#failure().
  ValidationResult.raise_if_invalid: ValidationResult#raise_if_invalid().
  ValidationResult.errors: ValidationResult#errors.
  ValidationResult.warnings: ValidationResult#warnings.
  ValidationResult.__bool__: ValidationResult#__bool__().
  ValidationResult.is_valid: ValidationResult#is_valid.
  RequirementsValidator: RequirementsValidator#
  RequirementsValidator.cache_type: RequirementsValidator#cache_type.
  RequirementsValidator.available_metadata: RequirementsValidator#available_metadata.
  __all__: __all__.
  RequirementsValidator.mode: RequirementsValidator#mode.
---
# Module: [`easydel/operations/requirements/validation.py`](../../../../../../../raw/code/EasyDeL/easydel/operations/requirements/validation.py)

## Classes
### `RequirementsValidator`
- def: [`easydel/operations/requirements/validation.py:148`](../../../../../../../raw/code/EasyDeL/easydel/operations/requirements/validation.py#L148)
- doc: Validator for operation requirements against runtime configuration.
- signature: `class RequirementsValidator:`
- members:
  - `__init__(self, cache_type: CacheType, available_metadata: MetadataField, mode: ExecutionMode = ExecutionMode.MIXED)` — [`L155`](../../../../../../../raw/code/EasyDeL/easydel/operations/requirements/validation.py#L155) — Initialize the validator.
  - `get_combined_metadata_requirements(self, requirements_list: Sequence[OperationRequirements])` — [`L222`](../../../../../../../raw/code/EasyDeL/easydel/operations/requirements/validation.py#L222) — Get the union of all required metadata fields.
  - `get_common_cache_types(self, requirements_list: Sequence[OperationRequirements])` — [`L239`](../../../../../../../raw/code/EasyDeL/easydel/operations/requirements/validation.py#L239) — Get cache types supported by all operations.
  - `validate(self, requirements: OperationRequirements)` — [`L172`](../../../../../../../raw/code/EasyDeL/easydel/operations/requirements/validation.py#L172) — Validate operation requirements against runtime configuration.
  - `validate_all(self, requirements_list: Sequence[OperationRequirements])` — [`L198`](../../../../../../../raw/code/EasyDeL/easydel/operations/requirements/validation.py#L198) — Validate multiple operation requirements.
  - `available_metadata` — [`L169`](../../../../../../../raw/code/EasyDeL/easydel/operations/requirements/validation.py#L169)
  - `cache_type` — [`L168`](../../../../../../../raw/code/EasyDeL/easydel/operations/requirements/validation.py#L168)
  - `mode` — [`L170`](../../../../../../../raw/code/EasyDeL/easydel/operations/requirements/validation.py#L170)
- uses (calls/refs, reference-scoped): [`MetadataField`](types.md#MetadataField), [`CacheType`](types.md#CacheType), [`ExecutionMode`](types.md#ExecutionMode), [`OperationRequirements`](requirements.md#OperationRequirements), [`cache`](requirements.md#OperationRequirements.cache), [`supported`](requirements.md#CacheRequirements.supported), [`validate_metadata_availability`](validation.md#validate_metadata_availability), [`MIXED`](types.md#ExecutionMode.MIXED), [`validate_cache_compatibility`](validation.md#validate_cache_compatibility), [`any`](types.md#CacheType.any), [`required`](requirements.md#MetadataRequirements.required), [`ValidationResult`](validation.md#ValidationResult), [`metadata`](requirements.md#OperationRequirements.metadata), [`NONE`](types.md#MetadataField.NONE), [`failure`](validation.md#ValidationResult.failure), [`success`](validation.md#ValidationResult.success), [`errors`](validation.md#ValidationResult.errors), [`warnings`](validation.md#ValidationResult.warnings)

### `ValidationResult`
- def: [`easydel/operations/requirements/validation.py:38`](../../../../../../../raw/code/EasyDeL/easydel/operations/requirements/validation.py#L38)
- doc: Result of a validation check.
- signature: `class ValidationResult:`
- members:
  - `__bool__(self)` — [`L61`](../../../../../../../raw/code/EasyDeL/easydel/operations/requirements/validation.py#L61) — Allow using result in boolean context.
  - `failure(cls, errors: list[str], warnings: list[str] | None = None)` — [`L57`](../../../../../../../raw/code/EasyDeL/easydel/operations/requirements/validation.py#L57) — Create a failed validation result.
  - `raise_if_invalid(self, error_class: type[Exception] = ValueError)` — [`L65`](../../../../../../../raw/code/EasyDeL/easydel/operations/requirements/validation.py#L65) — Raise an exception if validation failed.
  - `success(cls, warnings: list[str] | None = None)` — [`L52`](../../../../../../../raw/code/EasyDeL/easydel/operations/requirements/validation.py#L52) — Create a successful validation result.
  - `errors` — [`L48`](../../../../../../../raw/code/EasyDeL/easydel/operations/requirements/validation.py#L48)
  - `is_valid` — [`L47`](../../../../../../../raw/code/EasyDeL/easydel/operations/requirements/validation.py#L47)
  - `warnings` — [`L49`](../../../../../../../raw/code/EasyDeL/easydel/operations/requirements/validation.py#L49)
- used by: [`validate_metadata_availability`](validation.md#validate_metadata_availability), [`validate_cache_compatibility`](validation.md#validate_cache_compatibility), [`validate`](validation.md#RequirementsValidator.validate), [`validate_all`](validation.md#RequirementsValidator.validate_all)

## Functions
- `get_metadata_field_names(fields: MetadataField)` — [`L78`](../../../../../../../raw/code/EasyDeL/easydel/operations/requirements/validation.py#L78) — Get human-readable names for metadata fields.
- `validate_cache_compatibility(requirements: OperationRequirements, cache_type: CacheType)` — [`L94`](../../../../../../../raw/code/EasyDeL/easydel/operations/requirements/validation.py#L94) — Validate that a cache type is compatible with operation requirements.
- `validate_metadata_availability(requirements: OperationRequirements, available: MetadataField)` — [`L119`](../../../../../../../raw/code/EasyDeL/easydel/operations/requirements/validation.py#L119) — Validate that all required metadata fields are available.

## Module values
- `__all__` — [`L28`](../../../../../../../raw/code/EasyDeL/easydel/operations/requirements/validation.py#L28)

