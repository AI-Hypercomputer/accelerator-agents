---
title: 'Module: src/maxdiffusion/utils/dynamic_modules_utils.py'
type: catalog
provenance: extracted
module: src/maxdiffusion/utils/dynamic_modules_utils.py
status: fresh
symbol_base: scip-python python maxdiffusion 0.0.0 `src.maxdiffusion.utils.dynamic_modules_utils`/
symbols:
  get_cached_module_file: get_cached_module_file().
  create_dynamic_module: create_dynamic_module().
  get_class_from_dynamic_module: get_class_from_dynamic_module().
  logger: logger.
  init_hf_modules: init_hf_modules().
  check_imports: check_imports().
  get_class_in_module: get_class_in_module().
  get_relative_import_files: get_relative_import_files().
  get_relative_imports: get_relative_imports().
  COMMUNITY_PIPELINES_URL: COMMUNITY_PIPELINES_URL.
  get_diffusers_versions: get_diffusers_versions().
  find_pipeline_class: find_pipeline_class().
---
# Module: [`src/maxdiffusion/utils/dynamic_modules_utils.py`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/utils/dynamic_modules_utils.py)

## Functions
- `check_imports(filename)` — [`L127`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/utils/dynamic_modules_utils.py#L127) — Check if the current Python environment contains all the libraries that are imported in a file.
- `create_dynamic_module(name: Union[str, os.PathLike])` — [`L65`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/utils/dynamic_modules_utils.py#L65) — Creates a dynamic module in the cache directory for modules.
- `find_pipeline_class(loaded_module)` — [`L171`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/utils/dynamic_modules_utils.py#L171) — Retrieve pipeline class that inherits from `DiffusionPipeline`. Note that there has to be exactly one class
- `get_cached_module_file(pretrained_model_name_or_path: Union[str, os.PathLike], module_file: str, cache_dir: Optional[Union[str, os.PathLike]] = None, force_download: bool = False, resume_download: bool = False, proxies: Optional[Dict[str, str]] = None, use_auth_token: Optional[Union[bool, str]] = None, revision: Optional[str] = None, local_files_only: bool = False)` — [`L198`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/utils/dynamic_modules_utils.py#L198) — Prepares Downloads a module from a local folder or a distant repo and returns its path inside the cached
- `get_class_from_dynamic_module(pretrained_model_name_or_path: Union[str, os.PathLike], module_file: str, class_name: Optional[str] = None, cache_dir: Optional[Union[str, os.PathLike]] = None, force_download: bool = False, resume_download: bool = False, proxies: Optional[Dict[str, str]] = None, use_auth_token: Optional[Union[bool, str]] = None, revision: Optional[str] = None, local_files_only: bool = False, **kwargs)` — [`L372`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/utils/dynamic_modules_utils.py#L372) — Extracts a class from a module file, present in the local folder or repository of a model.
- `get_class_in_module(class_name, module_path)` — [`L159`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/utils/dynamic_modules_utils.py#L159) — Import a module on the cache directory for modules and extract a class from it.
- `get_diffusers_versions()` — [`L44`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/utils/dynamic_modules_utils.py#L44)
- `get_relative_import_files(module_file)` — [`L98`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/utils/dynamic_modules_utils.py#L98) — Get the list of all files that are needed for a given module. Note that this function recurses through the relative
- `get_relative_imports(module_file)` — [`L80`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/utils/dynamic_modules_utils.py#L80) — Get the list of modules that are relatively imported in a module file.
- `init_hf_modules()` — [`L50`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/utils/dynamic_modules_utils.py#L50) — Creates the cache directory for modules with an init, and adds it to the Python path.

## Module values
- `COMMUNITY_PIPELINES_URL` — [`L36`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/utils/dynamic_modules_utils.py#L36)
- `logger` — [`L41`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/utils/dynamic_modules_utils.py#L41)

