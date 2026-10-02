---
title: 'Module: src/maxdiffusion/models/wan/wan_utils.py'
type: catalog
provenance: extracted
module: src/maxdiffusion/models/wan/wan_utils.py
status: fresh
symbol_base: scip-python python maxdiffusion 0.0.0 `src.maxdiffusion.models.wan.wan_utils`/
symbols:
  load_base_wan_transformer: load_base_wan_transformer().
  load_fusionx_transformer: load_fusionx_transformer().
  load_causvid_transformer: load_causvid_transformer().
  load_wan_transformer: load_wan_transformer().
  load_wan_animate_transformer: load_wan_animate_transformer().
  load_wan_vae: load_wan_vae().
  get_key_and_value: get_key_and_value().
  _rename_wan_animate_pt_tuple_key: _rename_wan_animate_pt_tuple_key().
  get_wan_animate_key_and_value: get_wan_animate_key_and_value().
  CAUSVID_TRANSFORMER_MODEL_NAME_OR_PATH: CAUSVID_TRANSFORMER_MODEL_NAME_OR_PATH.
  WAN_21_FUSION_X_MODEL_NAME_OR_PATH: WAN_21_FUSION_X_MODEL_NAME_OR_PATH.
  _tuple_str_to_int: _tuple_str_to_int().
  _normalize_animate_list_key: _normalize_animate_list_key().
  rename_for_custom_trasformer: rename_for_custom_trasformer().
  _build_random_flax_state_dict: _build_random_flax_state_dict().
  _rename_common_wan_transformer_key: _rename_common_wan_transformer_key().
  rename_for_nnx: rename_for_nnx().
  _is_motion_encoder_custom_weight: _is_motion_encoder_custom_weight().
---
# Module: [`src/maxdiffusion/models/wan/wan_utils.py`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/wan/wan_utils.py)

## Functions
- `_build_random_flax_state_dict(eval_shapes)` — [`L112`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/wan/wan_utils.py#L112)
- `_is_motion_encoder_custom_weight(pt_key: str)` — [`L357`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/wan/wan_utils.py#L357) — Returns True for FlaxMotionConv2d/FlaxMotionLinear weight keys that must NOT be renamed to kernel.
- `_normalize_animate_list_key(key)` — [`L42`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/wan/wan_utils.py#L42) — Convert flattened animate list names into nnx.List-style tuple paths.
- `_rename_common_wan_transformer_key(renamed_pt_key: str)` — [`L120`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/wan/wan_utils.py#L120)
- `_rename_wan_animate_pt_tuple_key(pt_key: str)` — [`L150`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/wan/wan_utils.py#L150)
- `_tuple_str_to_int(in_tuple)` — [`L32`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/wan/wan_utils.py#L32)
- `get_key_and_value(pt_tuple_key, tensor, flax_state_dict, random_flax_state_dict, scan_layers, num_layers=40)` — [`L91`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/wan/wan_utils.py#L91)
- `get_wan_animate_key_and_value(pt_tuple_key, tensor, flax_state_dict, random_flax_state_dict, scan_layers, is_motion_custom_weight=False, num_layers=40)` — [`L161`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/wan/wan_utils.py#L161)
- `load_base_wan_transformer(pretrained_model_name_or_path: str, eval_shapes: dict, device: str, hf_download: bool = True, num_layers: int = 40, scan_layers: bool = True, subfolder: str = "")` — [`L287`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/wan/wan_utils.py#L287)
- `load_causvid_transformer(pretrained_model_name_or_path: str, eval_shapes: dict, device: str, hf_download: bool = True, num_layers: int = 40, scan_layers: bool = True)` — [`L226`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/wan/wan_utils.py#L226)
- `load_fusionx_transformer(pretrained_model_name_or_path: str, eval_shapes: dict, device: str, hf_download: bool = True, num_layers: int = 40, scan_layers: bool = True)` — [`L181`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/wan/wan_utils.py#L181)
- `load_wan_animate_transformer(pretrained_model_name_or_path: str, eval_shapes: dict, device: str, hf_download: bool = True, num_layers: int = 40, scan_layers: bool = True, subfolder: str = "transformer")` — [`L372`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/wan/wan_utils.py#L372) — Loads WanAnimate transformer weights from a HuggingFace checkpoint.
- `load_wan_transformer(pretrained_model_name_or_path: str, eval_shapes: dict, device: str, hf_download: bool = True, num_layers: int = 40, scan_layers: bool = True, subfolder: str = "")` — [`L268`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/wan/wan_utils.py#L268)
- `load_wan_vae(pretrained_model_name_or_path: str, eval_shapes: dict, device: str, hf_download: bool = True)` — [`L452`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/wan/wan_utils.py#L452)
- `rename_for_custom_trasformer(key)` — [`L65`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/wan/wan_utils.py#L65)
- `rename_for_nnx(key)` — [`L58`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/wan/wan_utils.py#L58)

## Module values
- `CAUSVID_TRANSFORMER_MODEL_NAME_OR_PATH` — [`L28`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/wan/wan_utils.py#L28)
- `WAN_21_FUSION_X_MODEL_NAME_OR_PATH` — [`L29`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/wan/wan_utils.py#L29)

