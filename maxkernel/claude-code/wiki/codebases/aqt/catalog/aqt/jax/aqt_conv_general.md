---
title: 'Module: aqt/jax/aqt_conv_general.py'
type: catalog
provenance: extracted
module: aqt/jax/aqt_conv_general.py
status: fresh
symbol_base: scip-python python aqt 0.0.0 `aqt.jax.v2.aqt_conv_general`/
symbols:
  conv_general_dilated_make: conv_general_dilated_make().
  make_conv_general_dilated_with_qt: make_conv_general_dilated_with_qt().
  make_conv_general_dilated: make_conv_general_dilated().
  make_conv_general_dilated_with_qt.my_conv_general_dilated: make_conv_general_dilated_with_qt().my_conv_general_dilated().
  make_conv_general_dilated.my_conv_general_dilated: make_conv_general_dilated().my_conv_general_dilated().
---
# Module: [`aqt/jax/aqt_conv_general.py`](../../../../../../raw/code/aqt/aqt/jax/aqt_conv_general.py)

## Functions
- `conv_general_dilated_make(spatial_dimensions: int, lhs_bits: None | int = None, rhs_bits: None | int = None, initialize_calibration: bool = True)` — [`L193`](../../../../../../raw/code/aqt/aqt/jax/aqt_conv_general.py#L193) — Create quantization config conv_general_dilated.
- `make_conv_general_dilated(cfg: aqt_dot_general.DotGeneralRaw)` — [`L154`](../../../../../../raw/code/aqt/aqt/jax/aqt_conv_general.py#L154) — Makes quantized lax.make_conv_general_dilated replacement.
- `make_conv_general_dilated_with_qt(cfg: aqt_dot_general.DotGeneralRaw)` — [`L33`](../../../../../../raw/code/aqt/aqt/jax/aqt_conv_general.py#L33) — Makes quantized lax.make_conv_general_dilated replacement which returns QTensors.
- `my_conv_general_dilated(lhs, rhs, window_strides, padding, lhs_qt: None | aqt_tensor.QTensor = None, rhs_qt: None | aqt_tensor.QTensor = None, lhs_dilation=None, rhs_dilation=None, dimension_numbers=None, feature_group_count=1, batch_group_count=1, precision=None, preferred_element_type=None)` — [`L38`](../../../../../../raw/code/aqt/aqt/jax/aqt_conv_general.py#L38)
- `my_conv_general_dilated(lhs, rhs, window_strides, padding, lhs_dilation=None, rhs_dilation=None, dimension_numbers=None, feature_group_count=1, batch_group_count=1, precision=None, preferred_element_type=None)` — [`L158`](../../../../../../raw/code/aqt/aqt/jax/aqt_conv_general.py#L158)

