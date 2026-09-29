---
title: 'Module: torchax/ops/jtorch.py'
type: catalog
provenance: extracted
module: torchax/ops/jtorch.py
status: fresh
symbol_base: scip-python python torchax 0.0.0 `torchax.ops.jtorch`/
symbols:
  register_function: register_function().
  getitem: getitem().
  _as_tensor: _as_tensor().
  _tensor: _tensor().
  functional_interpolate: functional_interpolate().
  _pad_sequence: _pad_sequence().
  empty: empty().
  _einsum: _einsum().
  scaled_dot_product_attention: scaled_dot_product_attention().
  _ones: _ones().
  _zeros: _zeros().
  _eye: _eye().
  _full: _full().
  arange: arange().
  empty_strided: empty_strided().
  rand: rand().
  randn: randn().
  randint: randint().
  logdet: logdet().
  linalg_slogdet: linalg_slogdet().
  linalg_solve: linalg_solve().
  linalg_solve_ex: linalg_solve_ex().
  linalg_svd: linalg_svd().
  svd: svd().
  _cdist: _cdist().
  lu: lu().
  _functional_max_pool2d: _functional_max_pool2d().
  _aten_allclose: _aten_allclose().
  _torch_angle: _torch_angle().
  _torch_argsort: _torch_argsort().
  _diag: _diag().
  one_hot: one_hot().
  pad: pad().
  _corrcoef: _corrcoef().
  _sparse_mm: _sparse_mm().
  _aten_isclose: _aten_isclose().
  linalg_det: linalg_det().
  unravel_index: unravel_index().
  tensor_split: tensor_split().
  matrix_power: matrix_power().
  lu_solve: lu_solve().
  linalg_tensorsolve: linalg_tensorsolve().
  functional_linear: functional_linear().
  torch_Tensor_repeat_interleave: torch_Tensor_repeat_interleave().
  _einsum.get_params: _einsum().get_params().
  _sdpa_reference: _sdpa_reference().
  getitem.is_narrow_slicing: getitem().is_narrow_slicing().
---
# Module: [`torchax/ops/jtorch.py`](../../../../../../raw/code/torchax/torchax/ops/jtorch.py)

## Functions
- `_as_tensor(data, dtype=None, device=None, env=None)` — [`L40`](../../../../../../raw/code/torchax/torchax/ops/jtorch.py#L40) — documented in [torchax-ops-jtorch](../../../concepts/torchax-ops-jtorch.md)
- `_aten_allclose(input, other, rtol=0.00001, atol=1e-8, equal_nan=False)` — [`L68`](../../../../../../raw/code/torchax/torchax/ops/jtorch.py#L68)
- `_aten_isclose(input, other, rtol=0.00001, atol=1e-8, equal_nan=False)` — [`L275`](../../../../../../raw/code/torchax/torchax/ops/jtorch.py#L275)
- `_cdist(x1, x2, p=2, compute_mode="use_mm_for_euclid_dist_if_necessary")` — [`L446`](../../../../../../raw/code/torchax/torchax/ops/jtorch.py#L446)
- `_corrcoef(x)` — [`L263`](../../../../../../raw/code/torchax/torchax/ops/jtorch.py#L263)
- `_diag(input, diagonal=0)` — [`L96`](../../../../../../raw/code/torchax/torchax/ops/jtorch.py#L96)
- `_einsum(equation, *operands)` — [`L102`](../../../../../../raw/code/torchax/torchax/ops/jtorch.py#L102) — documented in [torchax-ops-jtorch](../../../concepts/torchax-ops-jtorch.md)
- `_eye(n: int, m: int | None = None, *, dtype=None, **kwargs)` — [`L304`](../../../../../../raw/code/torchax/torchax/ops/jtorch.py#L304)
- `_full(size: Sequence[int], fill_value, *, dtype=None, **kwargs)` — [`L310`](../../../../../../raw/code/torchax/torchax/ops/jtorch.py#L310)
- `_functional_max_pool2d(input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False)` — [`L551`](../../../../../../raw/code/torchax/torchax/ops/jtorch.py#L551)
- `_ones(*size: int, dtype=None, **kwargs)` — [`L285`](../../../../../../raw/code/torchax/torchax/ops/jtorch.py#L285) — documented in [torchax-ops-jtorch](../../../concepts/torchax-ops-jtorch.md)
- `_pad_sequence(sequences, batch_first=False, padding_value=0, padding_side="right")` — [`L579`](../../../../../../raw/code/torchax/torchax/ops/jtorch.py#L579) — documented in [torchax-ops-jtorch](../../../concepts/torchax-ops-jtorch.md)
- `_sdpa_reference(query, key, value, attn_mask=None, dropout_p=0, is_causal=False, scale=None, enable_gqa=False)` — [`L119`](../../../../../../raw/code/torchax/torchax/ops/jtorch.py#L119)
- `_sparse_mm(mat1, mat2, reduce="sum")` — [`L270`](../../../../../../raw/code/torchax/torchax/ops/jtorch.py#L270) — documented in [torchax-ops-jtorch](../../../concepts/torchax-ops-jtorch.md)
- `_tensor(data, *, dtype=None, **kwargs)` — [`L52`](../../../../../../raw/code/torchax/torchax/ops/jtorch.py#L52)
- `_torch_angle(input)` — [`L73`](../../../../../../raw/code/torchax/torchax/ops/jtorch.py#L73)
- `_torch_argsort(input, dim=-1, descending=False, stable=False)` — [`L80`](../../../../../../raw/code/torchax/torchax/ops/jtorch.py#L80)
- `_zeros(*size: int, dtype=None, **kwargs)` — [`L294`](../../../../../../raw/code/torchax/torchax/ops/jtorch.py#L294) — documented in [torchax-ops-jtorch](../../../concepts/torchax-ops-jtorch.md)
- `arange(start, end=None, step=None, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False, pin_memory=None)` — [`L326`](../../../../../../raw/code/torchax/torchax/ops/jtorch.py#L326) — documented in [torchax-ops-jtorch](../../../concepts/torchax-ops-jtorch.md)
- `empty(*size: Sequence[int], dtype=None, **kwargs)` — [`L317`](../../../../../../raw/code/torchax/torchax/ops/jtorch.py#L317)
- `empty_strided(size, stride, *, dtype=None, layout=None, device=None, requires_grad=False, pin_memory=False)` — [`L346`](../../../../../../raw/code/torchax/torchax/ops/jtorch.py#L346) — documented in [torchax-ops-jtorch](../../../concepts/torchax-ops-jtorch.md)
- `functional_interpolate(input, size: tuple[int, int], scale_factor: float | None, mode: str, align_corners: bool, recompute_scale_factor: bool, antialias: bool)` — [`L502`](../../../../../../raw/code/torchax/torchax/ops/jtorch.py#L502) — documented in [torchax-ops-jtorch](../../../concepts/torchax-ops-jtorch.md)
- `functional_linear(self, weights, bias=None)` — [`L494`](../../../../../../raw/code/torchax/torchax/ops/jtorch.py#L494)
- `get_params(*a)` — [`L103`](../../../../../../raw/code/torchax/torchax/ops/jtorch.py#L103)
- `getitem(self, indexes)` — [`L238`](../../../../../../raw/code/torchax/torchax/ops/jtorch.py#L238) — documented in [torchax-ops-jtorch](../../../concepts/torchax-ops-jtorch.md)
- `is_narrow_slicing()` — [`L245`](../../../../../../raw/code/torchax/torchax/ops/jtorch.py#L245)
- `linalg_det(input)` — [`L280`](../../../../../../raw/code/torchax/torchax/ops/jtorch.py#L280)
- `linalg_slogdet(input)` — [`L402`](../../../../../../raw/code/torchax/torchax/ops/jtorch.py#L402)
- `linalg_solve(a, b)` — [`L413`](../../../../../../raw/code/torchax/torchax/ops/jtorch.py#L413)
- `linalg_solve_ex(a, b)` — [`L419`](../../../../../../raw/code/torchax/torchax/ops/jtorch.py#L419)
- `linalg_svd(a, full_matrices=True)` — [`L425`](../../../../../../raw/code/torchax/torchax/ops/jtorch.py#L425)
- `linalg_tensorsolve(A, b, dims=None)` — [`L471`](../../../../../../raw/code/torchax/torchax/ops/jtorch.py#L471)
- `logdet(input)` — [`L396`](../../../../../../raw/code/torchax/torchax/ops/jtorch.py#L396)
- `lu(A, **kwargs)` — [`L451`](../../../../../../raw/code/torchax/torchax/ops/jtorch.py#L451)
- `lu_solve(b, LU_data, LU_pivots, **kwargs)` — [`L463`](../../../../../../raw/code/torchax/torchax/ops/jtorch.py#L463)
- `matrix_power(A, n, *, out=None)` — [`L430`](../../../../../../raw/code/torchax/torchax/ops/jtorch.py#L430)
- `one_hot(tensor, num_classes=-1)` — [`L155`](../../../../../../raw/code/torchax/torchax/ops/jtorch.py#L155)
- `pad(tensor, pad, mode="constant", value=None)` — [`L162`](../../../../../../raw/code/torchax/torchax/ops/jtorch.py#L162)
- `rand(*size, **kwargs)` — [`L365`](../../../../../../raw/code/torchax/torchax/ops/jtorch.py#L365) — documented in [torchax-ops-jtorch](../../../concepts/torchax-ops-jtorch.md)
- `randint(*args, **kwargs)` — [`L391`](../../../../../../raw/code/torchax/torchax/ops/jtorch.py#L391) — documented in [torchax-ops-jtorch](../../../concepts/torchax-ops-jtorch.md)
- `randn(*size, generator=None, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False, pin_memory=False, env=None)` — [`L374`](../../../../../../raw/code/torchax/torchax/ops/jtorch.py#L374) — documented in [torchax-ops-jtorch](../../../concepts/torchax-ops-jtorch.md)
- `register_function(torch_func, **kwargs)` — [`L34`](../../../../../../raw/code/torchax/torchax/ops/jtorch.py#L34) — documented in [torchax-ops-jtorch](../../../concepts/torchax-ops-jtorch.md)
- `scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0, is_causal=False, scale=None, enable_gqa=False, env=None)` — [`L221`](../../../../../../raw/code/torchax/torchax/ops/jtorch.py#L221) — documented in [torchax-ops-jtorch](../../../concepts/torchax-ops-jtorch.md)
- `svd(a, some=True, compute_uv=True)` — [`L435`](../../../../../../raw/code/torchax/torchax/ops/jtorch.py#L435)
- `tensor_split(input, indices_or_sections, dim=0)` — [`L408`](../../../../../../raw/code/torchax/torchax/ops/jtorch.py#L408)
- `torch_Tensor_repeat_interleave(self, repeats, dim=None, *, output_size=None)` — [`L546`](../../../../../../raw/code/torchax/torchax/ops/jtorch.py#L546)
- `unravel_index(indices, shape)` — [`L360`](../../../../../../raw/code/torchax/torchax/ops/jtorch.py#L360)

