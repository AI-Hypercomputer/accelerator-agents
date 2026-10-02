---
title: 'Module: jax/_src/random/core.py'
type: catalog
provenance: extracted
module: jax/_src/random/core.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax._src.random.core`/
symbols:
  _check_prng_key: _check_prng_key().
  uniform: uniform().
  _randint: _randint().
  choice: choice().
  _gamma_one: _gamma_one().
  normal: normal().
  randint: randint().
  _uniform: _uniform().
  Shape: Shape.
  multinomial: multinomial().
  multivariate_normal: multivariate_normal().
  poisson: poisson().
  loggamma: loggamma().
  gumbel: gumbel().
  RealArray: RealArray.
  beta: beta().
  categorical: categorical().
  pareto: pareto().
  t: t().
  rayleigh: rayleigh().
  wald: wald().
  triangular: triangular().
  lognormal: lognormal().
  _binomial: _binomial().
  bits: bits().
  permutation: permutation().
  bernoulli: bernoulli().
  gamma: gamma().
  chisquare: chisquare().
  canonicalize_sharding_for_samplers: canonicalize_sharding_for_samplers().
  dirichlet: dirichlet().
  exponential: exponential().
  truncated_normal: truncated_normal().
  f: f().
  orthogonal: orthogonal().
  generalized_normal: generalized_normal().
  cauchy: cauchy().
  laplace: laplace().
  logistic: logistic().
  ball: ball().
  geometric: geometric().
  _geometric: _geometric().
  DTypeLikeFloat: DTypeLikeFloat.
  maxwell: maxwell().
  key: key().
  maybe_auto_axes: maybe_auto_axes().
  _gumbel: _gumbel().
  rademacher: rademacher().
  _split: _split().
  _truncated_normal: _truncated_normal().
  _gamma_grad: _gamma_grad().
  binomial: binomial().
  _check_shape: _check_shape().
  _beta: _beta().
  _laplace: _laplace().
  _t: _t().
  double_sided_maxwell: double_sided_maxwell().
  weibull_min: weibull_min().
  _wald: _wald().
  split: split().
  _cauchy: _cauchy().
  _poisson_rejection: _poisson_rejection().
  _poisson: _poisson().
  _categorical: _categorical().
  _binomial_inversion: _binomial_inversion().
  _shuffle: _shuffle().
  _check_broadcast_shapes: _check_broadcast_shapes().
  _gamma: _gamma().
  _check_all_safe_to_cast: _check_all_safe_to_cast().
  _safe_int_to_float: _safe_int_to_float().
  _logistic: _logistic().
  _rayleigh: _rayleigh().
  _btrs: _btrs().
  resolve_prng_impl: resolve_prng_impl().
  random_gamma_p: random_gamma_p.
  _key: _key().
  _multivariate_normal: _multivariate_normal().
  _bernoulli: _bernoulli().
  _poisson_knuth: _poisson_knuth().
  _chisquare: _chisquare().
  _double_sided_maxwell: _double_sided_maxwell().
  _triangular: _triangular().
  _convert_and_clip_integer: _convert_and_clip_integer().
  _normal_real: _normal_real().
  _gamma_impl: _gamma_impl().
  _random_bits: _random_bits().
  fold_in: fold_in().
  random_insert_pvary: random_insert_pvary().
  _stirling_approx_tail: _stirling_approx_tail().
  _dirichlet: _dirichlet().
  _exponential: _exponential().
  _gamma_one._cond_fn: _gamma_one()._cond_fn().
  _gamma_one._body_fn: _gamma_one()._body_fn().
  _poisson_rejection.body_fn: _poisson_rejection().body_fn().
  _f: _f().
  _key_impl: _key_impl().
  random_clone_p: random_clone_p.
  PRNGKey: PRNGKey().
  _normal: _normal().
  _pareto: _pareto().
  _weibull_min: _weibull_min().
  _ball: _ball().
  _lognormal: _lognormal().
  _btrs.body_fn: _btrs().body_fn().
  wrap_key_data: wrap_key_data().
  PRNGSpecDesc: PRNGSpecDesc.
  key_data: key_data().
  key_impl: key_impl().
  _key_data: _key_data().
  _gamma_one._body_fn._next_kxv: _gamma_one()._body_fn()._next_kxv().
  _rademacher: _rademacher().
  _generalized_normal: _generalized_normal().
  _isnan: _isnan().
  _return_prng_keys: _return_prng_keys().
  PRNGSpec._impl: PRNGSpec#_impl.
  _key_spec: _key_spec().
  _permutation: _permutation().
  _poisson_knuth.body_fn: _poisson_knuth().body_fn().
  _maxwell: _maxwell().
  _orthogonal: _orthogonal().
  _binomial_inversion.body_fn: _binomial_inversion().body_fn().
  DTypeLikeInt: DTypeLikeInt.
  PRNGSpec: PRNGSpec#
  IntegerArray: IntegerArray.
  key_dtype: key_dtype().
  _gamma_batching_rule: _gamma_batching_rule().
  clone: clone().
  default_prng_impl: default_prng_impl().
  PRNGImpl: PRNGImpl.
  PRNGSpec.__eq__: PRNGSpec#__eq__().
  DTypeLikeUInt: DTypeLikeUInt.
  KeyDTypeLike: KeyDTypeLike.
  _random_gamma_abstract_eval: _random_gamma_abstract_eval().
  multinomial.f: multinomial().f().
  PRNGSpec.__init__: PRNGSpec#__init__().
  PRNGSpec.__repr__: PRNGSpec#__repr__().
  PRNGSpec.__str__: PRNGSpec#__str__().
  PRNGSpec.__hash__: PRNGSpec#__hash__().
  UINT_DTYPES: UINT_DTYPES.
  _poisson_knuth.cond_fn: _poisson_knuth().cond_fn().
  _poisson_rejection.cond_fn: _poisson_rejection().cond_fn().
  _binomial_inversion.cond_fn: _binomial_inversion().cond_fn().
  _btrs.cond_fn: _btrs().cond_fn().
  PRNGSpec.__slots__: PRNGSpec#__slots__.
---
# Module: [`jax/_src/random/core.py`](../../../../../../../raw/code/jax/jax/_src/random/core.py)

## Classes
### `PRNGSpec`
- def: [`jax/_src/random/core.py:160`](../../../../../../../raw/code/jax/jax/_src/random/core.py#L160)
- doc: Specifies a PRNG key implementation.
- signature: `class PRNGSpec:`
- protocol/private: `__eq__`[`L178`](../../../../../../../raw/code/jax/jax/_src/random/core.py#L178), `__hash__`[`L175`](../../../../../../../raw/code/jax/jax/_src/random/core.py#L175), `__init__`[`L166`](../../../../../../../raw/code/jax/jax/_src/random/core.py#L166), `__repr__`[`L169`](../../../../../../../raw/code/jax/jax/_src/random/core.py#L169), `__slots__`[`L163`](../../../../../../../raw/code/jax/jax/_src/random/core.py#L163), `__str__`[`L172`](../../../../../../../raw/code/jax/jax/_src/random/core.py#L172), `_impl`[`L164`](../../../../../../../raw/code/jax/jax/_src/random/core.py#L164)
- uses (calls/refs, reference-scoped): [`PRNGImpl`](core.md#PRNGImpl)
- used by: [`resolve_prng_impl`](core.md#resolve_prng_impl), [`PRNGSpecDesc`](core.md#PRNGSpecDesc), [`key_impl`](core.md#key_impl), [`_key_spec`](core.md#_key_spec)

## Functions
- `PRNGKey(seed: int | ArrayLike, *, impl: PRNGSpecDesc | None = None)` — [`L260`](../../../../../../../raw/code/jax/jax/_src/random/core.py#L260) — Create a legacy PRNG key given an integer seed.
- `_ball(key, p, d, shape, dtype)` — [`L2804`](../../../../../../../raw/code/jax/jax/_src/random/core.py#L2804)
- `_bernoulli(key: Array, p: Array, shape: Shape | None, mode: str)` — [`L1207`](../../../../../../../raw/code/jax/jax/_src/random/core.py#L1207)
- `_beta(key, a, b, shape, dtype)` — [`L1278`](../../../../../../../raw/code/jax/jax/_src/random/core.py#L1278)
- `_binomial(key, count, prob, shape, dtype)` — [`L3224`](../../../../../../../raw/code/jax/jax/_src/random/core.py#L3224)
- `_binomial_inversion(key, count, prob, shape, dtype, max_iters)` — [`L3149`](../../../../../../../raw/code/jax/jax/_src/random/core.py#L3149)
- `_body_fn(kXVU)` — [`L1525`](../../../../../../../raw/code/jax/jax/_src/random/core.py#L1525)
- `_btrs(key, count, prob, shape, dtype, max_iters)` — [`L3176`](../../../../../../../raw/code/jax/jax/_src/random/core.py#L3176)
- `_categorical(key, logits_arr, shape, batch_shape, axis, replace, mode)` — [`L2065`](../../../../../../../raw/code/jax/jax/_src/random/core.py#L2065)
- `_cauchy(key, shape, dtype)` — [`L1344`](../../../../../../../raw/code/jax/jax/_src/random/core.py#L1344)
- `_check_all_safe_to_cast(name: str, dtype: DTypeLike, *args)` — [`L82`](../../../../../../../raw/code/jax/jax/_src/random/core.py#L82)
- `_check_broadcast_shapes(name: str, shape: tuple | Shape | None, *args: ArrayLike)` — [`L69`](../../../../../../../raw/code/jax/jax/_src/random/core.py#L69)
- `_check_prng_key(name: str, key: ArrayLike, *, allow_batched: bool = False)` — [`L92`](../../../../../../../raw/code/jax/jax/_src/random/core.py#L92)
- `_check_shape(name: str, shape: Shape, *param_shapes)` — [`L402`](../../../../../../../raw/code/jax/jax/_src/random/core.py#L402)
- `_chisquare(key, df, shape, dtype, out_sharding)` — [`L2364`](../../../../../../../raw/code/jax/jax/_src/random/core.py#L2364)
- `_cond_fn(kXVU)` — [`L1513`](../../../../../../../raw/code/jax/jax/_src/random/core.py#L1513)
- `_convert_and_clip_integer(val: Array, dtype: DType)` — [`L557`](../../../../../../../raw/code/jax/jax/_src/random/core.py#L557) — Convert integer-typed val to specified integer dtype, clipping to dtype
- `_dirichlet(key, alpha, shape, dtype)` — [`L1409`](../../../../../../../raw/code/jax/jax/_src/random/core.py#L1409)
- `_double_sided_maxwell(key, loc, scale, shape, dtype)` — [`L2581`](../../../../../../../raw/code/jax/jax/_src/random/core.py#L2581)
- `_exponential(key, shape, dtype)` — [`L1472`](../../../../../../../raw/code/jax/jax/_src/random/core.py#L1472)
- `_f(key, dfnum, dfden, shape, dtype, out_sharding)` — [`L2429`](../../../../../../../raw/code/jax/jax/_src/random/core.py#L2429)
- `_gamma(key, a, shape, dtype, log_space=False)` — [`L1736`](../../../../../../../raw/code/jax/jax/_src/random/core.py#L1736)
- `_gamma_batching_rule(batched_args, batch_dims, *, log_space)` — [`L1596`](../../../../../../../raw/code/jax/jax/_src/random/core.py#L1596)
- `_gamma_grad(sample, a, *, log_space)` — [`L1558`](../../../../../../../raw/code/jax/jax/_src/random/core.py#L1558)
- `_gamma_impl(key, a, *, log_space, use_vmap=False)` — [`L1579`](../../../../../../../raw/code/jax/jax/_src/random/core.py#L1579)
- `_gamma_one(key: Array, alpha, log_space)` — [`L1479`](../../../../../../../raw/code/jax/jax/_src/random/core.py#L1479)
- `_generalized_normal(key, p, shape, dtype)` — [`L2756`](../../../../../../../raw/code/jax/jax/_src/random/core.py#L2756)
- `_geometric(key, p, shape, dtype, out_sharding)` — [`L2985`](../../../../../../../raw/code/jax/jax/_src/random/core.py#L2985)
- `_gumbel(key, shape, dtype, mode)` — [`L1971`](../../../../../../../raw/code/jax/jax/_src/random/core.py#L1971)
- `_isnan(x: ArrayLike)` — [`L88`](../../../../../../../raw/code/jax/jax/_src/random/core.py#L88)
- `_key(ctor_name: str, seed: int | ArrayLike, impl_spec: PRNGSpecDesc | None)` — [`L220`](../../../../../../../raw/code/jax/jax/_src/random/core.py#L220)
- `_key_data(keys: Array)` — [`L348`](../../../../../../../raw/code/jax/jax/_src/random/core.py#L348)
- `_key_impl(keys: Array)` — [`L334`](../../../../../../../raw/code/jax/jax/_src/random/core.py#L334)
- `_key_spec(keys: Array)` — [`L339`](../../../../../../../raw/code/jax/jax/_src/random/core.py#L339)
- `_laplace(key, shape, dtype)` — [`L2138`](../../../../../../../raw/code/jax/jax/_src/random/core.py#L2138)
- `_logistic(key, shape, dtype)` — [`L2187`](../../../../../../../raw/code/jax/jax/_src/random/core.py#L2187)
- `_lognormal(key, sigma, shape, dtype)` — [`L3117`](../../../../../../../raw/code/jax/jax/_src/random/core.py#L3117)
- `_maxwell(key, shape, dtype, out_sharding)` — [`L2534`](../../../../../../../raw/code/jax/jax/_src/random/core.py#L2534)
- `_multivariate_normal(key, mean, cov, shape, dtype, method)` — [`L1041`](../../../../../../../raw/code/jax/jax/_src/random/core.py#L1041)
- `_next_kxv(kxv)` — [`L1526`](../../../../../../../raw/code/jax/jax/_src/random/core.py#L1526)
- `_normal(key, shape, dtype)` — [`L955`](../../../../../../../raw/code/jax/jax/_src/random/core.py#L955)
- `_normal_real(key, shape, dtype)` — [`L968`](../../../../../../../raw/code/jax/jax/_src/random/core.py#L968)
- `_orthogonal(key, n, _m, shape, dtype)` — [`L2699`](../../../../../../../raw/code/jax/jax/_src/random/core.py#L2699)
- `_pareto(key, b, shape, dtype)` — [`L2243`](../../../../../../../raw/code/jax/jax/_src/random/core.py#L2243)
- `_permutation(key, x, axis, independent)` — [`L785`](../../../../../../../raw/code/jax/jax/_src/random/core.py#L785)
- `_poisson(key, lam, shape, dtype)` — [`L1816`](../../../../../../../raw/code/jax/jax/_src/random/core.py#L1816)
- `_poisson_knuth(key, lam, shape, dtype, max_iters)` — [`L1750`](../../../../../../../raw/code/jax/jax/_src/random/core.py#L1750)
- `_poisson_rejection(key, lam, shape, dtype, max_iters)` — [`L1773`](../../../../../../../raw/code/jax/jax/_src/random/core.py#L1773)
- `_rademacher(key, shape, dtype, out_sharding)` — [`L2482`](../../../../../../../raw/code/jax/jax/_src/random/core.py#L2482)
- `_randint(key, minval, maxval, shape, dtype)` — [`L666`](../../../../../../../raw/code/jax/jax/_src/random/core.py#L666)
- `_random_bits(key: Array, bit_width: int, shape: Shape)` — [`L136`](../../../../../../../raw/code/jax/jax/_src/random/core.py#L136)
- `_random_gamma_abstract_eval(key, a, **_)` — [`L1608`](../../../../../../../raw/code/jax/jax/_src/random/core.py#L1608)
- `_rayleigh(key, scale, shape, dtype)` — [`L2862`](../../../../../../../raw/code/jax/jax/_src/random/core.py#L2862)
- `_return_prng_keys(was_wrapped, key)` — [`L127`](../../../../../../../raw/code/jax/jax/_src/random/core.py#L127)
- `_safe_int_to_float(bits, dtype)` — [`L1944`](../../../../../../../raw/code/jax/jax/_src/random/core.py#L1944) — Converts bits: u32[2,...] into f32[...] in the range (0,1).
- `_shuffle(key, x, axis)` — [`L793`](../../../../../../../raw/code/jax/jax/_src/random/core.py#L793)
- `_split(key: Array, num: int | tuple[int, ...] = 2)` — [`L308`](../../../../../../../raw/code/jax/jax/_src/random/core.py#L308)
- `_stirling_approx_tail(k)` — [`L3123`](../../../../../../../raw/code/jax/jax/_src/random/core.py#L3123)
- `_t(key, df, shape, dtype)` — [`L2299`](../../../../../../../raw/code/jax/jax/_src/random/core.py#L2299)
- `_triangular(key, left, mode, right, shape, dtype)` — [`L3058`](../../../../../../../raw/code/jax/jax/_src/random/core.py#L3058)
- `_truncated_normal(key, lower, upper, shape, dtype)` — [`L1128`](../../../../../../../raw/code/jax/jax/_src/random/core.py#L1128)
- `_uniform(key, minval, maxval, shape, dtype)` — [`L512`](../../../../../../../raw/code/jax/jax/_src/random/core.py#L512)
- `_wald(key, mean, shape, dtype)` — [`L2923`](../../../../../../../raw/code/jax/jax/_src/random/core.py#L2923)
- `_weibull_min(key, scale, concentration, shape, dtype)` — [`L2633`](../../../../../../../raw/code/jax/jax/_src/random/core.py#L2633)
- `ball(key: ArrayLike, d: int, p: float = 2, shape: Shape = (), dtype: DTypeLikeFloat | None = None, *, out_sharding: NamedSharding | P | None = None)` — [`L2762`](../../../../../../../raw/code/jax/jax/_src/random/core.py#L2762) — Sample uniformly from the unit Lp ball.
- `bernoulli(key: ArrayLike, p: RealArray = 0.5, shape: Shape | None = None, mode: str = 'low', *, out_sharding: NamedSharding | P | None = None)` — [`L1151`](../../../../../../../raw/code/jax/jax/_src/random/core.py#L1151) — Sample Bernoulli random values with given shape and mean.
- `beta(key: ArrayLike, a: RealArray, b: RealArray, shape: Shape | None = None, dtype: DTypeLikeFloat | None = None, *, out_sharding: NamedSharding | P | None = None)` — [`L1224`](../../../../../../../raw/code/jax/jax/_src/random/core.py#L1224) — Sample Beta random values with given shape and float dtype.
- `binomial(key: Array, n: RealArray, p: RealArray, shape: Shape | None = None, dtype: DTypeLikeFloat | None = None)` — [`L3282`](../../../../../../../raw/code/jax/jax/_src/random/core.py#L3282) — Sample Binomial random values with given shape and float dtype.
- `bits(key: ArrayLike, shape: Shape = (), dtype: DTypeLikeUInt | None = None, *, out_sharding: NamedSharding | P | None = None)` — [`L421`](../../../../../../../raw/code/jax/jax/_src/random/core.py#L421) — Sample uniform bits in the form of unsigned integers.
- `body_fn(carry)` — [`L1755`](../../../../../../../raw/code/jax/jax/_src/random/core.py#L1755)
- `body_fn(carry)` — [`L1783`](../../../../../../../raw/code/jax/jax/_src/random/core.py#L1783)
- `body_fn(carry)` — [`L3155`](../../../../../../../raw/code/jax/jax/_src/random/core.py#L3155)
- `body_fn(carry)` — [`L3188`](../../../../../../../raw/code/jax/jax/_src/random/core.py#L3188)
- `canonicalize_sharding_for_samplers(out_sharding, name, shape)` — [`L461`](../../../../../../../raw/code/jax/jax/_src/random/core.py#L461)
- `categorical(key: ArrayLike, logits: RealArray, axis: int = -1, shape: Shape | None = None, replace: bool = True, mode: str | None = None, *, out_sharding: NamedSharding | P | None = None)` — [`L2001`](../../../../../../../raw/code/jax/jax/_src/random/core.py#L2001) — Sample random values from categorical distributions.
- `cauchy(key: ArrayLike, shape: Shape = (), dtype: DTypeLikeFloat | None = None, *, out_sharding: NamedSharding | P | None = None)` — [`L1300`](../../../../../../../raw/code/jax/jax/_src/random/core.py#L1300) — Sample Cauchy random values with given shape and float dtype.
- `chisquare(key: ArrayLike, df: RealArray, shape: Shape | None = None, dtype: DTypeLikeFloat | None = None, *, out_sharding: NamedSharding | P | None = None)` — [`L2314`](../../../../../../../raw/code/jax/jax/_src/random/core.py#L2314) — Sample Chisquare random values with given shape and float dtype.
- `choice(key: ArrayLike, a: int | ArrayLike, shape: Shape = (), replace: bool = True, p: RealArray | None = None, axis: int = 0, mode: str | None = None)` — [`L824`](../../../../../../../raw/code/jax/jax/_src/random/core.py#L824) — Generates a random sample from a given array.
- `clone(key)` — [`L3397`](../../../../../../../raw/code/jax/jax/_src/random/core.py#L3397) — Clone a key for reuse
- `cond_fn(carry)` — [`L1762`](../../../../../../../raw/code/jax/jax/_src/random/core.py#L1762)
- `cond_fn(carry)` — [`L1805`](../../../../../../../raw/code/jax/jax/_src/random/core.py#L1805)
- `cond_fn(carry)` — [`L3164`](../../../../../../../raw/code/jax/jax/_src/random/core.py#L3164)
- `cond_fn(carry)` — [`L3214`](../../../../../../../raw/code/jax/jax/_src/random/core.py#L3214)
- `default_prng_impl()` — [`L143`](../../../../../../../raw/code/jax/jax/_src/random/core.py#L143) — Get the default PRNG implementation.
- `dirichlet(key: ArrayLike, alpha: RealArray, shape: Shape | None = None, dtype: DTypeLikeFloat | None = None, *, out_sharding: NamedSharding | P | None = None)` — [`L1351`](../../../../../../../raw/code/jax/jax/_src/random/core.py#L1351) — Sample Dirichlet random values with given shape and float dtype.
- `double_sided_maxwell(key: ArrayLike, loc: RealArray, scale: RealArray, shape: Shape = (), dtype: DTypeLikeFloat | None = None)` — [`L2544`](../../../../../../../raw/code/jax/jax/_src/random/core.py#L2544) — Sample from a double sided Maxwell distribution.
- `exponential(key: ArrayLike, shape: Shape = (), dtype: DTypeLikeFloat | None = None, *, out_sharding: NamedSharding | P | None = None)` — [`L1428`](../../../../../../../raw/code/jax/jax/_src/random/core.py#L1428) — Sample Exponential random values with given shape and float dtype.
- `f(key: ArrayLike, dfnum: RealArray, dfden: RealArray, shape: Shape | None = None, dtype: DTypeLikeFloat | None = None, *, out_sharding: NamedSharding | P | None = None)` — [`L2373`](../../../../../../../raw/code/jax/jax/_src/random/core.py#L2373) — Sample F-distribution random values with given shape and float dtype.
- `f(remainder, ratio_key)` — [`L3380`](../../../../../../../raw/code/jax/jax/_src/random/core.py#L3380)
- `fold_in(key: ArrayLike, data: IntegerArray)` — [`L289`](../../../../../../../raw/code/jax/jax/_src/random/core.py#L289) — Folds in data to a PRNG key to form a new PRNG key.
- `gamma(key: ArrayLike, a: RealArray, shape: Shape | None = None, dtype: DTypeLikeFloat | None = None, *, out_sharding: NamedSharding | P | None = None)` — [`L1624`](../../../../../../../raw/code/jax/jax/_src/random/core.py#L1624) — Sample Gamma random values with given shape and float dtype.
- `generalized_normal(key: ArrayLike, p: float, shape: Shape = (), dtype: DTypeLikeFloat | None = None, *, out_sharding: NamedSharding | P | None = None)` — [`L2710`](../../../../../../../raw/code/jax/jax/_src/random/core.py#L2710) — Sample from the generalized normal distribution.
- `geometric(key: ArrayLike, p: RealArray, shape: Shape | None = None, dtype: DTypeLikeInt | None = None, *, out_sharding: NamedSharding | P | None = None)` — [`L2937`](../../../../../../../raw/code/jax/jax/_src/random/core.py#L2937) — Sample Geometric random values with given shape and float dtype.
- `gumbel(key: ArrayLike, shape: Shape = (), dtype: DTypeLikeFloat | None = None, mode: str | None = None, *, out_sharding: NamedSharding | P | None = None)` — [`L1892`](../../../../../../../raw/code/jax/jax/_src/random/core.py#L1892) — Sample Gumbel random values with given shape and float dtype.
- `key(seed: int | ArrayLike, *, impl: PRNGSpecDesc | None = None, dtype: KeyDTypeLike | None = None)` — [`L232`](../../../../../../../raw/code/jax/jax/_src/random/core.py#L232) — Create a pseudo-random number generator (PRNG) key given an integer seed.
- `key_data(keys: ArrayLike)` — [`L352`](../../../../../../../raw/code/jax/jax/_src/random/core.py#L352) — Recover the bits of key data underlying a PRNG key array.
- `key_dtype(impl_spec: PRNGSpecDesc | None = None)` — [`L215`](../../../../../../../raw/code/jax/jax/_src/random/core.py#L215) — Get the dtype corresponding to a PRNG implementation.
- `key_impl(keys: ArrayLike)` — [`L343`](../../../../../../../raw/code/jax/jax/_src/random/core.py#L343)
- `laplace(key: ArrayLike, shape: Shape = (), dtype: DTypeLikeFloat | None = None, *, out_sharding: NamedSharding | P | None = None)` — [`L2096`](../../../../../../../raw/code/jax/jax/_src/random/core.py#L2096) — Sample Laplace random values with given shape and float dtype.
- `loggamma(key: ArrayLike, a: RealArray, shape: Shape | None = None, dtype: DTypeLikeFloat | None = None, *, out_sharding: NamedSharding | P | None = None)` — [`L1682`](../../../../../../../raw/code/jax/jax/_src/random/core.py#L1682) — Sample log-gamma random values with given shape and float dtype.
- `logistic(key: ArrayLike, shape: Shape = (), dtype: DTypeLikeFloat | None = None, *, out_sharding: NamedSharding | P | None = None)` — [`L2145`](../../../../../../../raw/code/jax/jax/_src/random/core.py#L2145) — Sample logistic random values with given shape and float dtype.
- `lognormal(key: ArrayLike, sigma: RealArray = np.float32(1), shape: Shape | None = None, dtype: DTypeLikeFloat | None = None, *, out_sharding: NamedSharding | P | None = None)` — [`L3071`](../../../../../../../raw/code/jax/jax/_src/random/core.py#L3071) — Sample lognormal random values with given shape and float dtype.
- `maxwell(key: ArrayLike, shape: Shape = (), dtype: DTypeLikeFloat | None = None, *, out_sharding: NamedSharding | P | None = None)` — [`L2488`](../../../../../../../raw/code/jax/jax/_src/random/core.py#L2488) — Sample from a one sided Maxwell distribution.
- `maybe_auto_axes(f, out_sharding, **hoist_kwargs)` — [`L412`](../../../../../../../raw/code/jax/jax/_src/random/core.py#L412)
- `multinomial(key: Array, n: RealArray, p: RealArray, *, shape: Shape | None = None, dtype: DTypeLikeFloat | None = None, unroll: int | bool = 1)` — [`L3337`](../../../../../../../raw/code/jax/jax/_src/random/core.py#L3337) — Sample from a multinomial distribution.
- `multivariate_normal(key: ArrayLike, mean: RealArray, cov: RealArray, shape: Shape | None = None, dtype: DTypeLikeFloat | None = None, method: str = 'cholesky', *, out_sharding: NamedSharding | P | None = None)` — [`L976`](../../../../../../../raw/code/jax/jax/_src/random/core.py#L976) — Sample multivariate normal random values with given mean and covariance.
- `normal(key: ArrayLike, shape: Shape = (), dtype: DTypeLikeFloat | None = None, *, out_sharding: NamedSharding | P | None = None)` — [`L912`](../../../../../../../raw/code/jax/jax/_src/random/core.py#L912) — Sample standard normal random values with given shape and float dtype.
- `orthogonal(key: ArrayLike, n: int, shape: Shape = (), dtype: DTypeLikeFloat | None = None, m: int | None = None, *, out_sharding: NamedSharding | P | None = None)` — [`L2641`](../../../../../../../raw/code/jax/jax/_src/random/core.py#L2641) — Sample uniformly from the orthogonal group O(n).
- `pareto(key: ArrayLike, b: RealArray, shape: Shape | None = None, dtype: DTypeLikeFloat | None = None, *, out_sharding: NamedSharding | P | None = None)` — [`L2193`](../../../../../../../raw/code/jax/jax/_src/random/core.py#L2193) — Sample Pareto random values with given shape and float dtype.
- `permutation(key: ArrayLike, x: int | ArrayLike, axis: int = 0, independent: bool = False, *, out_sharding: NamedSharding | P | None = None)` — [`L745`](../../../../../../../raw/code/jax/jax/_src/random/core.py#L745) — Returns a randomly permuted array or range.
- `poisson(key: ArrayLike, lam: RealArray, shape: Shape | None = None, dtype: DTypeLikeInt | None = None, *, out_sharding: NamedSharding | P | None = None)` — [`L1836`](../../../../../../../raw/code/jax/jax/_src/random/core.py#L1836) — Sample Poisson random values with given shape and integer dtype.
- `rademacher(key: ArrayLike, shape: Shape = (), dtype: DTypeLikeInt | None = None, *, out_sharding: NamedSharding | P | None = None)` — [`L2441`](../../../../../../../raw/code/jax/jax/_src/random/core.py#L2441) — Sample from a Rademacher distribution.
- `randint(key: ArrayLike, shape: Shape, minval: IntegerArray, maxval: IntegerArray, dtype: DTypeLikeInt | None = None, *, out_sharding: NamedSharding | P | None = None)` — [`L593`](../../../../../../../raw/code/jax/jax/_src/random/core.py#L593) — Sample uniform random values in [minval, maxval) with given shape/dtype.
- `random_insert_pvary(name, key, *args)` — [`L3415`](../../../../../../../raw/code/jax/jax/_src/random/core.py#L3415)
- `rayleigh(key: ArrayLike, scale: RealArray, shape: Shape | None = None, dtype: DTypeLikeFloat | None = None, *, out_sharding: NamedSharding | P | None = None)` — [`L2811`](../../../../../../../raw/code/jax/jax/_src/random/core.py#L2811) — Sample Rayleigh random values with given shape and float dtype.
- `resolve_prng_impl(impl_spec: PRNGSpecDesc | KeyDTypeLike | None)` — [`L188`](../../../../../../../raw/code/jax/jax/_src/random/core.py#L188)
- `split(key: ArrayLike, num: int | tuple[int, ...] = 2)` — [`L319`](../../../../../../../raw/code/jax/jax/_src/random/core.py#L319) — Splits a PRNG key into `num` new keys by adding a leading axis.
- `t(key: ArrayLike, df: RealArray, shape: Shape | None = None, dtype: DTypeLikeFloat | None = None, *, out_sharding: NamedSharding | P | None = None)` — [`L2249`](../../../../../../../raw/code/jax/jax/_src/random/core.py#L2249) — Sample Student's t random values with given shape and float dtype.
- `triangular(key: ArrayLike, left: RealArray, mode: RealArray, right: RealArray, shape: Shape | None = None, dtype: DTypeLikeFloat | None = None, *, out_sharding: NamedSharding | P | None = None)` — [`L3000`](../../../../../../../raw/code/jax/jax/_src/random/core.py#L3000) — Sample Triangular random values with given shape and float dtype.
- `truncated_normal(key: ArrayLike, lower: RealArray, upper: RealArray, shape: Shape | None = None, dtype: DTypeLikeFloat | None = None, *, out_sharding: NamedSharding | P | None = None)` — [`L1073`](../../../../../../../raw/code/jax/jax/_src/random/core.py#L1073) — Sample truncated standard normal random values with given shape and dtype.
- `uniform(key: ArrayLike, shape: Shape = (), dtype: DTypeLikeFloat | None = None, minval: RealArray = 0, maxval: RealArray = 1, *, out_sharding: NamedSharding | P | None = None)` — [`L470`](../../../../../../../raw/code/jax/jax/_src/random/core.py#L470) — Sample uniform random values in [minval, maxval) with given shape/dtype.
- `wald(key: ArrayLike, mean: RealArray, shape: Shape | None = None, dtype: DTypeLikeFloat | None = None, *, out_sharding: NamedSharding | P | None = None)` — [`L2872`](../../../../../../../raw/code/jax/jax/_src/random/core.py#L2872) — Sample Wald random values with given shape and float dtype.
- `weibull_min(key: ArrayLike, scale: RealArray, concentration: RealArray, shape: Shape = (), dtype: DTypeLikeFloat | None = None)` — [`L2596`](../../../../../../../raw/code/jax/jax/_src/random/core.py#L2596) — Sample from a Weibull distribution.
- `wrap_key_data(key_bits_array: Array, *, impl: PRNGSpecDesc | None = None, dtype: KeyDTypeLike | None = None)` — [`L358`](../../../../../../../raw/code/jax/jax/_src/random/core.py#L358) — Wrap an array of key data bits into a PRNG key array.

## Module values
- `DTypeLikeFloat` — [`L59`](../../../../../../../raw/code/jax/jax/_src/random/core.py#L59)
- `DTypeLikeInt` — [`L57`](../../../../../../../raw/code/jax/jax/_src/random/core.py#L57)
- `DTypeLikeUInt` — [`L58`](../../../../../../../raw/code/jax/jax/_src/random/core.py#L58)
- `IntegerArray` — [`L54`](../../../../../../../raw/code/jax/jax/_src/random/core.py#L54)
- `KeyDTypeLike` — [`L185`](../../../../../../../raw/code/jax/jax/_src/random/core.py#L185)
- `PRNGImpl` — [`L62`](../../../../../../../raw/code/jax/jax/_src/random/core.py#L62)
- `PRNGSpecDesc` — [`L184`](../../../../../../../raw/code/jax/jax/_src/random/core.py#L184)
- `RealArray` — [`L53`](../../../../../../../raw/code/jax/jax/_src/random/core.py#L53)
- `Shape` — [`L60`](../../../../../../../raw/code/jax/jax/_src/random/core.py#L60)
- `UINT_DTYPES` — [`L64`](../../../../../../../raw/code/jax/jax/_src/random/core.py#L64)
- `random_clone_p` — [`L3330`](../../../../../../../raw/code/jax/jax/_src/random/core.py#L3330)
- `random_gamma_p` — [`L1605`](../../../../../../../raw/code/jax/jax/_src/random/core.py#L1605)

