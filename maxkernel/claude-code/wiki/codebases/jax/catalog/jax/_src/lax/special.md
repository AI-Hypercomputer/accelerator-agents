---
title: 'Module: jax/_src/lax/special.py'
type: catalog
provenance: extracted
module: jax/_src/lax/special.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax._src.lax.special`/
symbols:
  igamma_impl: igamma_impl().
  regularized_incomplete_beta_impl: regularized_incomplete_beta_impl().
  igamma_grad_a_impl: igamma_grad_a_impl().
  random_gamma_grad_impl: random_gamma_grad_impl().
  igammac_impl: igammac_impl().
  _igammac_continued_fraction: _igammac_continued_fraction().
  _igamma_series: _igamma_series().
  _igammac_continued_fraction.body_fn: _igammac_continued_fraction().body_fn().
  erf_p: erf_p.
  lentz_thompson_barnett_algorithm.while_body_fn: lentz_thompson_barnett_algorithm().while_body_fn().
  erfc_p: erfc_p.
  erf_inv_p: erf_inv_p.
  lgamma: lgamma().
  igamma_grad_a_p: igamma_grad_a_p.
  regularized_incomplete_beta_p: regularized_incomplete_beta_p.
  digamma_p: digamma_p.
  igammac_p: igammac_p.
  lgamma_p: lgamma_p.
  polygamma_p: polygamma_p.
  bessel_i0e_p: bessel_i0e_p.
  lentz_thompson_barnett_algorithm: lentz_thompson_barnett_algorithm().
  _bessel_i1e_jvp: _bessel_i1e_jvp().
  _i0e_impl32: _i0e_impl32().
  igamma_p: igamma_p.
  IgammaMode: IgammaMode#
  _igamma_series.body_fn: _igamma_series().body_fn().
  _i0e_impl64: _i0e_impl64().
  bessel_i0e_impl: bessel_i0e_impl().
  polygamma: polygamma().
  _any: _any().
  random_gamma_grad: random_gamma_grad().
  digamma: digamma().
  bessel_i1e_p: bessel_i1e_p.
  _up_and_broadcast.up_and_broadcast: _up_and_broadcast().up_and_broadcast().
  betainc: betainc().
  igamma: igamma().
  igammac: igammac().
  igamma_grad_a: igamma_grad_a().
  zeta: zeta().
  regularized_incomplete_beta_impl.nth_partial_betainc_numerator: regularized_incomplete_beta_impl().nth_partial_betainc_numerator().
  bessel_i0e: bessel_i0e().
  erf: erf().
  erf_inv: erf_inv().
  zeta_p: zeta_p.
  bessel_i1e: bessel_i1e().
  erfc: erfc().
  igamma_gradx: igamma_gradx().
  betainc_gradx: betainc_gradx().
  regularized_incomplete_beta_impl.nth_partial_betainc_denominator: regularized_incomplete_beta_impl().nth_partial_betainc_denominator().
  polygamma_gradx: polygamma_gradx().
  _igammac_continued_fraction.cond_fn: _igammac_continued_fraction().cond_fn().
  IgammaMode.VALUE: IgammaMode#VALUE.
  _up_and_broadcast: _up_and_broadcast().
  evaluate_chebyshev_polynomial: evaluate_chebyshev_polynomial().
  lentz_thompson_barnett_algorithm.while_cond_fn: lentz_thompson_barnett_algorithm().while_cond_fn().
  igamma_grada: igamma_grada().
  IgammaMode.DERIVATIVE: IgammaMode#DERIVATIVE.
  IgammaMode.SAMPLE_DERIVATIVE: IgammaMode#SAMPLE_DERIVATIVE.
  igammac_gradx: igammac_gradx().
  igammac_grada: igammac_grada().
  _igamma_series.cond_fn: _igamma_series().cond_fn().
  betainc_grad_not_implemented: betainc_grad_not_implemented().
  polygamma_gradm: polygamma_gradm().
---
# Module: [`jax/_src/lax/special.py`](../../../../../../../raw/code/jax/jax/_src/lax/special.py)

## Classes
### `IgammaMode`  ·  implements/extends Enum
- def: [`jax/_src/lax/special.py:299`](../../../../../../../raw/code/jax/jax/_src/lax/special.py#L299)
- signature: `class IgammaMode(Enum):`
- members:
  - `DERIVATIVE` — [`L301`](../../../../../../../raw/code/jax/jax/_src/lax/special.py#L301)
  - `SAMPLE_DERIVATIVE` — [`L302`](../../../../../../../raw/code/jax/jax/_src/lax/special.py#L302)
  - `VALUE` — [`L300`](../../../../../../../raw/code/jax/jax/_src/lax/special.py#L300)
- used by: [`igamma_impl`](special.md#igamma_impl), [`igamma_grad_a_impl`](special.md#igamma_grad_a_impl), [`random_gamma_grad_impl`](special.md#random_gamma_grad_impl), [`igammac_impl`](special.md#igammac_impl), [`_igammac_continued_fraction`](special.md#_igammac_continued_fraction), [`_igamma_series`](special.md#_igamma_series), [`body_fn`](special.md#_igammac_continued_fraction.body_fn), [`body_fn`](special.md#_igamma_series.body_fn)

## Functions
- `_any(predicates: Array)` — [`L304`](../../../../../../../raw/code/jax/jax/_src/lax/special.py#L304)
- `_bessel_i1e_jvp(g, y, x)` — [`L744`](../../../../../../../raw/code/jax/jax/_src/lax/special.py#L744)
- `_i0e_impl32(x: Array)` — [`L597`](../../../../../../../raw/code/jax/jax/_src/lax/special.py#L597) — Computes an approximation to the modified Bessel function of the first kind,
- `_i0e_impl64(x: Array)` — [`L632`](../../../../../../../raw/code/jax/jax/_src/lax/special.py#L632)
- `_igamma_series(ax, x, a, enabled, dtype, mode)` — [`L310`](../../../../../../../raw/code/jax/jax/_src/lax/special.py#L310)
- `_igammac_continued_fraction(ax, x, a, enabled, dtype, mode)` — [`L403`](../../../../../../../raw/code/jax/jax/_src/lax/special.py#L403)
- `_up_and_broadcast(doit)` — [`L42`](../../../../../../../raw/code/jax/jax/_src/lax/special.py#L42)
- `bessel_i0e(x: ArrayLike)` — [`L104`](../../../../../../../raw/code/jax/jax/_src/lax/special.py#L104) — Exponentially scaled modified Bessel function of order 0:
- `bessel_i0e_impl(x: Array)` — [`L677`](../../../../../../../raw/code/jax/jax/_src/lax/special.py#L677)
- `bessel_i1e(x: ArrayLike)` — [`L110`](../../../../../../../raw/code/jax/jax/_src/lax/special.py#L110) — Exponentially scaled modified Bessel function of order 1:
- `betainc(a: ArrayLike, b: ArrayLike, x: ArrayLike)` — [`L60`](../../../../../../../raw/code/jax/jax/_src/lax/special.py#L60) — Elementwise regularized incomplete beta integral.
- `betainc_grad_not_implemented(g, a, b, x)` — [`L135`](../../../../../../../raw/code/jax/jax/_src/lax/special.py#L135)
- `betainc_gradx(g, a, b, x)` — [`L129`](../../../../../../../raw/code/jax/jax/_src/lax/special.py#L129)
- `body_fn(vals)` — [`L314`](../../../../../../../raw/code/jax/jax/_src/lax/special.py#L314)
- `body_fn(vals)` — [`L410`](../../../../../../../raw/code/jax/jax/_src/lax/special.py#L410)
- `cond_fn(vals)` — [`L311`](../../../../../../../raw/code/jax/jax/_src/lax/special.py#L311)
- `cond_fn(vals)` — [`L406`](../../../../../../../raw/code/jax/jax/_src/lax/special.py#L406)
- `digamma(x: ArrayLike)` — [`L69`](../../../../../../../raw/code/jax/jax/_src/lax/special.py#L69) — Elementwise digamma: :math:`\psi(x)`.
- `erf(x: ArrayLike)` — [`L116`](../../../../../../../raw/code/jax/jax/_src/lax/special.py#L116) — Elementwise error function: :math:`\mathrm{erf}(x)`.
- `erf_inv(x: ArrayLike)` — [`L125`](../../../../../../../raw/code/jax/jax/_src/lax/special.py#L125) — Elementwise inverse error function: :math:`\mathrm{erf}^{-1}(x)`.
- `erfc(x: ArrayLike)` — [`L120`](../../../../../../../raw/code/jax/jax/_src/lax/special.py#L120) — Elementwise complementary error function:
- `evaluate_chebyshev_polynomial(x, coefficients)` — [`L587`](../../../../../../../raw/code/jax/jax/_src/lax/special.py#L587)
- `igamma(a: ArrayLike, x: ArrayLike)` — [`L78`](../../../../../../../raw/code/jax/jax/_src/lax/special.py#L78) — Elementwise regularized incomplete gamma function.
- `igamma_grad_a(a: ArrayLike, x: ArrayLike)` — [`L88`](../../../../../../../raw/code/jax/jax/_src/lax/special.py#L88) — Elementwise derivative of the regularized incomplete gamma function.
- `igamma_grad_a_impl(a, x, *, dtype)` — [`L546`](../../../../../../../raw/code/jax/jax/_src/lax/special.py#L546)
- `igamma_grada(g, a, x)` — [`L141`](../../../../../../../raw/code/jax/jax/_src/lax/special.py#L141)
- `igamma_gradx(g, a, x)` — [`L138`](../../../../../../../raw/code/jax/jax/_src/lax/special.py#L138)
- `igamma_impl(a, x, *, dtype)` — [`L361`](../../../../../../../raw/code/jax/jax/_src/lax/special.py#L361)
- `igammac(a: ArrayLike, x: ArrayLike)` — [`L83`](../../../../../../../raw/code/jax/jax/_src/lax/special.py#L83) — Elementwise complementary regularized incomplete gamma function.
- `igammac_grada(g, a, x)` — [`L147`](../../../../../../../raw/code/jax/jax/_src/lax/special.py#L147)
- `igammac_gradx(g, a, x)` — [`L144`](../../../../../../../raw/code/jax/jax/_src/lax/special.py#L144)
- `igammac_impl(a, x, *, dtype)` — [`L508`](../../../../../../../raw/code/jax/jax/_src/lax/special.py#L508)
- `lentz_thompson_barnett_algorithm(*, num_iterations, small, threshold, nth_partial_numerator, nth_partial_denominator, inputs)` — [`L159`](../../../../../../../raw/code/jax/jax/_src/lax/special.py#L159)
- `lgamma(x: ArrayLike)` — [`L65`](../../../../../../../raw/code/jax/jax/_src/lax/special.py#L65) — Elementwise log gamma: :math:`\mathrm{log}(\Gamma(x))`.
- `nth_partial_betainc_denominator(iteration, a, b, x)` — [`L242`](../../../../../../../raw/code/jax/jax/_src/lax/special.py#L242)
- `nth_partial_betainc_numerator(iteration, a, b, x)` — [`L213`](../../../../../../../raw/code/jax/jax/_src/lax/special.py#L213) — The partial numerator for the incomplete beta function is given
- `polygamma(m: ArrayLike, x: ArrayLike)` — [`L73`](../../../../../../../raw/code/jax/jax/_src/lax/special.py#L73) — Elementwise polygamma: :math:`\psi^{(m)}(x)`.
- `polygamma_gradm(g, m, x)` — [`L150`](../../../../../../../raw/code/jax/jax/_src/lax/special.py#L150)
- `polygamma_gradx(g, m, x)` — [`L153`](../../../../../../../raw/code/jax/jax/_src/lax/special.py#L153)
- `random_gamma_grad(a: ArrayLike, x: ArrayLike, *, dtype)` — [`L94`](../../../../../../../raw/code/jax/jax/_src/lax/special.py#L94) — Elementwise derivative of samples from `Gamma(a, 1)`.
- `random_gamma_grad_impl(a, x, *, dtype)` — [`L566`](../../../../../../../raw/code/jax/jax/_src/lax/special.py#L566)
- `regularized_incomplete_beta_impl(a, b, x, *, dtype)` — [`L210`](../../../../../../../raw/code/jax/jax/_src/lax/special.py#L210)
- `up_and_broadcast(*args)` — [`L43`](../../../../../../../raw/code/jax/jax/_src/lax/special.py#L43)
- `while_body_fn(values)` — [`L177`](../../../../../../../raw/code/jax/jax/_src/lax/special.py#L177)
- `while_cond_fn(values)` — [`L171`](../../../../../../../raw/code/jax/jax/_src/lax/special.py#L171)
- `zeta(x: ArrayLike, q: ArrayLike)` — [`L99`](../../../../../../../raw/code/jax/jax/_src/lax/special.py#L99) — Elementwise Hurwitz zeta function: :math:`\zeta(x, q)`

## Module values
- `bessel_i0e_p` — [`L734`](../../../../../../../raw/code/jax/jax/_src/lax/special.py#L734)
- `bessel_i1e_p` — [`L740`](../../../../../../../raw/code/jax/jax/_src/lax/special.py#L740)
- `digamma_p` — [`L705`](../../../../../../../raw/code/jax/jax/_src/lax/special.py#L705)
- `erf_inv_p` — [`L763`](../../../../../../../raw/code/jax/jax/_src/lax/special.py#L763)
- `erf_p` — [`L753`](../../../../../../../raw/code/jax/jax/_src/lax/special.py#L753)
- `erfc_p` — [`L758`](../../../../../../../raw/code/jax/jax/_src/lax/special.py#L758)
- `igamma_grad_a_p` — [`L717`](../../../../../../../raw/code/jax/jax/_src/lax/special.py#L717)
- `igamma_p` — [`L713`](../../../../../../../raw/code/jax/jax/_src/lax/special.py#L713)
- `igammac_p` — [`L724`](../../../../../../../raw/code/jax/jax/_src/lax/special.py#L724)
- `lgamma_p` — [`L701`](../../../../../../../raw/code/jax/jax/_src/lax/special.py#L701)
- `polygamma_p` — [`L709`](../../../../../../../raw/code/jax/jax/_src/lax/special.py#L709)
- `regularized_incomplete_beta_p` — [`L690`](../../../../../../../raw/code/jax/jax/_src/lax/special.py#L690)
- `zeta_p` — [`L731`](../../../../../../../raw/code/jax/jax/_src/lax/special.py#L731)

