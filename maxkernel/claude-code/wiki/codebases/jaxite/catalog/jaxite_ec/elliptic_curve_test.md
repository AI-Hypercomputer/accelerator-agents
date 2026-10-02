---
title: 'Module: jaxite_ec/elliptic_curve_test.py'
type: catalog
provenance: extracted
module: jaxite_ec/elliptic_curve_test.py
status: fresh
symbol_base: scip-python python jaxite 0.0.0 `jaxite_ec.elliptic_curve_test`/TestEllipticCurve#
symbols:
  TestEllipticCurve.point_a: point_a.
  TestEllipticCurve.point_b: point_b.
  TestEllipticCurve.true_result_pdub_a: true_result_pdub_a.
  TestEllipticCurve.true_result_padd: true_result_padd.
  TestEllipticCurve.point_b_sys: point_b_sys.
  TestEllipticCurve.coordinate_num: coordinate_num.
  TestEllipticCurve.point_a_sys: point_a_sys.
  TestEllipticCurve.true_result_pdub_a_affine: true_result_pdub_a_affine.
  TestEllipticCurve.test_jit_pdul_barrett_xyzz_pack_two_no_batch: test_jit_pdul_barrett_xyzz_pack_two_no_batch().
  TestEllipticCurve.test_jit_pdul_barrett_xyzz_pack_two_batch: test_jit_pdul_barrett_xyzz_pack_two_batch().
  TestEllipticCurve.true_result_pdub_b: true_result_pdub_b.
  TestEllipticCurve.true_result_padd_affine: true_result_padd_affine.
  TestEllipticCurve.test_padd_barrett_xyzz_pack: test_padd_barrett_xyzz_pack().
  TestEllipticCurve.test_padd_lazy_xyzz_pack: test_padd_lazy_xyzz_pack().
  TestEllipticCurve.test_padd_rns_xyzz: test_padd_rns_xyzz().
  TestEllipticCurve.test_padd_rns_xyzz_batch: test_padd_rns_xyzz_batch().
  TestEllipticCurve.test_padd_rns_twisted_pack_new_twist_two_batch: test_padd_rns_twisted_pack_new_twist_two_batch().
  TestEllipticCurve.test_pdul_rns_twisted_pack_new_twist_two_batch: test_pdul_rns_twisted_pack_new_twist_two_batch().
  TestEllipticCurve.test_pdul_barrett_xyzz: test_pdul_barrett_xyzz().
  TestEllipticCurve.test_pdul_lazy_xyzz_pack: test_pdul_lazy_xyzz_pack().
  TestEllipticCurve.test_padd_lazy_twisted_pack: test_padd_lazy_twisted_pack().
  TestEllipticCurve.test_padd_lazy_twisted_pack_batch: test_padd_lazy_twisted_pack_batch().
  TestEllipticCurve.test_jit_pneg_lazy_twisted_pack: test_jit_pneg_lazy_twisted_pack().
  TestEllipticCurve.test_pdul_rns_xyzz_pack: test_pdul_rns_xyzz_pack().
  TestEllipticCurve.test_padd_rns_twisted_pack: test_padd_rns_twisted_pack().
  TestEllipticCurve.test_padd_rns_twisted_pack_new_twisted: test_padd_rns_twisted_pack_new_twisted().
  TestEllipticCurve.true_result_pdub_b_affine: true_result_pdub_b_affine.
  TestEllipticCurve.test_padd_same_lazy_twisted_pack: test_padd_same_lazy_twisted_pack().
  TestEllipticCurve.test_pdul_lazy_twisted_pack: test_pdul_lazy_twisted_pack().
  TestEllipticCurve.test_pdul_rns_twisted_pack_new_twisted: test_pdul_rns_twisted_pack_new_twisted().
  TestEllipticCurve.test_padd_zero_twisted_pack_new_twisted: test_padd_zero_twisted_pack_new_twisted().
  TestEllipticCurve.test_padd_zero_rns_twisted_pack_new_twisted: test_padd_zero_rns_twisted_pack_new_twisted().
  TestEllipticCurve.test_pdul_rns_twisted_pack: test_pdul_rns_twisted_pack().
  TestEllipticCurve.zero_twisted: zero_twisted.
  TestEllipticCurve.ec_sys: ec_sys.
  TestEllipticCurve.x1_int_: x1_int_.
  TestEllipticCurve.y1_int_: y1_int_.
  TestEllipticCurve.x2_int_: x2_int_.
  TestEllipticCurve.y2_int_: y2_int_.
  TestEllipticCurve: ''
  TestEllipticCurve.setUp: setUp().
  TestEllipticCurve.batch_size: batch_size.
  TestEllipticCurve.test_padd_rns_a_point_add_zero_correctness: test_padd_rns_a_point_add_zero_correctness().
---
# Module: [`jaxite_ec/elliptic_curve_test.py`](../../../../../raw/code/jaxite/jaxite_ec/elliptic_curve_test.py)

## Classes
### `TestEllipticCurve`  ·  implements/extends TestCase
- def: [`jaxite_ec/elliptic_curve_test.py:13`](../../../../../raw/code/jaxite/jaxite_ec/elliptic_curve_test.py#L13)
- signature: `class TestEllipticCurve(absltest.TestCase):`
- members:
  - `setUp(self)` — [`L15`](../../../../../raw/code/jaxite/jaxite_ec/elliptic_curve_test.py#L15)
  - `test_jit_pdul_barrett_xyzz_pack_two_batch(self)` — [`L115`](../../../../../raw/code/jaxite/jaxite_ec/elliptic_curve_test.py#L115)
  - `test_jit_pdul_barrett_xyzz_pack_two_no_batch(self)` — [`L84`](../../../../../raw/code/jaxite/jaxite_ec/elliptic_curve_test.py#L84)
  - `test_jit_pneg_lazy_twisted_pack(self)` — [`L374`](../../../../../raw/code/jaxite/jaxite_ec/elliptic_curve_test.py#L374)
  - `test_padd_barrett_xyzz_pack(self)` — [`L40`](../../../../../raw/code/jaxite/jaxite_ec/elliptic_curve_test.py#L40) — documented in [jaxite_ec-elliptic_curve_test](../../concepts/jaxite_ec-elliptic_curve_test.md)
  - `test_padd_lazy_twisted_pack(self)` — [`L216`](../../../../../raw/code/jaxite/jaxite_ec/elliptic_curve_test.py#L216)
  - `test_padd_lazy_twisted_pack_batch(self)` — [`L247`](../../../../../raw/code/jaxite/jaxite_ec/elliptic_curve_test.py#L247)
  - `test_padd_lazy_xyzz_pack(self)` — [`L143`](../../../../../raw/code/jaxite/jaxite_ec/elliptic_curve_test.py#L143) — documented in [jaxite_ec-elliptic_curve_test](../../concepts/jaxite_ec-elliptic_curve_test.md)
  - `test_padd_rns_a_point_add_zero_correctness(self)` — [`L877`](../../../../../raw/code/jaxite/jaxite_ec/elliptic_curve_test.py#L877)
  - `test_padd_rns_twisted_pack(self)` — [`L542`](../../../../../raw/code/jaxite/jaxite_ec/elliptic_curve_test.py#L542)
  - `test_padd_rns_twisted_pack_new_twist_two_batch(self)` — [`L698`](../../../../../raw/code/jaxite/jaxite_ec/elliptic_curve_test.py#L698)
  - `test_padd_rns_twisted_pack_new_twisted(self)` — [`L624`](../../../../../raw/code/jaxite/jaxite_ec/elliptic_curve_test.py#L624) — documented in [jaxite_ec-elliptic_curve_test](../../concepts/jaxite_ec-elliptic_curve_test.md)
  - `test_padd_rns_xyzz(self)` — [`L412`](../../../../../raw/code/jaxite/jaxite_ec/elliptic_curve_test.py#L412) — documented in [jaxite_ec-elliptic_curve_test](../../concepts/jaxite_ec-elliptic_curve_test.md)
  - `test_padd_rns_xyzz_batch(self)` — [`L454`](../../../../../raw/code/jaxite/jaxite_ec/elliptic_curve_test.py#L454)
  - `test_padd_same_lazy_twisted_pack(self)` — [`L304`](../../../../../raw/code/jaxite/jaxite_ec/elliptic_curve_test.py#L304)
  - `test_padd_zero_rns_twisted_pack_new_twisted(self)` — [`L842`](../../../../../raw/code/jaxite/jaxite_ec/elliptic_curve_test.py#L842)
  - `test_padd_zero_twisted_pack_new_twisted(self)` — [`L808`](../../../../../raw/code/jaxite/jaxite_ec/elliptic_curve_test.py#L808) — documented in [jaxite_ec-elliptic_curve_test](../../concepts/jaxite_ec-elliptic_curve_test.md)
  - `test_pdul_barrett_xyzz(self)` — [`L64`](../../../../../raw/code/jaxite/jaxite_ec/elliptic_curve_test.py#L64) — documented in [jaxite_ec-elliptic_curve_test](../../concepts/jaxite_ec-elliptic_curve_test.md)
  - `test_pdul_lazy_twisted_pack(self)` — [`L342`](../../../../../raw/code/jaxite/jaxite_ec/elliptic_curve_test.py#L342) — documented in [jaxite_ec-elliptic_curve_test](../../concepts/jaxite_ec-elliptic_curve_test.md)
  - `test_pdul_lazy_xyzz_pack(self)` — [`L181`](../../../../../raw/code/jaxite/jaxite_ec/elliptic_curve_test.py#L181)
  - `test_pdul_rns_twisted_pack(self)` — [`L586`](../../../../../raw/code/jaxite/jaxite_ec/elliptic_curve_test.py#L586)
  - `test_pdul_rns_twisted_pack_new_twist_two_batch(self)` — [`L753`](../../../../../raw/code/jaxite/jaxite_ec/elliptic_curve_test.py#L753)
  - `test_pdul_rns_twisted_pack_new_twisted(self)` — [`L664`](../../../../../raw/code/jaxite/jaxite_ec/elliptic_curve_test.py#L664)
  - `test_pdul_rns_xyzz_pack(self)` — [`L503`](../../../../../raw/code/jaxite/jaxite_ec/elliptic_curve_test.py#L503) — documented in [jaxite_ec-elliptic_curve_test](../../concepts/jaxite_ec-elliptic_curve_test.md)
  - `batch_size` — [`L18`](../../../../../raw/code/jaxite/jaxite_ec/elliptic_curve_test.py#L18)
  - `coordinate_num` — [`L17`](../../../../../raw/code/jaxite/jaxite_ec/elliptic_curve_test.py#L17)
  - `ec_sys` — [`L26`](../../../../../raw/code/jaxite/jaxite_ec/elliptic_curve_test.py#L26) — documented in [jaxite_ec-elliptic_curve_test](../../concepts/jaxite_ec-elliptic_curve_test.md)
  - `point_a` — [`L23`](../../../../../raw/code/jaxite/jaxite_ec/elliptic_curve_test.py#L23) — documented in [jaxite_ec-elliptic_curve_test](../../concepts/jaxite_ec-elliptic_curve_test.md)
  - `point_a_sys` — [`L27`](../../../../../raw/code/jaxite/jaxite_ec/elliptic_curve_test.py#L27) — documented in [jaxite_ec-elliptic_curve_test](../../concepts/jaxite_ec-elliptic_curve_test.md)
  - `point_b` — [`L24`](../../../../../raw/code/jaxite/jaxite_ec/elliptic_curve_test.py#L24) — documented in [jaxite_ec-elliptic_curve_test](../../concepts/jaxite_ec-elliptic_curve_test.md)
  - `point_b_sys` — [`L28`](../../../../../raw/code/jaxite/jaxite_ec/elliptic_curve_test.py#L28) — documented in [jaxite_ec-elliptic_curve_test](../../concepts/jaxite_ec-elliptic_curve_test.md)
  - `true_result_padd` — [`L33`](../../../../../raw/code/jaxite/jaxite_ec/elliptic_curve_test.py#L33) — documented in [jaxite_ec-elliptic_curve_test](../../concepts/jaxite_ec-elliptic_curve_test.md)
  - `true_result_padd_affine` — [`L34`](../../../../../raw/code/jaxite/jaxite_ec/elliptic_curve_test.py#L34)
  - `true_result_pdub_a` — [`L35`](../../../../../raw/code/jaxite/jaxite_ec/elliptic_curve_test.py#L35) — documented in [jaxite_ec-elliptic_curve_test](../../concepts/jaxite_ec-elliptic_curve_test.md)
  - `true_result_pdub_a_affine` — [`L36`](../../../../../raw/code/jaxite/jaxite_ec/elliptic_curve_test.py#L36)
  - `true_result_pdub_b` — [`L37`](../../../../../raw/code/jaxite/jaxite_ec/elliptic_curve_test.py#L37)
  - `true_result_pdub_b_affine` — [`L38`](../../../../../raw/code/jaxite/jaxite_ec/elliptic_curve_test.py#L38)
  - `x1_int_` — [`L19`](../../../../../raw/code/jaxite/jaxite_ec/elliptic_curve_test.py#L19) — documented in [jaxite_ec-elliptic_curve_test](../../concepts/jaxite_ec-elliptic_curve_test.md)
  - `x2_int_` — [`L21`](../../../../../raw/code/jaxite/jaxite_ec/elliptic_curve_test.py#L21) — documented in [jaxite_ec-elliptic_curve_test](../../concepts/jaxite_ec-elliptic_curve_test.md)
  - `y1_int_` — [`L20`](../../../../../raw/code/jaxite/jaxite_ec/elliptic_curve_test.py#L20) — documented in [jaxite_ec-elliptic_curve_test](../../concepts/jaxite_ec-elliptic_curve_test.md)
  - `y2_int_` — [`L22`](../../../../../raw/code/jaxite/jaxite_ec/elliptic_curve_test.py#L22) — documented in [jaxite_ec-elliptic_curve_test](../../concepts/jaxite_ec-elliptic_curve_test.md)
  - `zero_twisted` — [`L25`](../../../../../raw/code/jaxite/jaxite_ec/elliptic_curve_test.py#L25)

