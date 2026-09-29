---
title: 'Module: jaxite_ec/elliptic_curve.py'
type: catalog
provenance: extracted
module: jaxite_ec/elliptic_curve.py
status: fresh
symbol_base: scip-python python jaxite 0.0.0 `jaxite_ec.elliptic_curve`/
symbols:
  mod_mul_lazy_2u16: mod_mul_lazy_2u16.
  mod_mul_barrett_2u16: mod_mul_barrett_2u16.
  negate_rns_for_var_add: negate_rns_for_var_add.
  pdul_barret_xyzz: pdul_barret_xyzz().
  mod_mul_rns_2u16: mod_mul_rns_2u16.
  add_sub_rns_var: add_sub_rns_var.
  pdul_lazy_xyzz: pdul_lazy_xyzz().
  pdul_rns_xyzz: pdul_rns_xyzz().
  add_2u16: add_2u16.
  cond_sub_2u16: cond_sub_2u16.
  padd_barret_xyzz: padd_barret_xyzz().
  padd_lazy_xyzz: padd_lazy_xyzz().
  padd_lazy_twisted: padd_lazy_twisted().
  pdul_lazy_twisted: pdul_lazy_twisted().
  pdul_rns_twisted: pdul_rns_twisted().
  cond_sub_mod_u16: cond_sub_mod_u16.
  padd_rns_twisted_pack: padd_rns_twisted_pack().
  padd_rns_xyzz_pack: padd_rns_xyzz_pack().
  add_rns_2u16: add_rns_2u16.
  add_3u16: add_3u16.
  rns_constant: rns_constant.
  pneg_lazy_twisted: pneg_lazy_twisted().
  negate_rns_for_var_add_zero_check: negate_rns_for_var_add_zero_check.
  pdul_barrett_xyzz_pack: pdul_barrett_xyzz_pack().
  padd_barrett_xyzz_pack: padd_barrett_xyzz_pack().
  pdul_barrett_xyzz_pack_batch_first: pdul_barrett_xyzz_pack_batch_first().
  padd_barrett_xyzz_pack_batch_first: padd_barrett_xyzz_pack_batch_first().
  padd_lazy_xyzz_pack: padd_lazy_xyzz_pack().
  pdul_lazy_xyzz_pack: pdul_lazy_xyzz_pack().
  padd_lazy_twisted_pack: padd_lazy_twisted_pack().
  pdul_lazy_twisted_pack: pdul_lazy_twisted_pack().
  pneg_lazy_twisted_pack: pneg_lazy_twisted_pack().
  pdul_rns_xyzz_pack: pdul_rns_xyzz_pack().
  pdul_rns_twisted_pack: pdul_rns_twisted_pack().
  rns_twist_zero: rns_twist_zero().
  sub_2u16: sub_2u16.
  add_rns_3u16: add_rns_3u16.
---
# Module: [`jaxite_ec/elliptic_curve.py`](../../../../../raw/code/jaxite/jaxite_ec/elliptic_curve.py)

## Functions
- `padd_barret_xyzz(x1: jax.Array, y1: jax.Array, zz1: jax.Array, zzz1: jax.Array, x2: jax.Array, y2: jax.Array, zz2: jax.Array, zzz2: jax.Array)` — [`L44`](../../../../../raw/code/jaxite/jaxite_ec/elliptic_curve.py#L44) — PADD-BARRETT elliptic curve operation with packed arguments.
- `padd_barrett_xyzz_pack(x1_y1_zz1_zzz1: jax.Array, x2_y2_zz2_zzz2: jax.Array)` — [`L172`](../../../../../raw/code/jaxite/jaxite_ec/elliptic_curve.py#L172)
- `padd_barrett_xyzz_pack_batch_first(x1_y1_zz1_zzz1: jax.Array, x2_y2_zz2_zzz2: jax.Array, transpose=(0, 1, 2))` — [`L200`](../../../../../raw/code/jaxite/jaxite_ec/elliptic_curve.py#L200)
- `padd_lazy_twisted(x1: jax.Array, y1: jax.Array, z1: jax.Array, t1: jax.Array, x2: jax.Array, y2: jax.Array, z2: jax.Array, t2: jax.Array, twisted_d_chunk=util.TWIST_D_INT_CHUNK)` — [`L389`](../../../../../raw/code/jaxite/jaxite_ec/elliptic_curve.py#L389) — PADD-LAZY elliptic curve operation with packed arguments.
- `padd_lazy_twisted_pack(x1_y1_z1_t1: jax.Array, x2_y2_z2_t2: jax.Array)` — [`L572`](../../../../../raw/code/jaxite/jaxite_ec/elliptic_curve.py#L572)
- `padd_lazy_xyzz(x1: jax.Array, y1: jax.Array, zz1: jax.Array, zzz1: jax.Array, x2: jax.Array, y2: jax.Array, zz2: jax.Array, zzz2: jax.Array)` — [`L217`](../../../../../raw/code/jaxite/jaxite_ec/elliptic_curve.py#L217) — PADD-LAZY elliptic curve operation with packed arguments.
- `padd_lazy_xyzz_pack(x1_y1_zz1_zzz1: jax.Array, x2_y2_zz2_zzz2: jax.Array)` — [`L363`](../../../../../raw/code/jaxite/jaxite_ec/elliptic_curve.py#L363)
- `padd_rns_twisted_pack(x1_y1_zz1_zzz1: jax.Array, x2_y2_zz2_zzz2: jax.Array, rns_mat=util.RNS_MAT, twist_d=util.TWIST_D_RNS)` — [`L773`](../../../../../raw/code/jaxite/jaxite_ec/elliptic_curve.py#L773) — PADD-RNS elliptic curve operation with packed arguments.
- `padd_rns_xyzz_pack(x1_y1_zz1_zzz1: jax.Array, x2_y2_zz2_zzz2: jax.Array, rns_mat=util.RNS_MAT)` — [`L606`](../../../../../raw/code/jaxite/jaxite_ec/elliptic_curve.py#L606) — PADD-RNS elliptic curve operation with packed arguments.
- `pdul_barret_xyzz(x1: jax.Array, y1: jax.Array, zz1: jax.Array, zzz1: jax.Array)` — [`L111`](../../../../../raw/code/jaxite/jaxite_ec/elliptic_curve.py#L111) — PDUL-BARRET elliptic curve operation with packed arguments.
- `pdul_barrett_xyzz_pack(x1_y1_zz1_zzz1: jax.Array)` — [`L165`](../../../../../raw/code/jaxite/jaxite_ec/elliptic_curve.py#L165)
- `pdul_barrett_xyzz_pack_batch_first(x1_y1_zz1_zzz1: jax.Array, transpose=(0, 1, 2))` — [`L188`](../../../../../raw/code/jaxite/jaxite_ec/elliptic_curve.py#L188)
- `pdul_lazy_twisted(x1: jax.Array, y1: jax.Array, z1: jax.Array, t1: jax.Array)` — [`L466`](../../../../../raw/code/jaxite/jaxite_ec/elliptic_curve.py#L466) — PDUL-LAZY elliptic curve operation with packed arguments.
- `pdul_lazy_twisted_pack(x1_y1_z1_t1: jax.Array)` — [`L585`](../../../../../raw/code/jaxite/jaxite_ec/elliptic_curve.py#L585)
- `pdul_lazy_xyzz(x1: jax.Array, y1: jax.Array, zz1: jax.Array, zzz1: jax.Array)` — [`L296`](../../../../../raw/code/jaxite/jaxite_ec/elliptic_curve.py#L296) — PDUL-BARRET elliptic curve operation with packed arguments.
- `pdul_lazy_xyzz_pack(x1_y1_zz1_zzz1: jax.Array)` — [`L377`](../../../../../raw/code/jaxite/jaxite_ec/elliptic_curve.py#L377)
- `pdul_rns_twisted(x1: jax.Array, y1: jax.Array, z1: jax.Array, t1: jax.Array, rns_mat=util.RNS_MAT)` — [`L835`](../../../../../raw/code/jaxite/jaxite_ec/elliptic_curve.py#L835) — PDUL-RNS elliptic curve operation with packed arguments.
- `pdul_rns_twisted_pack(x1_y1_zz1_zzz1: jax.Array, rns_mat=util.RNS_MAT)` — [`L882`](../../../../../raw/code/jaxite/jaxite_ec/elliptic_curve.py#L882)
- `pdul_rns_xyzz(x1: jax.Array, y1: jax.Array, zz1: jax.Array, zzz1: jax.Array, rns_mat=util.RNS_MAT)` — [`L690`](../../../../../raw/code/jaxite/jaxite_ec/elliptic_curve.py#L690) — PDUL-RNS elliptic curve operation with packed arguments.
- `pdul_rns_xyzz_pack(x1_y1_zz1_zzz1: jax.Array, rns_mat=util.RNS_MAT)` — [`L760`](../../../../../raw/code/jaxite/jaxite_ec/elliptic_curve.py#L760)
- `pneg_lazy_twisted(x1: jax.Array, y1: jax.Array, z1: jax.Array, t1: jax.Array)` — [`L530`](../../../../../raw/code/jaxite/jaxite_ec/elliptic_curve.py#L530) — PDUL-LAZY elliptic curve operation with packed arguments.
- `pneg_lazy_twisted_pack(x1_y1_z1_t1: jax.Array)` — [`L594`](../../../../../raw/code/jaxite/jaxite_ec/elliptic_curve.py#L594)
- `rns_twist_zero()` — [`L893`](../../../../../raw/code/jaxite/jaxite_ec/elliptic_curve.py#L893)

## Module values
- `add_2u16` — [`L25`](../../../../../raw/code/jaxite/jaxite_ec/elliptic_curve.py#L25)
- `add_3u16` — [`L24`](../../../../../raw/code/jaxite/jaxite_ec/elliptic_curve.py#L24)
- `add_rns_2u16` — [`L32`](../../../../../raw/code/jaxite/jaxite_ec/elliptic_curve.py#L32)
- `add_rns_3u16` — [`L33`](../../../../../raw/code/jaxite/jaxite_ec/elliptic_curve.py#L33)
- `add_sub_rns_var` — [`L34`](../../../../../raw/code/jaxite/jaxite_ec/elliptic_curve.py#L34)
- `cond_sub_2u16` — [`L27`](../../../../../raw/code/jaxite/jaxite_ec/elliptic_curve.py#L27)
- `cond_sub_mod_u16` — [`L28`](../../../../../raw/code/jaxite/jaxite_ec/elliptic_curve.py#L28)
- `mod_mul_barrett_2u16` — [`L29`](../../../../../raw/code/jaxite/jaxite_ec/elliptic_curve.py#L29)
- `mod_mul_lazy_2u16` — [`L30`](../../../../../raw/code/jaxite/jaxite_ec/elliptic_curve.py#L30)
- `mod_mul_rns_2u16` — [`L31`](../../../../../raw/code/jaxite/jaxite_ec/elliptic_curve.py#L31)
- `negate_rns_for_var_add` — [`L35`](../../../../../raw/code/jaxite/jaxite_ec/elliptic_curve.py#L35)
- `negate_rns_for_var_add_zero_check` — [`L36`](../../../../../raw/code/jaxite/jaxite_ec/elliptic_curve.py#L36)
- `rns_constant` — [`L39`](../../../../../raw/code/jaxite/jaxite_ec/elliptic_curve.py#L39)
- `sub_2u16` — [`L26`](../../../../../raw/code/jaxite/jaxite_ec/elliptic_curve.py#L26)

