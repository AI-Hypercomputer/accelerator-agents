---
title: 'Module: python/sgl_jax/srt/kernels/fused_moe/v2/bench_se_mlp.py'
type: catalog
provenance: extracted
module: python/sgl_jax/srt/kernels/fused_moe/v2/bench_se_mlp.py
status: fresh
symbol_base: scip-python python sglang-jax 0.0.0 `python.sgl_jax.srt.kernels.fused_moe.v2.bench_se_mlp`/
symbols:
  run: run().
  toks: toks.
  per: per.
  w2: w2.
  s1: s1.
  s3: s3.
  s2: s2.
  P: P.
  repl_put: repl_put().
  a: a.
  tok_local: tok_local.
  repl: repl.
  shard: shard.
  d: d.
  w1: w1.
  w3: w3.
  body: body().
  ts: ts.
  log: log().
  se_inter: se_inter.
  kt: kt.
  mesh: mesh.
  REPS: REPS.
  q_pc: q_pc().
  ntok: ntok.
  t0: t0.
  nd: nd.
  key: key.
  body.one: body().one().
  warmup: warmup.
  iters: iters.
  k1: k1.
  k2: k2.
  k3: k3.
  w1q: w1q.
  w3q: w3q.
  w2q: w2q.
  use_fp8: use_fp8.
---
# Module: [`python/sgl_jax/srt/kernels/fused_moe/v2/bench_se_mlp.py`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_se_mlp.py)

## Functions
- `body(x, w1, w3, w2, s1, s3, s2)` — [`L83`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_se_mlp.py#L83)
- `log(m)` — [`L19`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_se_mlp.py#L19)
- `one(xf)` — [`L84`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_se_mlp.py#L84)
- `q_pc(k, shp)` — [`L49`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_se_mlp.py#L49)
- `repl_put(x)` — [`L39`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_se_mlp.py#L39)
- `run()` — [`L106`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_se_mlp.py#L106)

## Module values
- `P` — [`L25`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_se_mlp.py#L25)
- `REPS` — [`L35`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_se_mlp.py#L35)
- `a` — [`L114`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_se_mlp.py#L114)
- `d` — [`L31`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_se_mlp.py#L31)
- `iters` — [`L36`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_se_mlp.py#L36)
- `k1` — [`L46`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_se_mlp.py#L46)
- `k2` — [`L46`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_se_mlp.py#L46)
- `k3` — [`L46`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_se_mlp.py#L46)
- `key` — [`L45`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_se_mlp.py#L45)
- `kt` — [`L46`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_se_mlp.py#L46)
- `mesh` — [`L27`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_se_mlp.py#L27)
- `nd` — [`L26`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_se_mlp.py#L26)
- `ntok` — [`L33`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_se_mlp.py#L33)
- `per` — [`L118`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_se_mlp.py#L118)
- `repl` — [`L28`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_se_mlp.py#L28)
- `s1` — [`L56`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_se_mlp.py#L56)
- `s2` — [`L58`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_se_mlp.py#L58)
- `s3` — [`L57`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_se_mlp.py#L57)
- `se_inter` — [`L32`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_se_mlp.py#L32)
- `shard` — [`L29`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_se_mlp.py#L29)
- `t0` — [`L16`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_se_mlp.py#L16)
- `tok_local` — [`L66`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_se_mlp.py#L66)
- `toks` — [`L67`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_se_mlp.py#L67)
- `ts` — [`L112`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_se_mlp.py#L112)
- `use_fp8` — [`L34`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_se_mlp.py#L34)
- `w1` — [`L59`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_se_mlp.py#L59)
- `w1q` — [`L56`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_se_mlp.py#L56)
- `w2` — [`L61`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_se_mlp.py#L61)
- `w2q` — [`L58`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_se_mlp.py#L58)
- `w3` — [`L60`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_se_mlp.py#L60)
- `w3q` — [`L57`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_se_mlp.py#L57)
- `warmup` — [`L36`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/bench_se_mlp.py#L36)

