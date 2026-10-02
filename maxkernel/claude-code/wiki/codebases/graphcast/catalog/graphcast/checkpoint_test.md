---
title: 'Module: graphcast/checkpoint_test.py'
type: catalog
provenance: extracted
module: graphcast/checkpoint_test.py
status: fresh
symbol_base: scip-python python graphcast 0.0.0 `graphcast.checkpoint_test`/
symbols:
  DataclassTest.test_serialize_dataclass: DataclassTest#test_serialize_dataclass().
  SubConfig: SubConfig#
  Checkpoint.params: Checkpoint#params.
  Checkpoint.config: Checkpoint#config.
  Config.ldc: Config#ldc.
  Config.t: Config#t.
  Config.tdc: Config#tdc.
  Config.dc: Config#dc.
  Config.dco: Config#dco.
  Config.ddc: Config#ddc.
  Config: Config#
  Checkpoint: Checkpoint#
  Config.bt: Config#bt.
  Config.bf: Config#bf.
  Config.i: Config#i.
  Config.f: Config#f.
  Config.o1: Config#o1.
  Config.o2: Config#o2.
  Config.o3: Config#o3.
  Config.o4: Config#o4.
  Config.o5: Config#o5.
  Config.o6: Config#o6.
  Config.li: Config#li.
  Config.ls: Config#ls.
  Config.tf: Config#tf.
  Config.ts: Config#ts.
  Config.dsi: Config#dsi.
  Config.dss: Config#dss.
  Config.dis: Config#dis.
  Config.dsdis: Config#dsdis.
  SubConfig.a: SubConfig#a.
  SubConfig.b: SubConfig#b.
  DataclassTest: DataclassTest#
---
# Module: [`graphcast/checkpoint_test.py`](../../../../../raw/code/graphcast/graphcast/checkpoint_test.py)

## Classes
### `Checkpoint`
- def: [`graphcast/checkpoint_test.py:60`](../../../../../raw/code/graphcast/graphcast/checkpoint_test.py#L60)
- signature: `class Checkpoint:`
- members:
  - `config` — [`L62`](../../../../../raw/code/graphcast/graphcast/checkpoint_test.py#L62)
  - `params` — [`L61`](../../../../../raw/code/graphcast/graphcast/checkpoint_test.py#L61)
- uses (calls/refs, reference-scoped): [`Config`](checkpoint_test.md#Config)
- used by: [`test_serialize_dataclass`](checkpoint_test.md#DataclassTest.test_serialize_dataclass)

### `Config`
- def: [`graphcast/checkpoint_test.py:32`](../../../../../raw/code/graphcast/graphcast/checkpoint_test.py#L32)
- signature: `class Config:`
- members:
  - `bf` — [`L34`](../../../../../raw/code/graphcast/graphcast/checkpoint_test.py#L34)
  - `bt` — [`L33`](../../../../../raw/code/graphcast/graphcast/checkpoint_test.py#L33)
  - `dc` — [`L54`](../../../../../raw/code/graphcast/graphcast/checkpoint_test.py#L54)
  - `dco` — [`L55`](../../../../../raw/code/graphcast/graphcast/checkpoint_test.py#L55)
  - `ddc` — [`L56`](../../../../../raw/code/graphcast/graphcast/checkpoint_test.py#L56)
  - `dis` — [`L52`](../../../../../raw/code/graphcast/graphcast/checkpoint_test.py#L52)
  - `dsdis` — [`L53`](../../../../../raw/code/graphcast/graphcast/checkpoint_test.py#L53)
  - `dsi` — [`L50`](../../../../../raw/code/graphcast/graphcast/checkpoint_test.py#L50)
  - `dss` — [`L51`](../../../../../raw/code/graphcast/graphcast/checkpoint_test.py#L51)
  - `f` — [`L36`](../../../../../raw/code/graphcast/graphcast/checkpoint_test.py#L36)
  - `i` — [`L35`](../../../../../raw/code/graphcast/graphcast/checkpoint_test.py#L35)
  - `ldc` — [`L45`](../../../../../raw/code/graphcast/graphcast/checkpoint_test.py#L45)
  - `li` — [`L43`](../../../../../raw/code/graphcast/graphcast/checkpoint_test.py#L43)
  - `ls` — [`L44`](../../../../../raw/code/graphcast/graphcast/checkpoint_test.py#L44)
  - `o1` — [`L37`](../../../../../raw/code/graphcast/graphcast/checkpoint_test.py#L37)
  - `o2` — [`L38`](../../../../../raw/code/graphcast/graphcast/checkpoint_test.py#L38)
  - `o3` — [`L39`](../../../../../raw/code/graphcast/graphcast/checkpoint_test.py#L39)
  - `o4` — [`L40`](../../../../../raw/code/graphcast/graphcast/checkpoint_test.py#L40)
  - `o5` — [`L41`](../../../../../raw/code/graphcast/graphcast/checkpoint_test.py#L41)
  - `o6` — [`L42`](../../../../../raw/code/graphcast/graphcast/checkpoint_test.py#L42)
  - `t` — [`L48`](../../../../../raw/code/graphcast/graphcast/checkpoint_test.py#L48)
  - `tdc` — [`L49`](../../../../../raw/code/graphcast/graphcast/checkpoint_test.py#L49)
  - `tf` — [`L46`](../../../../../raw/code/graphcast/graphcast/checkpoint_test.py#L46)
  - `ts` — [`L47`](../../../../../raw/code/graphcast/graphcast/checkpoint_test.py#L47)
- uses (calls/refs, reference-scoped): [`SubConfig`](checkpoint_test.md#SubConfig)
- used by: [`test_serialize_dataclass`](checkpoint_test.md#DataclassTest.test_serialize_dataclass), [`config`](checkpoint_test.md#Checkpoint.config)

### `DataclassTest`  ·  implements/extends TestCase
- def: [`graphcast/checkpoint_test.py:65`](../../../../../raw/code/graphcast/graphcast/checkpoint_test.py#L65)
- signature: `class DataclassTest(absltest.TestCase):`
- members:
  - `test_serialize_dataclass(self)` — [`L67`](../../../../../raw/code/graphcast/graphcast/checkpoint_test.py#L67)
- uses (calls/refs, reference-scoped): [`SubConfig`](checkpoint_test.md#SubConfig), [`params`](checkpoint_test.md#Checkpoint.params), [`load`](checkpoint.md#load), [`config`](checkpoint_test.md#Checkpoint.config), [`dc`](checkpoint_test.md#Config.dc), [`dco`](checkpoint_test.md#Config.dco), [`ddc`](checkpoint_test.md#Config.ddc), [`dump`](checkpoint.md#dump), [`ldc`](checkpoint_test.md#Config.ldc), [`t`](checkpoint_test.md#Config.t), [`tdc`](checkpoint_test.md#Config.tdc), [`Checkpoint`](checkpoint_test.md#Checkpoint), [`Config`](checkpoint_test.md#Config), [`bf`](checkpoint_test.md#Config.bf), [`bt`](checkpoint_test.md#Config.bt), [`dis`](checkpoint_test.md#Config.dis), [`dsdis`](checkpoint_test.md#Config.dsdis), [`dsi`](checkpoint_test.md#Config.dsi), [`dss`](checkpoint_test.md#Config.dss), [`f`](checkpoint_test.md#Config.f), [`i`](checkpoint_test.md#Config.i), [`li`](checkpoint_test.md#Config.li), [`ls`](checkpoint_test.md#Config.ls), [`o1`](checkpoint_test.md#Config.o1), [`o2`](checkpoint_test.md#Config.o2), [`o3`](checkpoint_test.md#Config.o3), [`o4`](checkpoint_test.md#Config.o4), [`o5`](checkpoint_test.md#Config.o5), [`o6`](checkpoint_test.md#Config.o6), [`tf`](checkpoint_test.md#Config.tf), [`ts`](checkpoint_test.md#Config.ts)

### `SubConfig`
- def: [`graphcast/checkpoint_test.py:26`](../../../../../raw/code/graphcast/graphcast/checkpoint_test.py#L26)
- signature: `class SubConfig:`
- members:
  - `a` — [`L27`](../../../../../raw/code/graphcast/graphcast/checkpoint_test.py#L27)
  - `b` — [`L28`](../../../../../raw/code/graphcast/graphcast/checkpoint_test.py#L28)
- used by: [`test_serialize_dataclass`](checkpoint_test.md#DataclassTest.test_serialize_dataclass), [`dc`](checkpoint_test.md#Config.dc), [`dco`](checkpoint_test.md#Config.dco), [`ddc`](checkpoint_test.md#Config.ddc), [`ldc`](checkpoint_test.md#Config.ldc), [`t`](checkpoint_test.md#Config.t), [`tdc`](checkpoint_test.md#Config.tdc)

