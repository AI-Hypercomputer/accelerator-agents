import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

tools_dir = os.path.dirname(__file__)
sys.path.insert(0, tools_dir)

import jax
import jax.numpy as jnp
import make_vjp_base

# A forward with a static argument used as a shape, an integer index input
# (never differentiated) and a tuple output with an integer leaf -- the
# shapes that break a naive `jax.vjp(computation, *args)`.
FORWARD = """
import jax
import jax.numpy as jnp


def _helper(x, w):
  return jnp.tanh(x @ w)


def computation(x, w, idx, n):
  h = _helper(x, w)[idx]
  return h.reshape(n, -1), jnp.sum(h), jnp.argmax(h, axis=-1)
"""


def _exec(src):
  ns = {}
  exec(compile(src, "base.py", "exec"), ns)
  return ns


def _inputs():
  k1, k2, k3 = jax.random.split(jax.random.PRNGKey(0), 3)
  x = jax.random.normal(k1, (8, 4), dtype=jnp.float32)
  w = jax.random.normal(k2, (4, 6), dtype=jnp.float32)
  idx = jnp.array([3, 1, 0, 2], dtype=jnp.int32)
  ct = (
    jax.random.normal(k3, (2, 12), dtype=jnp.float32),
    jnp.float32(0.5),
    jnp.zeros((4,), dtype=jnp.int32),
  )
  return x, w, idx, ct


def _expected(x, w, idx, ct):
  fwd = _exec(FORWARD)["computation"]

  def loss(x, w):
    y, s, _ = fwd(x, w, idx, 2)
    return jnp.sum(y * ct[0]) + s * ct[1]

  return jax.grad(loss, argnums=(0, 1))(x, w)


class TestMakeVjpBase(unittest.TestCase):
  def test_default_wrt_matches_grad(self):
    x, w, idx, ct = _inputs()
    bwd = _exec(make_vjp_base.generate(FORWARD))["computation"]
    _, got = jax.jit(bwd, static_argnums=(4,))(ct, x, w, idx, 2)
    want = _expected(x, w, idx, ct)
    self.assertEqual(len(got), 2)
    for g, e in zip(got, want):
      self.assertTrue(jnp.allclose(g, e, atol=1e-5))

  def test_explicit_wrt(self):
    x, w, idx, ct = _inputs()
    bwd = _exec(make_vjp_base.generate(FORWARD, ["w"]))["computation"]
    _, got = jax.jit(bwd, static_argnums=(4,))(ct, x, w, idx, 2)
    self.assertEqual(len(got), 1)
    self.assertTrue(
      jnp.allclose(got[0], _expected(x, w, idx, ct)[1], atol=1e-5)
    )

  def test_default_returns_forward_output(self):
    x, w, idx, ct = _inputs()
    step = _exec(make_vjp_base.generate(FORWARD))["computation"]
    out, _ = jax.jit(step, static_argnums=(4,))(ct, x, w, idx, 2)
    want = _exec(FORWARD)["computation"](x, w, idx, 2)
    for o, e in zip(out, want):
      self.assertTrue(jnp.allclose(o, e))

  def test_passes_are_kept_apart(self):
    x, w, idx, ct = _inputs()
    step = _exec(make_vjp_base.generate(FORWARD))["computation"]
    hlo = jax.jit(step, static_argnums=(4,)).lower(ct, x, w, idx, 2).as_text()
    self.assertIn("optimization_barrier", hlo)

  def test_passes_form_a_custom_vjp(self):
    x, w, idx, ct = _inputs()
    ns = _exec(make_vjp_base.generate(FORWARD))
    f = jax.custom_vjp(lambda x, w: ns["forward_computation"](x, w, idx, 2))
    f.defvjp(lambda x, w: ns["forward_pass"](x, w, idx, 2), ns["backward_pass"])

    def loss(x, w):
      y, s, _ = f(x, w)
      return jnp.sum(y * ct[0]) + s * ct[1]

    got = jax.grad(loss, argnums=(0, 1))(x, w)
    for g, e in zip(got, _expected(x, w, idx, ct)):
      self.assertTrue(jnp.allclose(g, e, atol=1e-5))

  def test_static_values_in_saved_stay_static(self):
    step = _exec(make_vjp_base.generate(FORWARD))["training_step"]
    x, w, _, _ = _inputs()

    def forward_pass(x, w, eps):
      return x * w[0, 0], (w, eps)

    def backward_pass(saved, cotangent):
      w, eps = saved
      self.assertIsInstance(eps, float)  # not traced by the barrier
      return (cotangent * w[0, 0] * eps,)

    run = jax.jit(step(forward_pass, backward_pass), static_argnums=(3,))
    run(jnp.ones_like(x), x, w, 0.5)

  def test_fixed_forward_returns_gradients_only(self):
    x, w, idx, ct = _inputs()
    bwd = _exec(make_vjp_base.generate(FORWARD, optimize_forward=False))
    got = jax.jit(bwd["computation"], static_argnums=(4,))(ct, x, w, idx, 2)
    self.assertEqual(len(got), 2)
    for g, e in zip(got, _expected(x, w, idx, ct)):
      self.assertTrue(jnp.allclose(g, e, atol=1e-5))

  def test_forward_stays_readable_and_callable(self):
    ns = _exec(make_vjp_base.generate(FORWARD))
    x, w, idx, _ = _inputs()
    y, _, _ = ns["forward_computation"](x, w, idx, 2)
    self.assertEqual(y.shape, (2, 12))

  def test_alias(self):
    src = FORWARD.replace("def computation(", "def workload(") + (
      "\ncomputation = workload\n"
    )
    self.assertIn("(1,)", make_vjp_base.generate(src, ["w"]))

  def test_errors(self):
    with self.assertRaises(make_vjp_base.VjpError):
      make_vjp_base.generate(FORWARD, ["nope"])
    with self.assertRaises(make_vjp_base.VjpError):
      make_vjp_base.generate("import jax\n")
    with self.assertRaises(make_vjp_base.VjpError):
      make_vjp_base.generate(make_vjp_base.generate(FORWARD))
    with self.assertRaises(make_vjp_base.VjpError):
      make_vjp_base.generate("def computation(*a):\n  return a\n", ["x"])
    with self.assertRaises(make_vjp_base.VjpError):
      make_vjp_base.generate(
        "def computation(x):\n  return pl.pallas_call(k)(x)\n"
      )
    # A mention in a docstring is not a call.
    make_vjp_base.generate(
      '"""No pallas_call here."""\ndef computation(x):\n  return x\n'
    )

  def test_default_wrt_pytrees_and_statics(self):
    # A params pytree is differentiated; a NumPy scalar static is not.
    src = (
      "import jax.numpy as jnp\n"
      "def computation(x, params, eps):\n"
      "  return x @ params['w'] * eps\n"
    )
    import numpy as np

    x, w, _, _ = _inputs()
    bwd = _exec(make_vjp_base.generate(src, optimize_forward=False))
    run = jax.jit(bwd["computation"], static_argnums=(3,))
    got = run(jnp.ones((8, 6)), x, {"w": w}, np.float32(0.5))
    self.assertEqual(len(got), 2)
    self.assertEqual(got[1]["w"].shape, w.shape)

  def test_mixed_pytree_differentiates_float_leaves(self):
    src = (
      "import jax.numpy as jnp\n"
      "def computation(x, params):\n"
      "  return (x @ params['w'])[params['idx']]\n"
    )
    x, w, _, _ = _inputs()
    params = {"w": w, "idx": jnp.array([2, 0])}
    for optimize_forward in (True, False):
      gen = make_vjp_base.generate(src, optimize_forward=optimize_forward)
      step = _exec(gen)
      got = jax.jit(step["computation"])(jnp.ones((2, 6)), x, params)
      gx, gp = got[1] if optimize_forward else got
      self.assertIsNone(gp["idx"])  # int leaf: held fixed, no float0 gradient
      want = jax.grad(lambda w: jnp.sum((x @ w)[params["idx"]]))(w)
      self.assertTrue(jnp.allclose(gp["w"], want, atol=1e-5))

  def test_python_number_outputs(self):
    # jax.vjp passes Python-number outputs through as plain floats and ints.
    src = "def computation(x):\n  return x * 2.0, 3.0, 4\n"
    x = jnp.ones((3,))
    for optimize_forward in (True, False):
      gen = make_vjp_base.generate(src, optimize_forward=optimize_forward)
      got = _exec(gen)["computation"]((jnp.ones((3,)), 0.0, 0), x)
      (g,) = got[1] if optimize_forward else got
      self.assertTrue(jnp.allclose(g, 2.0))

  def test_nothing_to_differentiate_fails(self):
    src = "import jax.numpy as jnp\ndef computation(i):\n  return i * 2.0\n"
    bwd = _exec(make_vjp_base.generate(src, optimize_forward=False))
    with self.assertRaises(ValueError):
      bwd["computation"](jnp.ones((3,)), jnp.arange(3))

  def test_bound_computation_without_wrt(self):
    src = "import jax\ndef f(x):\n  return x * x\ncomputation = jax.jit(f)\n"
    bwd = _exec(make_vjp_base.generate(src, optimize_forward=False))
    (g,) = bwd["computation"](jnp.ones((3,)), jnp.full((3,), 2.0))
    self.assertTrue(jnp.allclose(g, 4.0))

  def test_does_not_rebind_forward_names(self):
    src = FORWARD.replace("import jax.numpy as jnp", "import jax.numpy as np")
    src = src.replace("jnp.", "np.")
    x, w, idx, ct = _inputs()
    bwd = _exec(make_vjp_base.generate(src))["computation"]
    _, got = jax.jit(bwd, static_argnums=(4,))(ct, x, w, idx, 2)
    self.assertTrue(
      jnp.allclose(got[1], _expected(x, w, idx, ct)[1], atol=1e-5)
    )

  def test_cli_failure_leaves_forward_untouched(self):
    with tempfile.TemporaryDirectory() as d:
      base = Path(d) / "base.py"
      base.write_text(FORWARD)
      cmd = [
        sys.executable,
        os.path.join(tools_dir, "make_vjp_base.py"),
        str(base),
        "--wrt",
        "nope",
      ]
      r = subprocess.run(cmd, capture_output=True)
      self.assertEqual(r.returncode, 1)
      self.assertFalse((Path(d) / "base_fwd.py").exists())
      self.assertEqual(base.read_text(), FORWARD)

  def test_cli_is_idempotent(self):
    with tempfile.TemporaryDirectory() as d:
      base = Path(d) / "base.py"
      base.write_text(FORWARD)
      cmd = [
        sys.executable,
        os.path.join(tools_dir, "make_vjp_base.py"),
        str(base),
      ]
      for _ in range(2):
        subprocess.run(cmd, check=True, capture_output=True)
      self.assertEqual((Path(d) / "base_fwd.py").read_text(), FORWARD)
      self.assertEqual(base.read_text().count(make_vjp_base.MARKER), 1)


if __name__ == "__main__":
  unittest.main()
