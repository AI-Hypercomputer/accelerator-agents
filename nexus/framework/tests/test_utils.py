"""Unit tests for shared utils module."""

import json
import pathlib
import shutil
import tempfile
import unittest

try:
  from accelerator_agents.tpu_nexus.framework import utils
except ImportError:
  try:
    from .. import utils
  except ImportError:
    import utils

Path = pathlib.Path


class TestUtils(unittest.TestCase):
  """Unit tests for utils functions and config managers."""

  def setUp(self):
    super().setUp()
    self.temp_dir = Path(tempfile.mkdtemp())

  def tearDown(self):
    shutil.rmtree(self.temp_dir, ignore_errors=True)
    super().tearDown()

  def test_semver_parsing_and_ranges(self):
    v1 = utils.SemVer.parse("1.2.3")
    v2 = utils.SemVer.parse("1.2.4")
    v3 = utils.SemVer.parse("2.0.0")
    self.assertTrue(v1 < v2 < v3)
    self.assertTrue(utils.version_satisfies(v1, ">=1.0.0,<2.0.0"))
    self.assertTrue(utils.version_satisfies(v1, "==1.2.3"))
    self.assertFalse(utils.version_satisfies(v1, ">1.2.3"))
    self.assertTrue(utils.version_satisfies(v2, ">=1.2.0"))

  def test_schema_validator(self):
    schema = {
        "type": "object",
        "required": ["name", "age"],
        "properties": {
            "name": {
                "type": "string",
                "minLength": 2,
                "pattern": r"^[A-Z][a-z]+$",
            },
            "age": {"type": "integer", "minimum": 0, "maximum": 120},
            "role": {"type": "string", "enum": ["admin", "user"]},
            "tags": {"type": "array", "items": {"type": "string"}},
        },
        "additionalProperties": False,
    }
    # Valid instance
    self.assertEqual(
        utils.validate_instance(
            {"name": "Alice", "age": 30, "role": "admin", "tags": ["lead"]},
            schema,
        ),
        [],
    )

    # Missing required property
    errs = utils.validate_instance({"name": "Alice"}, schema)
    self.assertIn("$/age: required property is missing", errs)

    # Type discrimination: boolean vs integer
    errs = utils.validate_instance({"name": "Alice", "age": True}, schema)
    self.assertIn("$/age: expected integer, got bool", errs)

    # Pattern failure
    errs = utils.validate_instance({"name": "alice123", "age": 30}, schema)
    self.assertIn("$/name: string does not match '^[A-Z][a-z]+$'", errs)

    # Additional property rejection
    errs = utils.validate_instance(
        {"name": "Alice", "age": 30, "extra": 123}, schema
    )
    self.assertIn("$/extra: additional property is not allowed", errs)

  def test_toml_manager(self):
    toml_file = self.temp_dir / "config.toml"
    toml_file.write_text('[section1]\nfoo = "bar"\n', encoding="utf-8")

    # Apply setting in existing section
    record = utils.TomlConfigManager.apply_set(
        toml_file, "section1", "key1", "val1"
    )
    self.assertIsNotNone(record)

    # Apply setting in new section
    record2 = utils.TomlConfigManager.apply_set(
        toml_file, "section2", "key2", True
    )
    self.assertIsNotNone(record2)

    # Verify content
    text = toml_file.read_text(encoding="utf-8")
    self.assertIn('key1 = "val1"', text)
    self.assertIn("[section2]", text)
    self.assertIn("key2 = true", text)

  def test_json_config_manager(self):
    doc = {"a": {"b": 1}}
    records = []
    utils.JsonConfigManager.merge(doc, {"a": {"c": 2}, "d": 3}, [], records)
    self.assertEqual(doc, {"a": {"b": 1, "c": 2}, "d": 3})
    self.assertEqual(len(records), 2)


if __name__ == "__main__":
  unittest.main()
