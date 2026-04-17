import unittest
from unittest.mock import MagicMock, patch

from google import genai

from evaluation.code_adapter import code_adapter
from evaluation.custom_types.kernel_task import KernelTask


class TestCodeAdapter(unittest.TestCase):
  def setUp(self):
    self.mock_client = MagicMock(spec=genai.Client)
    # Use 2 retries to make the test run faster
    self.adapter = code_adapter.CodeAdapter(
      client=self.mock_client, max_retries=2
    )

  @patch.object(code_adapter.CodeAdapter, "_get_adapt_reference_prompt")
  def test_adapt_reference_success(self, mock_prompt):
    mock_prompt.return_value = "mock reference prompt"
    mock_response = MagicMock()
    # Test that it properly strips python markdown formatting
    mock_response.text = "```python\n# Imports\nimport jax\n# Initialization\nx = 1\n# Computation\ndef comp(): pass\n```"
    self.mock_client.models.generate_content.return_value = mock_response

    result = self.adapter.adapt("original code")

    self.assertEqual(
      result,
      "# Imports\nimport jax\n# Initialization\nx = 1\n# Computation\ndef comp(): pass",
    )
    self.mock_client.models.generate_content.assert_called_once()

  @patch.object(code_adapter.CodeAdapter, "_get_adapt_optimized_prompt")
  def test_adapt_optimized_success(self, mock_prompt):
    mock_prompt.return_value = "mock optimized prompt"
    mock_response = MagicMock()
    # Test with no markdown backticks
    mock_response.text = "# Imports\nimport jax\n# Initialization\nx = 1\n# Computation\ndef comp(): pass"
    self.mock_client.models.generate_content.return_value = mock_response

    result = self.adapter.adapt(
      "original code",
      adapt_optimized=True,
      get_inputs_code="def get_inputs(): pass",
    )

    self.assertEqual(
      result,
      "# Imports\nimport jax\n# Initialization\nx = 1\n# Computation\ndef comp(): pass",
    )
    self.mock_client.models.generate_content.assert_called_once()

  def test_adapt_optimized_missing_get_inputs(self):
    with self.assertRaisesRegex(ValueError, "get_inputs_code must be provided"):
      self.adapter.adapt("original code", adapt_optimized=True)

  @patch.object(code_adapter.time, "sleep")
  @patch.object(code_adapter.CodeAdapter, "_get_adapt_reference_prompt")
  def test_adapt_retries_and_fails_on_missing_sections(
    self, mock_prompt, mock_sleep
  ):
    mock_prompt.return_value = "mock prompt"
    mock_response = MagicMock()
    # Missing the required # Imports, # Initialization, # Computation sections
    mock_response.text = "```python\ndef bad_format(): pass\n```"
    self.mock_client.models.generate_content.return_value = mock_response

    with self.assertRaisesRegex(RuntimeError, "Failed to refactor code"):
      self.adapter.adapt("original code")

    # max_retries = 2, so it should attempt 2 times
    self.assertEqual(self.mock_client.models.generate_content.call_count, 2)
    mock_sleep.assert_called()

  @patch.object(code_adapter.time, "sleep")
  @patch.object(code_adapter.CodeAdapter, "_get_adapt_reference_prompt")
  def test_adapt_retries_and_fails_on_exception(self, mock_prompt, mock_sleep):
    mock_prompt.return_value = "mock prompt"
    self.mock_client.models.generate_content.side_effect = Exception(
      "API Error"
    )

    with self.assertRaisesRegex(RuntimeError, "Failed to refactor code"):
      self.adapter.adapt("original code")

    self.assertEqual(self.mock_client.models.generate_content.call_count, 2)
    mock_sleep.assert_called()

  def test_extract_input_gen_code_success(self):
    sample_code = (
      "# Imports\n"
      "import jax\n"
      "import jax.numpy as jnp\n"
      "# Initialization\n"
      "BATCH = 8\n"
      "\n"
      "def get_inputs():\n"
      "    x = jnp.zeros(BATCH)\n"
      "    return [x], []\n"
      "# Computation\n"
      "def computation(x):\n"
      "    return x\n"
    )
    expected_extracted = (
      "def get_inputs():\n"
      "    import jax\n"
      "    import jax.numpy as jnp\n"
      "\n"
      "    BATCH = 8\n"
      "\n"
      "    x = jnp.zeros(BATCH)\n"
      "    return [x], []"
    )

    result = self.adapter._extract_input_gen_code(sample_code)
    self.assertEqual(result, expected_extracted)

  def test_extract_input_gen_code_missing_get_inputs(self):
    sample_code = (
      "# Imports\n"
      "import jax\n"
      "# Initialization\n"
      "BATCH = 8\n"
      "# Computation\n"
      "def computation(x):\n"
      "    return x\n"
    )
    result = self.adapter._extract_input_gen_code(sample_code)
    self.assertEqual(result, "")

  def test_generate_kernel_task(self):
    sample_code = (
      "# Imports\n"
      "import jax\n"
      "# Initialization\n"
      "def get_inputs():\n"
      "    return [], []\n"
      "# Computation\n"
      "def computation(): pass\n"
    )

    task = self.adapter.generate_kernel_task(
      "test_id", "test desc", sample_code
    )

    self.assertIsInstance(task, KernelTask)
    self.assertEqual(task.task_id, "test_id")
    self.assertEqual(task.description, "test desc")
    self.assertIn("def get_inputs():", task.input_gen_code)
    self.assertIn("import jax", task.input_gen_code)
