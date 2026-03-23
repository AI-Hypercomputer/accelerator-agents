import unittest
from unittest.mock import MagicMock, patch

from google.api_core import exceptions
from src.bq_helpers import add_entry, query_entries, remove_entry


class TestBigQueryHelpers(unittest.TestCase):
  @patch("src.bq_helpers.bigquery.Client")
  def test_add_entry(self, mock_client_constructor):
    mock_client = mock_client_constructor.return_value
    table_id = "project.dataset.table"
    row_to_add = {"col1": "val1", "col2": "val2"}

    # Test successful insertion
    mock_client.insert_rows_json.return_value = []  # No errors
    add_entry(table_id, row_to_add)
    mock_client.insert_rows_json.assert_called_once_with(table_id, [row_to_add])

    # Test error handling
    mock_client.insert_rows_json.reset_mock()
    mock_client.insert_rows_json.side_effect = exceptions.GoogleAPICallError("API error")
    with self.assertRaises(exceptions.GoogleAPICallError):
      add_entry(table_id, row_to_add)
    mock_client.insert_rows_json.assert_called_once_with(table_id, [row_to_add])

  @patch("src.bq_helpers.bigquery.Client")
  def test_remove_entry(self, mock_client_constructor):
    mock_client = mock_client_constructor.return_value
    mock_query_job = MagicMock()
    mock_client.query.return_value = mock_query_job
    table_id = "project.dataset.table"
    condition = "col1 = 'val1'"
    expected_query = f"DELETE FROM `{table_id}` WHERE {condition}"

    # Test successful removal
    remove_entry(table_id, condition)
    mock_client.query.assert_called_once_with(expected_query)
    mock_query_job.result.assert_called_once()

    # Test error handling when client.query raises
    mock_client.query.reset_mock()
    mock_query_job.result.reset_mock()
    mock_client.query.side_effect = exceptions.GoogleAPICallError("API error from query")
    with self.assertRaises(exceptions.GoogleAPICallError):
      remove_entry(table_id, condition)
    mock_client.query.assert_called_once_with(expected_query)
    mock_query_job.result.assert_not_called()  # Should not be called if query fails

    # Test error handling when query_job.result() raises
    mock_client.query.reset_mock()
    mock_client.query.side_effect = None  # Reset side effect
    mock_client.query.return_value = mock_query_job  # Re-assign mock_query_job
    mock_query_job.result.reset_mock()
    mock_query_job.result.side_effect = exceptions.GoogleAPICallError("API error from result")
    with self.assertRaises(exceptions.GoogleAPICallError):
      remove_entry(table_id, condition)
    mock_client.query.assert_called_once_with(expected_query)
    mock_query_job.result.assert_called_once()

  @patch("src.bq_helpers.bigquery.Client")
  def test_query_entries(self, mock_client_constructor):
    mock_client = mock_client_constructor.return_value
    mock_query_job = MagicMock()
    mock_client.query.return_value = mock_query_job
    query = "SELECT col1 FROM project.dataset.table WHERE col2 = 'val2'"

    # Mock rows - BigQuery rows are not simple dicts, they behave like tuples/lists
    # and have a .to_api_repr() or can be accessed by keys/index.
    # We'll mock the iterator behavior and dict conversion.
    mock_row1 = MagicMock()
    mock_row1.keys.return_value = ["col1"]
    mock_row1.__getitem__ = lambda s, k: "val1" if k == "col1" or k == 0 else None

    mock_row2 = MagicMock()
    mock_row2.keys.return_value = ["col1"]
    mock_row2.__getitem__ = lambda s, k: "val2" if k == "col1" or k == 0 else None

    mock_rows_iterable = [mock_row1, mock_row2]
    expected_results = [{"col1": "val1"}, {"col1": "val2"}]

    # Test successful query
    mock_query_job.result.return_value = mock_rows_iterable
    results = query_entries(query)
    mock_client.query.assert_called_once_with(query)
    mock_query_job.result.assert_called_once()
    self.assertEqual(results, expected_results)

    # Test error handling when client.query raises
    mock_client.query.reset_mock()
    mock_query_job.result.reset_mock()
    mock_client.query.side_effect = exceptions.GoogleAPICallError("API error from query")
    # As per current implementation, query_entries prints error and returns []
    # It does not re-raise the exception for query_entries.
    results_on_error = query_entries(query)
    mock_client.query.assert_called_once_with(query)
    mock_query_job.result.assert_not_called()
    self.assertEqual(results_on_error, [])

    # Test error handling when query_job.result() raises
    mock_client.query.reset_mock()
    mock_client.query.side_effect = None  # Reset side effect
    mock_client.query.return_value = mock_query_job
    mock_query_job.result.reset_mock()
    mock_query_job.result.side_effect = exceptions.GoogleAPICallError("API error from result")
    results_on_error_result = query_entries(query)
    mock_client.query.assert_called_once_with(query)
    mock_query_job.result.assert_called_once()
    self.assertEqual(results_on_error_result, [])


if __name__ == "__main__":
  unittest.main()
