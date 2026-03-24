import json
import logging
import os
import tempfile

import pandas as pd
from bq_upload import prepare_dataframe, upload_to_bigquery
from google.cloud import bigquery

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_test_data():
  """Create sample test data that matches the expected schema."""
  test_data = [
    {
      "operation_name": "test_operation_1",
      "code": "def test_func(): return 1",
      "framework": "jax",
      "operation_class": "math",
      "hardware": "tpu",
      "associated_operations": json.dumps(["op1", "op2"]),
      "embedding": json.dumps([0.1, 0.2, 0.3, 0.4]),
      "embedding_model": "test_model",
    },
    {
      "operation_name": "test_operation_2",
      "code": "def test_func2(): return 2",
      "framework": "pytorch",
      "operation_class": "linear",
      "hardware": "gpu",
      "associated_operations": json.dumps(["op3", "op4"]),
      "embedding": json.dumps([0.5, 0.6, 0.7, 0.8]),
      "embedding_model": "test_model",
    },
  ]
  return pd.DataFrame(test_data)


def verify_and_cleanup_bigquery_entries(table_name, test_operation_names, project_id=None):
  """Verify test entries exist in BigQuery and then delete them."""
  try:
    # Initialize BigQuery client
    if project_id:
      client = bigquery.Client(project=project_id)
    else:
      client = bigquery.Client()

    # First, verify entries exist
    logger.info("Verifying test entries exist in BigQuery...")
    operation_names_str = "', '".join(test_operation_names)
    query = f"""
        SELECT uuid, operation_name 
        FROM `{table_name}` 
        WHERE operation_name IN ('{operation_names_str}')
        """

    query_job = client.query(query)
    results = list(query_job.result())

    if len(results) != len(test_operation_names):
      logger.error(f"Expected {len(test_operation_names)} entries, found {len(results)}")
      raise AssertionError("Test entries not found in BigQuery table")

    logger.info(f"✓ Verified {len(results)} test entries exist in BigQuery")
    for row in results:
      logger.info(f"  - Found: {row.operation_name} (UUID: {row.uuid})")

    # Now delete the test entries
    logger.info("Cleaning up test entries from BigQuery...")
    delete_query = f"""
        DELETE FROM `{table_name}` 
        WHERE operation_name IN ('{operation_names_str}')
        """

    delete_job = client.query(delete_query)
    delete_job.result()  # Wait for delete to complete

    # Verify deletion
    verify_query = f"""
        SELECT COUNT(*) as count 
        FROM `{table_name}` 
        WHERE operation_name IN ('{operation_names_str}')
        """

    verify_job = client.query(verify_query)
    verify_result = list(verify_job.result())[0]

    if verify_result.count == 0:
      logger.info("✓ Successfully deleted all test entries from BigQuery")
    else:
      logger.warning(f"Warning: {verify_result.count} test entries still remain in table")

  except Exception as e:
    logger.error(f"Error during BigQuery verification/cleanup: {str(e)}")
    raise


def test_prepare_dataframe():
  """Test the prepare_dataframe function."""
  logger.info("Testing prepare_dataframe function...")

  # Test with complete data
  df = create_test_data()
  prepared_df = prepare_dataframe(df)

  logger.info(f"Original columns: {list(df.columns)}")
  logger.info(f"Prepared columns: {list(prepared_df.columns)}")

  assert len(prepared_df) == 2

  # Check for all expected columns (including uuid which should be added)
  expected_columns = [
    "uuid",
    "operation_name",
    "code",
    "framework",
    "operation_class",
    "hardware",
    "associated_operations",
    "embedding",
    "embedding_model",
  ]

  missing_columns = [col for col in expected_columns if col not in prepared_df.columns]
  if missing_columns:
    logger.error(f"Missing columns: {missing_columns}")
    assert False, f"Missing columns: {missing_columns}"

  logger.info("✓ All expected columns present including UUID")

  # Test with missing columns
  partial_df = pd.DataFrame({"code": ["test code"], "embedding": [json.dumps([1.0, 2.0])]})
  prepared_partial = prepare_dataframe(partial_df)
  logger.info(f"Partial prepared columns: {list(prepared_partial.columns)}")

  # Should have all expected columns after preparation
  assert len(prepared_partial.columns) == len(expected_columns)
  assert prepared_partial["operation_name"].isna().iloc[0]
  assert prepared_partial["uuid"].notna().iloc[0]  # UUID should be generated

  logger.info("prepare_dataframe tests passed!")


def test_bigquery_upload(table_name, project_id=None):
  """Test the actual BigQuery upload."""
  logger.info(f"Testing BigQuery upload to table: {table_name}")

  # Create test CSV file
  test_df = create_test_data()
  test_operation_names = test_df["operation_name"].tolist()

  with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
    test_df.to_csv(f.name, index=False)
    csv_file_path = f.name

  try:
    # Test upload
    upload_to_bigquery(csv_file_path, table_name, project_id)
    logger.info("BigQuery upload test passed!")

    # Verify entries exist and then clean them up
    verify_and_cleanup_bigquery_entries(table_name, test_operation_names, project_id)

  finally:
    # Clean up temp file
    os.unlink(csv_file_path)


def main():
  """Main test function."""
  import argparse

  parser = argparse.ArgumentParser(description="Test BigQuery upload functionality")
  parser.add_argument(
    "--table-name",
    "-t",
    required=True,
    help="BigQuery table name in format dataset.table",
  )
  parser.add_argument("--project-id", "-p", help="GCP project ID (optional)")
  parser.add_argument("--skip-upload", action="store_true", help="Skip actual BigQuery upload test")

  args = parser.parse_args()

  try:
    # Test dataframe preparation
    test_prepare_dataframe()

    # Test BigQuery upload (if not skipped)
    if not args.skip_upload:
      test_bigquery_upload(args.table_name, args.project_id)
    else:
      logger.info("Skipping BigQuery upload test")

    logger.info("All tests passed!")

  except Exception as e:
    logger.error(f"Test failed: {str(e)}")
    raise


if __name__ == "__main__":
  main()
