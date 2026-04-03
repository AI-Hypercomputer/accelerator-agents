import argparse
import json
import logging
import os
import uuid

import pandas as pd
from google.cloud import bigquery
from google.cloud.bigquery import LoadJobConfig

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_bigquery_schema(client, table_id):
  """
  Get the BigQuery schema from an existing table.

  Args:
      client: BigQuery client
      table_id (str): BigQuery table ID in format 'dataset.table'

  Returns:
      list: List of SchemaField objects from the existing table

  Raises:
      Exception: If the table doesn't exist or schema cannot be retrieved
  """
  try:
    table = client.get_table(table_id)
    return table.schema
  except Exception as e:
    logger.error(f"Could not get schema from table {table_id}: {str(e)}")
    raise Exception(f"Failed to retrieve schema from table {table_id}: {str(e)}")


def get_max_uuid_from_table(client, table_id):
  """
  Get the maximum UUID value from the existing BigQuery table.

  Args:
      client: BigQuery client
      table_id (str): BigQuery table ID in format 'dataset.table'

  Returns:
      int: Maximum UUID value (0 if table doesn't exist or is empty)
  """
  try:
    # Check if table exists
    try:
      table = client.get_table(table_id)
    except Exception:
      logger.info(f"Table {table_id} doesn't exist, starting UUIDs from 1")
      return 0

    # Query for maximum UUID value
    query = f"""
        SELECT MAX(CAST(uuid AS INT64)) as max_uuid
        FROM `{table_id}`
        WHERE REGEXP_CONTAINS(uuid, r'^[0-9]+$')
        """

    query_job = client.query(query)
    results = query_job.result()

    for row in results:
      max_uuid = row.max_uuid
      if max_uuid is not None:
        logger.info(f"Found maximum UUID: {max_uuid}")
        return max_uuid

    logger.info("No numeric UUIDs found in table, starting from 1")
    return 0

  except Exception as e:
    logger.warning(f"Error querying max UUID: {str(e)}, starting from 1")
    return 0


def prepare_kernel_dataframe(df, client=None, table_id=None):
  """Prepare the dataframe for BigQuery upload to kernel dataset."""
  # Map kernel_name to operation_name if it exists
  if "kernel_name" in df.columns:
    df["operation_name"] = df["kernel_name"]
    df = df.drop("kernel_name", axis=1)

  # Define expected columns from schema
  expected_columns = [
    "uuid",
    "operation_name",
    "file_path",
    "code",
    "framework",
    "operation_class",
    "hardware",
    "call_lines",
    "def_lines",
    "associated_operations",
    "embedding",
    "embedding_model",
  ]

  # Add UUID column if missing
  if "uuid" not in df.columns:
    if client and table_id:
      # Get the maximum UUID from existing table
      max_uuid = get_max_uuid_from_table(client, table_id)
      # Generate incremental UUIDs starting from max_uuid + 1
      df["uuid"] = [str(max_uuid + i + 1) for i in range(len(df))]
      logger.info(f"Generated UUIDs from {max_uuid + 1} to {max_uuid + len(df)}")
    else:
      # Fallback to regular UUIDs if no client/table_id provided
      df["uuid"] = [str(uuid.uuid4()) for _ in range(len(df))]

  # Add missing columns with null values
  for col in expected_columns:
    if col not in df.columns:
      df[col] = None

  # Convert JSON columns if they're strings
  json_columns = ["associated_operations", "embedding"]
  for col in json_columns:
    if col in df.columns:
      df[col] = df[col].apply(lambda x: json.loads(x) if isinstance(x, str) and pd.notna(x) else x)

  # Reorder columns to match schema
  df = df[expected_columns]

  return df


def prepare_documentation_dataframe(df, client=None, table_id=None):
  """Prepare the dataframe for BigQuery upload to documentation dataset."""

  # Define expected columns for documentation table based on the CSV structure
  expected_columns = [
    "chunk_id",
    "content",
    "title",
    "url",
    "section",
    "subsection",
    "word_count",
    "char_count",
    "metadata",
    "embedding",
    "embedding_model",
  ]

  # Add missing columns with null values
  for col in expected_columns:
    if col not in df.columns:
      df[col] = None

  # Convert JSON columns if they're strings, and ensure proper JSON format for BigQuery
  json_columns = ["metadata", "embedding"]
  for col in json_columns:
    if col in df.columns:
      df[col] = df[col].apply(lambda x: json.loads(x) if isinstance(x, str) and pd.notna(x) else x)

  # Reorder columns to match schema
  df = df[expected_columns]

  return df


def upload_to_bigquery(csv_file_path, table_id, project_id=None, data_type="kernel"):
  """
  Upload CSV data to BigQuery table.

  Args:
      csv_file_path (str): Path to the CSV file
      table_id (str): BigQuery table ID in format 'dataset.table'
      project_id (str): GCP project ID (optional, uses default if not provided)
      data_type (str): Type of data to upload: 'kernel' or 'documentation'
  """
  try:
    # Initialize BigQuery client
    if project_id:
      client = bigquery.Client(project=project_id)
    else:
      client = bigquery.Client()

    # Read CSV file
    logger.info(f"Reading CSV file: {csv_file_path}")
    df = pd.read_csv(csv_file_path)

    if data_type == "kernel":
      # Prepare dataframe with incremental UUIDs
      logger.info("Preparing kernel dataframe for upload")
      df = prepare_kernel_dataframe(df, client, table_id)
    elif data_type == "documentation":
      # Prepare documentation dataframe
      logger.info("Preparing documentation dataframe for upload")
      df = prepare_documentation_dataframe(df, client, table_id)
    else:
      raise ValueError(f"Invalid data type: {data_type}. Must be 'kernel' or 'documentation'.")

    # Configure load job
    job_config = LoadJobConfig(
      write_disposition="WRITE_TRUNCATE",  # Overwrite table
      autodetect=True,  # Let BigQuery auto-detect the schema
    )

    # Upload to BigQuery
    logger.info(f"Uploading data to table: {table_id}")
    job = client.load_table_from_dataframe(df, table_id, job_config=job_config)

    # Wait for job to complete
    job.result()

    # Get table info
    table = client.get_table(table_id)
    logger.info(f"Successfully uploaded {table.num_rows} rows to {table_id}")

  except Exception as e:
    logger.error(f"Error uploading to BigQuery: {str(e)}")
    raise


def main():
  """Main function to run the upload process."""
  parser = argparse.ArgumentParser(description="Upload CSV data to BigQuery")
  parser.add_argument(
    "--csv-file",
    "-f",
    default="pallas_kernels.csv",
    help="Path to the CSV file (default: pallas_kernels.csv)",
  )
  parser.add_argument(
    "--table-name",
    "-t",
    required=True,
    help="BigQuery table name in format dataset.table",
  )
  parser.add_argument(
    "--project-id",
    "-p",
    help="GCP project ID (optional, uses default if not provided)",
  )
  parser.add_argument(
    "--data-type",
    "-d",
    choices=["kernel", "documentation"],
    default="kernel",
    help="Type of data to upload: kernel or documentation (default: kernel)",
  )

  args = parser.parse_args()

  # Check if CSV file exists
  if not os.path.exists(args.csv_file):
    raise FileNotFoundError(f"CSV file not found: {args.csv_file}")

  upload_to_bigquery(args.csv_file, args.table_name, args.project_id, args.data_type)


if __name__ == "__main__":
  main()
