import argparse
from collections import Counter

from google.cloud import bigquery


def count_operation_words(project_id, dataset_id, table_id):
  """
  Query BigQuery table for operation_name column and count occurrences of specific words.

  Args:
      project_id (str): Your Google Cloud project ID
      dataset_id (str): BigQuery dataset ID
      table_id (str): BigQuery table ID
  """
  # Initialize BigQuery client
  client = bigquery.Client(project=project_id)

  # SQL query to select operation_name column
  query = f"""
        SELECT operation_name, file_path
        FROM `{project_id}.{dataset_id}.{table_id}`
        WHERE operation_name IS NOT NULL
    """

  print(f"Executing query on {project_id}.{dataset_id}.{table_id}...")

  # Execute query
  query_job = client.query(query)
  results = query_job.result()

  # Initialize counters
  non_test_files_count = 0
  total_rows = 0
  word_counter = Counter()

  # Process results
  for row in results:
    operation_name = row.operation_name.lower() if row.operation_name else ""
    file_path = row.file_path.lower() if row.file_path else ""
    total_rows += 1

    # Count occurrences (case-insensitive)
    if "test" not in file_path:
      non_test_files_count += 1

    # Split operation_name by underscores and count words
    if operation_name:
      words = operation_name.split("_")
      for word in words:
        if word:  # Skip empty strings
          word_counter[word] += 1

  # Display results
  print("\nResults:")
  print(f"Total rows processed: {total_rows}")
  print(f"Rows with file paths NOT containing 'test': {non_test_files_count}")

  # Display most common words
  print("\nMost common words in operation_name:")
  for word, count in word_counter.most_common(20):
    print(f"  {word}: {count}")

  return {
    "total_rows": total_rows,
    "non_test_files_count": non_test_files_count,
    "most_common_words": word_counter.most_common(20),
  }


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
    description="Count occurrences of 'attention' and 'matmul' in BigQuery operation names"
  )
  parser.add_argument("--project-id", required=True, help="Google Cloud project ID")
  parser.add_argument("--dataset-id", required=True, help="BigQuery dataset ID")
  parser.add_argument("--table-id", required=True, help="BigQuery table ID")

  args = parser.parse_args()

  try:
    results = count_operation_words(args.project_id, args.dataset_id, args.table_id)
  except Exception as e:
    print(f"Error: {e}")
    print("Make sure to:")
    print("1. Set up Google Cloud authentication")
    print("2. Install google-cloud-bigquery: pip install google-cloud-bigquery")
