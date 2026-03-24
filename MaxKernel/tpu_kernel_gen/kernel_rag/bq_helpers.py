# This file will contain helper functions for interacting with BigQuery.
from google.api_core import exceptions
from google.cloud import bigquery


def add_entry(table_id: str, row_to_add: dict) -> None:
  """Adds a new row to the specified BigQuery table.

  Args:
    table_id: The fully qualified ID of the BigQuery table to add the entry to.
      (e.g., "your-project.your_dataset.your_table")
    row_to_add: A dictionary representing the row to add.
      The keys of the dictionary should correspond to the column names
      in the BigQuery table.

  Raises:
    google.api_core.exceptions.GoogleAPICallError: If the insertion fails.
      An error message is printed to stdout in this case.
  """
  client = bigquery.Client()
  try:
    errors = client.insert_rows_json(table_id, [row_to_add])
    if errors:
      print(f"Encountered errors while inserting rows: {errors}")
    else:
      print("New row added successfully.")
  except exceptions.GoogleAPICallError as e:
    print(f"Failed to insert row: {e}")
    raise


def remove_entry(table_id: str, condition: str) -> None:
  """Removes entries from the specified BigQuery table based on a condition.

  Args:
    table_id: The fully qualified ID of the BigQuery table to remove entries from.
      (e.g., "your-project.your_dataset.your_table")
    condition: A SQL condition string to filter the rows to be deleted.
      (e.g., "column_name = 'value'")

  Raises:
    google.api_core.exceptions.GoogleAPICallError: If the deletion fails.
      An error message is printed to stdout in this case.
  """
  client = bigquery.Client()
  query = f"DELETE FROM `{table_id}` WHERE {condition}"
  try:
    query_job = client.query(query)
    query_job.result()  # Wait for the job to complete
    print(f"Entries matching condition '{condition}' removed successfully.")
  except exceptions.GoogleAPICallError as e:
    print(f"Failed to remove entries: {e}")
    raise


def query_entries(query: str) -> list[dict]:
  """Executes a query and returns the results as a list of dictionaries.

  Args:
    query: The SQL query string to execute.

  Returns:
    A list of dictionaries, where each dictionary represents a row
    from the query results. Returns an empty list if the query fails and
    prints an error message to stdout.

  Raises:
    google.api_core.exceptions.GoogleAPICallError: If the query execution fails.
      An error message is printed to stdout, and an empty list is returned.
  """
  client = bigquery.Client()
  results = []
  try:
    query_job = client.query(query)
    for row in query_job.result():
      results.append(dict(row))
    print("Query executed successfully.")
  except exceptions.GoogleAPICallError as e:
    print(f"Failed to execute query: {e}")
    # Optionally re-raise the exception if the caller should handle it
    # raise
  return results
