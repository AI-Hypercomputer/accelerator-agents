import argparse
import time
from typing import Any, Dict, List

from langchain_google_community import BigQueryVectorStore

from tpu_kernel_gen.unixcoder_embeddings import UniXcoderEmbeddings


def initialize_vector_store(project_id: str, dataset_name: str, table_name: str) -> BigQueryVectorStore:
  """
  Initialize BigQuery Vector Store with text embeddings for documentation

  Args:
      project_id: Google Cloud Project ID
      dataset_name: BigQuery dataset name
      table_name: BigQuery table name

  Returns:
      Initialized BigQueryVectorStore instance
  """
  # Use text embeddings for documentation instead of code embeddings
  embedding = UniXcoderEmbeddings()

  store = BigQueryVectorStore(
    project_id=project_id,
    dataset_name=dataset_name,
    table_name=table_name,
    location="US",
    embedding=embedding,
    distance_type="COSINE",
    content_field="content",
    embedding_field="embedding",
    metadata_fields=[
      "source",
      "title",
      "section",
      "doc_type",
      "url",
      "last_updated",
      "chunk_id",
      "embedding_model",
    ],
  )

  return store


def retrieve_similar_docs(store: BigQueryVectorStore, query: str, k: int = 5) -> List[Dict[str, Any]]:
  """
  Retrieve k most similar documentation chunks for a given query

  Args:
      store: Initialized BigQueryVectorStore instance
      query: Search query string
      k: Number of similar documentation chunks to retrieve (default: 5)

  Returns:
      List of dictionaries containing search results with metadata and similarity scores
  """
  try:
    results = store.similarity_search_with_score(query=query, k=k, with_scores=True)

    # Convert results to more structured format
    formatted_results = []
    for i, result in enumerate(results):
      doc, score = result[0], result[1]
      formatted_result = {
        "rank": i + 1,
        "content": doc.page_content,
        "metadata": doc.metadata,
        "similarity_score": score,
      }
      formatted_results.append(formatted_result)

    return formatted_results

  except Exception as e:
    print(f"Error during similarity search: {e}")
    return []


def print_search_results(
  results: List[Dict[str, Any]],
  query: str,
  init_duration: float = None,
  search_duration: float = None,
):
  """
  Pretty print documentation search results

  Args:
      results: List of search results
      query: Original search query
      init_duration: Time taken to initialize the vector store in seconds
      search_duration: Time taken for the search in seconds
  """
  print(f"\nDocumentation Search Results for query: '{query}'")
  if init_duration is not None:
    print(f"Vector Store initialized in {init_duration:.3f} seconds")
  if search_duration is not None:
    print(f"Search completed in {search_duration:.3f} seconds")
  print("=" * 80)

  if not results:
    print("No results found.")
    return

  for result in results:
    print(f"\nRank {result['rank']}:")
    print("-" * 40)

    # Print metadata
    metadata = result.get("metadata", {})
    if "title" in metadata:
      print(f"Title: {metadata['title']}")
    if "section" in metadata:
      print(f"Section: {metadata['section']}")
    if "doc_type" in metadata:
      print(f"Document Type: {metadata['doc_type']}")
    if "source" in metadata:
      print(f"Source: {metadata['source']}")
    if "url" in metadata:
      print(f"URL: {metadata['url']}")
    if "chunk_id" in metadata:
      print(f"Chunk ID: {metadata['chunk_id']}")
    if "last_updated" in metadata:
      print(f"Last Updated: {metadata['last_updated']}")

    if result.get("similarity_score"):
      print(f"Similarity Score: {result['similarity_score']:.4f}")

    # Print content preview (first 200 characters)
    content = result.get("content", "")
    print("\nContent:")
    print(f"{content}")


def main():
  parser = argparse.ArgumentParser(description="Retrieve similar documentation using BigQuery Vector Search")
  parser.add_argument("--project-id", required=True, help="Google Cloud Project ID")
  parser.add_argument("--dataset-name", required=True, help="BigQuery dataset name")
  parser.add_argument("--table-name", required=True, help="BigQuery table name")
  parser.add_argument("--query", required=True, help="Search query")
  parser.add_argument(
    "--k",
    type=int,
    default=5,
    help="Number of similar documentation chunks to retrieve (default: 5)",
  )
  parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

  args = parser.parse_args()

  if args.verbose:
    print("Initializing Documentation Vector Search with the following parameters:")
    print(f"Project ID: {args.project_id}")
    print(f"Dataset Name: {args.dataset_name}")
    print(f"Table Name: {args.table_name}")
    print(f"Query: {args.query}")
    print(f"K: {args.k}")

  try:
    start_time = time.time()
    store = initialize_vector_store(args.project_id, args.dataset_name, args.table_name)
    init_duration = time.time() - start_time

    # Retrieve similar documentation
    start_time = time.time()
    results = retrieve_similar_docs(store, args.query, args.k)
    search_duration = time.time() - start_time

    # Print results
    print_search_results(results, args.query, init_duration, search_duration)

  except Exception as e:
    print(f"Error: {e}")
    return 1

  return 0


if __name__ == "__main__":
  exit(main())
