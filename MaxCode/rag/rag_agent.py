"""Tool for performing retrieval augmented generation."""

import ast
import logging
import os
import sqlite3
from typing import Any, Dict, List

import models
from agents import base
from rag import embedding
from rag import prompts
from rag import vector_db
import numpy as np

logger = logging.getLogger(__name__)


# We use a hardcoded character limit for the full code context to avoid
# exceeding the model's token limit. While the Gemini API does not provide a
# way to get the max context length in characters, 20000 characters
# (roughly 5000-7000 tokens) is a safe limit for models with 32k token limits,
# when considering that the prompt sends file content in two fields.
_MAX_CONTEXT_LENGTH = 100_000


def _extract_component_signatures(code: str) -> list[str]:
  """Extracts focused query strings per top-level class/function using AST.

  For classes: "JAX Flax {ClassName} {base_classes} {method_names} {init_params}"
  For functions: "JAX Flax {func_name} {param_names}"

  Args:
    code: Python source code to parse.

  Returns:
    A list of query strings, one per top-level component.
  """
  try:
    tree = ast.parse(code)
  except SyntaxError:
    return []

  signatures = []
  for node in ast.iter_child_nodes(tree):
    if isinstance(node, ast.ClassDef):
      bases = [
          ast.unparse(b) if hasattr(ast, "unparse") else getattr(b, "id", "")
          for b in node.bases
      ]
      methods = [
          n.name for n in ast.iter_child_nodes(node)
          if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
      ]
      init_params = []
      for n in ast.iter_child_nodes(node):
        if isinstance(n, ast.FunctionDef) and n.name == "__init__":
          init_params = [
              a.arg for a in n.args.args if a.arg != "self"
          ]
          break
      parts = ["JAX Flax", node.name] + bases + methods + init_params
      signatures.append(" ".join(parts))
    elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
      params = [a.arg for a in node.args.args if a.arg != "self"]
      parts = ["JAX Flax", node.name] + params
      signatures.append(" ".join(parts))
  return signatures


class RAGAgent(base.Agent):
  """Tool for performing retrieval augmented generation."""

  def __init__(
      self,
      model: Any,
      embedding_model_name: models.EmbeddingModel,
      db_path: str = vector_db.RAG_DB_FILE,
      api_key: str | None = None,
  ):
    """Initializes the agent.

    Args:
      model: The base model to use for generation.
      embedding_model_name: Name of the embedding model to use.
      db_path: Path to the RAG SQLite database.
      api_key: The API key for Google AI services.
    """
    super().__init__(model=model)
    self._db_path = db_path
    self._embedding_agent = embedding.EmbeddingAgent(
        model_name=embedding_model_name.value, api_key=api_key
    )
    vector_db.create_db(db_path)
    (
        self._ids,
        self._names,
        self._texts,
        self._files,
        self._index,
    ) = vector_db.make_embedding_index(db_path)

  def build_from_directory(self, source_path: str):
    """Builds RAG database from files in a source directory."""
    for root, _, files in os.walk(source_path):
      for filename in files:
        if filename.endswith(".py"):
          file_path = os.path.join(root, filename)
          try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
              content = f.read()
            doc_name = os.path.relpath(file_path, source_path)
            print(f"Adding {doc_name} to RAG database...")
            description = self.generate(
                prompts.CODE_DESCRIPTION,
                {
                    "code_block": content,
                    "full_code_context": content[:_MAX_CONTEXT_LENGTH],
                },
            )
            embedding_vector = self._embedding_agent.embed(description)
            vector_db.save_document(
                name=doc_name,
                text=content,
                desc=description,
                file=file_path,
                embedding=np.array(embedding_vector),
                db_path=self._db_path,
            )
          except (OSError, sqlite3.Error) as e:
            print(f"Skipping {file_path}: {e}")
    # Refresh index
    self._ids, self._names, self._texts, self._files, self._index = (
        vector_db.make_embedding_index(self._db_path)
    )
    print("Finished building RAG database.")

  def retrieve_context(
      self, query: str, top_k: int = 3
  ) -> List[Dict[str, Any]]:
    """Retrieves relevant context from the vector DB based on the query.

    Args:
      query: The query string to search for.
      top_k: The number of top results to return.

    Returns:
      A list of dictionaries, each containing 'name', 'text', 'file',
      and 'distance' for a retrieved document.
    """
    if self._index is None:
      return []
    query_embedding = self._embedding_agent.embed(query)
    results = vector_db.search_embedding(
        np.array(query_embedding), self._index, self._texts, top_k=top_k
    )
    retrieved_context = []
    for text, distance, i in results:
      retrieved_context.append({
          "name": self._names[i],
          "text": text,
          "file": self._files[i],
          "distance": distance,
      })
    return retrieved_context

  def retrieve_per_component_context(
      self,
      source_code: str,
      top_k_per_component: int = 3,
      max_total: int = 15,
  ) -> List[Dict[str, Any]]:
    """Retrieves RAG context using a hybrid full-file + per-component strategy.

    Combines broad domain context from the full source code query with
    targeted results from per-component queries. This ensures the LLM gets
    both the overall architectural patterns AND component-specific examples.

    Args:
      source_code: The full Python source code to retrieve context for.
      top_k_per_component: Number of results per component query.
      max_total: Maximum total results to return after deduplication.

    Returns:
      A deduplicated, distance-sorted list of retrieved documents.
    """
    signatures = _extract_component_signatures(source_code)

    # Fall back to single-query if AST parsing yielded nothing
    if not signatures:
      logger.info("Per-component extraction failed, falling back to single query")
      return self.retrieve_context(source_code, top_k=max_total)

    # Start with full-file query for broad domain context
    best_by_file: Dict[str, Dict[str, Any]] = {}
    full_results = self.retrieve_context(source_code, top_k=max_total)
    for doc in full_results:
      best_by_file[doc["file"]] = doc

    # If >12 components, batch into groups of 3-4 to cap embedding calls
    if len(signatures) > 12:
      batched = []
      for i in range(0, len(signatures), 4):
        batched.append(" ".join(signatures[i:i + 4]))
      queries = batched
    else:
      queries = signatures

    logger.info("Per-component RAG: %d queries from %d components (+ full-file)",
                len(queries), len(signatures))

    # Add per-component results, keeping best distance per file
    for query in queries:
      results = self.retrieve_context(query, top_k=top_k_per_component)
      for doc in results:
        fpath = doc["file"]
        if fpath not in best_by_file or doc["distance"] < best_by_file[fpath]["distance"]:
          best_by_file[fpath] = doc

    # Sort by distance, truncate to max_total
    sorted_docs = sorted(best_by_file.values(), key=lambda d: d["distance"])
    return sorted_docs[:max_total]

  def run(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """Runs RAG to retrieve context for a query."""
    return self.retrieve_context(query, top_k)
