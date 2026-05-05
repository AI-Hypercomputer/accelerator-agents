"""Tool for performing retrieval augmented generation."""

import ast
import logging
import os
import sqlite3
from typing import Any, Dict, List, Optional

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

# Corpus tags supported by the RAG layer. Files whose basename starts with
# "maxtext_" are tagged "maxtext"; everything else falls back to "jax".
_KNOWN_CORPORA = ("jax", "maxtext")


def _query_prefix_for_target(target: str) -> str:
  """Returns the human-readable prefix used in component-signature queries."""
  if target == "maxtext":
    return "MaxText"
  return "JAX Flax"


def _corpus_for_filename(filename: str) -> str:
  """Auto-tags a file by its basename. `maxtext_*.py` -> 'maxtext', else 'jax'."""
  base = os.path.basename(filename)
  if base.startswith("maxtext_"):
    return "maxtext"
  return "jax"


def _extract_component_signatures(code: str, target: str = "jax") -> list[str]:
  """Extracts focused query strings per top-level class/function using AST.

  For classes: "{prefix} {ClassName} {base_classes} {method_names} {init_params}"
  For functions: "{prefix} {func_name} {param_names}"

  The prefix is "JAX Flax" for `target="jax"` and "MaxText" for
  `target="maxtext"`.

  Args:
    code: Python source code to parse.
    target: Conversion target — selects the query prefix.

  Returns:
    A list of query strings, one per top-level component.
  """
  try:
    tree = ast.parse(code)
  except SyntaxError:
    return []

  prefix = _query_prefix_for_target(target)
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
      parts = [prefix, node.name] + bases + methods + init_params
      signatures.append(" ".join(parts))
    elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
      params = [a.arg for a in node.args.args if a.arg != "self"]
      parts = [prefix, node.name] + params
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
      target: str = "jax",
  ):
    """Initializes the agent.

    Args:
      model: The base model to use for generation.
      embedding_model_name: Name of the embedding model to use.
      db_path: Path to the RAG SQLite database.
      api_key: The API key for Google AI services.
      target: Conversion target ("jax" or "maxtext"). Selects which corpus
        the agent retrieves from.
    """
    super().__init__(model=model)
    self._db_path = db_path
    self._target = target
    self._embedding_agent = embedding.EmbeddingAgent(
        model_name=embedding_model_name.value, api_key=api_key
    )
    vector_db.create_db(db_path)
    self._index_by_corpus: Dict[str, Dict[str, Any]] = {}
    self._refresh_indexes()

  def _refresh_indexes(self) -> None:
    """Rebuilds per-corpus indexes from the database."""
    self._index_by_corpus = {}
    for corpus in _KNOWN_CORPORA:
      ids, names, texts, files, index = vector_db.make_embedding_index(
          self._db_path, corpus=corpus
      )
      self._index_by_corpus[corpus] = {
          "ids": ids,
          "names": names,
          "texts": texts,
          "files": files,
          "index": index,
      }

  def _active_corpus(self) -> Dict[str, Any]:
    """Returns the corpus payload selected by `self._target`, with JAX fallback."""
    if self._target in self._index_by_corpus:
      return self._index_by_corpus[self._target]
    return self._index_by_corpus.get("jax", {
        "ids": [], "names": [], "texts": [], "files": [], "index": None,
    })

  def build_from_directory(self, source_path: str):
    """Builds RAG database from files in a source directory.

    Each file is auto-tagged with a corpus based on its basename — files
    starting with `maxtext_` are stored in the 'maxtext' corpus, all others
    in 'jax'. Retrieval at query time is filtered by the agent's `target`.
    """
    for root, _, files in os.walk(source_path):
      for filename in files:
        if filename.endswith(".py"):
          file_path = os.path.join(root, filename)
          corpus_tag = _corpus_for_filename(filename)
          try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
              content = f.read()
            doc_name = os.path.relpath(file_path, source_path)
            print(f"Adding {doc_name} to RAG database (corpus={corpus_tag})...")
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
                corpus=corpus_tag,
            )
          except (OSError, sqlite3.Error) as e:
            print(f"Skipping {file_path}: {e}")
    # Refresh per-corpus indexes
    self._refresh_indexes()
    print("Finished building RAG database.")

  def retrieve_context(
      self, query: str, top_k: int = 3
  ) -> List[Dict[str, Any]]:
    """Retrieves relevant context from the vector DB based on the query.

    Filters to the corpus selected by the agent's `target`.

    Args:
      query: The query string to search for.
      top_k: The number of top results to return.

    Returns:
      A list of dictionaries, each containing 'name', 'text', 'file',
      and 'distance' for a retrieved document.
    """
    payload = self._active_corpus()
    index = payload.get("index")
    if index is None:
      return []
    query_embedding = self._embedding_agent.embed(query)
    results = vector_db.search_embedding(
        np.array(query_embedding), index, payload["texts"], top_k=top_k
    )
    names = payload["names"]
    files = payload["files"]
    retrieved_context = []
    for text, distance, i in results:
      retrieved_context.append({
          "name": names[i],
          "text": text,
          "file": files[i],
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
    signatures = _extract_component_signatures(source_code, self._target)

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
