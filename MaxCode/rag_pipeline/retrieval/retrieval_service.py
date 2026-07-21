"""Service for retrieval using HyDE and hybrid retrieval strategies with AlloyDB."""

# pylint: disable=broad-except,invalid-name,g-import-not-at-top,pointless-string-statement,g-importing-member,line-too-long

import json
import os
import re
import ssl
from typing import Any, Dict, List

from agents import base
from rag_pipeline.retrieval import prompts as retrieval_prompts

pg8000: Any = None
aiplatform: Any = None
TextEmbeddingModel: Any = None

try:
  # pylint: disable=g-import-not-at-top
  import pg8000 as _pg8000

  pg8000 = _pg8000
except ImportError as e:
  print(f"[RAG] Warning: Failed to import pg8000: {e}")
  pg8000 = None

try:
  try:
    # pylint: disable=g-import-not-at-top
    import google.cloud.aiplatform.aiplatform as _aiplatform
    from google.cloud.aiplatform.vertexai.language_models import TextEmbeddingModel as _TextEmbeddingModel

    aiplatform = _aiplatform
    TextEmbeddingModel = _TextEmbeddingModel
  except ImportError:
    # pylint: disable=g-import-not-at-top
    from google.cloud import aiplatform as _aiplatform  # pytype: disable=import-error
    from vertexai.language_models import TextEmbeddingModel as _TextEmbeddingModel  # pytype: disable=import-error

    aiplatform = _aiplatform
    TextEmbeddingModel = _TextEmbeddingModel
except ImportError:
  aiplatform = None
  TextEmbeddingModel = None


class RetrievalService(base.Agent):
  """Service for performing advanced retrieval using HyDE and AlloyDB."""

  def __init__(
      self,
      model: Any,
      db_config: Dict[str, str] | None = None,
      embedding_model_name: str = "text-embedding-005",
      api_key: str | None = None,
  ):
    super().__init__(model=model)
    """Initializes the service."""
    self._model = model
    self._embedding_model_name = embedding_model_name
    try:
      if aiplatform and TextEmbeddingModel:
        init_fn = getattr(aiplatform, "init", None)
        if init_fn:
          init_fn(
              project=os.environ.get("GCP_PROJECT"),
              location=os.environ.get("GCP_LOCATION", "us-west1"),
          )
        self._embedding_model = TextEmbeddingModel.from_pretrained(
            self._embedding_model_name
        )
      else:
        self._embedding_model = None
    except Exception as e:
      print(f"[RAG] Embedding model init note: {e}")
      self._embedding_model = None

    self._conn = None

    # Configuration DB
    if db_config is None:
      host = os.environ.get("ALLOYDB_HOST", "localhost")
      password = os.environ.get("ALLOYDB_PASS", "")
      database = os.environ.get("ALLOYDB_DB", "postgres")
      user = os.environ.get("ALLOYDB_USER", "postgres")
      port = int(os.environ.get("ALLOYDB_PORT", "5432"))

      if not host or not password:
        config_paths = [
            os.path.join(os.getcwd(), "alloydb_config.json"),
            os.path.expanduser("~/.alloydb_config.json"),
        ]
        for config_file in config_paths:
          if os.path.exists(config_file):
            try:
              with open(config_file, "r", encoding="utf-8") as f:
                file_config = json.load(f)
                host = host or file_config.get("host")
                password = password or file_config.get("password")
                database = file_config.get("database", database)
                user = file_config.get("user", user)
                if host and password:
                  break
            except Exception as e:
              print(f"[RAG] Warning reading {config_file}: {e}")

      self._db_config = {
          "host": host,
          "database": database,
          "user": user,
          "password": password,
          "port": port,
      }
    else:
      self._db_config = db_config

  def _get_connection(self):
    """Creates or returns a standard PostgreSQL connection using pg8000."""
    global pg8000
    if self._conn is None:
      if pg8000 is None:
        import pg8000  # Force import here to raise the real ImportError traceback
      ssl_context = ssl.create_default_context()
      ssl_context.check_hostname = False
      ssl_context.verify_mode = ssl.CERT_NONE

      self._conn = pg8000.connect(
          host=self._db_config["host"],
          database=self._db_config["database"],
          user=self._db_config["user"],
          password=self._db_config["password"],
          port=self._db_config["port"],
          ssl_context=ssl_context,
      )
      print("[RAG] DB connection initialized.")
    return self._conn

  def generate_draft_code(self, query: str) -> str:
    """Generates a hypothetical draft code snippet based on the query."""
    hyde_prompt = retrieval_prompts.HYDE_PROMPT.format(query=query)
    try:
      response = self._model.generate(hyde_prompt)
      code_blocks = re.findall(
          r"```(?:python)?\n(.*?)\n```", response, re.DOTALL
      )
      if code_blocks:
        return code_blocks[0]
      return response
    except Exception as e:
      print(f"HyDE generation failed: {e}. Falling back to raw query.")
      return query

  def keyword_search(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
    """Performs a simple keyword search in AlloyDB using ILIKE."""
    search_query = """
        SELECT file_path, code_chunk, metadata
        FROM chunked_code_snippets
        WHERE code_chunk ILIKE %s AND repository = 'MaxText'
        LIMIT %s;
        """
    try:
      conn = self._get_connection()
      cursor = conn.cursor()
      cursor.execute(search_query, (f"%{query}%", top_k))
      results = cursor.fetchall()
      return [
          {
              "name": os.path.basename(row[0]),
              "text": row[1],
              "file": row[0],
              "metadata": (
                  json.loads(row[2]) if isinstance(row[2], str) else row[2]
              ),
          }
          for row in results
      ]
    except Exception as e:
      print(f"Keyword search failed: {e}")
      # If the database connection was broken, reset it for retry
      self._conn = None
      return []

  def vector_search(
      self, embedding: List[float], top_k: int = 20
  ) -> List[Dict[str, Any]]:
    """Performs a dense vector search in AlloyDB."""
    search_query = """
        SELECT file_path, code_chunk, metadata,
               (embedding <=> %s::vector) as distance
        FROM chunked_code_snippets
        WHERE repository = 'MaxText'
        ORDER BY distance ASC
        LIMIT %s;
        """
    # Convert embedding to PostgreSQL vector string format
    vector_str = f"[{','.join(map(str, embedding))}]"

    try:
      conn = self._get_connection()
      cursor = conn.cursor()
      cursor.execute(search_query, (vector_str, top_k))
      results = cursor.fetchall()
      return [
          {
              "name": os.path.basename(row[0]),
              "text": row[1],
              "file": row[0],
              "distance": float(row[3]),
              "metadata": (
                  json.loads(row[2]) if isinstance(row[2], str) else row[2]
              ),
          }
          for row in results
      ]
    except Exception as e:
      print(f"Vector search failed: {e}")
      self._conn = None
      return []

  def rrf(
      self,
      vector_results: List[Dict[str, Any]],
      keyword_results: List[Dict[str, Any]],
      k: int = 60,
  ) -> List[Dict[str, Any]]:
    """Combines results using Reciprocal Rank Fusion (RRF)."""
    scores = {}

    # Score for vector search
    for rank, res in enumerate(vector_results):
      file_path = res["file"]
      if file_path not in scores:
        scores[file_path] = {"doc": res, "score": 0}
      scores[file_path]["score"] += 1.0 / (k + rank + 1)

    # Score for keyword search
    for rank, res in enumerate(keyword_results):
      file_path = res["file"]
      if file_path not in scores:
        scores[file_path] = {"doc": res, "score": 0}
      scores[file_path]["score"] += 1.0 / (k + rank + 1)

    # Sort by descending RRF score
    fused_results = sorted(
        scores.values(), key=lambda x: x["score"], reverse=True
    )
    return [x["doc"] for x in fused_results]

  def search_and_retrieve(
      self, query: str, top_k: int = 20
  ) -> List[Dict[str, Any]]:
    """Retrieves relevant context from AlloyDB using Hybrid Search and RRF."""
    print(f"[RAG] search_and_retrieve called with query of length {len(query)}")

    # 1. HyDE (CPU bound/API call)
    draft_code = self.generate_draft_code(query)
    print(f"[RAG] Generated HyDE snippet:\n{draft_code}")

    # Embedding (API call)
    if self._embedding_model:
      try:
        response = self._embedding_model.get_embeddings([draft_code])
        snippet_embedding = response[0].values
      except Exception as e:
        print(f"[RAG] Embedding generation note: {e}")
        snippet_embedding = []
    else:
      snippet_embedding = []

    # 2. Parallel search runs (using ThreadPoolExecutor)
    print("[RAG] Launching parallel searches...")
    db_top_k = max(top_k * 2, 30)

    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
      vector_future = executor.submit(
          self.vector_search, snippet_embedding, db_top_k
      )
      keyword_future = executor.submit(self.keyword_search, query, db_top_k)

      vector_results = vector_future.result()
      keyword_results = keyword_future.result()

    print(
        f"[RAG] Searches completed. Vector: {len(vector_results)}, Keyword:"
        f" {len(keyword_results)}"
    )

    # 3. Fusion RRF
    fused_results = self.rrf(vector_results, keyword_results)
    print(f"[RAG] RRF fused into {len(fused_results)} unique results.")

    return fused_results[:top_k]

  def run(self, query: str, top_k: int = 20) -> str:
    """Runs RAG to retrieve context, augment prompt, and generate final answer."""
    context_list = self.search_and_retrieve(query, top_k=top_k)
    context_text = "\n\n".join([c["text"] for c in context_list])
    augmented_prompt = (
        "Use the following context to answer the"
        f" query:\nContext:\n{context_text}\n\nQuery: {query}"
    )
    return self.generate(augmented_prompt)
