"""Service for retrieval using HyDE and hybrid retrieval strategies with AlloyDB."""

import os
import re
import json
from typing import Any, Dict, List
import numpy as np
import psycopg2
from agents import base
from google import genai
from rag_pipeline.retrieval import prompts as retrieval_prompts

class RetrievalService(base.Agent):
    """Service for performing advanced retrieval using HyDE and AlloyDB."""

    def __init__(
        self,
        model: Any,
        db_config: Dict[str, str] = None,
        embedding_model_name: str = "text-embedding-005",
        api_key: str | None = None,
    ):
        super().__init__(model=model)
        """Initializes the service.

        Args:
          model: The language model to use for generation (HyDE).
          db_config: Optional dictionary containing host, database, user, password, port. If None, reads from environment.
          embedding_model_name: Name of the embedding model to use.
          api_key: The API key for Google AI services.
        """
        self._model = model
        self._embedding_model_name = embedding_model_name
        api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        self._genai_client = genai.Client(api_key=api_key)

        if db_config is None:
            self._db_config = {
                'host': os.environ.get('ALLOYDB_HOST'),
                'database': os.environ.get('ALLOYDB_DB', 'postgres'),
                'user': os.environ.get('ALLOYDB_USER', 'postgres'),
                'password': os.environ.get('ALLOYDB_PASS'),
                'port': int(os.environ.get('ALLOYDB_PORT', '5432'))
            }
            if not self._db_config['host'] or not self._db_config['password']:
                raise ValueError("Missing required AlloyDB environment variables (ALLOYDB_HOST, ALLOYDB_PASS).")
        else:
            self._db_config = db_config

    def generate_draft_code(self, query: str) -> str:
        """Generates a hypothetical draft code snippet based on the query."""
        hyde_prompt = retrieval_prompts.HYDE_PROMPT.format(query=query)
        try:
            response = self._model.generate(hyde_prompt)
            code_blocks = re.findall(r"```(?:python)?\n(.*?)\n```", response, re.DOTALL)
            if code_blocks:
                return code_blocks[0]
            return response
        except Exception as e:
            print(f"HyDE generation failed: {e}. Falling back to raw query.")
            return query

    def keyword_search(self, query: str, top_k: int = 4) -> List[Dict[str, Any]]:
        """Performs a simple keyword search in AlloyDB using ILIKE."""
        try:
            conn = psycopg2.connect(
                host=self._db_config['host'],
                database=self._db_config['database'],
                user=self._db_config['user'],
                password=self._db_config['password'],
                port=self._db_config['port'],
                sslmode='require'
            )
            cur = conn.cursor()
            
            # Simple keyword search
            search_query = """
            SELECT file_path, code_chunk, metadata
            FROM chunked_code_snippets
            WHERE code_chunk ILIKE %s
            LIMIT %s;
            """
            cur.execute(search_query, (f"%{query}%", top_k))
            results = cur.fetchall()
            
            retrieved_context = []
            for row in results:
                file_path, code_chunk, metadata_json = row
                metadata = json.loads(metadata_json) if isinstance(metadata_json, str) else metadata_json
                retrieved_context.append(
                    {
                        "name": os.path.basename(file_path),
                        "text": code_chunk,
                        "file": file_path,
                        "metadata": metadata
                    }
                )
            return retrieved_context
        except Exception as e:
            print(f"AlloyDB keyword search failed: {e}")
            return []
        finally:
            if 'cur' in locals():
                cur.close()
            if 'conn' in locals():
                conn.close()

    def rrf(self, vector_results: List[Dict], keyword_results: List[Dict], k: int = 60) -> List[Dict]:
        """Combines results using Reciprocal Rank Fusion (RRF)."""
        scores = {}
        
        # Score for vector search
        for rank, res in enumerate(vector_results):
            file_path = res['file']
            if file_path not in scores:
                scores[file_path] = {"doc": res, "score": 0}
            scores[file_path]["score"] += 1.0 / (k + rank + 1)
            
        # Score for keyword search
        for rank, res in enumerate(keyword_results):
            file_path = res['file']
            if file_path not in scores:
                scores[file_path] = {"doc": res, "score": 0}
            scores[file_path]["score"] += 1.0 / (k + rank + 1)
            
        # Sort by descending RRF score
        fused_results = sorted(scores.values(), key=lambda x: x["score"], reverse=True)
        return [x["doc"] for x in fused_results]

    def search_and_retrieve(
        self, query: str, top_k: int = 4
    ) -> List[Dict[str, Any]]:
        """Retrieves relevant context from AlloyDB using Hybrid Search and RRF."""
        
        # 1. HyDE: Generate draft code snippet
        draft_code = self.generate_draft_code(query)
        print(f"Generated HyDE snippet:\n{draft_code[:100]}...")

        # 2. Vector Search (Dense)
        response = self._genai_client.models.embed_content(
            model=self._embedding_model_name, contents=draft_code
        )
        snippet_embedding = response.embeddings[0].values

        vector_results = []
        try:
            conn = psycopg2.connect(
                host=self._db_config['host'],
                database=self._db_config['database'],
                user=self._db_config['user'],
                password=self._db_config['password'],
                port=self._db_config['port'],
                sslmode='require'
            )
            cur = conn.cursor()
            
            query_vector = snippet_embedding
            search_query = """
            SELECT file_path, code_chunk, metadata, 
                   (embedding <=> %s::vector) as distance
            FROM chunked_code_snippets
            ORDER BY distance ASC
            LIMIT %s;
            """
            
            vector_str = "[" + ",".join(map(str, query_vector)) + "]"
            cur.execute(search_query, (vector_str, top_k))
            results = cur.fetchall()
            
            for row in results:
                file_path, code_chunk, metadata_json, distance = row
                metadata = json.loads(metadata_json) if isinstance(metadata_json, str) else metadata_json
                vector_results.append(
                    {
                        "name": os.path.basename(file_path),
                        "text": code_chunk,
                        "file": file_path,
                        "distance": float(distance),
                        "metadata": metadata
                    }
                )
        except Exception as e:
            print(f"AlloyDB vector search failed: {e}")
        finally:
            if 'cur' in locals():
                cur.close()
            if 'conn' in locals():
                conn.close()

        # 3. Keyword Search (Sparse)
        keyword_results = self.keyword_search(query, top_k=top_k)

        # 4. Result Fusion (RRF)
        fused_results = self.rrf(vector_results, keyword_results)

        # 5. Re-ranking (Placeholder)
        # TODO: Integrate a Cross-Encoder model for final reranking
        print("Re-ranking requested but used as identity for now.")
        
        return fused_results[:top_k]

    def run(self, query: str, top_k: int = 4) -> str:
        """Runs RAG to retrieve context, augment prompt, and generate final answer."""
        context_list = self.search_and_retrieve(query, top_k=top_k)
        context_text = "\n\n".join([c['text'] for c in context_list])
        augmented_prompt = f"Use the following context to answer the query:\nContext:\n{context_text}\n\nQuery: {query}"
        return self.generate(augmented_prompt)
