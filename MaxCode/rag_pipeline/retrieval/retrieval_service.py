"""Service for retrieval using HyDE and hybrid retrieval strategies with AlloyDB."""

import os
import re
import json
import asyncio
import asyncpg
from typing import Any, Dict, List
import numpy as np
from agents import base
from google.cloud import aiplatform
from vertexai.language_models import TextEmbeddingModel
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
        """Initializes the service."""
        self._model = model
        self._embedding_model_name = embedding_model_name
        self._pool = None  # Initialisation du pool à None

        aiplatform.init(project=os.environ.get('GCP_PROJECT'), location=os.environ.get('GCP_LOCATION', 'us-west1'))
        self._embedding_model = TextEmbeddingModel.from_pretrained(self._embedding_model_name)

        # Configuration DB
        if db_config is None:
            self._db_config = {
                'host': os.environ.get('ALLOYDB_HOST'),
                'database': os.environ.get('ALLOYDB_DB', 'postgres'),
                'user': os.environ.get('ALLOYDB_USER', 'postgres'),
                'password': os.environ.get('ALLOYDB_PASS'),
                'port': int(os.environ.get('ALLOYDB_PORT', '5432'))
            }
        else:
            self._db_config = db_config

    async def init_pool(self):
        """Crée le pool de connexions (à appeler une seule fois au démarrage)."""
        if not self._pool:
            self._pool = await asyncpg.create_pool(
                host=self._db_config['host'],
                database=self._db_config['database'],
                user=self._db_config['user'],
                password=self._db_config['password'],
                port=self._db_config['port'],
                ssl='require',
                min_size=5,
                max_size=10
            )
            print("[RAG] Connection Pool initialized.")

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

    async def keyword_search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Performs a simple keyword search in AlloyDB using ILIKE."""
        search_query = """
        SELECT file_path, code_chunk, metadata
        FROM chunked_code_snippets
        WHERE code_chunk ILIKE $1 AND repository = 'MaxText'
        LIMIT $2;
        """
        try:
            async with self._pool.acquire() as conn:
                results = await conn.fetch(search_query, f"%{query}%", top_k)
                return [
                    {
                        "name": os.path.basename(row['file_path']),
                        "text": row['code_chunk'],
                        "file": row['file_path'],
                        "metadata": json.loads(row['metadata']) if isinstance(row['metadata'], str) else row['metadata']
                    } for row in results
                ]
        except Exception as e:
            print(f"Keyword search failed: {e}")
            return []

    async def vector_search(self, embedding: List[float], top_k: int = 10) -> List[Dict[str, Any]]:
        """Performs a dense vector search in AlloyDB."""
        search_query = """
        SELECT file_path, code_chunk, metadata, 
               (embedding <=> $1::vector) as distance
        FROM chunked_code_snippets
        WHERE repository = 'MaxText'
        ORDER BY distance ASC
        LIMIT $2;
        """
        # Conversion de l'embedding en string format PostgreSQL vector
        vector_str = f"[{','.join(map(str, embedding))}]"
        
        try:
            async with self._pool.acquire() as conn:
                results = await conn.fetch(search_query, vector_str, top_k)
                return [
                    {
                        "name": os.path.basename(row['file_path']),
                        "text": row['code_chunk'],
                        "file": row['file_path'],
                        "distance": float(row['distance']),
                        "metadata": json.loads(row['metadata']) if isinstance(row['metadata'], str) else row['metadata']
                    } for row in results
                ]
        except Exception as e:
            print(f"Vector search failed: {e}")
            return []

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

    async def search_and_retrieve(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Retrieves relevant context from AlloyDB using Hybrid Search and RRF."""
        print(f"[RAG] search_and_retrieve called with query of length {len(query)}")
        
        # 1. HyDE (CPU bound/API call)
        draft_code = self.generate_draft_code(query)
        print(f"[RAG] Generated HyDE snippet:\n{draft_code}")
        
        # Embedding (API call)
        response = self._embedding_model.get_embeddings([draft_code])
        snippet_embedding = response[0].values

        # 2. Exécution PARALLÈLE des deux recherches
        print("[RAG] Launching parallel searches...")
        vector_task = self.vector_search(snippet_embedding, top_k)
        keyword_task = self.keyword_search(query, top_k)
        
        vector_results, keyword_results = await asyncio.gather(vector_task, keyword_task)
        print(f"[RAG] Searches completed. Vector: {len(vector_results)}, Keyword: {len(keyword_results)}")

        # 3. Fusion RRF
        fused_results = self.rrf(vector_results, keyword_results)
        print(f"[RAG] RRF fused into {len(fused_results)} unique results.")
        
        return fused_results[:top_k]

    async def run(self, query: str, top_k: int = 10) -> str:
        """Runs RAG to retrieve context, augment prompt, and generate final answer."""
        context_list = await self.search_and_retrieve(query, top_k=top_k)
        context_text = "\n\n".join([c['text'] for c in context_list])
        augmented_prompt = f"Use the following context to answer the query:\nContext:\n{context_text}\n\nQuery: {query}"
        return self.generate(augmented_prompt)
