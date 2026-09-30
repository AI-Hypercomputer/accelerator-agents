"""Tool setup for HITL kernel generation agents."""

import logging

from google.adk.models import LlmRequest
from google.adk.tools import ToolContext
from google.adk.tools.retrieval.vertex_ai_rag_retrieval import (
  VertexAiRagRetrieval,
)
from vertexai.preview import rag

from auto_agent.config import RAG_CORPUS
from auto_agent.tools.search_api_tool import search_api_tool


# Custom VertexAiRagRetrieval that forces function_declarations mode to avoid
# incompatibility with MCPToolset when using Gemini 2.x models
class CompatibleVertexAiRagRetrieval(VertexAiRagRetrieval):
  """VertexAiRagRetrieval that uses function_declarations instead of retrieval mode.

  This avoids the 400 INVALID_ARGUMENT error when mixing with MCPToolset.
  """

  async def process_llm_request(
    self,
    *,
    tool_context: ToolContext,
    llm_request: LlmRequest,
  ) -> None:
    # Always use function_declarations mode, even for Gemini 2+
    # to maintain compatibility with MCPToolset
    from google.adk.tools.retrieval.base_retrieval_tool import BaseRetrievalTool

    await BaseRetrievalTool.process_llm_request(
      self, tool_context=tool_context, llm_request=llm_request
    )


# Vertex AI RAG Engine tool
vertex_ai_rag_tool = None
if RAG_CORPUS:
  vertex_ai_rag_tool = CompatibleVertexAiRagRetrieval(
    name="retrieval_tool",
    description="Use this tool to retrieve Pallas/JAX/TPU documentation and examples from the RAG corpus. This is helpful for answering questions about Pallas concepts, JAX APIs, TPU architecture, best practices, and implementation patterns.",
    rag_resources=[rag.RagResource(rag_corpus=RAG_CORPUS)],
    similarity_top_k=10,
    vector_distance_threshold=0.6,
  )
  logging.info(
    f"Initialized CompatibleVertexAiRagRetrieval with corpus: {RAG_CORPUS}"
  )
else:
  logging.warning(
    "RAG_CORPUS not set. VertexAiRagRetrieval tool will not be available."
  )

__all__ = [
  "search_api_tool",
  "vertex_ai_rag_tool",
  "CompatibleVertexAiRagRetrieval",
]
