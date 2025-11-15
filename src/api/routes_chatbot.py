"""
RAG (Retrieval-Augmented Generation) endpoints.

Provides AI-powered question answering, semantic search, and document indexing
using ChromaDB vector search and Google Gemini LLM.
"""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status

from ..models import RAGAnswerResponse, RAGQueryRequest, RAGSearchRequest, RAGStatsResponse
from ..services import RAGService
from .dependencies import get_rag_service, get_settings_dep

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/rag", tags=["rag"])


@router.post("/ask", response_model=RAGAnswerResponse)
async def rag_ask_question(
    query: RAGQueryRequest,
    rag_service: Optional[RAGService] = Depends(get_rag_service),
):
    """
    Ask a question about the indexed Trump rally speeches using Retrieval-Augmented Generation.

    Combines semantic search with Google Gemini AI to:
    1. Find relevant speech excerpts using vector similarity
    2. Generate comprehensive answers grounded in the retrieved context
    3. Provide confidence scores and source attribution

    Supports hybrid search (semantic + keyword) and cross-encoder re-ranking for best results.
    Returns the AI-generated answer with supporting context chunks and source documents.
    """
    if rag_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG service not initialized. Please try again later.",
        )

    try:
        result = rag_service.ask(query.question, top_k=query.top_k)
        return RAGAnswerResponse(**result)
    except Exception as e:
        logger.error(f"RAG question error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to answer question: {str(e)}",
        )


@router.post("/search")
async def rag_semantic_search(
    query: RAGSearchRequest,
    rag_service: Optional[RAGService] = Depends(get_rag_service),
):
    """
    Perform semantic search over the indexed Trump rally speeches.

    Finds speech excerpts similar to your query using vector embeddings from SentenceTransformers.
    Combines semantic similarity with optional keyword matching (BM25) for hybrid search.

    Returns the most relevant text chunks with similarity scores and source metadata.
    """
    if rag_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG service not initialized. Please try again later.",
        )

    try:
        results = rag_service.search(query.query, top_k=query.top_k)
        return {"query": query.query, "results": results}
    except Exception as e:
        logger.error(f"RAG search error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}",
        )


@router.get("/stats", response_model=RAGStatsResponse)
async def get_rag_statistics(rag_service: Optional[RAGService] = Depends(get_rag_service)):
    """
    Get statistics about the RAG knowledge base.

    Returns information about the indexed Trump rally speeches including:
    - Total number of indexed chunks
    - Unique source speeches
    - Embedding model details
    - Chunk configuration (size, overlap)
    """
    if rag_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG service not initialized. Please try again later.",
        )

    try:
        stats = rag_service.get_stats()
        return RAGStatsResponse(**stats)
    except Exception as e:
        logger.error(f"RAG stats error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get statistics: {str(e)}",
        )


@router.post("/index")
async def index_documents(
    data_dir: str = "data/Donald Trump Rally Speeches",
    rag_service: Optional[RAGService] = Depends(get_rag_service),
    settings=Depends(get_settings_dep),
):
    """
    Index or re-index documents into the RAG knowledge base.

    Loads all text files from the specified directory, chunks them using LangChain,
    generates vector embeddings, and stores them in ChromaDB for semantic search.

    This clears the existing index and rebuilds it. Use to update the knowledge base
    with new or modified speeches.
    """
    if rag_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG service not initialized. Please try again later.",
        )

    try:
        # Clear existing collection
        rag_service.clear_collection()
        # Load new documents
        docs_loaded = rag_service.load_documents(data_dir)
        stats = rag_service.get_stats()

        return {
            "status": "success",
            "documents_loaded": docs_loaded,
            "total_chunks": stats["total_chunks"],
            "sources": stats["sources"],
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        logger.error(f"RAG indexing error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Indexing failed: {str(e)}",
        )
