# RAG System Features

## Overview

This project implements a production-grade Retrieval-Augmented Generation (RAG) system for question-answering over a corpus of 35 political speeches (300,000+ words).

## Core Architecture

### Vector Database
- **ChromaDB** with persistent storage
- **MPNet embeddings** (768 dimensions) for semantic understanding
- **Hybrid search** combining dense embeddings with BM25 sparse retrieval
- **Cross-encoder reranking** for precision optimization

### LLM Integration
- **Google Gemini** (gemini-2.5-flash) for answer generation
- Context-aware prompt engineering
- Entity-focused generation for targeted queries
- Fallback extraction for robustness

### Advanced Features
- Multi-factor confidence scoring
- Entity extraction and analytics
- Sentiment analysis for entities
- Co-occurrence analysis
- Source attribution with citations

## Key Features

### 1. Intelligent Question Answering

Ask natural language questions and receive AI-generated answers with supporting evidence.

**Example:**
```python
response = rag.ask("What economic policies were discussed?", top_k=5)
```

**Response includes:**
- Generated answer from Gemini
- 5 supporting context chunks
- Confidence score with explanation
- Source document attribution
- Entity statistics (if applicable)

### 2. Multi-Factor Confidence Scoring

Sophisticated confidence assessment considering:
- **Retrieval Quality (40%)** — Semantic similarity of retrieved chunks
- **Consistency (25%)** — Low variance in scores = higher confidence
- **Coverage (20%)** — Number of supporting chunks
- **Entity Coverage (15%)** — For entity queries, mention frequency

**Example output:**
```json
{
  "confidence": "high",
  "confidence_score": 0.87,
  "confidence_explanation": "Overall confidence is HIGH (score: 0.87) based on excellent semantic match (similarity: 0.91), very consistent results (consistency: 0.93), 5 supporting context chunks",
  "confidence_factors": {
    "retrieval_score": 0.91,
    "consistency": 0.93,
    "chunk_coverage": 5,
    "entity_coverage": 0.84
  }
}
```

### 3. Entity Analytics

Automatic entity detection with comprehensive statistics:

- **Mention counts** across entire corpus
- **Speech coverage** — which documents mention the entity
- **Sentiment analysis** — average sentiment toward entity
- **Co-occurrence analysis** — most common associated terms

**Example output:**
```json
{
  "entity_statistics": {
    "Biden": {
      "mention_count": 524,
      "speech_count": 30,
      "corpus_percentage": 25.03,
      "speeches": ["OhioSep21_2020.txt", "BemidjiSep18_2020.txt", ...],
      "sentiment": {
        "average_score": -0.15,
        "classification": "Neutral",
        "sample_size": 50
      },
      "associations": ["people", "our", "right", "about", "say"]
    }
  }
}
```

### 4. Hybrid Search

Combines semantic and keyword search for optimal retrieval:

- **Semantic search** — Dense embeddings capture meaning and context
- **BM25 keyword search** — Ensures exact term matches aren't missed
- **Cross-encoder reranking** — Final precision optimization
- **Configurable weights** — Adjust semantic vs keyword importance

### 5. Optimized Chunking

- **2048 character chunks** (~512-768 tokens) for complete context
- **150 character overlap** to preserve continuity
- Smart splitting with RecursiveCharacterTextSplitter
- Maintains coherent context boundaries

## API Usage

### Basic Question

```python
import requests

response = requests.post(
    "http://localhost:8000/rag/ask",
    json={"question": "What was said about the economy?", "top_k": 5}
)

result = response.json()
print(result["answer"])
print(f"Confidence: {result['confidence']} ({result['confidence_score']:.2f})")
```

### Entity Query

```python
response = requests.post(
    "http://localhost:8000/rag/ask",
    json={"question": "What did Trump say about Biden?", "top_k": 10}
)

result = response.json()

# View entity statistics
if "entity_statistics" in result:
    for entity, stats in result["entity_statistics"].items():
        print(f"\n{entity}:")
        print(f"  Mentions: {stats['mention_count']}")
        print(f"  Sentiment: {stats['sentiment']['classification']}")
        print(f"  Associated: {', '.join(stats['associations'][:3])}")
```

### Semantic Search

```python
response = requests.post(
    "http://localhost:8000/rag/search",
    json={"query": "immigration policy", "top_k": 5}
)

results = response.json()["results"]
for i, result in enumerate(results, 1):
    print(f"\n{i}. Source: {result['source']}")
    print(f"   Similarity: {result['similarity']:.3f}")
    print(f"   Preview: {result['text'][:100]}...")
```

## Configuration

### RAGService Parameters

```python
from src.rag_service import RAGService

rag = RAGService(
    collection_name="speeches",
    persist_directory="./data/chromadb",
    embedding_model="all-mpnet-base-v2",      # 768d embeddings
    reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
    chunk_size=2048,                          # ~512-768 tokens
    chunk_overlap=150,                        # ~100-150 tokens
    use_llm=True,                             # Enable Gemini
    use_reranking=True,                       # Enable cross-encoder
    use_hybrid_search=True                    # Enable BM25 + semantic
)
```

### API Endpoint Configuration

- Default `top_k`: 5 chunks
- Maximum `top_k`: 15 chunks
- Increase for complex/entity queries

## Performance

### First Request
- ~30-60 seconds (model downloads + document indexing)
- Downloads ~1-2 GB of models (one-time)

### Subsequent Requests
- ~1-3 seconds for typical queries
- ~2-5 seconds for entity analytics (sentiment analysis)

### Optimization Opportunities
- Cache entity statistics
- Pre-compute embeddings
- Async sentiment analysis
- Redis for query caching

## Technical Details

### Models Used
- **Embeddings:** `sentence-transformers/all-mpnet-base-v2` (768d)
- **Reranker:** `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **LLM:** Google Gemini 2.5 Flash
- **Sentiment:** ProsusAI/finbert

### Database
- **ChromaDB 0.5.0** with SQLite persistence
- **Vector index:** HNSW for efficient similarity search
- **Metadata filtering:** Source, chunk index, timestamps

### Prompt Engineering
- Context-limited to 4000 characters max
- Source attribution in context
- Entity-focused instructions when entities detected
- Structured output format
- Safety settings for political content

## Limitations & Future Work

### Current Limitations
- Entity extraction uses simple heuristics (capitalization)
- Sentiment analysis may show neutral for complex political text
- No query caching (every request recomputes)
- Synchronous processing (no async optimization)

### Future Enhancements
- Integrate proper NER (spaCy or Hugging Face)
- Add query caching layer (Redis)
- Implement async processing
- Add temporal analysis (sentiment over time)
- Entity relationship graphs
- Fine-tune embeddings on domain data

## Migration

If you're upgrading from a previous version with different embeddings:

```powershell
poetry run python scripts/migrate_rag_embeddings.py
```

This will:
1. Clear existing ChromaDB collection
2. Reload documents with new embeddings
3. Re-index all 35 speeches (~1082 chunks)

## Testing

```powershell
# Run RAG tests
poetry run pytest tests/test_rag_service.py -v

# Run all tests
poetry run pytest -v
```

## Documentation

- **API Reference:** <http://localhost:8000/docs>
- **Quick Start:** `docs/QUICKSTART.md`
- **Deployment:** `docs/DEPLOYMENT.md`
- **Testing:** `docs/TESTING.md`

---

*This RAG system demonstrates production-ready AI engineering with vector databases, LLM integration, and sophisticated retrieval techniques.*
