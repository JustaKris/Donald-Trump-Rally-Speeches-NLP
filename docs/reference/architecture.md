# System Architecture

This document provides a comprehensive overview of the Trump Speeches NLP Chatbot API architecture, including system components, data flows, and deployment strategies.

## Table of Contents

- [High-Level Architecture](#high-level-architecture)
- [Component Architecture](#component-architecture)
- [RAG Pipeline](#rag-pipeline)
- [Data Flow](#data-flow)
- [API Architecture](#api-architecture)
- [Deployment Architecture](#deployment-architecture)
- [Technology Stack](#technology-stack)
- [Scalability Considerations](#scalability-considerations)

---

## High-Level Architecture

```mermaid
graph TB
    Client[Client/Browser]
    Frontend[Static Frontend<br/>HTML/CSS/JS]
    API[FastAPI Application<br/>REST API]
    
    subgraph "NLP Services"
        Sentiment[Sentiment Analyzer<br/>FinBERT]
        Preprocessing[Text Preprocessing<br/>NLTK]
        Topics[AI Topic Analysis<br/>Semantic Clustering + LLM]
        RAG[RAG Service<br/>ChromaDB + Embeddings]
    end
    
    subgraph "Data Layer"
        Speeches[Demo Dataset<br/>Political Speeches]
        VectorDB[(ChromaDB<br/>Vector Store)]
        Models[ML Models<br/>Transformers]
    end
    
    Client -->|HTTP Requests| Frontend
    Frontend -->|API Calls| API
    API --> Sentiment
    API --> Preprocessing
    API --> Topics
    API --> RAG
    
    Sentiment --> Models
    RAG --> VectorDB
    RAG --> Models
    Preprocessing --> Speeches
    Topics --> Speeches
```

---

## Component Architecture

### 1. **API Layer** (`src/api/`)

FastAPI application with modular route organization.

**Responsibilities:**
- HTTP request handling
- Input validation (Pydantic models)
- Error handling and logging
- CORS middleware
- Static file serving
- Dependency injection for services

**Route Modules:**
- `routes_chatbot.py` - RAG question-answering endpoints
- `routes_nlp.py` - Traditional NLP analysis endpoints
- `routes_health.py` - Health checks and system status
- `dependencies.py` - Service dependency injection

**Endpoints:**
- `/rag/ask` - RAG question answering
- `/rag/search` - Semantic search
- `/rag/stats` - Collection statistics
- `/rag/index` - Document indexing
- `/analyze/sentiment` - Sentiment analysis
- `/analyze/words` - Word frequency analysis
- `/analyze/topics` - AI-powered topic extraction with semantic clustering
- `/analyze/ngrams` - N-gram analysis
- `/health` - Health check

### 2. **Sentiment Analysis** (`src/services/sentiment_service.py`)

Transformer-based sentiment classification using FinBERT.

**Key Features:**
- Pre-trained FinBERT model (ProsusAI/finbert)
- Automatic text chunking for long documents
- Confidence scoring
- Three-class classification (positive/negative/neutral)

**Processing Flow:**
```mermaid
graph LR
    Input[Raw Text] --> Chunk[Text Chunking<br/>512 tokens max]
    Chunk --> Tokenize[Tokenization<br/>BERT Tokenizer]
    Tokenize --> Model[FinBERT Model<br/>Inference]
    Model --> Aggregate[Score Aggregation<br/>Mean Pooling]
    Aggregate --> Output[Sentiment + Confidence]
```

### 3. **RAG Service** (`src/services/rag_service.py`)

Orchestrates the RAG pipeline, coordinating modular components for intelligent question answering.

**Architecture:**
The RAG service now uses a modular design with dedicated components:
- **Orchestration:** Manages ChromaDB collection and coordinates components
- **Delegation:** Delegates to specialized services for search, confidence, entities, and loading

**Components Used:**
- `SearchEngine` (from `services/rag/search_engine.py`)
- `ConfidenceCalculator` (from `services/rag/confidence.py`)
- `EntityAnalyzer` (from `services/rag/entity_analyzer.py`)
- `DocumentLoader` (from `services/rag/document_loader.py`)
- `GeminiLLM` (from `services/llm_service.py`)

### 4. **RAG Components** (`src/services/rag/`)

Modular, testable components for RAG functionality.

#### 4.1 **SearchEngine** (`search_engine.py`)

Hybrid search engine combining multiple retrieval strategies.

**Features:**
- **Semantic Search:** MPNet embeddings (768d) with cosine similarity
- **BM25 Search:** Keyword-based sparse retrieval
- **Hybrid Search:** Configurable weighting of semantic + BM25 scores
- **Cross-encoder Reranking:** Final precision optimization
- **Deduplication:** Removes duplicate results by ID

**Search Modes:**
- `semantic` - Pure vector similarity search
- `hybrid` - Combines semantic + BM25 (default weights: 0.7/0.3)
- `reranking` - Optional cross-encoder for top results

#### 4.2 **ConfidenceCalculator** (`confidence.py`)

Multi-factor confidence scoring for RAG answers.

**Confidence Factors (weighted):**
- **Retrieval Quality (40%):** Average semantic similarity of results
- **Consistency (25%):** Score variance (low variance = high confidence)
- **Coverage (20%):** Normalized chunk count (more chunks = better coverage)
- **Entity Coverage (15%):** Percentage of results mentioning query entities

**Confidence Levels:**
- **High:** combined_score ≥ 0.7
- **Medium:** 0.4 ≤ combined_score < 0.7
- **Low:** combined_score < 0.4

**Output:**
- Confidence level (high/medium/low)
- Numeric confidence score (0-1)
- Detailed explanation
- Individual factor scores

#### 4.3 **EntityAnalyzer** (`entity_analyzer.py`)

Entity extraction and statistical analysis.

**Capabilities:**
- **Entity Extraction:** Identifies capitalized words (filtered for stopwords, question words)
- **Mention Counting:** Tracks entity mentions across corpus
- **Speech Coverage:** Identifies which documents mention each entity
- **Sentiment Analysis:** Average sentiment toward entity (optional)
- **Co-occurrence Analysis:** Most common words appearing near entity
- **Corpus Percentage:** Percentage of documents mentioning entity

**Statistics Output:**
```python
{
    "mention_count": 524,
    "speech_count": 30,
    "corpus_percentage": 85.7,
    "speeches": ["file1.txt", "file2.txt", ...],
    "sentiment": {
        "average_score": -0.15,
        "classification": "Neutral",
        "sample_size": 50
    },
    "associations": ["people", "country", "great", ...]
}
```

#### 4.4 **DocumentLoader** (`document_loader.py`)

Smart document loading and chunking.

**Features:**
- **Recursive Text Splitting:** LangChain RecursiveCharacterTextSplitter
- **Configurable Chunking:** Default 2048 chars (~512-768 tokens)
- **Overlap:** 150 char overlap for context continuity
- **Metadata Tracking:** Preserves source filename, chunk index, total chunks
- **Directory Loading:** Batch loading from directories with progress tracking

**Chunking Strategy:**
```python
chunk_size = 2048       # ~512-768 tokens (full context)
chunk_overlap = 150     # ~100-150 tokens (preserve continuity)
```

### 5. **LLM Service** (`src/services/llm_service.py`)

Google Gemini integration for answer generation.

**Features:**
- **Context-Aware Prompting:** Builds prompts with retrieved context
- **Entity-Focused Generation:** Emphasizes entity mentions when applicable
- **Fallback Extraction:** Returns context snippets if LLM fails
- **Source Attribution:** Tracks and cites source documents
- **Error Handling:** Graceful degradation with informative fallbacks

### 4. **Text Preprocessing** (`src/utils/text_preprocessing.py`)

Text cleaning and normalization utilities.

**Functions:**
- Stopword removal (NLTK)
- Tokenization
- Special character removal
- URL removal
- N-gram extraction

### 5. **Utilities** (`src/utils/`)

Data loading and analysis helpers.

**Modules:**
- `io_helpers.py` - Speech loading from directory
- `formatters.py` - Word frequency statistics
- `text_preprocessing.py` - Basic topic extraction (TF-IDF) and dataset statistics

### 6. **AI-Powered Topic Analysis** (`src/services/topic_service.py`)

Advanced topic extraction with semantic clustering and LLM-generated insights.

**Features:**
- **Semantic Clustering:** Groups related keywords using sentence embeddings (MPNet) and KMeans
- **AI-Generated Labels:** Uses Gemini LLM to create meaningful cluster labels (e.g., "Border Security" instead of just "wall")
- **Contextual Snippets:** Extracts text passages showing keywords in use with highlighting
- **Topic Summaries:** LLM-generated interpretation of main themes and patterns
- **Smart Filtering:** Excludes common verbs and low-relevance clusters (< 50% avg relevance)

**Processing Pipeline:**
```mermaid
graph LR
    Text[Input Text] --> Extract[Extract Keywords<br/>TF-IDF + Filtering]
    Extract --> Embed[Generate Embeddings<br/>MPNet]
    Embed --> Cluster[Semantic Clustering<br/>KMeans]
    Cluster --> Label[Generate Labels<br/>Gemini LLM]
    Label --> Snippets[Extract Snippets<br/>Context Windows]
    Snippets --> Summary[Generate Summary<br/>Gemini LLM]
    Summary --> Output[Clustered Topics<br/>+ Snippets + Summary]
```

**Key Advantages:**
- Groups synonyms and related concepts automatically (e.g., "economy", "jobs", "employment" → "Economic Policy")
- Provides real-world context with highlighted examples
- Ranks by semantic relevance, not just frequency
- Offers human-readable interpretation via AI
- Filters out generic verbs and weak clusters

---

## RAG Pipeline

Modular architecture for Retrieval-Augmented Generation.

```mermaid
graph TB
    subgraph "RAG Service (Orchestrator)"
        Orchestrator[RAGService<br/>Collection Management]
    end
    
    subgraph "Indexing Components"
        Loader[DocumentLoader<br/>Chunking & Metadata]
        Embedder[Embedding Model<br/>all-mpnet-base-v2]
        DB[(ChromaDB<br/>Vector Store)]
    end
    
    subgraph "Query Components"
        Search[SearchEngine<br/>Hybrid Retrieval]
        Entities[EntityAnalyzer<br/>Extraction & Stats]
        Confidence[ConfidenceCalculator<br/>Multi-factor Scoring]
        LLM[GeminiLLM<br/>Answer Generation]
    end
    
    Orchestrator --> Loader
    Loader --> Embedder
    Embedder --> DB
    
    Orchestrator --> Search
    Search --> DB
    Orchestrator --> Entities
    Entities --> DB
    Orchestrator --> Confidence
    Orchestrator --> LLM
```

### RAG Workflow Details

**1. Indexing (One-time or on-demand):**

```python
# DocumentLoader handles chunking
1. Load documents from directory
2. Split into chunks (RecursiveCharacterTextSplitter)
   - chunk_size: 2048 characters (~512-768 tokens)
   - chunk_overlap: 150 characters (~100-150 tokens)
3. Generate embeddings via ChromaDB
   - Model: all-mpnet-base-v2 (sentence-transformers)
   - Dimension: 768
4. Store in ChromaDB with metadata:
   - source: filename
   - chunk_index: position in document
   - total_chunks: document length
```

**2. Querying:**

```python
# Orchestrated by RAGService, delegated to components
1. Receive question from user
2. EntityAnalyzer extracts entities from question
3. SearchEngine performs hybrid retrieval:
   a. Semantic search: cosine similarity on embeddings
   b. BM25 search: keyword matching
   c. Combine results with weights (0.7 semantic, 0.3 BM25)
   d. Optional cross-encoder reranking
4. ConfidenceCalculator computes multi-factor score:
   - Retrieval quality (40%): average semantic similarity
   - Consistency (25%): low score variance
   - Coverage (20%): normalized chunk count
   - Entity coverage (15%): % chunks mentioning entities
5. EntityAnalyzer generates statistics (if entities found):
   - Mention counts across corpus
   - Speech coverage percentage
   - Sentiment analysis (optional)
   - Co-occurrence analysis
6. GeminiLLM generates answer:
   - Build context-aware prompt
   - Include entity focus if applicable
   - Fallback to context extraction if LLM fails
7. Return complete response:
   - Generated answer
   - Confidence level + score + explanation
   - Supporting context chunks
   - Source attribution
   - Entity statistics (if applicable)
```

**3. Confidence Scoring:**

Multi-factor calculation combining:
- **Retrieval Quality (40%):** Average semantic similarity (0-1)
- **Consistency (25%):** Score variance (low variance = high confidence)
- **Coverage (20%):** Number of supporting chunks (normalized)
- **Entity Coverage (15%):** % of chunks mentioning query entities

**Confidence Levels:**
- **High:** combined_score ≥ 0.7
- **Medium:** 0.4 ≤ combined_score < 0.7
- **Low:** combined_score < 0.4

---

## Data Flow

### Sentiment Analysis Flow

```mermaid
sequenceDiagram
    participant User
    participant Frontend
    participant API
    participant SentimentAnalyzer
    participant FinBERT
    
    User->>Frontend: Enter text
    Frontend->>API: POST /analyze/sentiment
    API->>SentimentAnalyzer: analyze_sentiment(text)
    SentimentAnalyzer->>SentimentAnalyzer: Chunk text (512 tokens)
    loop For each chunk
        SentimentAnalyzer->>FinBERT: Classify chunk
        FinBERT-->>SentimentAnalyzer: Scores
    end
    SentimentAnalyzer->>SentimentAnalyzer: Aggregate scores
    SentimentAnalyzer-->>API: Sentiment + Confidence
    API-->>Frontend: JSON Response
    Frontend-->>User: Display results
```

### RAG Question Answering Flow

```mermaid
sequenceDiagram
    participant User
    participant Frontend
    participant API
    participant RAGService
    participant ChromaDB
    participant Embeddings
    
    User->>Frontend: Ask question
    Frontend->>API: POST /rag/ask
    API->>RAGService: ask(question)
    RAGService->>Embeddings: Encode question
    Embeddings-->>RAGService: Query vector
    RAGService->>ChromaDB: Search similar vectors
    ChromaDB-->>RAGService: Top-k chunks
    RAGService->>RAGService: Generate answer
    RAGService-->>API: Answer + Context + Sources
    API-->>Frontend: JSON Response
    Frontend-->>User: Display answer & context
```

---

## API Architecture

### Request/Response Models (Pydantic)

```python
# Input Models
TextInput
NGramRequest
RAGQueryRequest
RAGSearchRequest

# Response Models
SentimentResponse
WordFrequencyResponse
TopicResponse
StatsResponse
RAGAnswerResponse
RAGStatsResponse
```

### Middleware Stack

```
User Request
    ↓
CORS Middleware (allow all origins in dev)
    ↓
FastAPI Routing
    ↓
Pydantic Validation
    ↓
Endpoint Handler
    ↓
Business Logic (Services)
    ↓
Response Serialization
    ↓
HTTP Response
```

### Error Handling Strategy

```python
try:
    # Business logic
except SpecificError:
    # Handle known errors
    raise HTTPException(status_code=4xx)
except Exception as e:
    # Log unexpected errors
    logger.error(f"Error: {e}")
    raise HTTPException(status_code=500)
```

---

## Deployment Architecture

### Docker Multi-Stage Build

```mermaid
graph TB
    subgraph "Stage 1: Builder"
        UV[uv Package Manager]
        Deps[Install Dependencies]
        UV --> Deps
    end
    
    subgraph "Stage 2: Runtime"
        Slim[Python 3.12-slim]
        Copy[Copy Dependencies]
        App[Copy Application Code]
        Models[Download Models<br/>NLTK + Transformers]
        
        Slim --> Copy
        Copy --> App
        App --> Models
    end
    
    Deps -.->|Python packages| Copy
```

### Deployment Options

#### Option 1: Render (via Docker Hub)

```mermaid
graph LR
    GH[GitHub Actions] -->|Build & Push| DH[Docker Hub]
    DH -->|Auto-deploy| Render[Render Platform]
    Render -->|Serve| Users[End Users]
```

**Flow:**
1. Push to `main` branch
2. GitHub Actions builds Docker image
3. Push to Docker Hub (`trump-speeches-nlp-chatbot:latest`)
4. Render detects new image
5. Render pulls and deploys
6. Health check `/health`

#### Option 2: Azure Web App

**Via ACR:**
```mermaid
graph LR
    GH[GitHub Actions] -->|Build & Push| ACR[Azure Container<br/>Registry]
    ACR -->|Deploy| Azure[Azure Web App]
    Azure -->|Serve| Users[End Users]
```

**Via Docker Hub:**
```mermaid
graph LR
    GH[GitHub Actions] -->|Build & Push| DH[Docker Hub]
    DH -->|Deploy| Azure[Azure Web App]
    Azure -->|Serve| Users[End Users]
```

### CI/CD Pipeline

```mermaid
graph TB
    Push[Push to main] --> CI[CI Workflow]
    Push --> Security[Security Scan]
    
    CI --> Tests[Unit Tests<br/>Pytest]
    CI --> Lint[Code Quality<br/>flake8, black, mypy]
    
    Security --> PipAudit[pip-audit<br/>Dependency Check]
    Security --> Bandit[bandit<br/>Security Analysis]
    
    Tests --> Build[Build Docker Image]
    Lint --> Build
    PipAudit --> Build
    Bandit --> Build
    
    Build --> DHPush[Push to Docker Hub]
    Build --> ACRPush[Push to ACR]
    
    DHPush --> RenderDeploy[Deploy to Render]
    ACRPush --> AzureDeploy[Deploy to Azure]
```

---

## Technology Stack

### Core Technologies

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **API Framework** | FastAPI 0.116+ | High-performance async API |
| **Web Server** | Uvicorn | ASGI server |
| **LLM Integration** | Google Gemini 2.5 Flash | Answer generation |
| **ML Framework** | PyTorch 2.5+ | Deep learning backend |
| **NLP Library** | Transformers 4.57+ | Pre-trained models |
| **Text Processing** | NLTK 3.9+ | Tokenization, stopwords |
| **Vector DB** | ChromaDB 0.5+ | Persistent embeddings storage |
| **Embeddings** | sentence-transformers 3.3+ | Semantic embeddings (MPNet) |
| **Reranking** | Cross-encoder | Precision optimization |
| **Keyword Search** | rank-bm25 | Sparse retrieval |
| **RAG Framework** | LangChain 0.3+ | Text splitting utilities |

### Supporting Technologies

| Category | Technology | Version |
|----------|-----------|---------|
| **Dependency Mgmt** | uv | 0.9+ |
| **Containerization** | Docker | Latest |
| **CI/CD** | GitHub Actions | - |
| **Testing** | pytest | 8.3+ |
| **Code Quality** | black, flake8, isort | Latest |
| **Security** | pip-audit, bandit | Latest |

**Testing Strategy:**
- **Unit Tests:** Component-level testing for SearchEngine, ConfidenceCalculator, EntityAnalyzer, DocumentLoader
- **Integration Tests:** Full RAG pipeline testing
- **Coverage:** 65%+ overall, 90%+ for core RAG components
- **Fixtures:** Modular pytest fixtures for isolated component testing

### Model Details

| Model | Task | Source | Size |
|-------|------|--------|------|
| **Gemini 2.5 Flash** | Answer Generation | Google AI | API-based |
| **FinBERT** | Sentiment Analysis | ProsusAI/finbert | ~440MB |
| **all-mpnet-base-v2** | Embeddings (768d) | sentence-transformers | ~420MB |
| **ms-marco-MiniLM** | Reranking | cross-encoder | ~80MB |

---

## Scalability Considerations

### Current Architecture

- **Compute:** Single-instance deployment
- **Storage:** Local filesystem + ChromaDB
- **Concurrency:** Async FastAPI (handles concurrent requests)

### Scaling Strategies

#### 1. Horizontal Scaling

```mermaid
graph TB
    LB[Load Balancer]
    API1[API Instance 1]
    API2[API Instance 2]
    API3[API Instance 3]
    SharedDB[(Shared ChromaDB<br/>Postgres pgvector)]
    
    LB --> API1
    LB --> API2
    LB --> API3
    
    API1 --> SharedDB
    API2 --> SharedDB
    API3 --> SharedDB
```

**Required Changes:**
- Replace ChromaDB with pgvector (Postgres) or Pinecone
- Use shared model storage (S3/Azure Blob)
- Add Redis for caching

#### 2. Vertical Scaling

**Current Requirements:**
- RAM: ~2GB (models + API)
- CPU: 1-2 cores
- Storage: ~1GB (models + data)

**Optimized for:**
- RAM: 4-8GB for concurrent requests
- CPU: 4+ cores for parallel processing
- Storage: 5GB+ for larger datasets

#### 3. Performance Optimizations

**Already Implemented:**
- Multi-stage Docker builds
- Model pre-loading on startup
- Async request handling
- Efficient text chunking

**Future Improvements:**
- Model quantization (reduce size)
- GPU acceleration (CUDA support)
- Response caching (Redis)
- CDN for static files
- Database connection pooling
- Background task queues (Celery)

### Resource Usage

| Component | RAM | CPU | Storage |
|-----------|-----|-----|---------|
| FastAPI | ~100MB | Low | - |
| FinBERT | ~1GB | Medium | 440MB |
| all-MiniLM-L6-v2 | ~200MB | Low | 80MB |
| ChromaDB | ~100MB | Low | Variable |
| NLTK Data | ~50MB | Low | 50MB |
| **Total** | **~2GB** | **1-2 cores** | **~1GB** |

---

## Security Architecture

### Current Security Measures

1. **Dependency Scanning:** pip-audit (weekly)
2. **Code Analysis:** bandit
3. **Input Validation:** Pydantic models
4. **Non-root Container:** User `appuser` (UID 1000)
5. **Health Checks:** `/health` endpoint

### Production Recommendations

1. **Authentication:** Add API key validation
2. **Rate Limiting:** Implement per-IP limits
3. **HTTPS:** Use reverse proxy (Nginx)
4. **CORS:** Restrict to specific origins
5. **Secrets Management:** Use environment variables
6. **Logging:** Centralized logging (ELK stack)
7. **Monitoring:** Prometheus + Grafana

---

## Monitoring & Observability

### Recommended Metrics

**Application Metrics:**
- Request count (by endpoint)
- Response time (p50, p95, p99)
- Error rate (4xx, 5xx)
- Model inference time

**System Metrics:**
- CPU usage
- Memory usage
- Disk I/O
- Network I/O

**Business Metrics:**
- Total analyses performed
- Most used endpoints
- Average sentiment scores
- RAG query accuracy

### Implementation Example (Prometheus)

```python
from prometheus_client import Counter, Histogram

request_count = Counter('api_requests_total', 'Total requests', ['endpoint'])
request_duration = Histogram('api_request_duration_seconds', 'Request duration')
```

---

## Future Architecture Enhancements

### Recently Completed (November 2025)

- ✅ **Modular RAG Architecture:** Separated RAG functionality into dedicated, testable components
- ✅ **Component Testing:** Achieved 65%+ test coverage with component-level unit tests
- ✅ **Type Safety:** Pydantic models for all RAG data structures
- ✅ **Production Logging:** Dual-format logging (JSON for cloud, colored for development)

### 1. Advanced RAG Features

- **Query Caching:** Redis layer for common questions
- **Multi-modal:** Support PDFs, images, audio transcripts
- **Temporal Analysis:** Sentiment trends over time
- **Entity Relationships:** Knowledge graph visualization
- **Fine-tuned Embeddings:** Domain-specific embedding models

### 2. Performance Optimizations

- **Async Processing:** Background tasks for entity analytics
- **GPU Acceleration:** CUDA support for faster inference
- **Model Quantization:** Reduce model sizes
- **Response Streaming:** WebSocket support for real-time answers

### 3. Enhanced NLP

- **Proper NER:** spaCy or Hugging Face transformers for entity extraction
- **Text Summarization:** Automatic speech summarization
- **Topic Modeling:** LDA or BERTopic for theme discovery
- **Fact Extraction:** Structured information extraction

### 4. Deployment & Scale

- **Kubernetes:** Container orchestration
- **Auto-scaling:** Based on request volume
- **Multi-region:** Global deployment
- **CDN:** Static asset delivery

---

## Development Workflow

```mermaid
graph LR
    Dev[Local Development] -->|Test| Test[pytest]
    Test -->|Lint| Lint[black, flake8, mypy]
    Lint -->|Commit| Git[Git Push]
    Git -->|Trigger| CI[GitHub Actions]
    CI -->|Build| Docker[Docker Build]
    Docker -->|Deploy| Env[Render/Azure]
```

---

## Testing Strategy

### Component-Level Testing

Each RAG component has dedicated unit tests ensuring isolation and reliability:

**Test Files:**
- `tests/test_search_engine.py` - SearchEngine component tests (18 tests)
- `tests/test_confidence.py` - ConfidenceCalculator tests (11 tests)  
- `tests/test_entity_analyzer.py` - EntityAnalyzer tests (20 tests)
- `tests/test_document_loader.py` - DocumentLoader tests (11 tests)
- `tests/test_rag_integration.py` - Full RAG pipeline integration tests (28 tests)

**Coverage:**
- Overall: 65%+
- Core RAG components: 90%+
- SearchEngine: 94%
- ConfidenceCalculator: 93%
- DocumentLoader: 93%
- EntityAnalyzer: 73%

**Testing Approach:**
- **Unit Tests:** Isolated component testing with mocked dependencies
- **Integration Tests:** Full pipeline testing with real ChromaDB
- **Fixtures:** Reusable pytest fixtures for component setup
- **Parametrized Tests:** Testing multiple scenarios efficiently
- **Edge Cases:** Empty collections, invalid inputs, boundary conditions

### Continuous Integration

GitHub Actions workflow runs on every push:
- Python 3.11, 3.12, 3.13 matrix testing
- Unit tests with coverage reporting
- Integration tests (excluding model loading)
- Linting (flake8, black, isort)
- Type checking (mypy for select modules)
- Security scanning (bandit, pip-audit)

---

## References

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Transformers Documentation](https://huggingface.co/docs/transformers)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [LangChain Documentation](https://python.langchain.com/)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [pytest Documentation](https://docs.pytest.org/)

---

**Last Updated:** November 2025  
**Version:** 0.2.0  
**Maintainer:** Kristiyan Bonev
