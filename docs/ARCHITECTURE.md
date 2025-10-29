# System Architecture

This document provides a comprehensive overview of the NLP Text Analysis API architecture, including system components, data flows, and deployment strategies.

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
        Topics[Topic Extraction<br/>TF-IDF]
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

### 1. **API Layer** (`src/api.py`)

FastAPI application serving as the main entry point.

**Responsibilities:**
- HTTP request handling
- Input validation (Pydantic models)
- Error handling and logging
- CORS middleware
- Static file serving
- Service orchestration

**Endpoints:**
- `/analyze/sentiment` - Sentiment analysis
- `/analyze/words` - Word frequency analysis
- `/analyze/topics` - Topic extraction
- `/analyze/ngrams` - N-gram analysis
- `/text/clean` - Text preprocessing
- `/rag/ask` - RAG question answering
- `/rag/search` - Semantic search
- `/rag/stats` - Collection statistics
- `/rag/index` - Document indexing
- `/speeches/stats` - Dataset statistics
- `/speeches/list` - List all speeches
- `/health` - Health check

### 2. **Sentiment Analysis** (`src/models.py`)

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

### 3. **RAG Service** (`src/rag_service.py`)

Retrieval-Augmented Generation for question answering.

**Components:**
- **Vector Store:** ChromaDB for persistent storage
- **Embeddings:** sentence-transformers (all-MiniLM-L6-v2)
- **Text Splitter:** LangChain RecursiveCharacterTextSplitter
- **Chunking:** 500 chars with 50 char overlap

**Capabilities:**
- Document loading and indexing
- Semantic search
- Context retrieval
- Answer generation (extraction-based)

### 4. **Text Preprocessing** (`src/preprocessing.py`)

Text cleaning and normalization utilities.

**Functions:**
- Stopword removal (NLTK)
- Tokenization
- Special character removal
- URL removal
- N-gram extraction

### 5. **Utilities** (`src/utils.py`)

Data loading and analysis helpers.

**Functions:**
- Speech loading from directory
- Word frequency statistics
- Topic extraction (TF-IDF)
- Dataset statistics calculation

---

## RAG Pipeline

Detailed architecture of the Retrieval-Augmented Generation system.

```mermaid
graph TB
    subgraph "Indexing Phase (Startup)"
        Docs[Text Documents<br/>*.txt files]
        Load[Document Loader]
        Split[Text Splitter<br/>500 chars, 50 overlap]
        Embed[Embedding Model<br/>all-MiniLM-L6-v2]
        Store[(ChromaDB<br/>Vector Store)]
        
        Docs --> Load
        Load --> Split
        Split -->|Text Chunks| Embed
        Embed -->|384-dim Vectors| Store
    end
    
    subgraph "Query Phase (Runtime)"
        Question[User Question]
        QEmbed[Query Embedding]
        Search[Similarity Search<br/>Cosine Distance]
        Retrieve[Top-K Retrieval]
        Generate[Answer Generation<br/>Context-based]
        Response[Answer + Context]
        
        Question --> QEmbed
        QEmbed --> Search
        Store -.->|Vector Lookup| Search
        Search --> Retrieve
        Retrieve --> Generate
        Generate --> Response
    end
```

### RAG Workflow Details

**1. Indexing (One-time or on-demand):**
```python
1. Load documents from directory
2. Split into chunks (RecursiveCharacterTextSplitter)
   - chunk_size: 500 characters
   - chunk_overlap: 50 characters
3. Generate embeddings (sentence-transformers)
   - Model: all-MiniLM-L6-v2
   - Dimension: 384
4. Store in ChromaDB with metadata:
   - source: filename
   - chunk_index: position in document
   - total_chunks: document length
```

**2. Querying:**
```python
1. Receive question from user
2. Generate question embedding (same model)
3. Perform similarity search (cosine distance)
4. Retrieve top-k most relevant chunks (default k=3)
5. Extract context from retrieved chunks
6. Generate answer (currently extraction-based)
7. Return answer + context + confidence + sources
```

**3. Confidence Scoring:**
- **High:** avg_distance < 0.5
- **Medium:** 0.5 ≤ avg_distance < 0.8
- **Low:** avg_distance ≥ 0.8

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
        Poetry[Poetry 2.2.1]
        Deps[Install Dependencies]
        Poetry --> Deps
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
3. Push to Docker Hub (`nlp-text-analysis-api:latest`)
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
| **ML Framework** | PyTorch 2.5+ | Deep learning backend |
| **NLP Library** | Transformers 4.57+ | Pre-trained models |
| **Text Processing** | NLTK 3.9+ | Tokenization, stopwords |
| **Vector DB** | ChromaDB 0.5+ | Embeddings storage |
| **Embeddings** | sentence-transformers 3.3+ | Semantic embeddings |
| **RAG Framework** | LangChain 0.3+ | Text splitting, RAG utils |

### Supporting Technologies

| Category | Technology | Version |
|----------|-----------|---------|
| **Dependency Mgmt** | Poetry | 2.2.1 |
| **Containerization** | Docker | Latest |
| **CI/CD** | GitHub Actions | - |
| **Testing** | pytest | 8.3+ |
| **Code Quality** | black, flake8, mypy, isort | Latest |
| **Security** | pip-audit, bandit | Latest |

### Model Details

| Model | Task | Source | Size |
|-------|------|--------|------|
| **FinBERT** | Sentiment Analysis | ProsusAI/finbert | ~440MB |
| **all-MiniLM-L6-v2** | Embeddings | sentence-transformers | ~80MB |

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

### 1. Advanced RAG

- **LLM Integration:** Add GPT-4/Claude for answer generation
- **Multi-modal:** Support PDFs, images, audio
- **Hybrid Search:** Combine semantic + keyword search
- **Re-ranking:** Improve retrieval accuracy

### 2. Fine-tuned Models

- Train DistilBERT on domain-specific data
- Custom sentiment classifiers
- Named entity recognition (NER)
- Text summarization

### 3. Real-time Processing

- WebSocket support for streaming
- Event-driven architecture (Kafka)
- Real-time analytics dashboard

### 4. Multi-language Support

- Translation API integration
- Multi-lingual embeddings
- Language detection

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

## References

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Transformers Documentation](https://huggingface.co/docs/transformers)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [LangChain Documentation](https://python.langchain.com/)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)

---

**Last Updated:** October 2025  
**Version:** 0.1.0  
**Maintainer:** Kristiyan Bonev
