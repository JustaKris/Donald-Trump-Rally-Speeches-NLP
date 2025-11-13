# Trump Speeches NLP Chatbot — Production-Ready Portfolio Project

[![CI/CD Pipeline](https://github.com/JustaKris/Trump-Rally-Speeches-NLP-Chatbot/actions/workflows/ci.yml/badge.svg)](https://github.com/JustaKris/Trump-Rally-Speeches-NLP-Chatbot/actions/workflows/ci.yml)
[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://justakris.github.io/Trump-Rally-Speeches-NLP-Chatbot/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-ready NLP API showcasing natural language processing and retrieval-augmented generation capabilities. This portfolio project demonstrates expertise in LLM integration, vector databases, semantic search, REST API development, and modern AI engineering practices.

## 🎯 Project Highlights

### 🤖 Core AI Features

- **RAG (Retrieval-Augmented Generation)** — Ask questions about 300,000+ words of political speeches using semantic search + state-of-the-art LLMs (Gemini, OpenAI GPT, Anthropic Claude)
- **Intelligent Q&A System** — ChromaDB vector database + MPNet embeddings (768d) + cross-encoder reranking for accurate context retrieval
- **Multi-Factor Confidence Scoring** — Sophisticated confidence calculation considering semantic similarity, consistency, coverage, and entity mentions
- **Entity Analytics** — Automatic entity detection with sentiment analysis and contextual associations
- **AI-Powered Sentiment Analysis** — Multi-model approach using FinBERT (sentiment), RoBERTa (emotion detection), and LLM (contextual interpretation)
- **Semantic Topic Extraction** — AI-powered topic clustering with sentence embeddings, semantic grouping, and contextual summaries
- **Flexible LLM Integration** — Model-agnostic architecture supporting multiple providers with easy switching

### 🛠️ Engineering Excellence

- **Production-ready FastAPI application** with 12+ RESTful endpoints
- **Hybrid search architecture** — Combining semantic embeddings with BM25 keyword search
- **Docker containerization** ready for cloud deployment
- **Comprehensive testing** with pytest (50%+ code coverage)
- **CI/CD pipelines** with GitHub Actions
- **Clean, documented code** following industry best practices

## 🚀 Live Demo

🔗 **[Try the API](https://trump-speeches-nlp-chatbot.azurewebsites.net)**

📚 **[API Documentation](https://trump-speeches-nlp-chatbot.azurewebsites.net/docs)** - Interactive Swagger UI

📖 **[ReDoc Documentation](https://trump-speeches-nlp-chatbot.azurewebsites.net/redoc)** - Alternative API docs

> NOTE: The Azure Web App may be cold and can take a minute or two to boot after a deploy or when it hasn't been used recently. If the demo or API docs are temporarily unavailable, please wait 1–2 minutes and retry.

📘 **[Documentation Site](https://justakris.github.io/Trump-Rally-Speeches-NLP-Chatbot/)** - Complete project documentation with guides and references

🏗️ **[System Architecture](https://justakris.github.io/Trump-Rally-Speeches-NLP-Chatbot/reference/architecture/)** - Detailed technical documentation with diagrams

## 📊 Technical Implementation

### RAG (Retrieval-Augmented Generation) System

Production-ready question-answering system over 35 political speeches (300,000+ words) built with a modular, testable architecture:

**Core Services:**
- **`services/rag_service.py`** — Orchestrates RAG pipeline, manages ChromaDB, coordinates components
- **`services/llm/`** — Pluggable LLM abstraction layer supporting multiple providers (Gemini, OpenAI, Anthropic)
  - **`base.py`** — Abstract LLMProvider interface
  - **`factory.py`** — Factory pattern with lazy imports for optional providers
  - **`gemini.py`** — Google Gemini implementation
  - **`openai.py`** — OpenAI GPT models (optional dependency)
  - **`anthropic.py`** — Anthropic Claude models (optional dependency)

**Modular RAG Components** (`services/rag/`):
- **`search_engine.py`** — Hybrid search combining semantic (MPNet 768d), BM25 keyword, and cross-encoder reranking
- **`confidence.py`** — Multi-factor confidence scoring (retrieval quality, consistency, coverage, entity mentions)
- **`entity_analyzer.py`** — Entity extraction with sentiment analysis, speech coverage, and co-occurrence analytics
- **`document_loader.py`** — Document chunking (2048 chars, 150 overlap) with metadata tracking
- **`models.py`** — Pydantic data models for type-safe RAG operations

**RAG API Endpoints:**
- `POST /rag/ask` — Ask natural language questions with AI-generated answers
- `POST /rag/search` — Semantic search over indexed documents
- `GET /rag/stats` — Vector database statistics and health check
- `POST /rag/index` — Index/re-index documents

### 📝 Traditional NLP Endpoints

**API Layer** (`api/`):
- **`routes_chatbot.py`** — RAG question-answering endpoints
- **`routes_nlp.py`** — Traditional NLP analysis endpoints
- **`routes_health.py`** — Health checks and system status
- **`dependencies.py`** — Dependency injection for services

**Core Services:**
- **`services/nlp_service.py`** — Word frequency, n-gram analysis
- **`services/sentiment_service.py`** — Enhanced AI-powered sentiment analysis with emotion detection and contextual interpretation
- **`services/topic_service.py`** — AI-powered topic extraction with semantic clustering and LLM-generated summaries

**Additional Endpoints:**
- `POST /analyze/sentiment` — Multi-model sentiment analysis (FinBERT + RoBERTa emotions + Gemini interpretation)
- `POST /analyze/words` — Word frequency
- `POST /analyze/topics` — AI-powered topic extraction with semantic clustering and contextual analysis
- `POST /analyze/ngrams` — N-gram analysis

### 📊 Demo Dataset

35 political rally speech transcripts (2019-2020) totaling 300,000+ words — indexed in ChromaDB for RAG queries. The dataset demonstrates the system's ability to handle real-world political text with nuanced language.

### 🎨 Interactive Web Interface

Single-page application at the root (`/`) for testing all API features including the RAG Q&A system.

### 📓 Analysis Notebooks

Jupyter notebooks showcasing statistical NLP and exploratory data analysis techniques on the speech corpus.

## 🚀 Key Skills Demonstrated

### AI/ML Engineering

- **RAG Systems**: End-to-end retrieval-augmented generation with ChromaDB vector database
- **LLM Integration**: Multi-provider abstraction layer with Gemini, OpenAI GPT, and Anthropic Claude support
- **Design Patterns**: Factory pattern with lazy imports, abstract base classes, dependency injection
- **Semantic Search**: Hybrid search combining dense embeddings (MPNet) and sparse retrieval (BM25)
- **Model Selection**: Cross-encoder reranking for precision optimization
- **Confidence Scoring**: Multi-factor confidence calculation for answer quality assessment
- **Transformer Models**: FinBERT sentiment analysis, sentence-transformers for embeddings
- **Entity Analytics**: NER-based entity extraction with sentiment and co-occurrence analysis

### Backend Engineering

- **API Development**: Production-grade FastAPI with 12+ RESTful endpoints, async request handling, modular route organization
- **Vector Databases**: ChromaDB with persistent storage, smart deduplication, efficient querying
- **Modular Architecture**: Separated concerns with dedicated services for search, confidence, entities, and document loading
- **Configuration Management**: Pydantic Settings with type validation and environment-based config
- **Production Logging**: JSON-formatted logs for cloud platforms, colored output for development
- **Error Handling**: Graceful fallbacks, comprehensive exception handling, structured error responses
- **Performance**: Efficient chunking (2048 chars), hybrid search, cross-encoder reranking
- **Type Safety**: Full Pydantic validation, Python 3.11+ type hints throughout
- **Dependency Injection**: Clean service initialization and testable architecture

### DevOps & Quality

- **Containerization**: Multi-stage Docker builds, non-root user, health checks
- **CI/CD**: GitHub Actions with automated testing, security scanning, code quality gates
- **Testing**: pytest with 65%+ coverage, unit and integration tests, parametrized test cases, modular component testing
- **Code Quality**: Black, flake8, isort enforced via CI (mypy for select modules)
- **Security**: pip-audit for vulnerability scanning, bandit for security analysis
- **Documentation**: Comprehensive MkDocs site, API docs via Swagger/ReDoc, inline docstrings
- **Observability**: Structured logging, health endpoints, startup configuration display

## � Example RAG Queries

Try asking the system natural language questions like:

- *"What economic policies were discussed in the speeches?"*
- *"How many times was Biden mentioned and in what context?"*
- *"What did the speaker say about immigration?"*
- *"Compare the themes between 2019 and 2020 speeches"*

The system retrieves relevant context, analyzes entities, calculates confidence scores, and generates coherent answers with source attribution.

### 🎯 Recent Improvements (November 2025)

### LLM Provider Abstraction
- **Multi-Provider Support**: Pluggable architecture supporting Gemini, OpenAI GPT, and Anthropic Claude
- **Model-Agnostic Configuration**: Single configuration interface for all providers (`LLM_API_KEY`, `LLM_MODEL_NAME`)
- **Factory Pattern**: Lazy imports with optional dependencies for clean provider switching
- **Type-Safe Interface**: Abstract base class ensuring consistent LLM behavior across providers
- **Easy Extension**: Add new providers by implementing the `LLMProvider` interface

### Enhanced AI-Powered NLP Features
- **Multi-Model Sentiment Analysis**:
  - FinBERT for sentiment classification (positive/negative/neutral)
  - RoBERTa emotion detection (anger, joy, fear, sadness, surprise, disgust)
  - Gemini LLM for contextual interpretation explaining WHY the models produced their results
  - Clean UI with AI interpretation as the focal point, compact score visualization
- **Semantic Topic Extraction**:
  - Sentence-transformer embeddings for semantic similarity
  - DBSCAN clustering for intelligent topic grouping
  - LLM-generated cluster labels and comprehensive topic summaries
  - Interactive snippets showing topic occurrences in context
- **Centralized Configuration**: All NLP parameters (thresholds, model names, excluded verbs) configurable via environment variables

### Modular RAG Architecture (Code Refactoring)
- **Component Separation**: Extracted RAG functionality into dedicated, testable modules
  - `SearchEngine`: Hybrid search with semantic, BM25, and cross-encoder reranking
  - `ConfidenceCalculator`: Multi-factor confidence scoring
  - `EntityAnalyzer`: Entity extraction, sentiment, and co-occurrence analysis
  - `DocumentLoader`: Smart chunking with metadata tracking
- **Improved Testability**: 65%+ test coverage with component-level unit tests
- **Type Safety**: Pydantic models for all RAG data structures
- **Maintainability**: Clear separation of concerns, easier to extend and debug

### Production-Ready Logging
- **Dual-format logging**: JSON for production/cloud, colored for development
- **Automatic environment detection**: Zero configuration needed
- **Cloud-native**: Works with Azure Application Insights, CloudWatch, ELK stack
- **Smart filtering**: Suppresses noisy third-party library logs

### Professional Configuration Management  
- **Type-safe settings**: Pydantic validation catches errors at startup
- **Environment-based**: Full `.env` file support for local and cloud deployment
- **Flexible**: Support for multiple LLM providers (Gemini, OpenAI, Anthropic)
- **Cloud-ready**: Seamless Azure/AWS deployment with environment variables

### Performance & Reliability
- **Smart deduplication**: Prevents re-indexing existing ChromaDB chunks
- **100x faster re-indexing**: Skip embedding computation for existing documents
- **Clean logs**: No more duplicate ID warnings or telemetry errors
- **Dependency injection**: Better testability and cleaner initialization

[See full changelog](docs/CHANGELOG.md)

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- uv ([install guide](https://docs.astral.sh/uv/getting-started/installation/))
- LLM API key from one of the supported providers:
  - **Google Gemini** ([get one free](https://ai.google.dev/)) — Default provider
  - **OpenAI** ([get API key](https://platform.openai.com/api-keys)) — Optional
  - **Anthropic** ([get API key](https://console.anthropic.com/)) — Optional

### Setup

1. **Install dependencies**

   ```powershell
   uv sync
   ```

2. **Configure LLM Provider**

   The project supports multiple LLM providers with a model-agnostic configuration approach.

   **Option A: Google Gemini (Default)**

   Create a `.env` file in the project root:
   ```bash
   # LLM Provider Configuration
   LLM_PROVIDER=gemini
   LLM_API_KEY=your_gemini_api_key_here
   LLM_MODEL_NAME=gemini-2.0-flash-exp
   
   # Optional: Adjust LLM parameters
   LLM_TEMPERATURE=0.7
   LLM_MAX_OUTPUT_TOKENS=2048
   ```

   **Option B: OpenAI**

   ```powershell
   # Install OpenAI support
   uv sync --group llm-openai
   ```

   Update `.env`:
   ```bash
   LLM_PROVIDER=openai
   LLM_API_KEY=sk-your_openai_api_key_here
   LLM_MODEL_NAME=gpt-4o-mini
   LLM_TEMPERATURE=0.7
   LLM_MAX_OUTPUT_TOKENS=2048
   ```

   **Option C: Anthropic (Claude)**

   ```powershell
   # Install Anthropic support
   uv sync --group llm-anthropic
   ```

   Update `.env`:
   ```bash
   LLM_PROVIDER=anthropic
   LLM_API_KEY=sk-ant-your_anthropic_api_key_here
   LLM_MODEL_NAME=claude-3-5-sonnet-20241022
   LLM_TEMPERATURE=0.7
   LLM_MAX_OUTPUT_TOKENS=2048
   ```

   **Install All Providers:**
   ```powershell
   uv sync --group llm-all
   ```

3. **Start the FastAPI server**

   ```powershell
   uv run uvicorn src.api:app --reload
   ```

4. **Access the application**
   - **Web Interface:** <https://trump-speeches-nlp-chatbot.azurewebsites.net>
   - **API Docs:** <https://trump-speeches-nlp-chatbot.azurewebsites.net/docs>
   - **ReDoc:** <https://trump-speeches-nlp-chatbot.azurewebsites.net/redoc>

### Try the RAG System

**Web Interface:** Navigate to the RAG tab and ask a question

**API Example:**
```powershell
curl -X POST http://localhost:8000/rag/ask `
  -H "Content-Type: application/json" `
  -d '{"question": "What was said about the economy?", "top_k": 5}'
```

**Python Example:**
```python
import requests

response = requests.post(
    "http://localhost:8000/rag/ask",
    json={"question": "What economic policies were discussed?", "top_k": 5}
)
print(response.json()["answer"])
```

### Alternative: Docker

**Note:** Add your Gemini API key to the Dockerfile or pass it as an environment variable.

### Run with Docker

1. **Build the Docker image**

   ```powershell
   docker build -t trump-speeches-nlp-chatbot .
   ```

2. **Run the container**

   ```powershell
   docker run -d -p 8000:8000 trump-speeches-nlp-chatbot
   ```

3. **Or use Docker Compose**

   ```powershell
   docker-compose up -d
   ```

### View Documentation Site (Optional)

The project includes comprehensive documentation built with MkDocs:

```powershell
# Install documentation dependencies
uv sync --group docs

# Serve documentation site locally (with live reload)
uv run mkdocs serve
```

Then open <http://localhost:8001> to browse the documentation with search and navigation.

**Build static site:**
```powershell
uv run mkdocs build
```

This generates a `site/` folder with the complete static documentation website.

### Explore Analysis Notebooks (Optional)

```powershell
# Install notebook dependencies (includes matplotlib, seaborn, plotly, etc.)
uv sync --group notebooks
uv run jupyter lab
```

Navigate to `notebooks/` to explore statistical NLP analysis and visualizations.

## 🧪 Testing & Code Quality

This project includes comprehensive testing and code quality tools to demonstrate professional software engineering practices.

### Run Tests

```powershell
# Install dev dependencies
uv sync --group dev

# Run all tests with coverage
uv run pytest

# Run only unit tests
uv run pytest -m unit

# Run only integration tests
uv run pytest -m integration

# Generate HTML coverage report
uv run pytest --cov=src --cov-report=html
```

### Code Quality Checks

```powershell
# Format code
uv run black src/

# Lint code
uv run flake8 src/

# Sort imports
uv run isort src/

# Type checking
uv run mypy src/

# Run all checks
uv run black src/ && uv run isort src/ && uv run flake8 src/ && uv run pytest
```

### CI/CD Pipeline

The project uses GitHub Actions for continuous integration:
- ✅ **Automated testing** on Python 3.11, 3.12, 3.13
- ✅ **Code quality checks** (black, flake8, isort, mypy)
- ✅ **Security scanning** (safety, bandit)
- ✅ **Docker image build** and health checks

See [`.github/workflows/ci.yml`](.github/workflows/ci.yml) for the full pipeline configuration.

For detailed testing documentation, see [`docs/howto/testing.md`](docs/howto/testing.md).

## 📦 Dependencies

## 📦 Core Dependencies

**RAG & LLM:**
- `chromadb` — Vector database for embeddings
- `google-generativeai` — Gemini LLM integration (default provider)
- `openai` — OpenAI GPT models (optional: `uv sync --group llm-openai`)
- `anthropic` — Anthropic Claude models (optional: `uv sync --group llm-anthropic`)
- `sentence-transformers` — MPNet embeddings (768d)
- `rank-bm25` — Keyword search for hybrid retrieval
- `langchain` — Text splitting utilities

**NLP & ML:**
- `transformers` + `torch` — FinBERT sentiment analysis, RoBERTa emotion detection
- `nltk` — Text preprocessing
- `scikit-learn` — DBSCAN clustering, cosine similarity

**API & Infrastructure:**
- `fastapi` — REST API framework
- `uvicorn` — ASGI server
- `pydantic` — Data validation

See `pyproject.toml` for complete dependency list.

## 💡 Project Structure

```
Trump-Rally-Speeches-NLP-Chatbot/
│
├── src/                          # Production API code
│   ├── api.py                   # FastAPI with RAG & NLP endpoints
│   ├── services/
│   │   ├── rag_service.py       # ⭐ RAG orchestration
│   │   ├── llm/                 # ⭐ Pluggable LLM providers
│   │   │   ├── base.py          #    Abstract LLMProvider interface
│   │   │   ├── factory.py       #    Factory pattern with lazy imports
│   │   │   ├── gemini.py        #    Google Gemini implementation
│   │   │   ├── openai.py        #    OpenAI GPT (optional)
│   │   │   └── anthropic.py     #    Anthropic Claude (optional)
│   │   ├── rag/                 # Modular RAG components
│   │   │   ├── search_engine.py #    Hybrid search
│   │   │   ├── confidence.py    #    Confidence scoring
│   │   │   ├── entity_analyzer.py #  Entity extraction
│   │   │   └── document_loader.py #  Document chunking
│   │   ├── sentiment_service.py # Multi-model sentiment
│   │   └── topic_service.py     # Semantic topic extraction
│   ├── models.py                # FinBERT sentiment analysis
│   ├── preprocessing.py         # Text preprocessing
│   └── utils.py                 # Data loading utilities
│
├── data/
│   ├── Donald Trump Rally Speeches/  # 35 speech transcripts
│   └── chromadb/                     # Vector database persistence
│
├── static/
│   └── index.html               # Web interface
│
├── notebooks/                   # Exploratory analysis
├── tests/                       # pytest test suite
├── docs/                        # Documentation (MkDocs site)
│   ├── index.md                # Docs homepage
│   ├── guides/                 # Getting started guides
│   ├── howto/                  # Task-oriented guides
│   └── reference/              # Technical reference
├── mkdocs.yml                   # Documentation site config
└── pyproject.toml               # Dependencies
```

## � Documentation

**📘 [Full Documentation Site](https://justakris.github.io/Trump-Rally-Speeches-NLP-Chatbot/)** — Complete guides, tutorials, and API reference

### View Documentation Locally (Optional)

```powershell
# Install docs dependencies
uv sync --group docs

# Serve docs with live reload (use port 8001 to avoid API conflict)
uv run mkdocs serve --dev-addr localhost:8001
```

Then open <http://localhost:8001> in your browser.

For more information on working with the documentation, see the [Documentation Guide](https://justakris.github.io/Trump-Rally-Speeches-NLP-Chatbot/howto/documentation/).

## �📄 License & Attribution

This repository is for educational and portfolio purposes. The speech transcripts are publicly available data used for demonstrative NLP analysis.

**Technologies Used:**

- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [FinBERT](https://huggingface.co/ProsusAI/finbert) for sentiment analysis
- [Plotly](https://plotly.com/python/) for interactive visualizations
- [uv](https://docs.astral.sh/uv/) for dependency management

---

## 📫 Contact

**Kristiyan Bonev** | [GitHub](https://github.com/JustaKris)

*This project showcases practical NLP skills and modern data science workflows. Feel free to explore the notebooks and reach out with questions!*
