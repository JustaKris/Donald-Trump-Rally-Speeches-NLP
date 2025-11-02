# NLP Text Analysis API — Production-Ready Portfolio Project

[![CI/CD Pipeline](https://github.com/JustaKris/Donald-Trump-Rally-Speeches-NLP/actions/workflows/ci.yml/badge.svg)](https://github.com/JustaKris/Donald-Trump-Rally-Speeches-NLP/actions/workflows/ci.yml)
[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://justakris.github.io/Donald-Trump-Rally-Speeches-NLP/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-ready NLP API showcasing natural language processing and retrieval-augmented generation capabilities. This portfolio project demonstrates expertise in LLM integration, vector databases, semantic search, REST API development, and modern AI engineering practices.

## 🎯 Project Highlights

### 🤖 Core AI Features

- **RAG (Retrieval-Augmented Generation)** — Ask questions about 300,000+ words of political speeches using semantic search + Google Gemini LLM
- **Intelligent Q&A System** — ChromaDB vector database + MPNet embeddings (768d) + cross-encoder reranking for accurate context retrieval
- **Multi-Factor Confidence Scoring** — Sophisticated confidence calculation considering semantic similarity, consistency, coverage, and entity mentions
- **Entity Analytics** — Automatic entity detection with sentiment analysis and contextual associations
- **Transformer-based Sentiment Analysis** — FinBERT model for accurate sentiment classification

### 🛠️ Engineering Excellence

- **Production-ready FastAPI application** with 12+ RESTful endpoints
- **Hybrid search architecture** — Combining semantic embeddings with BM25 keyword search
- **Docker containerization** ready for cloud deployment
- **Comprehensive testing** with pytest (50%+ code coverage)
- **CI/CD pipelines** with GitHub Actions
- **Clean, documented code** following industry best practices

## 🚀 Live Demo

🔗 **[Try the API](http://localhost:8000)** (Coming soon: deployed version)

📚 **[API Documentation](http://localhost:8000/docs)** - Interactive Swagger UI

📖 **[ReDoc Documentation](http://localhost:8000/redoc)** - Alternative API docs

📘 **[Documentation Site](https://justakris.github.io/Donald-Trump-Rally-Speeches-NLP/)** - Complete project documentation with guides and references

🏗️ **[System Architecture](https://justakris.github.io/Donald-Trump-Rally-Speeches-NLP/reference/architecture/)** - Detailed technical documentation with diagrams

## 📊 What's Inside

### 🤖 RAG System — The Star Feature

**Ask questions about 35 political speeches** using state-of-the-art retrieval and generation:

- **`rag_service.py`** — Complete RAG implementation with:
  - ChromaDB vector database for persistent embeddings
  - MPNet sentence-transformers (768-dimensional embeddings)
  - Hybrid search (semantic + BM25 keyword matching)
  - Cross-encoder reranking for precision
  - Multi-factor confidence scoring
  - Entity extraction and analytics
  
- **`llm_service.py`** — Google Gemini integration:
  - Context-aware prompt engineering
  - Entity-focused answer generation
  - Fallback extraction for robustness
  - Source attribution and citations

**RAG API Endpoints:**
- `POST /rag/ask` — Ask natural language questions with AI-generated answers
- `POST /rag/search` — Semantic search over indexed documents
- `GET /rag/stats` — Vector database statistics and health check
- `POST /rag/index` — Index/re-index documents

### 📝 Traditional NLP Endpoints

- **`api.py`** — FastAPI application with RESTful design
- **`models.py`** — FinBERT sentiment analysis
- **`preprocessing.py`** — Text cleaning and tokenization
- **`utils.py`** — Data loading and statistics

**Additional Endpoints:**
- `POST /analyze/sentiment` — Sentiment analysis
- `POST /analyze/words` — Word frequency
- `POST /analyze/topics` — Topic extraction
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
- **LLM Integration**: Google Gemini API integration with context-aware prompting
- **Semantic Search**: Hybrid search combining dense embeddings (MPNet) and sparse retrieval (BM25)
- **Model Selection**: Cross-encoder reranking for precision optimization
- **Confidence Scoring**: Multi-factor confidence calculation for answer quality assessment
- **Transformer Models**: FinBERT sentiment analysis, sentence-transformers for embeddings
- **Entity Analytics**: NER-based entity extraction with sentiment and co-occurrence analysis

### Backend Engineering

- **API Development**: Production-grade FastAPI with 12+ RESTful endpoints
- **Vector Databases**: ChromaDB with persistent storage and efficient querying
- **Error Handling**: Graceful fallbacks and comprehensive logging
- **Performance**: Efficient chunking, caching strategies, optimized retrieval
- **Type Safety**: Full Pydantic models and Python type hints

### DevOps & Quality

- **Containerization**: Docker with multi-stage builds
- **CI/CD**: GitHub Actions with automated testing and code quality checks
- **Testing**: pytest with 50%+ coverage, unit and integration tests
- **Code Quality**: Black, flake8, isort, mypy for consistent, maintainable code
- **Documentation**: API docs via Swagger/ReDoc, comprehensive README

## � Example RAG Queries

Try asking the system natural language questions like:

- *"What economic policies were discussed in the speeches?"*
- *"How many times was Biden mentioned and in what context?"*
- *"What did the speaker say about immigration?"*
- *"Compare the themes between 2019 and 2020 speeches"*

The system retrieves relevant context, analyzes entities, calculates confidence scores, and generates coherent answers with source attribution.

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- uv ([install guide](https://docs.astral.sh/uv/getting-started/installation/))
- Google Gemini API key ([get one free](https://ai.google.dev/))

### Setup

1. **Install dependencies**

   ```powershell
   uv sync
   ```

2. **Configure environment variables**

   Create a `.env` file in the project root:
   ```bash
   GEMINI_API_KEY=your_api_key_here
   ```

3. **Start the FastAPI server**

   ```powershell
   uv run uvicorn src.api:app --reload
   ```

4. **Access the application**
   - **Web Interface:** <http://localhost:8000>
   - **API Docs:** <http://localhost:8000/docs>
   - **ReDoc:** <http://localhost:8000/redoc>

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
   docker build -t trump-speeches-nlp-api .
   ```

2. **Run the container**

   ```powershell
   docker run -d -p 8000:8000 trump-speeches-nlp-api
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
- `google-generativeai` — Gemini LLM integration
- `sentence-transformers` — MPNet embeddings (768d)
- `rank-bm25` — Keyword search for hybrid retrieval
- `langchain` — Text splitting utilities

**NLP & ML:**
- `transformers` + `tensorflow` — FinBERT sentiment analysis
- `nltk` — Text preprocessing
- `scikit-learn` — ML utilities

**API & Infrastructure:**
- `fastapi` — REST API framework
- `uvicorn` — ASGI server
- `pydantic` — Data validation

See `pyproject.toml` for complete dependency list.

## 💡 Project Structure

```
Donald-Trump-Rally-Speeches-NLP/
│
├── src/                          # Production API code
│   ├── api.py                   # FastAPI with RAG & NLP endpoints
│   ├── rag_service.py           # ⭐ RAG implementation (ChromaDB + hybrid search)
│   ├── llm_service.py           # ⭐ Gemini LLM integration
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

**📘 [Full Documentation Site](https://justakris.github.io/Donald-Trump-Rally-Speeches-NLP/)** — Complete guides, tutorials, and API reference

### View Documentation Locally (Optional)

```powershell
# Install docs dependencies
uv sync --group docs

# Serve docs with live reload (use port 8001 to avoid API conflict)
uv run mkdocs serve --dev-addr localhost:8001
```

Then open <http://localhost:8001> in your browser.

For more information on working with the documentation, see the [Documentation Guide](https://justakris.github.io/Donald-Trump-Rally-Speeches-NLP/howto/documentation/).

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
