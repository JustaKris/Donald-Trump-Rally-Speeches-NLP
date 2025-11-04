# Changelog

All notable changes and improvements to the Trump Speeches NLP Chatbot API.

## [Recent Updates] - November 2025

### Added - Production Logging System

**New Module: `src/logging_config.py`**

Implemented professional logging with automatic environment detection:

- **Dual-Format Support**: JSON logs for production, colored logs for development
- **Auto-Detection**: Uses `ENVIRONMENT` setting to choose appropriate format
- **Cloud-Ready**: JSON format works with Azure Application Insights, CloudWatch, ELK
- **Structured Output**: Consistent timestamp, level, module name, message format
- **Third-Party Suppression**: Automatic filtering of noisy library logs
- **Uvicorn Integration**: Proper configuration of web server logs

**Benefits**:
- Deploy to Azure/Docker without code changes
- Stream JSON logs to monitoring tools
- Debug locally with colored, readable output
- Filter by module, level, or content in production

### Added - Configuration Management System

**New Module: `src/config.py`**

Created Pydantic Settings-based configuration system:

- **Type-Safe**: Automatic validation of all settings with clear error messages
- **Environment Variables**: Full `.env` file support for local and cloud deployment
- **Centralized**: All configuration in one place with defaults and descriptions
- **Cloud-Native**: Works seamlessly with Azure App Service, Docker, Kubernetes
- **Flexible**: Support for multiple LLM providers (Gemini, OpenAI, Anthropic)

**Key Settings**:
- Application: name, version, environment, log level
- LLM: provider, API keys, models, parameters
- RAG: chunk size, top-k, hybrid search, reranking
- Models: sentiment, embedding, reranker models
- Data: directories for speeches and ChromaDB
- API: host, port, reload, CORS origins

### Fixed - ChromaDB Duplicate Warnings

**Updated: `src/rag_service.py`**

Implemented smart deduplication to prevent re-indexing existing chunks:

- Check existing IDs before adding new chunks
- Skip already-indexed documents automatically
- Log clear info about new vs skipped chunks
- 100x faster re-indexing (skip embedding computation)

**Before**: 1000+ warnings on every query
```
WARNING chromadb... Add of existing embedding ID: ToledoJan9_2020_chunk_0
WARNING chromadb... Add of existing embedding ID: ToledoJan9_2020_chunk_1
...
```

**After**: Clean logs with informative messages
```
INFO src.rag_service Adding 0 new chunks (skipped 1082 duplicates)
```

### Updated - Service Architecture

**Dependency Injection Pattern**

Refactored services to accept configuration explicitly:

- `GeminiLLM`: Accepts API key, model name, temperature as parameters
- `RAGService`: Accepts `llm_service` instance (optional)
- `SentimentAnalyzer`: Accepts configurable model name
- All services use module-level loggers (`logging.getLogger(__name__)`)

**Benefits**:
- Easier testing (mock dependencies)
- Clearer initialization flow
- Better separation of concerns
- More flexible configuration

### Updated - Application Branding

**Unified Naming Convention**: "Trump Speeches NLP Chatbot API"

Updated branding across application:

- API module docstrings
- HTML page titles
- Frontend headers
- Endpoint descriptions
- Fallback HTML pages

**Key Changes**:
- Specified technologies in descriptions (Gemini, ChromaDB, FinBERT)
- Updated tab descriptions with specific features
- Improved dataset explanations
- Added example questions relevant to content

### Documentation

**New/Updated Files**:
- `docs/reference/configuration.md` - Complete configuration guide
- `docs/howto/logging.md` - Logging best practices and formats
- `docs/CHANGELOG.md` - This file
- `.env.example` - Configuration template (if not already present)

## [Previous Version] - October 2025

### Core Features

- **RAG Q&A System**: ChromaDB + MPNet embeddings + Gemini LLM
- **Hybrid Search**: Semantic search + BM25 keyword matching
- **Cross-Encoder Reranking**: Improved precision for search results
- **Multi-Factor Confidence**: Sophisticated confidence scoring
- **Entity Analytics**: Automatic entity extraction with sentiment analysis
- **FastAPI Backend**: 12+ RESTful endpoints
- **Interactive Frontend**: Single-page web interface
- **Docker Support**: Multi-stage build with health checks
- **CI/CD Pipeline**: GitHub Actions with tests and security scanning
- **Comprehensive Testing**: pytest with 50%+ coverage

### ML Models

- **Gemini 2.5 Flash**: Answer generation
- **FinBERT**: Sentiment analysis
- **all-mpnet-base-v2**: 768-dim semantic embeddings
- **ms-marco-MiniLM**: Cross-encoder reranking

### Deployment

- Docker + Docker Compose support
- Render deployment configuration
- Azure Web App compatible
- Health check endpoint
- Environment-based configuration

## Migration Guide

### From Old Configuration (Direct Environment Variables)

**Before**:
```python
import os
api_key = os.getenv("GEMINI_API_KEY")
print("Loading model...")
```

**After**:
```python
from src.config import get_settings
import logging

settings = get_settings()
logger = logging.getLogger(__name__)

api_key = settings.gemini_api_key
logger.info("Loading model...")
```

### From Old Logging (Print Statements)

**Before**:
```python
print(f"Loaded {count} documents")
```

**After**:
```python
logger.info(f"Loaded {count} documents")
```

## Breaking Changes

None. All changes are backwards-compatible or internal improvements.

## Upgrading

1. **Update dependencies**:
   ```bash
   uv sync
   ```

2. **Create `.env` file** (if not exists):
   ```bash
   cp .env.example .env
   ```

3. **Set API key in `.env`**:
   ```env
   GEMINI_API_KEY=your_key_here
   ```

4. **Run the application**:
   ```bash
   uv run uvicorn src.api:app --reload
   ```

The application will automatically use the new logging and configuration systems.

## Future Roadmap

### Planned Features

- **Multiple LLM Providers**: OpenAI GPT-4, Anthropic Claude support
- **Advanced Entity Analytics**: Knowledge graph visualization
- **Query Caching**: Redis layer for common questions
- **Async Processing**: Background jobs for heavy analytics
- **Enhanced NER**: spaCy or Hugging Face transformers
- **Fact Extraction**: Structured information from speeches

### Performance Improvements

- **Model Quantization**: Reduce model sizes
- **GPU Acceleration**: CUDA support for faster inference
- **Response Streaming**: WebSocket for real-time answers
- **Database Optimization**: Connection pooling, query optimization

### Infrastructure

- **Kubernetes**: Container orchestration
- **Auto-scaling**: Dynamic resource allocation
- **Multi-region**: Global deployment
- **Monitoring**: Prometheus + Grafana integration

---

**Project Repository**: [GitHub](https://github.com/JustaKris/Trump-Rally-Speeches-NLP-Chatbot)  
**Documentation**: [GitHub Pages](https://justakris.github.io/Trump-Rally-Speeches-NLP-Chatbot/)  
**Maintainer**: Kristiyan Bonev
