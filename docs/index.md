# Trump Speeches NLP Chatbot API — Technical Documentation

Production-ready AI/ML platform demonstrating enterprise-grade practices in natural language processing, retrieval-augmented generation, and modern backend development.

## Project Overview

This portfolio project demonstrates expertise in:

- **Advanced RAG Architecture** — Modular design with dedicated components for search, confidence scoring, entity analytics, and document loading
- **Hybrid Retrieval Systems** — Semantic search (MPNet 768d) combined with BM25 keyword matching and cross-encoder reranking
- **Production API Development** — FastAPI with 12+ RESTful endpoints, modular route organization, type-safe Pydantic models
- **Entity Analytics** — Automated entity extraction with sentiment analysis and contextual associations
- **Professional DevOps** — Docker containerization, CI/CD pipelines, automated testing (65%+ coverage), code quality enforcement
- **LLM Integration** — Google Gemini for context-aware answer generation with fallback extraction

## Documentation Structure

### Getting Started Guides

Quick setup and deployment:

- **[Quickstart Guide](guides/quickstart.md)** — Local setup and first API calls
- **[Deployment Guide](guides/deployment.md)** — Production deployment to Render, Azure, or Docker

### How-To Guides

Implementation details for specific features:

- **[Testing & CI/CD](howto/testing.md)** — Testing strategy, code quality tools, continuous integration
- **[Logging](howto/logging.md)** — Production logging with JSON formatting and cloud integration
- **[Configuration](reference/configuration.md)** — Environment-based configuration management
- **[Entity Analytics](howto/entity-analytics.md)** — Entity extraction and sentiment analysis
- **[Documentation](howto/documentation.md)** — Contributing to project documentation

### Technical Reference

In-depth technical documentation:

- **[System Architecture](reference/architecture.md)** — Component design, data flows, deployment patterns
- **[RAG Features](reference/rag-features.md)** — RAG implementation details and optimization
- **[Configuration Reference](reference/configuration.md)** — Complete configuration options
- **[Changelog](CHANGELOG.md)** — Version history and recent improvements

## Quick Links

- **[GitHub Repository](https://github.com/JustaKris/Trump-Rally-Speeches-NLP-Chatbot)** — Source code and issue tracking
- **[API Documentation (Swagger)](https://trump-speeches-nlp-chatbot.azurewebsites.net/docs)** — Interactive API documentation (local)
- **[API Documentation (ReDoc)](https://trump-speeches-nlp-chatbot.azurewebsites.net/redoc)** — Alternative API documentation

> NOTE: The Azure Web App links above point to a deployed instance that may take a minute or two to start (cold start) after a deployment or if idle. If the site doesn't load immediately, please wait 1–2 minutes and try again.

## Core Features

### RAG Q&A System

Ask natural language questions about 35 political speeches (300,000+ words):

```bash
curl -X POST http://localhost:8000/rag/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What economic policies were discussed?", "top_k": 5}'
```

**Features:**
- Modular RAG components: SearchEngine, ConfidenceCalculator, EntityAnalyzer, DocumentLoader
- Semantic search using MPNet embeddings (768-dimensional)
- Hybrid search combining vector similarity and BM25 keyword matching
- Cross-encoder reranking for improved precision
- Multi-factor confidence scoring (retrieval quality, consistency, coverage, entity mentions)
- Entity extraction and analytics with sentiment analysis
- Google Gemini LLM for answer generation

### NLP Endpoints

Traditional NLP analysis:
- **Sentiment Analysis** — FinBERT transformer model
- **Topic Extraction** — TF-IDF based topic modeling
- **Word Frequency** — Statistical text analysis
- **N-gram Analysis** — Bigram and trigram extraction

### Interactive Web Interface

Single-page application at the root (`/`) for testing all features without writing code.

## 🛠️ Technology Stack

**AI/ML:**
- ChromaDB (vector database)
- sentence-transformers (MPNet)
- Google Gemini (LLM)
- Hugging Face Transformers (FinBERT)

**Backend:**
- FastAPI (REST API)
- Pydantic (validation)
- NLTK (preprocessing)

**DevOps:**
- Docker + Docker Compose
- GitHub Actions (CI/CD)
- pytest (testing)
- Black, flake8, mypy (code quality)

## 💡 Example Use Cases

1. **Political Speech Analysis** — Extract themes, sentiment, and talking points
2. **RAG System Demo** — Show how to build Q&A over large text corpora
3. **Entity Analytics** — Track mentions of people, places, and topics
4. **Hybrid Search** — Demonstrate combining semantic and keyword search

## 🎓 Learning Resources

- **Architecture diagrams** in the [Architecture](reference/architecture.md) doc
- **RAG implementation details** in [RAG Features](reference/rag-features.md)
- **Testing strategy** in [Testing Guide](howto/testing.md)
- **Deployment options** in [Deployment Guide](guides/deployment.md)

## 📞 Support & Contributing

- **Issues:** [GitHub Issues](https://github.com/JustaKris/Trump-Rally-Speeches-NLP-Chatbot/issues)
- **Author:** Kristiyan Bonev
- **License:** MIT

---

**Ready to get started?** Head to the **[Quickstart Guide](guides/quickstart.md)** →
