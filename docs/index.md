# Trump Speeches NLP Chatbot — Documentation

Welcome to the documentation for the **Trump Speeches NLP Chatbot** project, a production-ready FastAPI application demonstrating modern AI engineering practices with RAG (Retrieval-Augmented Generation), semantic search, and sentiment analysis.

## 🎯 What This Project Demonstrates

This portfolio project showcases:

- **RAG System Architecture** — ChromaDB vector database + MPNet embeddings + Google Gemini LLM
- **Hybrid Search** — Combining semantic search with BM25 keyword matching and cross-encoder reranking
- **Production FastAPI Development** — RESTful API design with 12+ endpoints
- **Entity Analytics** — Automatic entity extraction with sentiment analysis
- **DevOps Practices** — Docker, CI/CD, comprehensive testing, code quality tools

## 📚 Documentation Structure

### Getting Started

New to the project? Start here:

- **[Quickstart Guide](guides/quickstart.md)** — Get the API running in 5 minutes
- **[Deployment Guide](guides/deployment.md)** — Deploy to Render, Azure, or Docker

### How-To Guides

Task-oriented guides for specific features:

- **[Testing Guide](howto/testing.md)** — Run tests, code quality checks, and CI/CD
- **[Entity Analytics](howto/entity-analytics.md)** — Analyze entities mentioned in speeches

### Reference Documentation

Deep technical documentation:

- **[System Architecture](reference/architecture.md)** — System design, components, and diagrams
- **[RAG Features](reference/rag-features.md)** — Detailed RAG implementation documentation

## 🚀 Quick Links

- **[GitHub Repository](https://github.com/JustaKris/Trump-Rally-Speeches-NLP-Chatbot)** — Source code and issues
- **[API Documentation (Swagger)](http://localhost:8000/docs)** — Interactive API docs (when running locally)
- **[API Documentation (ReDoc)](http://localhost:8000/redoc)** — Alternative API docs

## 🤖 Core Features

### RAG Q&A System

Ask natural language questions about 35 political speeches (300,000+ words):

```bash
curl -X POST http://localhost:8000/rag/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What economic policies were discussed?", "top_k": 5}'
```

**Features:**
- Semantic search using MPNet embeddings (768-dimensional)
- Hybrid search combining vector similarity and BM25 keyword matching
- Cross-encoder reranking for improved precision
- Multi-factor confidence scoring
- Entity extraction and analytics
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
