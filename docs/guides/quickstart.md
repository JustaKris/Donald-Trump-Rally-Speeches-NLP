# Quick Start Guide

Get the Trump Speeches NLP Chatbot API running locally in minutes.

## Prerequisites

- Python 3.11+ installed
- uv installed ([install guide](https://docs.astral.sh/uv/getting-started/installation/))
- Google Gemini API key ([get one free](https://ai.google.dev/))

### Setup

1. **Install Dependencies**

   ```powershell
   uv sync

   uv venv --python 3.12  # if you need to set a specific vesion
   ```

2. **Configure Environment**

   Create a `.env` file in the project root:
   ```bash
   GEMINI_API_KEY=your_api_key_here
   ```

3. **Run the API**

   ```powershell
   uv run uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
   ```

   The API will automatically:
   - Load configuration from `.env`
   - Initialize logging (colored output in development)
   - Load FinBERT sentiment model (~440MB)
   - Initialize Gemini LLM service
   - Load ChromaDB vector database with existing embeddings
   - Start FastAPI server

   **Expected startup output**:
   ```
   2025-11-04 12:34:56 | INFO     | src.api              | Application: Trump Speeches NLP Chatbot API v0.1.0
   2025-11-04 12:34:56 | INFO     | src.api              | Environment: development
   2025-11-04 12:34:56 | INFO     | src.api              | ✓ Sentiment analysis model loaded successfully
   2025-11-04 12:34:57 | INFO     | src.api              | ✓ LLM service initialized and tested successfully
   2025-11-04 12:34:58 | INFO     | src.api              | ✓ RAG service initialized with 1082 existing chunks
   2025-11-04 12:34:58 | INFO     | src.api              | Application startup complete
   ```

4. **Access the Application**
   - Web UI: <http://localhost:8000>
   - API Docs: <http://localhost:8000/docs>
   - Health Check: <http://localhost:8000/health>

## Running with Docker

### Build and Run

```powershell
docker build -t trump-speeches-nlp-chatbot .
docker run --rm -it -p 8000:8000 --env-file .env --name nlp-chatbot trump-speeches-nlp-chatbot


docker run --rm -it -p 8000:8000 -v "${PWD}/data/chromadb:/app/data/chromadb" --env-file .env --name nlp-chatbot trump-speeches-nlp-api:torch-2.6.0

### Using Docker Compose

```powershell
docker-compose up
```

## Testing the RAG System

### Using the Web Interface

1. Open <http://localhost:8000>
2. Navigate to the "RAG Q&A" tab
3. Ask a question like *"What economic policies were discussed?"*
4. View the AI-generated answer with confidence scores and sources

### Using curl

```powershell
# Ask a question (RAG)
curl -X POST http://localhost:8000/rag/ask `
  -H "Content-Type: application/json" `
  -d '{"question": "What was said about the economy?", "top_k": 5}'

# Semantic search
curl -X POST http://localhost:8000/rag/search `
  -H "Content-Type: application/json" `
  -d '{"query": "immigration policy", "top_k": 5}'

# Get RAG statistics
curl http://localhost:8000/rag/stats

# Sentiment analysis (traditional NLP)
curl -X POST http://localhost:8000/analyze/sentiment `
  -H "Content-Type: application/json" `
  -d '{"text": "The economy is doing great!"}'
```

### Using Python

```python
import requests

# RAG Question Answering
response = requests.post(
    "http://localhost:8000/rag/ask",
    json={
        "question": "What were the main themes in the 2020 speeches?",
        "top_k": 5
    }
)
result = response.json()
print(f"Answer: {result['answer']}")
print(f"Confidence: {result['confidence']} ({result['confidence_score']:.2f})")
print(f"Sources: {', '.join(result['sources'])}")

# Traditional NLP - Sentiment
response = requests.post(
    "http://localhost:8000/analyze/sentiment",
    json={"text": "This is incredible! Best economy ever."}
)
print(response.json())
```

## Troubleshooting

### "RAG service not initialized"

The API auto-indexes documents on first startup. This takes ~30-60 seconds. Check the logs for progress:
```
INFO:     Loading documents into RAG service...
INFO:     Loaded 35 documents into RAG service!
```

### Gemini API Errors

Ensure your `.env` file exists with a valid `GEMINI_API_KEY`. Get a free key at <https://ai.google.dev/>.

### Model Download Taking Long

First run downloads ~1-2 GB of models (FinBERT, MPNet embeddings). Subsequent runs are fast.

### Port Already in Use

```powershell
uv run uvicorn src.api:app --reload --port 8001
```

### Module Not Found

Ensure you're in the project root directory and have run `uv sync`.

## Next Steps

- Try the interactive web interface at <http://localhost:8000>
- Explore API documentation at <http://localhost:8000/docs>
- Read about RAG improvements in `docs/RAG_IMPROVEMENTS.md`
- Deploy to production with `docs/DEPLOYMENT.md`
