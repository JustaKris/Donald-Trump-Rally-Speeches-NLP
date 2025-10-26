# Quick Start Guide

## Running the NLP API Locally

### Prerequisites

- Python 3.12+ installed
- Poetry installed (<https://python-poetry.org/docs/#installation>)

### Steps

1. **Install Dependencies**

   ```powershell
   poetry install
   ```

2. **Download NLTK Data** (first time only)

   ```powershell
   poetry run python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('punkt_tab')"
   ```

3. **Run the API**

   ```powershell
   poetry run uvicorn src.api:app --reload
   ```

4. **Access the Application**
   - Web UI: <http://localhost:8000>
   - API Docs: <http://localhost:8000/docs>
   - Health Check: <http://localhost:8000/health>

## Running with Docker

### Build and Run

```powershell
docker build -t trump-speeches-nlp-api .
docker run -p 8000:8000 trump-speeches-nlp-api
```

### Using Docker Compose

```powershell
docker-compose up
```

## Testing the API

### Using the Web Interface

1. Open <http://localhost:8000>
2. Enter text in the "Sentiment Analysis" tab
3. Click "Analyze Sentiment"

### Using curl

```powershell
# Health check
curl http://localhost:8000/health

# Analyze sentiment
curl -X POST http://localhost:8000/analyze/sentiment `
  -H "Content-Type: application/json" `
  -d '{"text": "We are going to make America great again!"}'

# Get dataset statistics
curl http://localhost:8000/speeches/stats
```

### Using Python requests

```python
import requests

# Analyze sentiment
response = requests.post(
    "http://localhost:8000/analyze/sentiment",
    json={"text": "This is amazing! The economy is booming."}
)
print(response.json())
```

## Troubleshooting

### Port Already in Use

If port 8000 is busy, run on a different port:

```powershell
poetry run uvicorn src.api:app --reload --port 8001
```

### Model Loading Errors

The first request may take 30-60 seconds as the model downloads and loads.
Check logs for progress.

### Module Not Found

Ensure you're in the project directory and have run `poetry install`.

## Next Steps

- Explore the Jupyter notebooks in `notebooks/`
- Read the deployment guide in `docs/DEPLOYMENT.md`
- Check out the API documentation at <http://localhost:8000/docs>
