# Donald Trump Rally Speeches — NLP Portfolio Project

[![CI/CD Pipeline](https://github.com/JustaKris/Donald-Trump-Rally-Speeches-NLP/actions/workflows/ci.yml/badge.svg)](https://github.com/JustaKris/Donald-Trump-Rally-Speeches-NLP/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A professional data science portfolio project demonstrating advanced Natural Language Processing (NLP) techniques through comprehensive analysis of Donald Trump's 2019-2020 rally speeches. This project showcases production-ready code, interactive visualizations, REST API deployment, and actionable insights that would be valuable to employers seeking NLP and data analysis expertise.

## 🎯 Project Highlights

- **35 rally speeches** (2019-2020) totaling 300,000+ words analyzed
- **Interactive Plotly dashboards** for exploratory data analysis
- **Production FastAPI application** with sentiment analysis endpoints
- **Deep learning sentiment analysis** using FinBERT (BERT-based transformer model)
- **Temporal trend analysis** revealing sentiment patterns over time
- **Docker containerization** for easy deployment to any cloud platform
- **Clean, documented, reusable code** following best practices
- **Reproducible environment** managed with Poetry

## 🚀 Live Demo

🔗 **[Try the API](http://localhost:8000)** (Coming soon: deployed version)

📚 **[API Documentation](http://localhost:8000/docs)** - Interactive Swagger UI

📖 **[ReDoc Documentation](http://localhost:8000/redoc)** - Alternative API docs

## 📊 What's Inside

### `src/` — Production API Code

The project includes a production-ready FastAPI application for real-time NLP analysis:

- **`api.py`** — FastAPI application with RESTful endpoints
- **`models.py`** — Sentiment analysis model loading and inference
- **`preprocessing.py`** — Text cleaning, tokenization, and n-gram extraction
- **`utils.py`** — Data loading and statistical utilities

**API Endpoints:**
- `POST /analyze/sentiment` — Analyze sentiment of input text (FinBERT)
- `POST /analyze/words` — Word frequency analysis
- `POST /analyze/topics` — Topic/theme extraction
- `POST /analyze/ngrams` — N-gram extraction (bigrams, trigrams, etc.)
- `GET /speeches/stats` — Dataset statistics
- `GET /speeches/list` — List all speeches with metadata

### `static/` — Interactive Web Frontend

A beautiful, responsive web interface for interacting with the API without coding.

### `data/Donald Trump Rally Speeches/`

Plain text transcripts of 35 rally speeches from July 2019 to September 2020, providing rich corpus for NLP analysis.

### `notebooks/` — Three Production-Ready Jupyter Notebooks

#### 1. **Word Clouds.ipynb** — Exploratory Text Analysis

- Interactive word frequency visualizations with Plotly
- N-gram analysis (unigrams, bigrams, trigrams)
- Temporal word usage patterns comparing 2019 vs 2020
- Topic identification and frequency tracking
- Comprehensive statistical summaries

**Key Technologies:** pandas, NLTK, WordCloud, Plotly, tqdm

#### 2. **Sentiment Analysis.ipynb** — Deep Learning NLP

- FinBERT-based sentiment classification (positive/negative/neutral)
- Chunked processing for long documents
- Speech-by-speech sentiment scoring with chunk-level granularity
- Interactive dashboards: heatmaps, timelines, distributions
- Year-over-year statistical comparison
- Moving average trend analysis
- Results exported for downstream use

**Key Technologies:** TensorFlow, Transformers (Hugging Face), FinBERT, Plotly

#### 3. **Masked Language Modeling.ipynb** — Advanced Transformer Fine-tuning

- Custom DistilBERT fine-tuning on domain-specific corpus
- Whole-word masking implementation
- Model evaluation with perplexity metrics
- Comparative analysis: base model vs fine-tuned model
- Demonstrates understanding of transfer learning and model training

**Key Technologies:** TensorFlow, Transformers, Datasets (Hugging Face), DistilBERT

## 🚀 Key Skills Demonstrated

### Data Science & Analysis

- **Data wrangling**: Loading, parsing, and structuring unstructured text data
- **Exploratory Data Analysis**: Statistical summaries, distributions, temporal patterns
- **Feature engineering**: N-gram extraction, stopword filtering, custom metrics

### NLP & Machine Learning

- **Text preprocessing**: Tokenization, chunking, stopword removal
- **Sentiment analysis**: Deep learning classification with pre-trained transformers
- **Language modeling**: Fine-tuning BERT-based models on domain-specific data
- **Model evaluation**: Perplexity, accuracy metrics, comparative analysis

### Visualization & Communication

- **Interactive dashboards**: Plotly-based multi-panel visualizations
- **Time series analysis**: Trend lines, moving averages, year-over-year comparisons
- **Clear documentation**: Markdown cells, docstrings, and inline comments
- **Storytelling with data**: Extracting actionable insights from raw text

### Software Engineering

- **Production-ready code**: Modular functions, type hints, error handling
- **Environment management**: Poetry for dependency management
- **Best practices**: Progress bars, logging, reproducible workflows
- **Clean code principles**: DRY, separation of concerns, readability

## 📈 Sample Insights from the Analysis

- **Sentiment patterns**: Identified temporal shifts in speech sentiment between 2019 and 2020
- **Vocabulary evolution**: Tracked how language usage changed over the campaign period
- **Topic prevalence**: Quantified mentions of key themes (economy, immigration, media)
- **Speech characteristics**: Average speech length of 8,500+ words with consistent patterns

## 🚀 Quick Start

### Option 1: Run the API (Recommended for Demo)

1. **Install dependencies**

   ```powershell
   poetry install
   ```

2. **Start the FastAPI server**

   ```powershell
   poetry run uvicorn src.api:app --reload
   ```

3. **Access the application**
   - **Web Interface:** <http://localhost:8000>
   - **API Docs:** <http://localhost:8000/docs>
   - **ReDoc:** <http://localhost:8000/redoc>

### Option 2: Run with Docker

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

### Option 3: Explore Jupyter Notebooks

1. **Install dependencies**

   ```powershell
   poetry install
   ```

2. **Activate the environment**

   ```powershell
   poetry shell
   ```

3. **Launch Jupyter Lab**

   ```powershell
   poetry run jupyter lab
   ```

4. **Run notebooks in order**
   - Start with `1. Word Frequency & Topics Analysis.ipynb`
   - Then `2. Sentiment Analysis.ipynb`
   - Finally `3. Masked Language Modeling.ipynb` (optional)

### First-Time Setup

Some notebooks require NLTK data. Run once in a notebook cell:

```python
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
```

## 🧪 Testing & Code Quality

This project includes comprehensive testing and code quality tools to demonstrate professional software engineering practices.

### Run Tests

```powershell
# Install dev dependencies
poetry install --with dev

# Run all tests with coverage
poetry run pytest

# Run only unit tests
poetry run pytest -m unit

# Run only integration tests
poetry run pytest -m integration

# Generate HTML coverage report
poetry run pytest --cov=src --cov-report=html
```

### Code Quality Checks

```powershell
# Format code
poetry run black src/

# Lint code
poetry run flake8 src/

# Sort imports
poetry run isort src/

# Type checking
poetry run mypy src/

# Run all checks
poetry run black src/ && poetry run isort src/ && poetry run flake8 src/ && poetry run pytest
```

### CI/CD Pipeline

The project uses GitHub Actions for continuous integration:
- ✅ **Automated testing** on Python 3.11, 3.12, 3.13
- ✅ **Code quality checks** (black, flake8, isort, mypy)
- ✅ **Security scanning** (safety, bandit)
- ✅ **Docker image build** and health checks

See [`.github/workflows/ci.yml`](.github/workflows/ci.yml) for the full pipeline configuration.

For detailed testing documentation, see [`docs/TESTING.md`](docs/TESTING.md).

## 📦 Dependencies
```

## 📦 Core Dependencies

**API & Web Framework:**

```toml
fastapi = "^0.115.0"          # Modern async API framework
uvicorn[standard] = "^0.32.0" # ASGI server with auto-reload
python-multipart = "^0.0.12"  # Form data parsing
```

**NLP & Machine Learning:**

```toml
transformers = "^4.57.1"      # HuggingFace transformers (BERT, DistilBERT)
tensorflow = "^2.20.0"        # Deep learning framework
torch = "^2.9.0"              # PyTorch (transformers backend)
nltk = "^3.9.2"               # Natural Language Toolkit
scikit-learn = "^1.7.2"       # ML utilities
```

**Data & Visualization:**

```toml
pandas = "^2.3.3"             # Data manipulation
numpy = "^2.3.4"              # Numerical computing
plotly = "^6.3.1"             # Interactive visualizations
matplotlib = "^3.10.7"        # Static plots
seaborn = "^0.13.2"           # Statistical visualizations
wordcloud = "^1.9.4"          # Word cloud generation
```

**Development:**

```toml
jupyterlab = "^4.4.10"        # Interactive notebooks
ipykernel = "^7.0.1"          # Jupyter kernel
tqdm = "^4.67.1"              # Progress bars
```

**Note**: TensorFlow installs CPU-only by default. For GPU support, follow [PyTorch's official guide](https://pytorch.org/get-started/locally/) to install the appropriate CUDA-enabled version.

## 💡 Project Structure

```
Donald-Trump-Rally-Speeches-NLP/
│
├── src/                                # Production API code
│   ├── __init__.py
│   ├── api.py                         # FastAPI application
│   ├── models.py                      # ML model loading & inference
│   ├── preprocessing.py               # Text preprocessing utilities
│   └── utils.py                       # Data loading & statistics
│
├── static/
│   └── index.html                     # Interactive web frontend
│
├── data/
│   └── Donald Trump Rally Speeches/   # 35 .txt files (one per speech)
│
├── notebooks/
│   ├── 1. Word Frequency & Topics Analysis.ipynb
│   ├── 2. Sentiment Analysis.ipynb
│   └── 3. Masked Language Modeling.ipynb
│
├── docs/
│   ├── COMMANDS.md
│   ├── DEPLOYMENT.md                  # Deployment guide
│   └── TESTING.md                     # Testing & code quality guide
│
├── tests/                             # Test suite
│   ├── __init__.py
│   ├── test_preprocessing.py          # Unit tests for preprocessing
│   ├── test_utils.py                  # Unit tests for utilities
│   └── test_api.py                    # Integration tests for API
│
├── .github/
│   └── workflows/
│       └── ci.yml                     # GitHub Actions CI/CD pipeline
│
├── .azure/
│   └── deployment-guide.txt           # Azure deployment instructions
│
├── pytest.ini                         # Pytest configuration
├── .flake8                            # Flake8 linting rules
├── Dockerfile                         # Docker container definition
├── docker-compose.yml                 # Docker Compose configuration
├── render.yaml                        # Render deployment config
├── pyproject.toml                     # Poetry dependencies + tool configs
├── poetry.lock                        # Locked dependency versions
└── README.md                          # This file
```

## 🎓 Learning Outcomes & Portfolio Value

This project demonstrates capabilities that are directly applicable to industry roles:

### For Data Science Roles

- End-to-end analysis from raw text to insights
- Statistical rigor and proper evaluation methods
- Clear communication of findings
- Production deployment experience

### For ML Engineering Roles

- Working with state-of-the-art transformer models
- Efficient data processing pipelines
- Model fine-tuning and deployment considerations
- RESTful API design and implementation
- Docker containerization and cloud deployment

### For Analytics Roles

- Interactive visualization best practices
- Temporal trend analysis
- Stakeholder-ready reporting
- Web-based dashboards

### Full-Stack ML Skills

- **Backend:** FastAPI, Python, async programming
- **ML/NLP:** BERT models, sentiment analysis, text preprocessing
- **DevOps:** Docker, container orchestration, cloud deployment, CI/CD pipelines
- **Frontend:** HTML/CSS/JavaScript, responsive design
- **Data Engineering:** ETL pipelines, data validation
- **Testing:** pytest, unit/integration testing, code coverage
- **Code Quality:** Linting (flake8), formatting (black), type checking (mypy)

## 🚢 Deployment

This project is ready to deploy to multiple cloud platforms!

### Deployment Options

- **[Render](https://render.com)** — Free tier available, automatic deployments from Git
- **[Azure App Service](https://azure.microsoft.com/en-us/services/app-service/)** — Enterprise-grade, excellent for portfolio
- **[Railway](https://railway.app)** — Simple deployment with free credits
- **[Fly.io](https://fly.io)** — Global edge deployment

### Quick Deploy to Render

1. Push your code to GitHub
2. Connect repository to Render
3. Deploy with one click using `render.yaml`

### Deploy with Docker

```powershell
# Build image
docker build -t trump-speeches-nlp-api .

# Run locally
docker run -p 8000:8000 trump-speeches-nlp-api

# Deploy to any cloud platform that supports Docker
```

📖 **Full deployment instructions:** See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)

## 🔧 Troubleshooting

### Common Issues

#### SSL Certificate Errors (Hugging Face downloads)

```python
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
```

#### Out of Memory (TensorFlow/Sentiment Analysis)

- Reduce batch size in sentiment analysis
- Use smaller models (e.g., `distilbert-base-uncased`)
- Close other memory-intensive applications

#### Slow Performance

- MLM training: Use GPU if available, or reduce epochs
- Sentiment analysis: Already optimized with chunking and progress bars

## 📝 Next Steps & Enhancements

**Current features:**

- ✅ FastAPI REST API with sentiment analysis
- ✅ Interactive web frontend
- ✅ Docker containerization
- ✅ Deployment configurations for Render and Azure
- ✅ Comprehensive documentation

**Potential additions to strengthen the portfolio:**

1. **Cloud deployment:** Deploy to Render or Azure and add live demo link
2. **CI/CD pipeline:** GitHub Actions for automated testing and deployment
3. **Testing suite:** Pytest for API endpoints and data processing
4. **Enhanced frontend:** Add charts and visualizations using Plotly.js
5. **Authentication:** Add API keys or OAuth for production use
6. **Caching:** Redis for model predictions and frequently accessed data
7. **Rate limiting:** Protect API from abuse
8. **Monitoring:** Application performance monitoring (APM)
9. **Database integration:** PostgreSQL for storing analysis results

## 📄 License & Attribution

This repository is for educational and portfolio purposes. The speech transcripts are publicly available data used for demonstrative NLP analysis.

**Technologies Used:**

- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [FinBERT](https://huggingface.co/ProsusAI/finbert) for sentiment analysis
- [Plotly](https://plotly.com/python/) for interactive visualizations
- [Poetry](https://python-poetry.org/) for dependency management

---

## 📫 Contact

**Kristiyan Bonev** | [GitHub](https://github.com/JustaKris)

*This project showcases practical NLP skills and modern data science workflows. Feel free to explore the notebooks and reach out with questions!*
