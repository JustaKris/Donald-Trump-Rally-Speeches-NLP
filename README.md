# Donald Trump Rally Speeches — NLP Portfolio Project

A professional data science portfolio project demonstrating advanced Natural Language Processing (NLP) techniques through comprehensive analysis of Donald Trump's 2019-2020 rally speeches. This project showcases production-ready code, interactive visualizations, and actionable insights that would be valuable to employers seeking NLP and data analysis expertise.

## 🎯 Project Highlights

- **35 rally speeches** (2019-2020) totaling 300,000+ words analyzed
- **Interactive Plotly dashboards** for exploratory data analysis
- **Deep learning sentiment analysis** using FinBERT (BERT-based transformer model)
- **Temporal trend analysis** revealing sentiment patterns over time
- **Clean, documented, reusable code** following best practices
- **Reproducible environment** managed with Poetry

## 📊 What's Inside

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

## 🎯 Quick Start

### Prerequisites

- Python 3.12+
- Poetry for dependency management
- 2GB+ disk space for model weights

### Installation

1. **Clone the repository**
   ```powershell
   git clone https://github.com/JustaKris/Donald-Trump-Rally-Speeches-NLP.git
   cd Donald-Trump-Rally-Speeches-NLP
   ```

2. **Install dependencies with Poetry**
   ```powershell
   poetry install
   ```

3. **Activate the virtual environment**
   ```powershell
   poetry shell
   ```
   
   Or use the .venv directly:
   ```powershell
   & ".venv\Scripts\Activate.ps1"
   ```

4. **Launch Jupyter Lab**
   ```powershell
   poetry run jupyter lab
   ```

5. **Run notebooks in order**
   - Start with `1. Word Clouds.ipynb` (creates shared dataset)
   - Then `2. Sentiment Analysis.ipynb` (adds sentiment features)
   - Finally `3. Masked Language Modeling.ipynb` (optional deep dive)

### First-Time Setup

Some notebooks require NLTK data. Run once in a notebook cell:

```python
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
```

## 📦 Core Dependencies

```toml
python = "^3.12"
numpy = "^2.1.3"
pandas = "^2.2.3"
matplotlib = "^3.9.2"
seaborn = "^0.13.2"
wordcloud = "^1.9.4"
nltk = "^3.9.1"
plotly = "^5.24.1"
transformers = "^4.46.3"
tensorflow = "^2.18.0"
tqdm = "^4.67.0"
datasets = "^3.1.0"
```

**Note**: TensorFlow installs CPU-only by default. For GPU support, follow [PyTorch's official guide](https://pytorch.org/get-started/locally/) to install the appropriate CUDA-enabled version.

## 💡 Project Structure

```
Donald-Trump-Rally-Speeches-NLP/
│
├── data/
│   └── Donald Trump Rally Speeches/    # 35 .txt files (one per speech)
│
├── notebooks/
│   ├── 1. Word Clouds.ipynb            # Text analysis & visualization
│   ├── 2. Sentiment Analysis.ipynb     # Deep learning sentiment
│   └── 3. Masked Language Modeling.ipynb  # Transformer fine-tuning
│
├── pyproject.toml                      # Poetry dependencies
├── poetry.lock                         # Locked dependency versions
└── README.md                           # This file
```

## 🎓 Learning Outcomes & Portfolio Value

This project demonstrates capabilities that are directly applicable to industry roles:

### For Data Science Roles
- End-to-end analysis from raw text to insights
- Statistical rigor and proper evaluation methods
- Clear communication of findings

### For ML Engineering Roles
- Working with state-of-the-art transformer models
- Efficient data processing pipelines
- Model fine-tuning and deployment considerations

### For Analytics Roles
- Interactive visualization best practices
- Temporal trend analysis
- Stakeholder-ready reporting

## 🔧 Troubleshooting

### Common Issues

**SSL Certificate Errors (Hugging Face downloads)**
```python
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
```

**Out of Memory (TensorFlow/Sentiment Analysis)**
- Reduce batch size in sentiment analysis
- Use smaller models (e.g., `distilbert-base-uncased`)
- Close other memory-intensive applications

**Slow Performance**
- MLM training: Use GPU if available, or reduce epochs
- Sentiment analysis: Already optimized with chunking and progress bars

## 📝 Next Steps & Enhancements

**Potential additions to strengthen the portfolio:**

1. **Export visualizations**: Add HTML exports of key charts for non-technical viewers
2. **API deployment**: Wrap sentiment model in FastAPI for production demo
3. **Testing suite**: Add pytest tests for data processing functions
4. **CI/CD**: GitHub Actions for automated notebook execution
5. **Docker container**: Containerize for easy deployment
6. **Dashboard**: Streamlit/Dash app for interactive exploration

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
