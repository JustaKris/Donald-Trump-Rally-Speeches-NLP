# Donald Trump Rally Speeches — NLP Exercises

This repository is a small portfolio project intended to showcase practical Natural Language Processing (NLP) skills through exploratory analysis and modeling on a corpus of Donald Trump rally speeches. The goal is to provide clear, reproducible notebooks that demonstrate techniques useful to employers: data loading/cleaning, visualization, sentiment analysis, and masked language modeling.

## What you'll find here

- `data/Donald Trump Rally Speeches/` — plain text transcripts of rally speeches (source files used by the notebooks).
- `notebooks/` — three focused Jupyter notebooks:
  - `Word Clouds.ipynb` — EDA and word cloud visualizations to surface frequent words and themes.
  - `Sentiment Analysis.ipynb` — simple sentiment analysis experiments and visualizations.
  - `Masked Language Modeling.ipynb` — demonstration of masked language modeling using Hugging Face transformers.

## Quick project contract

- Inputs: plain text files (one rally transcript per file) under `data/Donald Trump Rally Speeches/`.
- Outputs: notebooks with plots, word clouds, and model outputs; helper artifacts produced while running notebooks (plots, cached models).
- Success criteria: notebooks run end-to-end in a reproducible Poetry environment and produce the visualizations and model outputs shown in the notebooks.

## Dependencies (Poetry)

The notebooks rely on a standard NLP / data-science stack plus Hugging Face Transformers and PyTorch for the modeling notebook. From PowerShell in the project root you can add the core packages with:

```powershell
poetry add numpy pandas matplotlib seaborn wordcloud scikit-learn transformers torch nltk tqdm jupyterlab notebook ipykernel
```

Notes:

- `torch` will install a CPU-only build by default unless you configure a CUDA-enabled wheel for your platform. If you have an NVIDIA GPU and want GPU support, follow PyTorch's official install selector and install the appropriate package before or after adding the other packages.
- `transformers` may download pretrained model weights the first time you run the masked LM notebook; ensure you have enough disk space (a few hundred MB to multiple GB depending on model choice).

After adding dependencies, either use `poetry shell` to enter the venv or run commands with `poetry run`.

Activate the venv from PowerShell if you prefer to use the created `.venv` directly:

```powershell
& ".venv\Scripts\Activate.ps1"
```

Start Jupyter Lab (recommended) inside the poetry environment:

```powershell
poetry run jupyter lab
```

Or use the classic notebook UI:

```powershell
poetry run jupyter notebook
```

## How to run the notebooks

1. Install dependencies (see above).
2. Start Jupyter Lab or Notebook in the project root so relative paths to `data/` work.
3. Open one of the notebooks under `notebooks/` and run cells in order. The notebooks include small preprocessing steps and inline notes about model sizes and runtime.

## Short guidance / edge cases

- If a notebook fails due to missing NLTK data (tokenizers/stopwords), run the following inside a Python cell or a short script once:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

- Large transformer models may run slowly or exhaust memory on CPU-only machines. For faster iteration use smaller models (e.g., `distilbert-base-uncased`) or a machine with a GPU.
- The dataset is plain text; if you add more files, the notebooks will pick them up if they follow the same filename conventions.

## Suggested next steps (for the portfolio)

- Add a short `requirements` section to the notebooks summarizing the exact package versions used for reproducibility.
- Add a small script in `scripts/` that pulls basic metrics (word counts, top n-grams) from `data/` as a programmatic demo.
- Add a short rendered HTML or exported PDF of a selected notebook to showcase the results to non-technical reviewers.

## License & attribution

This repository is for learning and portfolio purposes. Add a license if you plan to publish or share the repo widely.

---

If you'd like, I can also pin exact package versions in `pyproject.toml`, add `ipynb`->`html` export steps, or create a `scripts/` helper to run the notebooks programmatically.
