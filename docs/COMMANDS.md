---

# Project commands — dt-rally-speeches-nlp

This file contains the most relevant commands for working with this repository. It focuses on local development: Poetry environment setup, running Jupyter notebooks, basic testing and code-quality commands, and common Git workflows.

## Poetry / environment

Install Poetry (if not already installed):

```powershell
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -
```

Initialize (if you haven't already) — the project is named `dt-rally-speeches-nlp`:

```powershell
poetry init --name dt-rally-speeches-nlp --description "NLP exercise on a dataset of rally speeches from Donald Trump's first presidential campaign." --author "Kristiyan Bonev" --python ">=3.10,<3.15"
```

Add the core dependencies used by the notebooks and a few developer tools:

```powershell
poetry add numpy pandas matplotlib seaborn wordcloud scikit-learn transformers torch nltk tqdm
poetry add --group dev pytest black flake8 ipykernel
```

Install from `pyproject.toml`:

```powershell
poetry install
```

Activate the virtual environment (PowerShell):

```powershell
& ".venv\Scripts\Activate.ps1"
```

Or use Poetry's shell:

```powershell
poetry shell
```

Notes:
- If you want GPU-accelerated PyTorch, follow the official PyTorch install selector and install the appropriate `torch` wheel for your CUDA version.
- Transformers will download pretrained models on first run — make sure you have disk space.

## Jupyter / notebooks

Start Jupyter Lab (recommended):

```powershell
poetry run jupyter lab
```

Or the classic notebook UI:

```powershell
poetry run jupyter notebook
```

Open the notebooks in `notebooks/` and run cells top-to-bottom. If you see NLTK missing data errors, run once in a cell:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

## Testing & code quality

Run tests:

```powershell
poetry run pytest
```

Format code:

```powershell
poetry run black src/ && poetry run isort src/
```

Check code quality (linting and type checking):

```powershell
poetry run flake8 src/ && poetry run mypy src/
```

All-in-one quality check:

```powershell
poetry run black src/ && poetry run isort src/ && poetry run flake8 src/ && poetry run mypy src/ && poetry run pytest
```

## Common Git workflows

Clone and start working:

```powershell
git clone <repo-url>
cd Donald-Trump-Rally-Speeches-NLP
```

Daily workflow:

```powershell
git pull origin main
git checkout -b feature/your-feature
# make changes
git add .
git commit -m "Describe changes"
git push -u origin feature/your-feature
```

Undo local changes:

```powershell
git checkout -- <path/to/file>
git reset HEAD <path/to/file>
```

## Quick reference

- Install deps: `poetry install`
- Start Jupyter Lab: `poetry run jupyter lab`
- Run tests: `poetry run pytest`
- Format code: `poetry run black src/ && poetry run isort src/`
- Check quality: `poetry run flake8 src/ && poetry run mypy src/`

---