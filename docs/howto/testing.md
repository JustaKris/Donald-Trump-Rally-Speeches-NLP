# Testing & Development Guide

## Running Tests

### Using uv (recommended)

This repository uses uv to manage virtual environments and run commands in a reproducible project environment. If you've already been using `uv` in this project, the examples below will work as-is.

```powershell
# Install project dependencies (including dev groups defined in pyproject)
uv sync            # sync all default groups
uv sync --group dev  # sync only dev dependencies (if grouped)

# Run a command inside the project's environment
uv run <command>   # e.g. `uv run pytest` or `uv run black src/ tests/`
```

If you prefer to use Poetry directly, the original Poetry commands are still valid and left as alternatives in this document.

### Install Development Dependencies (alternative: Poetry)

```powershell
# With Poetry (alternative)
poetry install --with dev

# Or activate Poetry shell and run commands directly
poetry shell
```

### Run All Tests

```powershell
# Run all tests with coverage
uv run pytest

# Run only unit tests
uv run pytest -m unit

# Run only integration tests
uv run pytest -m integration

# Run with verbose output
uv run pytest -v

# Run specific test file
uv run pytest tests/test_preprocessing.py

# Run test with details on warnings
pytest -W default::Warning

# Run tests with enforced warnings
pytest -W error::DeprecationWarning
```

### Code Coverage

```powershell
# Generate coverage report
uv run pytest --cov=src --cov-report=html

# Open coverage report
start htmlcov/index.html  # Windows
```

## Code Quality Tools

### Formatting

```powershell
# Format code with Black (via uv)
uv run black src/ tests/

# Check formatting without changes
uv run black --check src/ tests/

# Sort imports with isort
uv run isort src/ tests/

# Check imports without changes
uv run isort --check-only src/ tests/
```

### Linting

```powershell
# Run flake8
uv run flake8 src/ tests/

# Show detailed statistics
uv run flake8 src/ tests/ --count --statistics --show-source
```

### Type Checking

```powershell
# Run mypy type checker
uv run mypy src/ tests/
```

### Run All Quality Checks

```powershell
# Run everything at once (uv wrapper)
uv run black src/ tests/ && uv run isort src/ tests/ && uv run flake8 src/ tests/ && uv run mypy src/ tests/ && uv run pytest
uv run black src/ tests/ ; uv run isort src/ tests/ ; uv run flake8 src/ tests/ ; uv run mypy src/ tests/ ; uv run pytest
```

## Pre-commit Setup (Optional)

Install pre-commit hooks to automatically run checks before commits. If you manage dev dependencies with `uv`, use `uv sync --group dev` to install dev deps (including `pre-commit`) if listed in `pyproject.toml`. Otherwise install pre-commit directly:

```powershell
# Ensure pre-commit is installed in the project environment
uv run pip install pre-commit

# Install the git hook
uv run pre-commit install
```

If you prefer Poetry:

```powershell
poetry add --group dev pre-commit
poetry run pre-commit install
```

## CI/CD Pipeline

The GitHub Actions workflow runs automatically on:
- **Push** to `main`, `develop`, or `feature/*` branches
- **Pull requests** to `main` or `develop`

### Pipeline Jobs:

1. **Test Suite** - Runs on Python 3.11, 3.12, 3.13
   - Unit tests
   - Integration tests (without ML model)
   - Coverage reporting

2. **Code Quality** - Linting and formatting checks
   - flake8
   - black
   - isort
   - mypy

3. **Security** - Security scanning
   - safety (dependency vulnerabilities)
   - bandit (code security issues)

4. **Build** - Docker image build (on main branch only)
   - Builds image
   - Tests health endpoint

## Test Structure

```
tests/
├── __init__.py
├── test_preprocessing.py  # Unit tests for text processing
├── test_utils.py          # Unit tests for utilities
└── test_api.py            # Integration tests for API endpoints
```

## Test Markers

- `@pytest.mark.unit` - Fast unit tests
- `@pytest.mark.integration` - API integration tests
- `@pytest.mark.requires_model` - Tests needing ML model (skipped in CI)
- `@pytest.mark.slow` - Slow-running tests

## Writing New Tests

### Unit Test Example

```python
import pytest
from src.preprocessing import clean_text

@pytest.mark.unit
def test_clean_text():
    text = "Hello World!"
    result = clean_text(text)
    assert isinstance(result, str)
```

### API Test Example

```python
import pytest
from fastapi.testclient import TestClient
from src.api import app

@pytest.mark.integration
def test_health_check():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
```

## Coverage Goals

- **Target**: 70%+ overall coverage
- **Focus**: Core logic in `src/ tests/`
- **Exclude**: ML model internals, notebooks

## Troubleshooting

### NLTK Data Missing

```powershell
uv run python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('stopwords', quiet=True); nltk.download('punkt_tab', quiet=True)"
```

### Import Errors

```powershell
# Reinstall dependencies (uv)
uv sync --group dev

# Or with Poetry
poetry install --with dev
```

### Slow Tests

```powershell
# Skip slow tests
uv run pytest -m "not slow"

# Skip model-dependent tests
uv run pytest -m "not requires_model"
```
