# Testing & Development Guide

## Running Tests

### Install Development Dependencies

```powershell
# With Poetry (recommended)
poetry install --with dev

# Or activate Poetry shell and run commands directly
poetry shell
```

### Run All Tests

```powershell
# Run all tests with coverage
poetry run pytest

# Run only unit tests
poetry run pytest -m unit

# Run only integration tests
poetry run pytest -m integration

# Run with verbose output
poetry run pytest -v

# Run specific test file
poetry run pytest tests/test_preprocessing.py
```

### Code Coverage

```powershell
# Generate coverage report
pytest --cov=src --cov-report=html

# Open coverage report
start htmlcov/index.html  # Windows
```

## Code Quality Tools

### Formatting

```powershell
# Format code with Black
black src/

# Check formatting without changes
black --check src/

# Sort imports with isort
isort src/

# Check imports without changes
isort --check-only src/
```

### Linting

```powershell
# Run flake8
flake8 src/

# Show detailed statistics
flake8 src/ --count --statistics --show-source
```

### Type Checking

```powershell
# Run mypy type checker
mypy src/
```

### Run All Quality Checks

```powershell
# Run everything at once
poetry run black src/ && poetry run isort src/ && poetry run flake8 src/ && poetry run mypy src/ && poetry run pytest
poetry run black src/ ; poetry run isort src/ ; poetry run flake8 src/ ; poetry run mypy src/ ; poetry run pytest
```

## Pre-commit Setup (Optional)

Install pre-commit hooks to automatically run checks before commits:

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
- **Focus**: Core logic in `src/`
- **Exclude**: ML model internals, notebooks

## Troubleshooting

### NLTK Data Missing

```powershell
poetry run python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('stopwords', quiet=True); nltk.download('punkt_tab', quiet=True)"
```

### Import Errors

```powershell
# Reinstall dependencies
poetry install --with dev
```

### Slow Tests

```powershell
# Skip slow tests
poetry run pytest -m "not slow"

# Skip model-dependent tests
poetry run pytest -m "not requires_model"
```
