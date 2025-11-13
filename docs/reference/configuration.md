# Configuration Guide

This project uses **Pydantic Settings** for type-safe, environment-based configuration. This is the industry-standard approach for modern Python applications, especially those deployed to cloud platforms like Azure.

## Configuration Architecture

### Core Components

1. **`src/config.py`** - Central configuration module with `Settings` class
2. **`.env` file** - Environment variables for local/cloud deployment
3. **Validation** - Automatic type checking and validation via Pydantic

### Benefits

- ✅ **Type-safe** - Compile-time checking of configuration values
- ✅ **Environment-aware** - Different configs for dev/staging/prod
- ✅ **Cloud-friendly** - Works seamlessly with Azure, AWS, GCP
- ✅ **Validated** - Invalid configs fail fast with clear error messages
- ✅ **Documented** - Self-documenting with type hints and descriptions

## Quick Start

### 1. Create Your `.env` File

Copy the example file:

```bash
cp .env.example .env
```

### 2. Set Your LLM Provider

Edit `.env` and configure your preferred LLM provider:

**Option A: Google Gemini (Default)**
```env
LLM_PROVIDER=gemini
LLM_API_KEY=your_gemini_api_key_here
LLM_MODEL_NAME=gemini-2.0-flash-exp
```

Get a free key at: https://ai.google.dev/

**Option B: OpenAI**
```bash
# Install OpenAI support
uv sync --group llm-openai
```

```env
LLM_PROVIDER=openai
LLM_API_KEY=sk-your_openai_api_key_here
LLM_MODEL_NAME=gpt-4o-mini
```

**Option C: Anthropic (Claude)**
```bash
# Install Anthropic support
uv sync --group llm-anthropic
```

```env
LLM_PROVIDER=anthropic
LLM_API_KEY=sk-ant-your_anthropic_api_key_here
LLM_MODEL_NAME=claude-3-5-sonnet-20241022
```

### 3. Run the Application

```bash
uv run uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```

The app will automatically:
- Load settings from `.env`
- Validate all configuration values
- Initialize services with configured parameters
- Display startup configuration in logs

## Configuration Options

### Application Settings

```env
APP_NAME="Trump Speeches NLP Chatbot API"
APP_VERSION="0.1.0"
ENVIRONMENT="development"  # development, staging, production
LOG_LEVEL="INFO"           # DEBUG, INFO, WARNING, ERROR, CRITICAL
```

### LLM Provider (Multi-Provider Support)

Configure which LLM provider to use for answer generation, sentiment interpretation, and topic analysis.

#### General LLM Settings

```env
LLM_PROVIDER="gemini"          # gemini | openai | anthropic | none
LLM_API_KEY="your_api_key"     # Single API key for active provider
LLM_MODEL_NAME="model-name"    # Model identifier
LLM_TEMPERATURE="0.7"          # 0.0-1.0 (lower = more focused, higher = more creative)
LLM_MAX_OUTPUT_TOKENS="2048"   # Maximum response length
LLM_ENABLED="true"             # Enable/disable LLM features
```

#### Provider-Specific Examples

**Gemini (Default - Always Available):**
```env
LLM_PROVIDER="gemini"
LLM_API_KEY="your_gemini_api_key"
LLM_MODEL_NAME="gemini-2.0-flash-exp"  # or gemini-1.5-pro
LLM_TEMPERATURE="0.7"
LLM_MAX_OUTPUT_TOKENS="2048"
```

**OpenAI (Optional - Install with `uv sync --group llm-openai`):**
```env
LLM_PROVIDER="openai"
LLM_API_KEY="sk-your_openai_api_key"
LLM_MODEL_NAME="gpt-4o-mini"  # or gpt-4o, gpt-4-turbo
LLM_TEMPERATURE="0.7"
LLM_MAX_OUTPUT_TOKENS="2048"
```

**Anthropic (Optional - Install with `uv sync --group llm-anthropic`):**
```env
LLM_PROVIDER="anthropic"
LLM_API_KEY="sk-ant-your_anthropic_api_key"
LLM_MODEL_NAME="claude-3-5-sonnet-20241022"  # or claude-3-opus-20240229
LLM_TEMPERATURE="0.7"
LLM_MAX_OUTPUT_TOKENS="2048"
```

**Disable LLM:**
```env
LLM_PROVIDER="none"
LLM_ENABLED="false"
```

#### Switching Providers

1. **Install optional provider** (if not already installed):
   ```bash
   uv sync --group llm-openai      # For OpenAI
   uv sync --group llm-anthropic   # For Anthropic
   uv sync --group llm-all         # For all providers
   ```

2. **Update `.env` file** with new provider settings

3. **Restart application**:
   ```bash
   uv run uvicorn src.api:app --reload
   ```

The application will automatically use the new provider without code changes.

### ML Models

Configure which models to use for different tasks:

```env
# Sentiment Analysis
SENTIMENT_MODEL_NAME="ProsusAI/finbert"

# Embeddings for RAG
EMBEDDING_MODEL_NAME="all-mpnet-base-v2"

# Re-ranking
RERANKER_MODEL_NAME="cross-encoder/ms-marco-MiniLM-L-6-v2"
```

### RAG Configuration

```env
# ChromaDB
CHROMADB_PERSIST_DIRECTORY="./data/chromadb"
CHROMADB_COLLECTION_NAME="speeches"

# Text Chunking
CHUNK_SIZE="2048"
CHUNK_OVERLAP="150"

# Search
DEFAULT_TOP_K="5"
USE_RERANKING="true"
USE_HYBRID_SEARCH="true"
```

### Data Directories

```env
DATA_ROOT_DIRECTORY="./data"
SPEECHES_DIRECTORY="./data/Donald Trump Rally Speeches"
```

### API Settings

```env
API_HOST="0.0.0.0"
API_PORT="8000"
API_RELOAD="false"  # true in development
CORS_ORIGINS="*"    # comma-separated list in production
```

## Environment-Specific Configs

### Development

```env
ENVIRONMENT="development"
LOG_LEVEL="DEBUG"
API_RELOAD="true"
CORS_ORIGINS="*"
```

### Production (Azure)

```env
ENVIRONMENT="production"
LOG_LEVEL="INFO"
API_RELOAD="false"
CORS_ORIGINS="https://yourdomain.com"
```

## Using Configuration in Code

### Accessing Settings

```python
from src.config import get_settings

settings = get_settings()

# Access values
print(settings.gemini_api_key)
print(settings.chunk_size)
print(settings.log_level)
```

### Type-Safe Access

```python
# All settings are type-checked
settings.chunk_size  # int
settings.gemini_temperature  # float
settings.use_reranking  # bool
settings.llm_provider  # Literal["gemini", "openai", "anthropic", "none"]
```

### Helper Methods

```python
# Check if LLM is configured
if settings.is_llm_configured():
    api_key = settings.get_llm_api_key()
    model = settings.get_llm_model_name()
    
# Create LLM provider
from src.services.llm import create_llm_provider
llm = create_llm_provider()  # Automatically uses LLM_PROVIDER setting

# Get Path objects
speeches_path = settings.get_speeches_path()
chromadb_path = settings.get_chromadb_path()

# Setup logging
settings.setup_logging()
```

## Logging Configuration

The project uses `src/logging_config.py` for production-ready logging with automatic format detection.

### Log Levels

- **DEBUG**: Detailed diagnostic information for troubleshooting
- **INFO**: Important application events (default, recommended for production)
- **WARNING**: Unexpected but recoverable situations
- **ERROR**: Application errors requiring attention
- **CRITICAL**: System-critical failures

### Log Formats

#### Development (Colored)

Automatically enabled when `ENVIRONMENT=development`:

```
2025-11-04 12:34:56 | INFO     | src.api              | Application startup complete
2025-11-04 12:34:57 | DEBUG    | src.rag_service      | Performing hybrid search
```

- ANSI colors by level (green=INFO, red=ERROR, etc.)
- Human-readable timestamps
- Module names right-aligned

#### Production (JSON)

Automatically enabled when `ENVIRONMENT=production`:

```json
{"timestamp": "2025-11-04 12:34:56", "level": "INFO", "name": "src.api", "message": "Application startup complete"}
{"timestamp": "2025-11-04 12:34:57", "level": "DEBUG", "name": "src.rag_service", "message": "Performing hybrid search"}
```

- Machine-parseable JSON
- Compatible with Azure Application Insights, CloudWatch, ELK stack
- Automatic exception field for errors

### Changing Log Settings

Edit `.env`:

```env
# Log level
LOG_LEVEL="INFO"   # Recommended for production
LOG_LEVEL="DEBUG"  # Verbose for debugging

# Environment (affects format)
ENVIRONMENT="development"  # Colored logs
ENVIRONMENT="production"   # JSON logs
```

The logging system automatically:
- Detects environment and chooses appropriate format
- Suppresses noisy third-party loggers (chromadb, httpx, transformers)
- Configures uvicorn logs
- Filters ChromaDB telemetry errors

For detailed logging documentation, see [`docs/howto/logging.md`](../howto/logging.md).

## Azure Deployment

Azure App Service automatically loads environment variables. Configure them in:

1. **Azure Portal**: App Service → Configuration → Application Settings
2. **Azure CLI**:
   ```bash
   az webapp config appsettings set --name myapp --resource-group mygroup \
     --settings GEMINI_API_KEY="your_key" LOG_LEVEL="INFO"
   ```

## Docker Deployment

### Using .env file

```bash
docker run --env-file .env -p 8000:8000 myapp
```

### Using environment variables

```bash
docker run \
  -e GEMINI_API_KEY="your_key" \
  -e LOG_LEVEL="INFO" \
  -p 8000:8000 \
  myapp
```

### Docker Compose

```yaml
services:
  api:
    build: .
    environment:
      - GEMINI_API_KEY=${GEMINI_API_KEY}
      - LOG_LEVEL=${LOG_LEVEL:-INFO}
    env_file:
      - .env
    ports:
      - "8000:8000"
```

## Validation

Pydantic automatically validates configuration:

### Example Validation Errors

```python
# Invalid log level
LOG_LEVEL="INVALID"
# ❌ Error: Invalid log level. Must be one of: DEBUG, INFO, WARNING, ERROR, CRITICAL

# Invalid chunk size
CHUNK_SIZE="not_a_number"
# ❌ Error: Input should be a valid integer

# Missing required API key (when LLM enabled)
LLM_ENABLED="true"
GEMINI_API_KEY=""
# ❌ Error: API key appears to be too short
```

## Best Practices

1. **Never commit `.env`** - Add to `.gitignore`
2. **Use `.env.example`** - Document all available options
3. **Validate early** - Settings load at startup, fail fast
4. **Environment-specific** - Different configs for dev/prod
5. **Security** - Use Azure Key Vault for sensitive values in production
6. **Logging** - Use appropriate log levels for each environment

## Troubleshooting

### Settings not loading

Check:
1. `.env` file exists in project root
2. File encoding is UTF-8
3. No syntax errors in `.env`

### Invalid configuration

Check logs at startup:
```
ERROR: ValidationError: 1 validation error for Settings
  Invalid log level. Must be one of: DEBUG, INFO, WARNING, ERROR, CRITICAL
```

### API key issues

```bash
# Check if API key is set
python -c "from src.config import get_settings; print(get_settings().gemini_api_key)"
```

## Migration from Old Code

If you were using environment variables directly:

**Before:**
```python
import os
api_key = os.getenv("GEMINI_API_KEY")
```

**After:**
```python
from src.config import get_settings
settings = get_settings()
api_key = settings.gemini_api_key  # Type-safe!
```

## Further Reading

- [Pydantic Settings Docs](https://docs.pydantic.dev/latest/concepts/pydantic_settings/)
- [12-Factor App Config](https://12factor.net/config)
- [Azure App Service Configuration](https://learn.microsoft.com/en-us/azure/app-service/configure-common)
