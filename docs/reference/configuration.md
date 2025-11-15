# Configuration Guide

This project uses **Pydantic Settings v2** for type-safe configuration, combining YAML config files with environment variable overrides. This is a modern, cloud-friendly pattern that works well for local development and deployments on AWS or other platforms.

## Configuration Architecture

### Core Components

1. **`src/config/settings.py`** - Central configuration module with `Settings` class
2. **YAML config files** in `configs/` (e.g., `configs/development.yaml`, `configs/production.yaml`)
3. **`.env` file** - Environment variables for sensitive values and overrides
4. **Validation** - Automatic type checking and validation via Pydantic

### Benefits

- ✅ **Type-safe** - Compile-time checking of configuration values
- ✅ **Environment-aware** - Different configs for dev/staging/prod
- ✅ **Cloud-friendly** - Works seamlessly with Azure, AWS, GCP
- ✅ **Validated** - Invalid configs fail fast with clear error messages
- ✅ **Documented** - Self-documenting with type hints and descriptions

## Quick Start

### 1. Choose Your Environment (YAML)

Configuration defaults live in YAML files under `configs/`:

- `configs/development.yaml` – for local development
- `configs/production.yaml` – for production deployments (AWS, Azure, etc.)

By default, the app uses the `development` environment. You can override this via the `ENVIRONMENT` environment variable:

```bash
ENVIRONMENT=production
```

The active environment name is used to pick `configs/<environment>.yaml`.

### 2. Create Your `.env` File

Copy the example file:

```bash
cp .env.example .env
```

Use `.env` for **secrets and overrides only** (API keys, tokens, one-off tweaks). All non-sensitive defaults should live in YAML.

### 3. Set Your LLM Provider

Edit `.env` and configure your preferred LLM provider (sensitive values like API keys stay here):

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

### 4. Run the Application

```bash
uv run uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```

The app will automatically:
- Load base defaults from `configs/<ENVIRONMENT>.yaml`
- Apply Pydantic model defaults for any missing values
- Override with environment variables / `.env` values
- Validate all configuration values
- Initialize services with configured parameters
- Display startup configuration in logs

## Configuration Options

### Application Settings

Core metadata and logging live primarily in YAML:

```yaml
# configs/development.yaml
environment: development
log_level: DEBUG
app_name: "Trump Speeches NLP Chatbot API (Development)"
app_version: "0.1.0"
```

You can still override via `.env` or environment variables if needed:

```env
ENVIRONMENT="production"   # selects configs/production.yaml
LOG_LEVEL="INFO"           # overrides YAML
APP_NAME="Custom Name"     # overrides YAML
```

### LLM Provider (Multi-Provider Support)

Configure which LLM provider to use for answer generation, sentiment interpretation, and topic analysis.

#### General LLM Settings

In YAML we configure non-sensitive defaults under the `llm` section:

```yaml
llm:
  provider: "gemini"          # gemini | openai | anthropic | none
  enabled: true
  model_name: "gemini-2.5-flash"
  temperature: 0.3
  max_output_tokens: 1024
```

Sensitive values like API keys are supplied via environment variables / `.env`:

```env
LLM_PROVIDER="gemini"          # optional override for provider
LLM_API_KEY="your_api_key"     # Single API key for active provider
LLM_MODEL_NAME="model-name"    # optional override for model
LLM_TEMPERATURE="0.7"          # optional override for temperature
LLM_MAX_OUTPUT_TOKENS="2048"   # optional override for max tokens
LLM_ENABLED="true"             # optional override
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

Configure which models to use for different tasks via YAML:

```yaml
models:
  sentiment_model_name: "ProsusAI/finbert"
  embedding_model_name: "all-mpnet-base-v2"
  reranker_model_name: "cross-encoder/ms-marco-MiniLM-L-6-v2"
  emotion_model_name: "j-hartmann/emotion-english-distilroberta-base"
```

You can override any of them via environment variables if needed:

```env
SENTIMENT_MODEL_NAME="ProsusAI/finbert"
EMBEDDING_MODEL_NAME="all-mpnet-base-v2"
RERANKER_MODEL_NAME="cross-encoder/ms-marco-MiniLM-L-6-v2"
EMOTION_MODEL_NAME="j-hartmann/emotion-english-distilroberta-base"
```

### RAG Configuration

These live under the `rag` section in YAML:

```yaml
rag:
  chromadb_persist_directory: "./data/chromadb"
  chromadb_collection_name: "speeches"
  chunk_size: 2048
  chunk_overlap: 150
  default_top_k: 5
  use_reranking: true
  use_hybrid_search: true
```

Environment variables can override them if necessary (e.g. for a one-off deployment):

```env
CHROMADB_PERSIST_DIRECTORY="./data/chromadb"
CHROMADB_COLLECTION_NAME="speeches"
CHUNK_SIZE="2048"
CHUNK_OVERLAP="150"
DEFAULT_TOP_K="5"
USE_RERANKING="true"
USE_HYBRID_SEARCH="true"
```

### Data Directories

Configured under `paths` in YAML:

```yaml
paths:
  data_root_directory: "./data"
  speeches_directory: "./data/Donald Trump Rally Speeches"
```

### API Settings

These are grouped under the `api` section in YAML:

```yaml
api:
  host: "0.0.0.0"
  port: 8000
  reload: true
  cors_origins:
    - "*"
```

In production (e.g. AWS) you might use something like:

```yaml
# configs/production.yaml
api:
  host: "0.0.0.0"
  port: 8000
  reload: false
  cors_origins:
    - "https://your-domain.com"
```

## Environment-Specific Configs

### Development

```yaml
# configs/development.yaml
environment: development
log_level: DEBUG
api:
  host: "0.0.0.0"
  port: 8000
  reload: true
  cors_origins:
    - "*"
```

### Production example

```yaml
# configs/production.yaml
environment: production
log_level: INFO
app_name: "Trump Speeches NLP Chatbot API"

api:
  host: "0.0.0.0"
  port: 8000
  reload: false
  cors_origins:
    - "https://your-domain.com"
```

## Using Configuration in Code

### Accessing Settings

```python
from src.config.settings import get_settings

settings = get_settings()

# Access values from nested sections
print(settings.llm.provider)
print(settings.rag.chunk_size)
print(settings.log_level)
```

### Type-Safe Access

```python
# All settings are type-checked
settings.rag.chunk_size              # int
settings.llm.temperature             # float
settings.rag.use_reranking           # bool
settings.llm.provider                # Literal["gemini", "openai", "anthropic", "none"]
```

### Helper Methods

```python
# Check if LLM is configured
if settings.is_llm_configured():
  api_key = settings.get_llm_api_key()
  model = settings.get_llm_model_name()

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
1. `.env` file exists in project root (for secrets/overrides)
2. A YAML config exists at `configs/<ENVIRONMENT>.yaml` (or `configs/development.yaml` by default)
3. File encoding is UTF-8
4. No syntax errors in `.env` or YAML files

### Invalid configuration

Check logs at startup:
```
ERROR: ValidationError: 1 validation error for Settings
  Invalid log level. Must be one of: DEBUG, INFO, WARNING, ERROR, CRITICAL
```

### API key issues

```bash
# Check if API key is set
python -c "from src.config.settings import get_settings; print(get_settings().get_llm_api_key())"
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
