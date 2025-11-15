# Production Logging Guide

This project implements professional logging with automatic environment detection, structured output, and cloud-ready JSON formatting for production deployment.

## Architecture

### Dual-Format Logging System

The application uses `src/logging_config.py` to provide:

- **Development Mode**: Colored, timestamped, human-readable logs
- **Production Mode**: Structured JSON logs for cloud platforms (Azure, Docker)
- **Automatic Detection**: Uses `ENVIRONMENT` setting to choose format

### Why Professional Logging?

- ✅ **Structured Data**: JSON format for log aggregation tools
- ✅ **Environment-Aware**: Different formats for dev/prod automatically
- ✅ **Cloud-Native**: Works with Azure Application Insights, CloudWatch, etc.
- ✅ **Performance**: Efficient formatting and filtering
- ✅ **Observability**: Proper levels, timestamps, module names
- ✅ **Thread-Safe**: Works correctly in async/concurrent code

## Log Levels

| Level    | Use Case | Example |
|----------|----------|---------|
| DEBUG    | Detailed diagnostic information | `logger.debug(f"Query embedding: {embedding[:5]}")` |
| INFO     | Important application events | `logger.info("RAG service initialized with 1082 chunks")` |
| WARNING  | Unexpected but recoverable situations | `logger.warning("LLM not configured, using extraction")` |
| ERROR    | Application errors requiring attention | `logger.error(f"Failed to generate answer: {e}")` |
| CRITICAL | System-critical failures | `logger.critical("Vector database connection lost")` |

## Logging Formats

### Development Format (Colored)

Automatically enabled when `ENVIRONMENT=development`:

```
2025-11-04 12:34:56 | INFO     | src.api              | Application startup complete
2025-11-04 12:34:57 | DEBUG    | src.rag_service      | Performing hybrid search
2025-11-04 12:34:58 | ERROR    | src.llm_service      | Gemini API error: rate limit
```

**Features**:
- ANSI color coding by level (INFO=green, ERROR=red, etc.)
- Timestamp in readable format
- Module name right-aligned
- Human-readable message

### Production Format (JSON)

Automatically enabled when `ENVIRONMENT=production`:

```json
{"timestamp": "2025-11-04 12:34:56", "level": "INFO", "name": "src.api", "message": "Application startup complete"}
{"timestamp": "2025-11-04 12:34:57", "level": "DEBUG", "name": "src.rag_service", "message": "Performing hybrid search"}
{"timestamp": "2025-11-04 12:34:58", "level": "ERROR", "name": "src.llm_service", "message": "Gemini API error: rate limit", "exception": "...traceback..."}
```

**Features**:
- Machine-parseable JSON
- Automatic exception field for errors
- Compatible with Azure Application Insights, CloudWatch, ELK stack
- Efficient for log aggregation

## Configuration

### Environment Variables

Set in `.env` file:

```env
# Choose environment (affects log format)
ENVIRONMENT=development  # colored logs
ENVIRONMENT=production   # JSON logs

# Choose log level
LOG_LEVEL=INFO   # Standard (recommended)
LOG_LEVEL=DEBUG  # Verbose debugging
LOG_LEVEL=ERROR  # Errors only
```

### Logging Module (`src/logging_config.py`)

The centralized logging configuration module provides:

```python
from src.logging_config import configure_logging

# Called automatically by Settings.setup_logging()
configure_logging(
    level="INFO",
    use_json=True,  # or False for colored
    include_uvicorn=True,  # configure uvicorn loggers
)
```

**Key Features**:
- `JsonFormatter`: Structured JSON output
- `ColoredFormatter`: ANSI-colored terminal output
- Automatic third-party logger suppression (chromadb, httpx, etc.)
- ChromaDB telemetry filter (removes noise)
- Uvicorn integration

## Usage Examples

### Basic Logging

```python
import logging

logger = logging.getLogger(__name__)

logger.debug("Detailed debug information")
logger.info("Informational message")
logger.warning("Warning message")
logger.error("Error occurred")
logger.critical("Critical error")
```

### With Context

```python
# Log with context
logger.info(f"Loading {doc_count} documents from {directory}")

# Log with variables
logger.debug(f"Chunk size: {chunk_size}, overlap: {overlap}")

# Log with stack traces
try:
    risky_operation()
except Exception as e:
    logger.error(f"Operation failed: {e}", exc_info=True)
```

### Module-Specific Loggers

Each module should have its own logger:

```python
# src/llm_service.py
import logging
logger = logging.getLogger(__name__)  # Creates 'src.llm_service' logger

# src/rag_service.py  
import logging
logger = logging.getLogger(__name__)  # Creates 'src.rag_service' logger
```

## Configuration

### Via Environment Variable

Set in `.env`:
```env
LOG_LEVEL="DEBUG"  # Show all messages
LOG_LEVEL="INFO"   # Show info and above (default)
LOG_LEVEL="WARNING"  # Show warnings and errors only
LOG_LEVEL="ERROR"  # Show errors only
```

### Via Code

```python
from src.config.settings import get_settings

settings = get_settings()
settings.setup_logging()  # Configures logging based on LOG_LEVEL / ENVIRONMENT
```

### Custom Configuration

```python
import logging

# Configure custom logger
logging.basicConfig(
    level=logging.DEBUG,
    format='%(levelname)-8s %(name)-20s %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
```

## Current Logging in Project

### Startup Sequence

```
INFO     src.config           Application: Trump Speeches NLP Chatbot API v0.1.0
INFO     src.config           Environment: development
INFO     src.config           Log Level: INFO
INFO     src.config           LLM Provider: gemini
INFO     src.config           LLM Model: gemini-2.5-flash
INFO     src.api              Loading sentiment analysis model: ProsusAI/finbert
INFO     src.models           Sentiment model loaded: ProsusAI/finbert
INFO     src.api              Initializing GEMINI LLM service...
DEBUG    src.llm_service      Initializing Gemini LLM with model: gemini-2.5-flash
INFO     src.llm_service      Gemini LLM initialized: model=gemini-2.5-flash, temp=0.3
INFO     src.llm_service      Gemini API connection test successful: OK
INFO     src.api              ✓ LLM service initialized and tested successfully
INFO     src.api              Initializing RAG service...
DEBUG    src.rag_service      Initializing RAG service: collection=speeches
INFO     src.rag_service      Loading embedding model: all-mpnet-base-v2
INFO     src.rag_service      Loading re-ranker model: cross-encoder/ms-marco-MiniLM-L-6-v2
INFO     src.rag_service      RAG service using LLM for answer generation
INFO     src.api              ✓ RAG service initialized with 1082 existing chunks
INFO     src.api              Application startup complete
```

### Query Processing

```
DEBUG    src.rag_service      Generating answer with 5 context chunks using LLM
DEBUG    src.llm_service      Sending prompt to Gemini (length: 3245 chars)
INFO     src.llm_service      Gemini generated answer successfully (length: 234 chars)
DEBUG    src.rag_service      LLM response received (length: 234 chars)
```

### Error Handling

```
ERROR    src.llm_service      Gemini generation failed: API key invalid
Traceback (most recent call last):
  File "src/llm_service.py", line 127, in generate_answer
    response = self.model.generate_content(prompt)
ValueError: Invalid API key
ERROR    src.rag_service      LLM generation failed, falling back to extraction
```

## Log Output Formats

### Default Format

```
LEVEL    MODULE_NAME          MESSAGE
INFO     src.api              Application startup complete
```

### With Timestamps

```python
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(name)-20s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
```

Output:
```
2025-11-04 10:30:45 INFO     src.api              Application startup complete
```

## Suppressing Noisy Loggers

Some third-party libraries are verbose. Suppress them:

```python
# Suppress ChromaDB telemetry errors
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.ERROR)

# Suppress httpx debug messages
logging.getLogger("httpx").setLevel(logging.WARNING)

# Suppress transformers warnings
logging.getLogger("transformers").setLevel(logging.ERROR)
```

This is already configured in `src/core/logging_config.py` and wired via
`Settings.setup_logging()` in `src/config/settings.py`:

```python
def setup_logging(self) -> None:
    """Configure application-wide logging based on settings."""
    from src.core.logging_config import configure_logging

    use_json = self.environment.lower() == "production"

    configure_logging(
        level=self.log_level,
        use_json=use_json,
        include_uvicorn=True,
    )
```

## Production Logging

### Azure App Service

Logs automatically stream to Azure Monitor:

```bash
# View live logs
az webapp log tail --name myapp --resource-group mygroup

# Download logs
az webapp log download --name myapp --resource-group mygroup
```

### Docker Logs

```bash
# View container logs
docker logs mycontainer

# Follow logs
docker logs -f mycontainer

# Save to file
docker logs mycontainer > app.log 2>&1
```

### Log Aggregation

For production, integrate with:
- **Azure Application Insights**
- **AWS CloudWatch**
- **Datadog**
- **Sentry** (for errors)
- **ELK Stack** (Elasticsearch, Logstash, Kibana)

Example with Azure Application Insights:

```python
from opencensus.ext.azure.log_exporter import AzureLogHandler

logger = logging.getLogger(__name__)
logger.addHandler(AzureLogHandler(
    connection_string='InstrumentationKey=your-key'
))
```

## Best Practices

### 1. Use Appropriate Levels

```python
# ❌ Wrong
logger.info(f"Debug variable: {x}")  # Should be DEBUG
logger.error("User clicked button")  # Should be INFO

# ✅ Correct
logger.debug(f"Variable value: {x}")
logger.info("User authentication successful")
logger.warning("Rate limit approaching")
logger.error("Database query failed")
```

### 2. Include Context

```python
# ❌ Not helpful
logger.error("Failed")

# ✅ Helpful
logger.error(f"Failed to load model {model_name}: {e}")
```

### 3. Use exc_info for Tracebacks

```python
# ❌ Manual traceback
try:
    risky_operation()
except Exception as e:
    import traceback
    logger.error(traceback.format_exc())

# ✅ Automatic traceback
try:
    risky_operation()
except Exception as e:
    logger.error(f"Operation failed: {e}", exc_info=True)
```

### 4. Don't Log Sensitive Data

```python
# ❌ Leaks secrets
logger.info(f"Using API key: {api_key}")

# ✅ Safe
logger.info("Using API key: ✓ Configured")
```

### 5. Performance-Conscious

```python
# ❌ Expensive string formatting even if not logged
logger.debug(f"Data: {expensive_computation()}")

# ✅ Only compute if needed
if logger.isEnabledFor(logging.DEBUG):
    logger.debug(f"Data: {expensive_computation()}")
```

## Migration from Print Statements

### Before

```python
print(f"Loading model: {model_name}")
print(f"ERROR: Failed to load: {e}")
import traceback
print(traceback.format_exc())
```

### After

```python
logger.info(f"Loading model: {model_name}")
logger.error(f"Failed to load: {e}", exc_info=True)
```

## Debugging

### Enable DEBUG Logging

```bash
# Temporary
LOG_LEVEL=DEBUG uv run uvicorn src.api:app

# Or edit .env
LOG_LEVEL="DEBUG"
```

### Filter Specific Modules

```python
# Only debug RAG service
logging.getLogger("src.rag_service").setLevel(logging.DEBUG)
logging.getLogger("src.llm_service").setLevel(logging.DEBUG)

# Suppress others
logging.getLogger("src.models").setLevel(logging.WARNING)
```

### Debug Startup Issues

```bash
# Enable all debug output
LOG_LEVEL=DEBUG uv run uvicorn src.api:app --reload
```

## Further Reading

- [Python Logging HOWTO](https://docs.python.org/3/howto/logging.html)
- [Python Logging Cookbook](https://docs.python.org/3/howto/logging-cookbook.html)
- [Structlog (advanced structured logging)](https://www.structlog.org/)
