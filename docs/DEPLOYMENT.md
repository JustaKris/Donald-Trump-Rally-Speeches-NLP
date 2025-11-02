# FastAPI Deployment Guide

This guide covers deploying the Trump Rally Speeches NLP API to various platforms.

## Table of Contents

- [Local Development](#local-development)
- [Docker Deployment](#docker-deployment)
- [Render Deployment](#render-deployment)
- [Azure App Service Deployment](#azure-app-service-deployment)
- [CI/CD with GitHub Actions](#cicd-with-github-actions)
- [Environment Variables](#environment-variables)

---

## Local Development

### Running Without Docker

1. **Install dependencies:**

   ```powershell
   uv sync
   ```

2. **Run the API:**

   ```powershell
   uv run uvicorn src.api:app --reload
   ```

3. **Access the application:**
   - Frontend: http://localhost:8000
   - API Docs: http://localhost:8000/docs
   - ReDoc: http://localhost:8000/redoc

---

## Docker Deployment

### Build and Run with Docker

1. **Build the Docker image:**

   ```powershell
   docker build -t nlp-text-analysis-api .
   ```

2. **Run the container:**

   ```powershell
   docker run --rm -it -p 8000:8000 --name nlp-api nlp-text-analysis-api
   ```

3. **View logs:**

   ```powershell
   docker logs -f nlp-api
   ```

### Using Docker Compose

**Production mode:**

```powershell
docker-compose up -d
```

**Development mode (with hot reload):**

```powershell
+
```

**Stop services:**

```powershell
docker-compose down
```

---

## Render Deployment

[Render](https://render.com) offers free hosting with automatic deployments using Docker images.

### Deployment Strategy

This project uses a **Docker-based deployment** approach for Render:

1. GitHub Actions builds and pushes Docker images to Docker Hub
2. Render pulls and deploys the pre-built images
3. Benefits: Faster deployments, consistent environments, easier rollbacks

### Prerequisites

1. **Docker Hub Account** (free)
   - Sign up at <https://hub.docker.com>
   - Create a repository named `trump-speeches-nlp-api`

2. **Render Account** (free)
   - Sign up at <https://render.com>

3. **GitHub Repository** connected to your account

### Step 1: Configure GitHub Secrets

Add these secrets to your GitHub repository (Settings → Secrets and variables → Actions):

1. **`DOCKERHUB_USERNAME`** - Your Docker Hub username
2. **`DOCKERHUB_TOKEN`** - Docker Hub access token
   - Go to Docker Hub → Account Settings → Security → New Access Token
   - Copy the token and save it as a GitHub secret

### Step 2: Update render.yaml

Edit `.render/render.yaml` and replace `<your-dockerhub-username>` with your actual Docker Hub username:

```yaml
image:
  url: docker.io/your-username/trump-speeches-nlp-api:latest
```

### Step 3: Deploy to Render

**Option A: Using Blueprint (Recommended)**

1. Go to Render Dashboard → "Blueprints"
2. Click "New Blueprint Instance"
3. Connect your GitHub repository
4. Render will detect `.render/render.yaml` and configure everything automatically
5. Click "Apply" to create the service

**Option B: Manual Configuration**

1. Go to Render Dashboard → "New +" → "Web Service"
2. Choose "Deploy an existing image from a registry"
3. Configure:
   - **Image URL:** `docker.io/your-username/trump-speeches-nlp-api:latest`
   - **Name:** `trump-speeches-nlp-api`
   - **Plan:** Free
4. Add environment variable:
   - `PORT` = `8000`
5. Set Health Check Path: `/health`
6. Click "Create Web Service"

### Step 4: Trigger Deployment

Push to `main` branch or manually trigger the workflow:

```powershell
git push origin main
```

This will:
1. Build the Docker image
2. Push it to Docker Hub
3. Render automatically detects the new image and deploys it

### Accessing Your Render App

Your API will be available at: `https://trump-speeches-nlp-api.onrender.com`

**Note:** Free tier apps spin down after 15 minutes of inactivity. First request may take 30-60 seconds to wake the service.

### Monitoring & Logs

- **Logs:** Render Dashboard → Your Service → Logs tab
- **Metrics:** Render Dashboard → Your Service → Metrics tab
- **Manual Deploy:** Render Dashboard → Your Service → Manual Deploy button

### Upgrading from Free Tier

For better performance:
- **Starter Plan:** $7/month (512 MB RAM, 0.5 CPU)
- No cold starts
- Faster response times
- Better for production use

---

## Azure App Service Deployment

### Deployment Strategy

This project supports two Azure deployment approaches:

1. **Azure Container Registry (ACR)** - Recommended for enterprise/production (default)
2. **Docker Hub** - Alternative for simpler setups or cross-platform deployments

Both use the same GitHub Actions workflow with conditional logic.

### Prerequisites

- Azure account (free tier available)
- Azure CLI installed: <https://docs.microsoft.com/en-us/cli/azure/install-azure-cli>
- Docker Hub account (if using Docker Hub approach)

### Option 1: Deploy with Azure Container Registry (Recommended)

#### Step 1: Create Azure Resources

```powershell
# Login to Azure
az login

# Create a resource group
az group create --name trump-nlp-rg --location eastus

# Create Azure Container Registry
az acr create `
  --resource-group trump-nlp-rg `
  --name trumpnlpacr `
  --sku Basic `
  --admin-enabled true

# Create an App Service plan (B1 or F1 tier)
az appservice plan create `
  --name trump-nlp-plan `
  --resource-group trump-nlp-rg `
  --sku B1 `
  --is-linux

# Create a web app for containers
az webapp create `
  --resource-group trump-nlp-rg `
  --plan trump-nlp-plan `
  --name trump-speeches-nlp `
  --deployment-container-image-name trumpnlpacr.azurecr.io/trump-speeches-nlp-api:latest

# Configure port
az webapp config appsettings set `
  --resource-group trump-nlp-rg `
  --name trump-speeches-nlp `
  --settings WEBSITES_PORT=8000

# Enable ACR integration
az webapp config container set `
  --resource-group trump-nlp-rg `
  --name trump-speeches-nlp `
  --docker-custom-image-name trumpnlpacr.azurecr.io/trump-speeches-nlp-api:latest `
  --docker-registry-server-url https://trumpnlpacr.azurecr.io
```

#### Step 2: Configure GitHub Secrets for ACR

Add these secrets to your GitHub repository:

1. **`AZURE_CREDENTIALS`** - Service principal credentials

   ```powershell
   az ad sp create-for-rbac --name "github-actions-trump-nlp" --sdk-auth --role contributor --scopes /subscriptions/{subscription-id}/resourceGroups/trump-nlp-rg
   ```

   Copy the entire JSON output and save as secret.

2. **`AZURE_REGISTRY_LOGIN_SERVER`** - e.g., `trumpnlpacr.azurecr.io`

3. **`AZURE_REGISTRY_USERNAME`** and **`AZURE_REGISTRY_PASSWORD`**

   ```powershell
   az acr credential show --name trumpnlpacr
   ```

4. **`AZURE_WEBAPP_NAME`** - e.g., `trump-speeches-nlp`

#### Step 3: Deploy

Push to `main` branch:

```powershell
git push origin main
```

GitHub Actions will automatically:
1. Build Docker image
2. Push to Azure Container Registry
3. Deploy to Azure Web App
4. Run health checks

---

### Option 2: Deploy with Docker Hub

This approach shares the same Docker images used for Render deployment.

#### Step 1: Create Azure Resources

```powershell
# Login to Azure
az login

# Create a resource group
az group create --name trump-nlp-rg --location eastus

# Create an App Service plan
az appservice plan create `
  --name trump-nlp-plan `
  --resource-group trump-nlp-rg `
  --sku B1 `
  --is-linux

# Create a web app using Docker Hub image
az webapp create `
  --resource-group trump-nlp-rg `
  --plan trump-nlp-plan `
  --name trump-speeches-nlp `
  --deployment-container-image-name docker.io/your-username/trump-speeches-nlp-api:latest

# Configure port
az webapp config appsettings set `
  --resource-group trump-nlp-rg `
  --name trump-speeches-nlp `
  --settings WEBSITES_PORT=8000
```

#### Step 2: Configure GitHub Secrets for Docker Hub

You'll need the same Docker Hub secrets as for Render:

1. **`DOCKERHUB_USERNAME`**
2. **`DOCKERHUB_TOKEN`**
3. **`AZURE_CREDENTIALS`** (same as Option 1)
4. **`AZURE_WEBAPP_NAME`**

#### Step 3: Deploy with Docker Hub

Manually trigger the workflow:

1. Go to GitHub → Actions → "Deploy to Azure"
2. Click "Run workflow"
3. Check "Use Docker Hub instead of ACR"
4. Click "Run workflow"

---

### Accessing Your Azure App

Your API will be available at: `https://trump-speeches-nlp.azurewebsites.net`

### Monitoring and Logs

```powershell
# Stream logs
az webapp log tail --resource-group trump-nlp-rg --name trump-speeches-nlp

# View metrics
az monitor metrics list `
  --resource /subscriptions/{subscription-id}/resourceGroups/trump-nlp-rg/providers/Microsoft.Web/sites/trump-speeches-nlp `
  --metric-names Requests,ResponseTime,Http5xx

# Open in Azure Portal
az webapp browse --resource-group trump-nlp-rg --name trump-speeches-nlp
```

### Updating the Deployment

Deployments are automatic on push to `main`. To manually update:

```powershell
# Trigger GitHub Actions workflow manually
# Or restart the web app to pull latest image
az webapp restart --resource-group trump-nlp-rg --name trump-speeches-nlp
```

---

## CI/CD with GitHub Actions

This project includes automated CI/CD pipelines using GitHub Actions. The workflows are split into multiple files for better organization:

### Workflow Files

All workflows are located in `.github/workflows/`:

- **`ci.yml`** - Tests & Linting
  - Runs on: All pushes and PRs to `main`, `develop`, `feature/*`
  - Jobs: Unit tests, integration tests, code quality checks (flake8, black, isort, mypy)
  - Python versions tested: 3.11, 3.12, 3.13

- **`security.yml`** - Security Scans
  - Runs on: All pushes and PRs, plus weekly schedule (Mondays 9 AM UTC)
  - Jobs: Dependency vulnerability scanning (pip-audit), code security analysis (bandit)

- **`deploy-render.yml`** - Render Deployment
  - Runs on: Push to `main` branch only
  - Jobs: Build Docker image, push to Docker Hub, test container
  - Render auto-deploys when new images are detected

- **`deploy-azure.yml`** - Azure Deployment
  - Runs on: Push to `main` branch (manual trigger also available)
  - Jobs: Build & push to ACR/Docker Hub, deploy to Azure Web App, health check
  - Supports both ACR and Docker Hub registries

### Deployment Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Push to main branch                     │
└────────────┬────────────────────────────────────────────────┘
             │
             ├──────────────────┐
             │                  │
             ▼                  ▼
    ┌────────────────┐   ┌──────────────────┐
    │  Build Docker  │   │  Build Docker    │
    │  Image         │   │  Image           │
    └────────┬───────┘   └────────┬─────────┘
             │                    │
             ▼                    ▼
    ┌────────────────┐   ┌──────────────────┐
    │  Push to       │   │  Push to ACR     │
    │  Docker Hub    │   │  (or Docker Hub) │
    └────────┬───────┘   └────────┬─────────┘
             │                    │
             ▼                    ▼
    ┌────────────────┐   ┌──────────────────┐
    │  Render        │   │  Azure Web App   │
    │  Auto-Deploy   │   │  Deploy          │
    └────────────────┘   └──────────────────┘
```

### Setting Up GitHub Secrets

#### Required for All Deployments

1. **`DOCKERHUB_USERNAME`** - Your Docker Hub username
2. **`DOCKERHUB_TOKEN`** - Docker Hub access token
   - Create at: Docker Hub → Account Settings → Security → New Access Token

#### Required for Azure Deployment

3. **`AZURE_CREDENTIALS`** - Service principal credentials (JSON format)

   ```powershell
   az ad sp create-for-rbac --name "github-actions-trump-nlp" --sdk-auth --role contributor --scopes /subscriptions/{subscription-id}/resourceGroups/trump-nlp-rg
   ```

4. **`AZURE_WEBAPP_NAME`** - Your Azure Web App name (e.g., `trump-speeches-nlp`)

#### Required for Azure Container Registry (Optional)

5. **`AZURE_REGISTRY_LOGIN_SERVER`** - ACR login server (e.g., `trumpnlpacr.azurecr.io`)
6. **`AZURE_REGISTRY_USERNAME`** - ACR admin username
7. **`AZURE_REGISTRY_PASSWORD`** - ACR admin password

   Get ACR credentials:

   ```powershell
   az acr credential show --name trumpnlpacr
   ```

### Workflow Triggers

| Workflow | Automatic Trigger | Manual Trigger | When to Use |
|----------|------------------|----------------|-------------|
| **CI** | Push/PR to main, develop, feature/* | ✅ Yes | Testing code changes |
| **Security** | Push/PR + Weekly (Mon 9AM) | ✅ Yes | Regular security audits |
| **Deploy Render** | Push to main | ✅ Yes | Deploy to Render |
| **Deploy Azure** | Push to main | ✅ Yes (with options) | Deploy to Azure |

### Manual Deployment Triggers

**For Azure:**

1. Go to GitHub → Actions → "Deploy to Azure"
2. Click "Run workflow"
3. Select options:
   - **Environment:** production/staging
   - **Use Docker Hub:** true/false (ACR is default)
4. Click "Run workflow"

**For Render:**

1. Go to GitHub → Actions → "Deploy to Render"
2. Click "Run workflow"
3. Click "Run workflow" to confirm

### Viewing Workflow Results

- Navigate to your repository → Actions tab
- Click on any workflow run to see detailed logs
- Failed workflows will show which step failed and why
- Deployment summaries are available in each workflow run

### Best Practices

1. **Always test locally** before pushing to `main`
2. **Use feature branches** for development
3. **Review CI results** before merging PRs
4. **Monitor deployments** after pushing to `main`
5. **Check security scans** weekly
6. **Use manual triggers** for testing deployment changes

---

## Environment Variables

### Required Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `PORT` | Port to run the application | `8000` |
| `PYTHONUNBUFFERED` | Disable Python output buffering | `1` |

### Optional Variables

| Variable | Description |
|----------|-------------|
| `PYTHON_VERSION` | Python version (for Render) | `3.12.0` |
| `WEBSITES_PORT` | Port for Azure App Service | `8000` |

### RAG-Specific Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `CHROMADB_PERSIST_DIR` | Directory for ChromaDB persistence | `./data/chromadb` |
| `RAG_COLLECTION_NAME` | Name of the vector collection | `speeches` |
| `RAG_CHUNK_SIZE` | Text chunk size for embeddings | `500` |
| `RAG_CHUNK_OVERLAP` | Overlap between chunks | `50` |
| `EMBEDDING_MODEL` | sentence-transformers model | `all-MiniLM-L6-v2` |

**Note:** ChromaDB data is persisted to disk. Ensure the persist directory is included in volume mounts for Docker deployments to maintain indexed documents across restarts.

---

## Performance Tips

### For Free Tiers

1. **Render Free Tier:**
   - Apps sleep after 15 min of inactivity
   - Use a health check service like [cron-job.org](https://cron-job.org) to keep it awake
   - Expect 30-60s cold start time
   - **RAG Impact:** First request will also load embedding model (~80MB) + index documents

2. **Azure Free Tier (F1):**
   - Limited to 60 CPU minutes/day
   - 1 GB RAM limit
   - Consider B1 tier ($13/month) for better performance
   - **RAG Requirements:** Minimum 2GB RAM recommended with RAG enabled

### Optimizing Docker Image

- Use multi-stage builds (already implemented)
- Remove unnecessary data files
- Use `.dockerignore` to exclude notebooks and docs
- **RAG Models:** Pre-download models during build to reduce startup time

### RAG Performance Considerations

**Memory Requirements:**
- **Without RAG:** ~1.5GB RAM
- **With RAG:** ~2.5GB RAM (includes embeddings model + vector store)
- **Recommended:** 4GB RAM for production with concurrent users

**Storage Requirements:**
- **Base application:** ~1GB
- **ChromaDB index:** ~50-100MB per 10,000 chunks
- **Embedding model:** ~80MB

**Startup Time:**
- **Without RAG:** 10-15 seconds
- **With RAG:** 30-60 seconds (model loading + document indexing)
- **Tip:** Documents are auto-indexed on first startup if collection is empty

**Query Performance:**
- **Semantic search:** ~50-200ms per query
- **Question answering:** ~200-500ms (includes search + answer generation)
- **Tip:** Results improve with more indexed documents

---

## Troubleshooting

### Docker Build Fails

```powershell
# Clean Docker cache and rebuild
docker system prune -a
docker build --no-cache -t trump-speeches-nlp-api .
```

### API Returns 503

- Model may still be loading (can take 30-60s on first request)
- Check logs for errors
- Ensure sufficient memory (minimum 1GB recommended)

### Azure Deployment Issues

```powershell
# Check app logs
az webapp log tail --resource-group trump-nlp-rg --name trump-speeches-nlp

# Restart the app
az webapp restart --resource-group trump-nlp-rg --name trump-speeches-nlp
```

### Render Deployment Issues

- Check build logs in Render dashboard
- Verify `.render/render.yaml` configuration
- Ensure all dependencies are in `pyproject.toml`

---

## Cost Comparison

| Platform | Free Tier | Paid Tier | Best For |
|----------|-----------|-----------|----------|
| **Render** | ✅ Yes (with limitations) | $7/month starter | Quick demos, portfolio |
| **Azure** | ✅ Yes (F1: 60 min/day) | $13/month (B1) | Enterprise, Azure ecosystem |
| **Railway** | ✅ $5 free credit | Pay-as-you-go | Simple projects |
| **Fly.io** | ✅ Free allowance | Pay-as-you-go | Global deployment |

---

## Next Steps

After deployment:

1. **Test the API:** Use the `/health` endpoint to verify
2. **Update README:** Add your live demo link
3. **Monitor performance:** Check logs and metrics
4. **Set up alerts:** Configure uptime monitoring
5. **Add custom domain:** (optional) Configure DNS

---

## Support

For issues or questions:

- Check the [main README](../README.md)
- Review API documentation at `/docs`
- Open an issue on GitHub
