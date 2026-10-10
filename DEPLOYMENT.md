# ERC Analyzer — Complete Production Deployment Guide

This guide walks you through deploying the **ERC Analyzer (Enterprise Regulatory Compliance Intelligence RAG Engine)** to the cloud or on your own server.

---

## Architecture Requirements

To run in production, ERC Analyzer requires:
1. **Python 3.11 Web Service**: FastAPI + PyTorch CPU + LangChain + Sentence-Transformers + Uvicorn
2. **PostgreSQL 15+ Database**: Must have the `pgvector` extension enabled (`CREATE EXTENSION IF NOT EXISTS vector;`)
3. **Google Gemini API Key**: For embedding generation (`models/gemini-embedding-001`) and compliant answer synthesis (`gemini-3.8-flash` / fallbacks)

---

## Deployment Options

Choose the option that best fits your needs:

| Option | Ease | Cost | Best For |
|---|---|---|---|
| **Option 1: Render (Blueprint)** | ⭐⭐⭐⭐⭐ Easiest (1-Click) | Free tier available / ~$7/mo | Fast, fully managed cloud deployment with automatic PostgreSQL |
| **Option 2: Railway** | ⭐⭐⭐⭐ Very Easy | Free trial / usage-based | Fast deployment with GitHub integration and managed Postgres |
| **Option 3: Docker Compose** | ⭐⭐⭐⭐ Simple (Single Command) | Depends on server (or free local) | Any VPS (DigitalOcean, AWS EC2, Hetzner, Linode) or local server |
| **Option 4: Cloud DB (Supabase/Neon) + Cloud Run / Fly.io** | ⭐⭐⭐ Moderate | Free tiers available | High scalability with decoupled database |

---

## Step 0: Initialize Git & Push to GitHub (Required for Cloud Platforms)

If you haven't published your code to GitHub yet, run these commands in your project terminal:

```bash
# 1. Initialize git
git init

# 2. Stage all files (sensitive files like .env and virtualenv are ignored by .gitignore)
git add .

# 3. Commit
git commit -m "Initial production deployment setup"

# 4. Link to your GitHub repository (replace with your repo URL)
git branch -M main
git remote add origin https://github.com/<YOUR_USERNAME>/<YOUR_REPO_NAME>.git

# 5. Push to GitHub
git push -u origin main
```

---

## Option 1: 1-Click Cloud Deployment via Render (Recommended)

The repository includes a ready-to-use [`render.yaml`](./render.yaml) blueprint that automatically provisions both the **Web Service** and a **PostgreSQL database with pgvector**.

### Instructions:
1. Go to [Render Dashboard](https://dashboard.render.com/) and sign up or log in.
2. Click **New +** in the top navigation bar and select **Blueprint**.
3. Connect your GitHub account and select your `ERC Analyzer` repository.
4. Render will detect `render.yaml` and configure:
   - **`erc-analyzer-web`**: Docker Web Service running FastAPI & the modern frontend.
   - **`erc-postgres`**: Managed PostgreSQL database with automatic connection credentials.
5. In the configuration screen, provide your secret:
   - `GOOGLE_API_KEY`: Your Google AI Studio API key (`AQ.Ab8...`).
6. Click **Apply**.
7. Render will build the Docker container and launch the database. Within a few minutes, your application will be live at `https://erc-analyzer-web.onrender.com`!

---

## Option 2: Cloud Deployment via Railway

1. Go to [Railway.app](https://railway.app/) and sign in with GitHub.
2. Click **New Project** → **Deploy from GitHub repo**.
3. Select your `ERC Analyzer` repository.
4. In your project canvas, click **+ New** → **Database** → **Add PostgreSQL**.
5. Click on the PostgreSQL service, go to **Connect**, and copy the **DATABASE_URL** (or **Postgres Connection URL**).
6. Click on your web application service, go to **Variables**, and add:
   - `DATABASE_URL`: `${{Postgres.DATABASE_URL}}` (or paste the connection string)
   - `GOOGLE_API_KEY`: Your Google Gemini API key
   - `PORT`: `8000`
7. In the PostgreSQL service query console, verify or enable pgvector:
   ```sql
   CREATE EXTENSION IF NOT EXISTS vector;
   ```
8. Railway will automatically build the `Dockerfile` and deploy your app with a public `*.up.railway.app` URL.

---

## Option 3: Docker Compose (Self-Hosted on VPS or Local Machine)

If you have a Linux server (Ubuntu/Debian on AWS EC2, DigitalOcean Droplet, Hetzner, etc.) with Docker installed, you can deploy the complete stack with one single command.

### 1. Ensure Docker & Docker Compose are installed:
```bash
docker --version
docker compose version
```

### 2. Set your environment variables:
Create a `.env` file in the root directory:
```bash
GOOGLE_API_KEY=your_google_api_key_here
POSTGRES_DB=erc_analyzer
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_secure_password
POSTGRES_PORT=5432
```

### 3. Launch the full stack:
```bash
docker compose up -d --build
```

### 4. Verify status:
```bash
# Check running containers
docker compose ps

# Check logs
docker compose logs -f

# Test the health check endpoint
curl http://localhost:8000/health
```

The web application is now accessible at `http://<your-server-ip>:8000`.

---

## Option 4: Decoupled Setup (Supabase / Neon DB + Any Web Host)

If you already use a cloud PostgreSQL provider like [Supabase](https://supabase.com/) or [Neon](https://neon.tech/):

1. **Enable pgvector** in your Supabase SQL editor:
   ```sql
   CREATE EXTENSION IF NOT EXISTS vector;
   ```
2. Copy your database connection string:
   ```
   postgresql://postgres:[PASSWORD]@[HOST]:[PORT]/postgres
   ```
3. Set the following environment variable on your web host (Render, Fly.io, Google Cloud Run, AWS ECS):
   ```
   DATABASE_URL=postgresql://postgres:[PASSWORD]@[HOST]:[PORT]/postgres
   GOOGLE_API_KEY=your_gemini_key
   ```
4. Deploy the `Dockerfile`. The app automatically normalizes the connection URL and bootstraps tables and HNSW indexes on startup.

---

## Post-Deployment Verification Checklist

Once deployed, run these quick checks:

- [ ] **Health Check**: Open `https://<YOUR_DEPLOYED_URL>/health` — it should return:
  ```json
  {"status":"healthy","database":"connected","pgvector":"ready"}
  ```
- [ ] **Swagger API Docs**: Open `https://<YOUR_DEPLOYED_URL>/docs` to view the interactive OpenAPI documentation.
- [ ] **Frontend Web Application**: Open `https://<YOUR_DEPLOYED_URL>/` — verify the Obsidian glassmorphic UI, live PostgreSQL telemetry dot (`Connected (pgvector)`), and document vault.
- [ ] **PDF Ingestion Test**: Upload a test PDF (e.g. `STI-BRC-CH-01-Board-Risk-Committee-Charter.pdf`) and confirm status shows `COMPLETED`.
- [ ] **Query Test**: Submit a compliance question (e.g., *"What is the primary role of the Committee?"*) and confirm you receive a grounded answer with page-level citations.
