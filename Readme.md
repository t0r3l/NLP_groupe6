# RAG Historian - African Civilizations

A Retrieval-Augmented Generation (RAG) system for answering questions about pre-colonial African civilizations using Wikipedia data.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  NETWORK: rag-frontend (bridge)                                             │
│  Description: Réseau exposé à l'hôte pour l'accès externe aux interfaces    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│    ┌──────────────────┐                      ┌──────────────────┐           │
│    │   Streamlit App  │                      │     MLflow UI    │           │
│    │   localhost:8501 │                      │   localhost:5000 │           │
│    │                  │                      │                  │           │
│    │  Interface RAG   │                      │  Suivi des       │           │
│    │  utilisateur     │                      │  expériences     │           │
│    └────────┬─────────┘                      └────────┬─────────┘           │
│             │                                         │                     │
└─────────────┼─────────────────────────────────────────┼─────────────────────┘
              │                                         │
              │  ┌──────────────────────────────────────┘
              │  │
┌─────────────┼──┼────────────────────────────────────────────────────────────┐
│  NETWORK: rag-backend (bridge, internal)                                    │
│  Description: Réseau interne pour la communication inter-services           │
│               Non accessible depuis l'hôte                                  │
├─────────────┼──┼────────────────────────────────────────────────────────────┤
│             │  │                                                            │
│             │  │         ┌──────────────────┐                               │
│             └──┴────────►│     ChromaDB     │                               │
│                          │   (port 8000)    │                               │
│                          │                  │                               │
│                          │  Base vectorielle│                               │
│                          │  persistante     │                               │
│                          └────────┬─────────┘                               │
│                                   │                                         │
│                          ┌────────┴─────────┐                               │
│                          │      VOLUME      │                               │
│                          │   chroma_data    │                               │
│                          │                  │                               │
│                          │  Embeddings +    │                               │
│                          │  Métadonnées     │                               │
│                          └──────────────────┘                               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Légende des réseaux

| Réseau | Type | Accès externe | Rôle |
|--------|------|---------------|------|
| `rag-frontend` | bridge | ✅ Oui (ports 8501, 5000) | Interfaces utilisateur (Streamlit, MLflow) |
| `rag-backend` | bridge, internal | ❌ Non | Communication sécurisée entre services |

## Quick Start

### Option 1: Local Development (without Docker)

```bash
# Install dependencies
uv sync

# Set Groq API key from config.yml
export GROQ_API_KEY=$(grep 'groq_key:' config.yml | cut -d' ' -f2)

# Launch Streamlit app
uv run streamlit run app.py
```

### Option 2: Docker Deployment

```bash
# 1. Create .env file from config.yml
echo "GROQ_API_KEY=$(grep 'groq_key:' config.yml | cut -d' ' -f2)" > .env

# 2. Build and start all services
docker-compose up -d --build

# 3. Access the services:
#    - Streamlit App: http://localhost:8501
#    - MLflow UI:     http://localhost:5000
#    - ChromaDB:      http://localhost:8000
```

## Docker Architecture

| Container | Image | Port | Network | Description |
|-----------|-------|------|---------|-------------|
| `rag-historian-app` | Custom (Dockerfile) | 8501 | frontend, backend | Streamlit RAG interface |
| `rag-historian-chromadb` | chromadb/chroma | 8000 | backend | Vector database |
| `rag-historian-mlflow` | ghcr.io/mlflow/mlflow | 5000 | frontend, backend | Experiment tracking |

### Volumes
- **chroma_data**: Persistent storage for vector embeddings

## Docker Commands

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f streamlit-app

# Stop all services
docker-compose down

# Stop and remove volumes (clean slate)
docker-compose down -v

# Rebuild after code changes
docker-compose up -d --build streamlit-app
```

## Evaluation

```bash
uv run python
```

```python
from src_rag import evaluate

# Run retrieval evaluation
evaluate.run_evaluate_retrieval(config={"model": {"chunk_size": 128, "overlap": 12}})

# Run reply evaluation
evaluate.run_evaluate_reply(config={"model": {"chunk_size": 256, "small2big": True, "add_metadata": True}})
```

## Configuration

Best performing configuration:
- **chunk_size**: 256
- **small2big**: True
- **add_metadata**: True
- **embedding**: miniLM (all-MiniLM-L6-v2)

## Project Structure

```
├── app.py                 # Streamlit interface
├── src_rag/
│   ├── models.py          # RAG implementation
│   ├── evaluate.py        # Evaluation utilities
│   └── vector_store.py    # Vector store abstraction
├── data/
│   └── raw/
│       ├── wikipedia_pages/   # Wikipedia text files
│       └── questions.csv      # Evaluation questions
├── Dockerfile             # App container definition
├── docker-compose.yml     # Multi-container orchestration
└── config.yml             # Application configuration
```
