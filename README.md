# Conversational RAG App

Chat with your PDFs. Upload documents, ask questions, get context-aware answers — a Retrieval-Augmented Generation stack with FastAPI, LangChain, Groq (Llama 3.1), Qdrant vector search, and a Streamlit UI, fully Dockerized.

## How It Works

```
PDF upload → extract text (pypdf) → chunk → embed (FastEmbed) → store in Qdrant
                                                                      │
User question → embed → top-3 vector search → context + history → Groq LLM → answer
```

## Features

- **PDF ingestion** — 500-char overlapping chunks, 384-dim embeddings
- **Vector search** — Qdrant with cosine similarity, auto-created collection
- **Conversational memory** — per-session chat history via LangChain
- **Groq LLM** — `llama-3.1-8b-instant`
- **Streamlit UI** — sidebar with session control, upload, and chat
- **Fully containerized** — one `docker compose up`

## Tech Stack

| Component | Tech |
|-----------|------|
| Backend | FastAPI + Uvicorn |
| LLM | LangChain + Groq (Llama 3.1 8B) |
| Vector DB | Qdrant |
| Embeddings | FastEmbed (`BAAI/bge-small-en-v1.5`) |
| Frontend | Streamlit |
| PDF parsing | pypdf |
| Containers | Docker + Docker Compose |

## Getting Started

### Prerequisites

- Docker + Docker Compose
- A [Groq API key](https://console.groq.com)

### Run (recommended)

```bash
git clone https://github.com/DONCHINGUEGUIM/conversational_rag_app.git
cd conversational_rag_app
echo "GROQ_API_KEY=your_key_here" > .env
docker compose up --build
```

| Service | URL |
|---------|-----|
| Streamlit UI | http://localhost:8501 |
| FastAPI backend | http://localhost:8000 |
| Qdrant dashboard | http://localhost:6333/dashboard |

### Run locally (without Docker)

```bash
pip install -r requirements.txt
export GROQ_API_KEY=your_key_here
export QDRANT_HOST=localhost

# terminal 1 — backend
uvicorn main:app --reload --port 8000

# terminal 2 — UI
streamlit run ui.py
```

## Configuration

| Env var | Default | Description |
|---------|---------|-------------|
| `GROQ_API_KEY` | — | required, Groq API key |
| `QDRANT_HOST` | `qdrant` (Docker) | Qdrant hostname |
| `QDRANT_PORT` | `6333` | Qdrant port |
| `BACKEND_URL` | `http://backend:8000` | UI → backend base URL |

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Liveness check |
| `POST` | `/upload` | Upload a PDF, chunk + embed into Qdrant |
| `POST` | `/chat` | `{session_id, message}` → RAG answer |

`/chat` body example:

```json
{
  "session_id": "user-1",
  "message": "What is this document about?"
}
```

## Project Structure

```
├── main.py              # FastAPI backend (upload + chat + Qdrant)
├── ui.py                # Streamlit frontend
├── requirements.txt
├── Dockerfile
└── docker-compose.yml   # qdrant + backend + frontend
```

## Author

**Donchi Ngueguim** — [github.com/DONCHINGUEGUIM](https://github.com/DONCHINGUEGUIM)

## License

MIT
