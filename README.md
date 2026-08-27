# RAG Learning Project

A transparent FastAPI project for learning how embeddings, semantic search, and
retrieval-augmented generation work without hiding the mechanics behind a RAG framework.

The project includes:

- OpenAI chat and embeddings calls;
- cosine similarity implemented from scratch;
- an educational in-memory vector store;
- semantic document search;
- grounded RAG answers with returned sources;
- Swagger UI and automated tests.

> This is a learning project. Stored documents live only in memory and disappear when
> the API process restarts. API calls use the key owner's OpenAI credits.

## Quick start with Docker

Requirements: Git, Docker Desktop, and an OpenAI API key.

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd rag-learning-project
cp .env.example .env
```

Add your real `OPENAI_API_KEY` to `.env`, then run:

```bash
docker compose up --build
```

Open http://127.0.0.1:8000/docs and try the endpoints in this order:

1. `POST /documents` to index text;
2. `POST /search` to inspect semantic retrieval;
3. `POST /rag/ask` to generate an answer from retrieved context.

Stop the app with `Ctrl+C`. Remove the container with:

```bash
docker compose down
```

## Local development

### 1. Install the environment

```bash
uv sync
```

`uv` creates `.venv` and installs both runtime and development dependencies from
`pyproject.toml`.

### 2. Configure the API key

Open `.env` and set:

```dotenv
OPENAI_API_KEY=your-real-key
OPENAI_MODEL=gpt-5-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

Never commit `.env`. It is already listed in `.gitignore`. `.env.example` documents
the required variables without containing secrets.

### 3. Run the API

```bash
uv run uvicorn app.main:app --reload
```

Useful URLs:

- API documentation: http://127.0.0.1:8000/docs
- Health check: http://127.0.0.1:8000/health

Try the model endpoint:

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"Explain embeddings in two sentences."}'
```

Create one embedding:

```bash
curl -X POST http://127.0.0.1:8000/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"text":"A cat is sleeping on the sofa."}'
```

Compare two texts using cosine similarity:

```bash
curl -X POST http://127.0.0.1:8000/similarity \
  -H 'Content-Type: application/json' \
  -d '{"text_a":"A cat is sleeping.","text_b":"A kitten is taking a nap."}'
```

The app sends both texts in one embeddings API request. It then calculates cosine
similarity locally in `app/vector_math.py`; OpenAI does not calculate that score.

## In-memory vector search

Index several short documents in one embeddings request:

```bash
curl -X POST http://127.0.0.1:8000/documents \
  -H 'Content-Type: application/json' \
  -d '{"documents":[
    {"text":"Cats sleep for many hours.","metadata":{"topic":"animals"}},
    {"text":"PostgreSQL stores relational data.","metadata":{"topic":"databases"}},
    {"text":"A kitten is a young cat.","metadata":{"topic":"animals"}}
  ]}'
```

Search the stored vectors:

```bash
curl -X POST http://127.0.0.1:8000/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"How do cats rest?","top_k":2}'
```

This store deliberately performs a linear scan: it compares the query embedding with
every stored embedding and sorts by cosine similarity. Its contents are lost whenever
the Uvicorn process restarts. A persistent vector database will replace it later.

## Ask with RAG

After indexing documents, retrieve relevant context and generate a grounded answer:

```bash
curl -X POST http://127.0.0.1:8000/rag/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"How long do cats sleep?","top_k":2}'
```

The response contains the generated answer, the exact source records retrieved, and
separate token counts for retrieval embeddings and answer generation. The model is
instructed to use only the supplied context and to cite blocks as `[Source N]`.

## Verify the project

```bash
uv run pytest
uv run ruff check .
```

## Why this structure?

- `app/main.py` owns HTTP concerns.
- `app/schemas.py` defines and validates the API contract.
- `app/config.py` loads configuration without exposing secret values.
- `app/openai_service.py` isolates the external SDK from the web layer.
- `tests/` verifies behavior without making paid API calls.
