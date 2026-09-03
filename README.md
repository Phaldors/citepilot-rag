# CitePilot

Upload a PDF or TXT file, ask a question, and inspect the source passages behind the answer.

## Why this project

CitePilot is a source-grounded RAG portfolio project. Version 0.1 focuses on the retrieval layer first: document parsing, text chunking, ranking, and readable citations. The next milestone adds LLM synthesis while preserving those citations.

## Current capabilities

- Upload PDF and TXT files
- Extract and chunk document text
- Search with TF-IDF cosine similarity
- Show document name, page, excerpt, and relevance for each result
- Run a minimal FastAPI web UI

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000`.

## Roadmap

- [x] Retrieval-first MVP with citations
- [ ] Add embeddings and a persistent vector store
- [ ] Add LLM answer synthesis with citation guardrails
- [ ] Add evaluation dataset and retrieval metrics
- [ ] Deploy a public demo
