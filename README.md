# CitePilot

> **Note:** This repository is an early retrieval-first prototype. It has been superseded by **[DocuGuard](https://github.com/Phaldors/docuguard)** (which implements hybrid BM25 + OpenAI embedding retrieval with Reciprocal Rank Fusion, LLM reranking, verbatim citation verification, and adversarial evaluation) and integrated as a tool inside **[AI Ops Assistant](https://github.com/Phaldors/ai-ops-assistant)**.

Upload a PDF or TXT file, ask a question, and inspect the source passages behind the answer.

## Overview

CitePilot was built as a foundational building block to isolate the retrieval layer before adding LLM synthesis: document parsing, overlapping text chunking, TF-IDF cosine ranking, and page-level source excerpts.

## Capabilities

- Upload PDF and TXT files
- Extract and chunk document text with page metadata
- Rank source passages with TF-IDF cosine similarity
- Display document name, page number, verbatim excerpt, and relevance score in a minimal FastAPI web UI

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000`.
