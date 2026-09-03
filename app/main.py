from __future__ import annotations

from io import BytesIO

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from pypdf import PdfReader

from app.retrieval import Retriever, chunk_text

app = FastAPI(title="CitePilot", version="0.1.0")
app.mount("/static", StaticFiles(directory="app/static"), name="static")
retriever = Retriever()


class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    return FileResponse("app/static/index.html")


@app.get("/health")
def health() -> dict[str, int | str]:
    return {"status": "ok", "indexed_chunks": len(retriever.chunks)}


@app.post("/api/documents")
async def upload_document(file: UploadFile = File(...)) -> dict[str, int | str]:
    filename = file.filename or "untitled"
    payload = await file.read()
    suffix = filename.lower().rsplit(".", maxsplit=1)[-1] if "." in filename else ""
    if suffix == "pdf":
        reader = PdfReader(BytesIO(payload))
        chunks = [chunk for page_number, page in enumerate(reader.pages, start=1) for chunk in chunk_text(page.extract_text() or "", filename, page_number)]
    elif suffix == "txt":
        chunks = chunk_text(payload.decode("utf-8", errors="ignore"), filename, 1)
    else:
        raise HTTPException(status_code=400, detail="Only PDF and TXT files are supported in v0.1.")
    if not chunks:
        raise HTTPException(status_code=400, detail="No readable text was found in this document.")
    retriever.add_chunks(chunks)
    return {"filename": filename, "indexed_chunks": len(chunks), "total_chunks": len(retriever.chunks)}


@app.post("/api/query")
def query_document(request: QueryRequest) -> dict[str, object]:
    results = retriever.search(request.question)
    if not results:
        raise HTTPException(status_code=404, detail="No relevant source found. Upload a document or ask a more specific question.")
    sources = [{"document": chunk.document_name, "page": chunk.page, "excerpt": chunk.text, "relevance": round(score, 2)} for chunk, score in results]
    answer = "The strongest evidence is in the sources below. CitePilot currently returns retrieved evidence; an LLM synthesis layer is the next milestone."
    return {"answer": answer, "sources": sources}
