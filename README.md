# Multi-PDF RAG Research Assistant

A Streamlit-based Retrieval-Augmented Generation (RAG) app that lets you upload multiple PDFs, index them into ChromaDB, and ask grounded questions across all files.

## Features

- Upload and index multiple PDFs from the UI
- Page-aware chunking with source metadata
- Hybrid retrieval (semantic + lexical reranking)
- Source previews with page numbers
- Retrieval debug mode (scores, reasons, semantic match)
- Multiple LLM providers in UI:
  - Groq
  - Google Gemini
  - Ollama (local)

## Tech Stack

- UI: Streamlit
- RAG framework: LangChain
- Vector store: ChromaDB (local persisted store)
- PDF parsing: PyMuPDF
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2`

## Project Structure

```text
multi-pdf-rag/
├─ app/
│  └─ streamlit_app.py
├─ src/
│  ├─ document_loader.py
│  ├─ text_chunking.py
│  ├─ embeddings.py
│  ├─ vector_store.py
│  ├─ retriever.py
│  └─ rag_pipeline.py
├─ data/
│  └─ pdfs/
├─ vector_store/
├─ requirements.txt
└─ README.md
```

## Setup

1. Create and activate a virtual environment.

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

2. Install dependencies.

```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root.

Example:

```env
# Required if using Groq provider
GROQ_API_KEY=your_groq_api_key

# Required if using Google provider
GOOGLE_API_KEY=your_google_api_key

# Optional (for Ollama; default shown)
OLLAMA_BASE_URL=http://localhost:11434
```

## Run the App

```bash
streamlit run app/streamlit_app.py
```

Then open the local URL shown in terminal (usually `http://localhost:8501`).

## How to Use

1. Upload one or more PDF files from the sidebar.
2. Select provider and model.
3. Click `Process PDFs` to build a fresh vector store.
4. Ask questions in chat input.
5. Expand `Sources used` (and optional debug panels) to inspect retrieval.
6. Use `Load Existing` to reuse previously persisted vectors.

## Notes

- Vector DB persists to `./vector_store`.
- Uploaded PDFs are saved to `./data/pdfs`.
- If provider/model access fails, switch to another model or provider.
- Groq free tier can hit rate limits on multi-question prompts.

## Troubleshooting

- `Missing ... API_KEY`:
  - Add the key to `.env` and restart Streamlit.
- `No PDFs found`:
  - Upload files and click `Process PDFs`.
- `Model unavailable`:
  - Choose a different model for the selected provider.
- Slow first run:
  - Initial embedding model download can take a few minutes.
