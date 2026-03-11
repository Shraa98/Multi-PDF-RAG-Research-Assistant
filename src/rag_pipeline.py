import os
import re

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables import RunnablePassthrough

from src.document_loader import load_all_pdfs
from src.retriever import debug_chunks, get_retriever, retrieve_chunks
from src.text_chunking import chunk_documents
from src.vector_store import create_vector_store, load_vector_store

load_dotenv()

MAX_CHARS_PER_DOC = 1200
MAX_CONTEXT_CHARS = 5000
DEFAULT_PROVIDER = "Groq"
DEFAULT_MODEL_NAME = "llama-3.1-8b-instant"
PROVIDER_MODELS = {
    "Groq": [
        "llama-3.1-8b-instant",
        "llama-3.3-70b-versatile",
        "meta-llama/llama-4-scout-17b-16e-instruct",
        "qwen/qwen3-32b",
    ],
    "Google": [
        "gemini-2.5-flash-lite-preview-09-2025",
        "gemini-2.5-flash",
        "gemini-2.0-flash",
    ],
    "Ollama": [
        "llama3.1:8b",
        "mistral:7b",
        "qwen2.5:7b",
    ],
}
AVAILABLE_PROVIDERS = list(PROVIDER_MODELS.keys())
AVAILABLE_MODELS = PROVIDER_MODELS[DEFAULT_PROVIDER]

PROMPT_TEMPLATE = """
You are a helpful research assistant.
Answer the question using ONLY the context from the PDFs provided.
If the answer requires combining multiple retrieved passages, synthesize them into one answer and mention all relevant PDFs/sources.
For comparison questions, compare only what is explicitly supported by the retrieved context.
If the question is awkwardly phrased but the context contains the closest equivalent fact, answer using the paper's exact terminology instead of refusing.
Prefer the most direct factual answer first, then a short explanation.
If the answer is not in the context, say:
"I couldn't find this information in the uploaded PDFs."
Do NOT make up any information.

Context:
{context}

Question: {question}

Answer (mention source PDF):
"""


def format_docs(docs):
    formatted_docs = []
    total_chars = 0

    for doc in docs:
        remaining = MAX_CONTEXT_CHARS - total_chars
        if remaining <= 0:
            break

        content_limit = min(MAX_CHARS_PER_DOC, remaining)
        snippet = doc.page_content[:content_limit]
        formatted = (
            f"[Source: {doc.metadata.get('source', 'Unknown')}"
            f"{_format_page_suffix(doc)}]\n{snippet}"
        )
        formatted_docs.append(formatted)
        total_chars += len(formatted)

    return "\n\n".join(formatted_docs)


def _format_page_suffix(doc):
    page = doc.metadata.get("page")
    return f", Page {page}" if page else ""


def get_sources_for_question(vector_store, question, k=6):
    retriever = get_retriever(vector_store, k=k)
    return retrieve_chunks(retriever, question)


def get_debug_for_question(vector_store, question, k=6):
    retriever = get_retriever(vector_store, k=k)
    return debug_chunks(retriever, question)


def select_unique_sources(docs, limit):
    unique_docs = []
    seen_sources = set()

    for doc in docs:
        source = doc.metadata.get("source", "Unknown")
        if source in seen_sources:
            continue
        seen_sources.add(source)
        unique_docs.append(doc)
        if len(unique_docs) >= limit:
            break

    return unique_docs


def split_questions(question_text):
    parts = []

    for block in question_text.splitlines():
        cleaned_block = block.strip()
        if not cleaned_block:
            continue

        for piece in re.split(r"(?<=[?])\s+", cleaned_block):
            candidate = piece.strip()
            if not candidate:
                continue
            if "?" in candidate:
                for subpiece in re.findall(r"[^?]+\?", candidate):
                    normalized = subpiece.strip()
                    if normalized:
                        parts.append(normalized)
            else:
                parts.append(candidate)

    deduped = []
    seen = set()
    for part in parts:
        key = part.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(part)

    return deduped or [question_text.strip()]


def build_rag_from_scratch(
    pdf_folder="./data/pdfs",
    model_name=DEFAULT_MODEL_NAME,
    provider=DEFAULT_PROVIDER,
):
    documents = load_all_pdfs(pdf_folder)
    if not documents:
        return None, None
    chunks = chunk_documents(documents)
    vector_store = create_vector_store(chunks)
    chain = _build_chain(vector_store, model_name=model_name, provider=provider)
    return chain, vector_store


def load_existing_rag(model_name=DEFAULT_MODEL_NAME, provider=DEFAULT_PROVIDER):
    vector_store = load_vector_store()
    chain = _build_chain(vector_store, model_name=model_name, provider=provider)
    return chain, vector_store


def _build_chain(vector_store, model_name=DEFAULT_MODEL_NAME, provider=DEFAULT_PROVIDER):
    llm = _build_llm(provider=provider, model_name=model_name)

    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"],
    )

    retriever = get_retriever(vector_store, k=6)

    rag_chain = (
        {
            "context": RunnableLambda(lambda question: format_docs(retrieve_chunks(retriever, question))),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    print(f"RAG chain ready with provider={provider}, model={model_name}")
    return rag_chain


def get_models_for_provider(provider):
    return PROVIDER_MODELS.get(provider, [])


def _build_llm(provider, model_name):
    if provider == "Groq":
        from langchain_groq import ChatGroq

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("Missing GROQ_API_KEY in environment.")
        return ChatGroq(api_key=api_key, model_name=model_name, temperature=0)

    if provider == "OpenAI":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise ImportError("OpenAI support requires installing langchain-openai.") from exc

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Missing OPENAI_API_KEY in environment.")
        return ChatOpenAI(api_key=api_key, model=model_name, temperature=0)

    if provider == "Google":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as exc:
            raise ImportError("Google support requires installing langchain-google-genai.") from exc

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Missing GOOGLE_API_KEY in environment.")
        return ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key, temperature=0)

    if provider == "Anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as exc:
            raise ImportError("Anthropic support requires installing langchain-anthropic.") from exc

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("Missing ANTHROPIC_API_KEY in environment.")
        return ChatAnthropic(api_key=api_key, model=model_name, temperature=0)

    if provider == "Ollama":
        try:
            from langchain_ollama import ChatOllama
        except ImportError as exc:
            raise ImportError("Ollama support requires installing langchain-ollama.") from exc

        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        return ChatOllama(model=model_name, base_url=base_url, temperature=0)

    raise ValueError(f"Unsupported provider: {provider}")
