import os
import re

from dotenv import load_dotenv
from langfuse import Langfuse
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
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
LANGFUSE_PROMPT_NAME = "multi-pdf-rag"
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

PROMPT_SYSTEM_TEMPLATE = """
You are a helpful research assistant.
Answer the question using ONLY the context from the PDFs provided.
If the answer requires combining multiple retrieved passages, synthesize them into one answer and mention all relevant PDFs/sources.
For comparison questions, compare only what is explicitly supported by the retrieved context.
If the question is awkwardly phrased but the context contains the closest equivalent fact, answer using the paper's exact terminology instead of refusing.
Prefer the most direct factual answer first, then a short explanation.
If the answer is not in the context, say:
"I couldn't find this information in the uploaded PDFs."
Do NOT make up any information.
"""

PROMPT_USER_TEMPLATE = """
Context:
{context}

Question: {question}

Answer (mention source PDF):
"""

langfuse = None
if all(
    os.getenv(key)
    for key in ("LANGFUSE_SECRET_KEY", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_BASE_URL")
):
    try:
        langfuse = Langfuse()
    except Exception:
        langfuse = None


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
    prompt_client = _get_prompt_client()

    retriever = get_retriever(vector_store, k=6)

    rag_chain = (
        {
            "context": RunnableLambda(lambda question: format_docs(retrieve_chunks(retriever, question))),
            "question": RunnablePassthrough(),
        }
        | RunnableLambda(
            lambda payload: _build_prompt_messages(
                prompt_client=prompt_client,
                context=payload["context"],
                question=payload["question"],
            )
        )
        | llm
    )

    print(f"RAG chain ready with provider={provider}, model={model_name}")
    return rag_chain

def _get_prompt_client():
    if langfuse is None:
        return None

    fallback = [
        {"role": "system", "content": PROMPT_SYSTEM_TEMPLATE.strip()},
        {"role": "user", "content": PROMPT_USER_TEMPLATE.strip().replace("{", "{{").replace("}", "}}")},
    ]

    try:
        return langfuse.get_prompt(
            LANGFUSE_PROMPT_NAME,
            type="chat",
            label="production",
            fallback=fallback,
        )
    except Exception:
        return None


def _build_prompt_messages(prompt_client, context, question):
    if prompt_client is None:
        return _build_fallback_messages(context=context, question=question)

    try:
        compiled_messages = prompt_client.compile(context=context, question=question)
        return [_to_langchain_message(message) for message in compiled_messages]
    except Exception:
        return _build_fallback_messages(context=context, question=question)


def _build_fallback_messages(context, question):
    return [
        SystemMessage(content=PROMPT_SYSTEM_TEMPLATE.strip()),
        HumanMessage(
            content=PROMPT_USER_TEMPLATE.format(
                context=context,
                question=question,
            ).strip()
        ),
    ]


def _to_langchain_message(message):
    role = message["role"]
    content = message["content"]

    if role == "system":
        return SystemMessage(content=content)
    if role == "assistant":
        return AIMessage(content=content)

    return HumanMessage(content=content)


def extract_answer_and_usage(result):
    if isinstance(result, str):
        return result, {}

    answer = getattr(result, "content", "")
    usage_details = {}

    usage_metadata = getattr(result, "usage_metadata", None) or {}
    if usage_metadata:
        prompt_tokens = usage_metadata.get("input_tokens")
        completion_tokens = usage_metadata.get("output_tokens")
        total_tokens = usage_metadata.get("total_tokens")

        if prompt_tokens is not None:
            usage_details["prompt_tokens"] = int(prompt_tokens)
        if completion_tokens is not None:
            usage_details["completion_tokens"] = int(completion_tokens)
        if total_tokens is not None:
            usage_details["total_tokens"] = int(total_tokens)

    if not usage_details:
        response_metadata = getattr(result, "response_metadata", None) or {}
        token_usage = response_metadata.get("token_usage", {}) if isinstance(response_metadata, dict) else {}
        prompt_tokens = token_usage.get("prompt_tokens") or token_usage.get("input_tokens")
        completion_tokens = token_usage.get("completion_tokens") or token_usage.get("output_tokens")
        total_tokens = token_usage.get("total_tokens")

        if prompt_tokens is not None:
            usage_details["prompt_tokens"] = int(prompt_tokens)
        if completion_tokens is not None:
            usage_details["completion_tokens"] = int(completion_tokens)
        if total_tokens is not None:
            usage_details["total_tokens"] = int(total_tokens)

    return answer, usage_details


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
