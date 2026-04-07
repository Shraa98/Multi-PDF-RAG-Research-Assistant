import os
import sys

import groq
import streamlit as st
from dotenv import load_dotenv
from langfuse import Langfuse
from langchain_core.messages import HumanMessage, SystemMessage

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import rag_pipeline as rag

AVAILABLE_PROVIDERS = rag.AVAILABLE_PROVIDERS
DEFAULT_MODEL_NAME = rag.DEFAULT_MODEL_NAME
DEFAULT_PROVIDER = rag.DEFAULT_PROVIDER
_get_prompt_client = rag._get_prompt_client
build_rag_from_scratch = rag.build_rag_from_scratch
extract_answer_and_usage = rag.extract_answer_and_usage
get_debug_for_question = rag.get_debug_for_question
get_models_for_provider = rag.get_models_for_provider
get_sources_for_question = rag.get_sources_for_question
load_existing_rag = rag.load_existing_rag
select_unique_sources = rag.select_unique_sources
split_questions = rag.split_questions
format_assistant_answer = getattr(rag, "format_assistant_answer", None)


def extract_model_name(result, fallback_model=None):
    extractor = getattr(rag, "extract_model_name", None)
    if extractor is None:
        return fallback_model
    return extractor(result, fallback_model=fallback_model)


def build_chat_llm(provider, model_name):
    if hasattr(rag, "build_chat_llm"):
        return rag.build_chat_llm(provider=provider, model_name=model_name)
    return rag._build_llm(provider=provider, model_name=model_name)


def build_general_chat_messages(question, recent_turns=None):
    if hasattr(rag, "build_general_chat_messages"):
        return rag.build_general_chat_messages(question, recent_turns=recent_turns)
    return [
        SystemMessage(content="You are a helpful AI assistant."),
        HumanMessage(content=question.strip()),
    ]


def should_route_to_rag(question, debug_items):
    if hasattr(rag, "should_route_to_rag"):
        return rag.should_route_to_rag(question, debug_items)
    return True, "routing helper unavailable; defaulting to RAG"


def get_recent_general_turns(max_turns=4):
    turns = []
    history = st.session_state.get("chat_history", [])

    for chat in reversed(history):
        responses = chat.get("responses") or []
        if not responses:
            continue

        for response in reversed(responses):
            if response.get("mode") != "general":
                continue
            user_q = (response.get("question") or chat.get("question") or "").strip()
            assistant_a = (response.get("answer") or "").strip()
            if user_q and assistant_a:
                turns.append({"user": user_q, "assistant": assistant_a})
                if len(turns) >= max_turns:
                    break
        if len(turns) >= max_turns:
            break

    turns.reverse()
    return turns

load_dotenv()

langfuse = None
if all(
    os.getenv(key)
    for key in ("LANGFUSE_SECRET_KEY", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_BASE_URL")
):
    try:
        langfuse = Langfuse()
    except Exception:
        langfuse = None


st.set_page_config(
    page_title="Multi-PDF Research Assistant",
    page_icon="📚",
    layout="wide",
)

st.title("📚 Multi-PDF Research Assistant")
st.caption("Upload multiple PDFs and ask questions across all of them")

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "debug_retrieval" not in st.session_state:
    st.session_state.debug_retrieval = False
if "source_limit" not in st.session_state:
    st.session_state.source_limit = 6
if "provider_name" not in st.session_state:
    st.session_state.provider_name = DEFAULT_PROVIDER
if "model_name" not in st.session_state:
    st.session_state.model_name = DEFAULT_MODEL_NAME
if "langfuse_prompt_client" not in st.session_state:
    st.session_state.langfuse_prompt_client = None
if "general_llm" not in st.session_state:
    st.session_state.general_llm = None
if "general_llm_provider" not in st.session_state:
    st.session_state.general_llm_provider = None
if "general_llm_model" not in st.session_state:
    st.session_state.general_llm_model = None

with st.sidebar:
    st.header("PDF Management")

    uploaded_files = st.file_uploader(
        "Upload your PDFs",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        os.makedirs("./data/pdfs", exist_ok=True)
        for file in uploaded_files:
            with open(f"./data/pdfs/{file.name}", "wb") as f:
                f.write(file.read())
        st.success(f"{len(uploaded_files)} PDF(s) uploaded.")

    col1, col2 = st.columns(2)

    st.session_state.provider_name = st.selectbox(
        "LLM provider",
        options=AVAILABLE_PROVIDERS,
        index=AVAILABLE_PROVIDERS.index(st.session_state.provider_name),
        help="Choose your provider backend (Groq, Google, OpenRouter, or local Ollama).",
    )

    provider_models = get_models_for_provider(st.session_state.provider_name)
    if st.session_state.model_name not in provider_models:
        st.session_state.model_name = provider_models[0]

    st.session_state.model_name = st.selectbox(
        "Model",
        options=provider_models,
        index=provider_models.index(st.session_state.model_name),
        help="Choose the model for the selected provider.",
    )

    with col1:
        if st.button("Process PDFs", use_container_width=True):
            with st.spinner("Processing PDFs..."):
                chain, vs = build_rag_from_scratch(
                    model_name=st.session_state.model_name,
                    provider=st.session_state.provider_name,
                )
                st.session_state.rag_chain = chain
                st.session_state.vector_store = vs
                st.session_state.langfuse_prompt_client = _get_prompt_client()
                st.session_state.chat_history = []
            if chain:
                st.success("Ready.")
            else:
                st.error("No PDFs found.")

    with col2:
        if st.button("Load Existing", use_container_width=True):
            with st.spinner("Loading..."):
                chain, vs = load_existing_rag(
                    model_name=st.session_state.model_name,
                    provider=st.session_state.provider_name,
                )
                st.session_state.rag_chain = chain
                st.session_state.vector_store = vs
                st.session_state.langfuse_prompt_client = _get_prompt_client()
            st.success("Loaded.")

    st.session_state.debug_retrieval = st.checkbox(
        "Show retrieval debug",
        value=st.session_state.debug_retrieval,
        help="Inspect which chunks were retrieved and why they were selected.",
    )

    st.session_state.source_limit = st.slider(
        "Unique PDFs to show",
        min_value=3,
        max_value=10,
        value=st.session_state.source_limit,
        help="Number of unique PDFs to show per question.",
    )

    st.divider()
    st.caption("Click 'Process PDFs' once per new upload.")

if st.session_state.rag_chain is None:
    st.info("Upload PDFs and click 'Process PDFs' to get started.")
else:
    for chat in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(chat["question"])
        with st.chat_message("assistant"):
            responses = chat.get("responses")
            if not responses:
                responses = [{
                    "question": chat["question"],
                    "answer": chat["answer"],
                    "sources": chat.get("sources", []),
                    "debug": chat.get("debug", []),
                }]

            for response in responses:
                if len(responses) > 1:
                    st.markdown(f"**{response['question']}**")
                st.write(response["answer"])
                mode = response.get("mode", "rag")
                routing_reason = response.get("routing_reason")
                if mode == "general":
                    st.caption("Routed to: General Chat")
                    if routing_reason:
                        st.caption(f"Reason: {routing_reason}")
                else:
                    st.caption("Routed to: RAG")
                    if routing_reason:
                        st.caption(f"Reason: {routing_reason}")
                    with st.expander(f"Sources used: {response['question']}"):
                        for i, src in enumerate(response["sources"], 1):
                            page = src.metadata.get("page")
                            page_label = f" (Page {page})" if page else ""
                            st.markdown(f"**Source {i}:** `{src.metadata.get('source', 'Unknown')}`{page_label}")
                            st.caption(src.page_content[:300] + "...")

                if st.session_state.debug_retrieval and response.get("debug"):
                    with st.expander(f"Debug retrieval: {response['question']}"):
                        for i, item in enumerate(response["debug"], 1):
                            doc = item["doc"]
                            page = doc.metadata.get("page")
                            page_label = f" (Page {page})" if page else ""
                            st.markdown(
                                f"**Chunk {i}:** `{doc.metadata.get('source', 'Unknown')}`{page_label} | "
                                f"lexical score `{item['lexical_score']}` | semantic match `{item['semantic_match']}`"
                            )
                            if item["reasons"]:
                                st.caption("Reasons: " + "; ".join(item["reasons"][:6]))
                            st.code(doc.page_content[:600], language="text")

    question = st.chat_input("Ask anything about your PDFs...")

    if question:
        subquestions = split_questions(question)
        responses = []
        prompt_client = st.session_state.langfuse_prompt_client

        try:
            with st.spinner("Searching PDFs..."):
                if (
                    st.session_state.general_llm is None
                    or st.session_state.general_llm_provider != st.session_state.provider_name
                    or st.session_state.general_llm_model != st.session_state.model_name
                ):
                    st.session_state.general_llm = build_chat_llm(
                        provider=st.session_state.provider_name,
                        model_name=st.session_state.model_name,
                    )
                    st.session_state.general_llm_provider = st.session_state.provider_name
                    st.session_state.general_llm_model = st.session_state.model_name

                for subquestion in subquestions:
                    debug_items = get_debug_for_question(
                        st.session_state.vector_store,
                        subquestion,
                        k=max(st.session_state.source_limit * 3, 6),
                    )
                    use_rag, routing_reason = should_route_to_rag(subquestion, debug_items)

                    if langfuse is not None:
                        with langfuse.start_as_current_observation(
                            name="rag-generation",
                            as_type="generation",
                            model=st.session_state.model_name,
                            input={"question": subquestion},
                            prompt=prompt_client,
                            metadata={
                                "provider": st.session_state.provider_name,
                                "model_name": st.session_state.model_name,
                                "source_limit": st.session_state.source_limit,
                                "route": "rag" if use_rag else "general",
                                "routing_reason": routing_reason,
                            },
                        ) as generation:
                            if use_rag:
                                llm_result = st.session_state.rag_chain.invoke(subquestion)
                            else:
                                recent_turns = get_recent_general_turns(max_turns=4)
                                llm_result = st.session_state.general_llm.invoke(
                                    build_general_chat_messages(subquestion, recent_turns=recent_turns)
                                )
                            answer, usage_details = extract_answer_and_usage(llm_result)
                            if format_assistant_answer is not None:
                                answer = format_assistant_answer(
                                    question=subquestion,
                                    answer=answer,
                                    mode="rag" if use_rag else "general",
                                )
                            resolved_model = extract_model_name(
                                llm_result,
                                fallback_model=st.session_state.model_name,
                            )
                            generation.update(
                                output=answer,
                                usage_details=usage_details if usage_details else None,
                                model=resolved_model,
                                metadata={
                                    "provider": st.session_state.provider_name,
                                    "model_name": st.session_state.model_name,
                                    "resolved_model": resolved_model,
                                    "source_limit": st.session_state.source_limit,
                                    "route": "rag" if use_rag else "general",
                                    "routing_reason": routing_reason,
                                },
                            )
                    else:
                        if use_rag:
                            llm_result = st.session_state.rag_chain.invoke(subquestion)
                        else:
                            recent_turns = get_recent_general_turns(max_turns=4)
                            llm_result = st.session_state.general_llm.invoke(
                                build_general_chat_messages(subquestion, recent_turns=recent_turns)
                            )
                        answer, _ = extract_answer_and_usage(llm_result)
                        if format_assistant_answer is not None:
                            answer = format_assistant_answer(
                                question=subquestion,
                                answer=answer,
                                mode="rag" if use_rag else "general",
                            )

                    if use_rag:
                        sources = get_sources_for_question(
                            st.session_state.vector_store,
                            subquestion,
                            k=max(st.session_state.source_limit * 3, 6),
                        )
                        unique_sources = select_unique_sources(
                            sources,
                            limit=st.session_state.source_limit,
                        )
                    else:
                        unique_sources = []

                    responses.append({
                        "question": subquestion,
                        "answer": answer,
                        "mode": "rag" if use_rag else "general",
                        "routing_reason": routing_reason,
                        "sources": unique_sources,
                        "debug": debug_items,
                    })
                if langfuse is not None:
                    langfuse.flush()

        except groq.RateLimitError as exc:
            st.error(
                "Groq rate limit reached. Your token budget is exhausted for now. "
                "Wait a few minutes and try again, or reduce the number of questions per message."
            )
            st.caption(str(exc))
            st.stop()
        except Exception as exc:
            error_text = str(exc)
            provider_lower = st.session_state.provider_name.lower()
            if any(token in error_text.lower() for token in [
                "model",
                "not found",
                "unavailable",
                "access",
                "permission",
            ]):
                st.error(
                    f"The selected {st.session_state.provider_name} model `{st.session_state.model_name}` is not "
                    "available for your account or is currently unavailable."
                )
                st.caption(error_text)
                st.stop()
            if "missing" in error_text.lower() and "_api_key" in error_text.lower():
                st.error(
                    f"Missing API key for provider `{st.session_state.provider_name}`. "
                    f"Set the required environment variable and try again."
                )
                st.caption(error_text)
                st.stop()
            if "requires installing" in error_text.lower():
                st.error(
                    f"Provider `{st.session_state.provider_name}` is not installed in this environment yet."
                )
                st.caption(error_text)
                st.stop()
            raise

        st.session_state.chat_history.append({
            "question": question,
            "answer": responses[0]["answer"] if len(responses) == 1 else "",
            "sources": responses[0]["sources"] if len(responses) == 1 else [],
            "debug": responses[0]["debug"] if len(responses) == 1 else [],
            "responses": responses,
        })
        st.rerun()
