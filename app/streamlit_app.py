import os
import sys

import groq
import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag_pipeline import (
    AVAILABLE_PROVIDERS,
    DEFAULT_MODEL_NAME,
    DEFAULT_PROVIDER,
    build_rag_from_scratch,
    get_debug_for_question,
    get_models_for_provider,
    get_sources_for_question,
    load_existing_rag,
    select_unique_sources,
    split_questions,
)


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
        help="Free-friendly options only: Groq free tier, Google Gemini free tier, or local Ollama.",
    )

    provider_models = get_models_for_provider(st.session_state.provider_name)
    if st.session_state.model_name not in provider_models:
        st.session_state.model_name = provider_models[0]

    st.session_state.model_name = st.selectbox(
        "Model",
        options=provider_models,
        index=provider_models.index(st.session_state.model_name),
        help="Choose a lower-cost free-tier model first; local Ollama avoids API costs entirely.",
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

        try:
            with st.spinner("Searching PDFs..."):
                for subquestion in subquestions:
                    answer = st.session_state.rag_chain.invoke(subquestion)
                    sources = get_sources_for_question(
                        st.session_state.vector_store,
                        subquestion,
                        k=max(st.session_state.source_limit * 3, 6),
                    )
                    unique_sources = select_unique_sources(
                        sources,
                        limit=st.session_state.source_limit,
                    )
                    debug_items = get_debug_for_question(
                        st.session_state.vector_store,
                        subquestion,
                        k=max(st.session_state.source_limit * 3, 6),
                    )
                    responses.append({
                        "question": subquestion,
                        "answer": answer,
                        "sources": unique_sources,
                        "debug": debug_items,
                    })
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
