from langchain_community.vectorstores import Chroma
from src.embeddings import get_embedding_model

VECTOR_STORE_PATH = "./vector_store"

def create_vector_store(chunks):
    """Create and save ChromaDB vector store"""
    print("💾 Creating vector store...")
    embeddings = get_embedding_model()

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTOR_STORE_PATH
    )

    print(f"✅ Vector store created with {len(chunks)} chunks!")
    return vector_store

def load_vector_store():
    """Load existing vector store from disk"""
    print("📂 Loading existing vector store...")
    embeddings = get_embedding_model()

    vector_store = Chroma(
        persist_directory=VECTOR_STORE_PATH,
        embedding_function=embeddings
    )

    print("✅ Vector store loaded!")
    return vector_store