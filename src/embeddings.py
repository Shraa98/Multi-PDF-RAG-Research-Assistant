from langchain_community.embeddings import HuggingFaceEmbeddings

def get_embedding_model():
    """Load the embedding model"""
    print("🔢 Loading embedding model...")
    print("⏳ First time takes 2-3 mins (downloading model)...")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},  # change to 'cuda' if you want GPU
        encode_kwargs={"normalize_embeddings": True}
    )

    print("✅ Embedding model loaded!")
    return embeddings