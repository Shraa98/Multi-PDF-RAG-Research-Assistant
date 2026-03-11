import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def chunk_documents(documents):
    """Split documents into smaller chunks with source and page metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    all_chunks = []

    for doc in documents:
        pages = doc.get("pages")
        source_title = os.path.splitext(doc["source"])[0]

        if pages:
            for page_data in pages:
                chunks = splitter.split_text(page_data["text"])
                for i, chunk in enumerate(chunks):
                    all_chunks.append(Document(
                        page_content=chunk,
                        metadata={
                            "source": doc["source"],
                            "source_title": source_title,
                            "page": page_data["page"],
                            "chunk_id": f"{page_data['page']}-{i}",
                        }
                    ))
            continue

        chunks = splitter.split_text(doc["text"])
        for i, chunk in enumerate(chunks):
            all_chunks.append(Document(
                page_content=chunk,
                metadata={
                    "source": doc["source"],
                    "source_title": source_title,
                    "chunk_id": i,
                }
            ))

    print(f"✅ Total chunks created: {len(all_chunks)}")
    return all_chunks
