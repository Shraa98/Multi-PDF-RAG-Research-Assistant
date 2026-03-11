import fitz  # PyMuPDF
import os

def load_single_pdf(pdf_path):
    """Extract text from a single PDF, preserving per-page content."""
    doc = fitz.open(pdf_path)
    text = ""
    pages = []
    for page_num, page in enumerate(doc):
        page_number = page_num + 1
        page_text = page.get_text()
        text += f"\n[Page {page_number}]\n"
        text += page_text
        pages.append({
            "page": page_number,
            "text": page_text,
        })
    doc.close()
    return {
        "text": text,
        "pages": pages,
    }

def load_all_pdfs(pdf_folder):
    """Extract text from all PDFs in a folder"""
    documents = []
    pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith('.pdf')]

    if not pdf_files:
        print("❌ No PDFs found in folder!")
        return []

    print(f"📄 Found {len(pdf_files)} PDF(s): {pdf_files}")

    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_folder, pdf_file)
        print(f"  → Loading: {pdf_file}")
        pdf_content = load_single_pdf(pdf_path)
        documents.append({
            "source": pdf_file,
            "text": pdf_content["text"],
            "pages": pdf_content["pages"],
        })

    print(f"✅ Loaded {len(documents)} PDF(s) successfully!")
    return documents
