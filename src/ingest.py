import os
from dotenv import load_dotenv
import fitz
import chromadb
from llama_index.core import Document, VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

load_dotenv()

DATA_DIR = "data/papers"
CHROMA_DIR = "data/chroma_db"
COLLECTION_NAME = "climate_papers"

def load_documents_by_page():
    docs = []
    for fname in os.listdir(DATA_DIR):
        if not fname.endswith('.pdf'):
            continue
        path = os.path.join(DATA_DIR, fname)
        pdf = fitz.open(path)
        for i, page in enumerate(pdf):
            text = page.get_text().strip()
            if len(text) < 100:  # skip nearly empty pages
                continue
            docs.append(Document(
                text=text,
                metadata={"file_name": fname, "page": i + 1}
            ))
        print(f"Loaded {fname}: {len(pdf)} pages")
    return docs

def main():
    print("Loading documents page by page...")
    documents = load_documents_by_page()
    print(f"Total chunks: {len(documents)}")

    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    chroma_collection = chroma_client.get_or_create_collection(COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    print("Building index...")
    VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True,
    )
    print(f"Done! Index saved to {CHROMA_DIR}/")

if __name__ == "__main__":
    main()