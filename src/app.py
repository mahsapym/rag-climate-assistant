import os
import streamlit as st
import chromadb
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from llama_index.core import Settings

load_dotenv()

CHROMA_DIR = "data/chroma_db"
COLLECTION_NAME = "climate_papers"

st.set_page_config(page_title="Climate Research Assistant", page_icon="🌍")
st.title("🌍 Atmospheric Science Research Assistant")
st.caption("Ask questions about the indexed research papers. Answers include source citations.")

@st.cache_resource
def load_index():
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    llm = Groq(model="llama-3.3-70b-versatile", api_key=os.environ.get("GROQ_API_KEY"))

    Settings.embed_model = embed_model
    Settings.llm = llm

    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    chroma_collection = chroma_client.get_or_create_collection(COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)
    return index

index = load_index()

query_engine = index.as_query_engine(
    similarity_top_k=8,
    response_mode="tree_summarize",
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask a question about the papers..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching papers..."):
            response = query_engine.query(prompt)
            answer = str(response)

            sources = set()
            for node in response.source_nodes:
                fname = node.metadata.get("file_name", "unknown")
                sources.add(fname)

            full_response = answer + "\n\n**Sources:** " + ", ".join(sorted(sources))
            st.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})