import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyMuPDFLoader
from document_loader import load_documents, split_documents

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

try:
    import streamlit as st
    if "GROQ_API_KEY" in st.secrets:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
except Exception:
    pass

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
VECTOR_DIR  = os.path.join(os.path.dirname(__file__), '..', 'vectorstore')

def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)

def build_vectorstore(pdf_folder: str):
    docs   = load_documents(pdf_folder)
    chunks = split_documents(docs)
    emb    = get_embeddings()
    db     = Chroma.from_documents(
                chunks, emb,
                persist_directory=VECTOR_DIR
             )
    print(f"Vectorstore built with {len(chunks)} chunks")
    return db

def load_vectorstore():
    if not os.path.exists(VECTOR_DIR):
        raise FileNotFoundError(
            "Vectorstore not found. Click 'Rebuild Default Index' in the sidebar to build it."
        )
    try:
        emb = get_embeddings()
        return Chroma(
            persist_directory=VECTOR_DIR,
            embedding_function=emb
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load vectorstore: {e}")

def add_pdf_to_vectorstore(pdf_path: str):
    try:
        loader = PyMuPDFLoader(pdf_path)
        docs   = loader.load()

        if not docs:
            raise ValueError("PDF appears to be empty or unreadable.")

        from langchain_text_splitters import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100
        )
        chunks = splitter.split_documents(docs)
        emb    = get_embeddings()
        db     = Chroma(
            persist_directory=VECTOR_DIR,
            embedding_function=emb
        )
        db.add_documents(chunks)
        print(f"Added {len(chunks)} chunks from {pdf_path}")
        return len(chunks)

    except Exception as e:
        raise RuntimeError(f"Failed to process PDF: {e}")

def get_retriever_with_scores(db, query: str, k: int = 4):
    """Returns (docs, scores) — scores are similarity 0 to 1, higher is better."""
    results = db.similarity_search_with_relevance_scores(query, k=k)
    docs    = [r[0] for r in results]
    scores  = [r[1] for r in results]
    return docs, scores

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)