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
    emb = get_embeddings()
    return Chroma(
        persist_directory=VECTOR_DIR,
        embedding_function=emb
    )

def add_pdf_to_vectorstore(pdf_path: str):
    """Add a single uploaded PDF to the existing vectorstore."""
    loader = PyMuPDFLoader(pdf_path)
    docs   = loader.load()

    from langchain_text_splitters import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(docs)

    emb = get_embeddings()
    db  = Chroma(
        persist_directory=VECTOR_DIR,
        embedding_function=emb
    )
    db.add_documents(chunks)
    print(f"Added {len(chunks)} chunks from {pdf_path}")
    return len(chunks)

def get_retriever_with_scores(db, query: str, k: int = 4):
    """Returns (docs, scores) — scores are similarity 0 to 1, higher is better."""
    results = db.similarity_search_with_relevance_scores(query, k=k)
    docs    = [r[0] for r in results]
    scores  = [r[1] for r in results]
    return docs, scores

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)