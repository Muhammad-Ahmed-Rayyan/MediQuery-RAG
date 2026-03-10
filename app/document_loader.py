import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_documents(pdf_folder: str):
    docs = []
    for file in os.listdir(pdf_folder):
        if file.endswith('.pdf'):
            path = os.path.join(pdf_folder, file)
            loader = PyMuPDFLoader(path)
            docs.extend(loader.load())
    print(f"Loaded {len(docs)} pages from {pdf_folder}")
    return docs

def split_documents(docs, chunk_size=800, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks")
    return chunks