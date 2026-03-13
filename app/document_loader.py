import os
import streamlit as st
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_documents(pdf_folder: str):
    docs = []
    files = [f for f in os.listdir(pdf_folder) if f.endswith('.pdf')]

    if not files:
        raise FileNotFoundError(f"No PDF files found in {pdf_folder}")

    for file in files:
        path = os.path.join(pdf_folder, file)
        try:
            loader = PyMuPDFLoader(path)
            docs.extend(loader.load())
        except Exception as e:
            print(f"Warning: Could not load {file} — {e}")
            continue

    if not docs:
        raise ValueError("All PDFs failed to load. Check your data/pdfs folder.")

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