import sys
import os
sys.path.append(os.path.dirname(__file__))

from rag_pipeline import build_vectorstore, load_vectorstore, get_qa_chain

print("--- Step 1: Building vectorstore ---")
db = build_vectorstore("../data/pdfs")

print("\n--- Step 2: Loading vectorstore ---")
db = load_vectorstore()

print("\n--- Step 3: Running a test query ---")
chain, retriever = get_qa_chain(db)

query = "What are the side effects of Metformin?"

answer = chain.invoke({"question": query})

print("\nANSWER:")
print(answer)

print("\nSOURCES:")
source_docs = retriever.invoke(query)
for doc in source_docs:
    src  = doc.metadata.get("source", "Unknown")
    pg   = doc.metadata.get("page", "?")
    print(f"  -> {os.path.basename(src)} -- Page {int(pg) + 1}")