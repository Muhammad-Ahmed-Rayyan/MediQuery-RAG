import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from document_loader import load_documents, split_documents

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
VECTOR_DIR  = os.path.join(os.path.dirname(__file__), '..', 'vectorstore')

def build_vectorstore(pdf_folder: str):
    docs   = load_documents(pdf_folder)
    chunks = split_documents(docs)
    emb    = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    db     = Chroma.from_documents(
                chunks, emb,
                persist_directory=VECTOR_DIR
             )
    print(f"Vectorstore built with {len(chunks)} chunks")
    return db

def load_vectorstore():
    emb = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    return Chroma(
        persist_directory=VECTOR_DIR,
        embedding_function=emb
    )

PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a helpful medical assistant.
Answer ONLY using the context provided below.
If the answer is not in the context, say "I don't have enough information in the documents to answer this."
Do not make up information.

Context: {context}

Question: {question}

Answer:"""
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def get_qa_chain(db):
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0
    )
    retriever = db.as_retriever(search_kwargs={"k": 4})

    chain = (
        RunnablePassthrough.assign(
            context=lambda x: format_docs(retriever.invoke(x["question"]))
        )
        | PROMPT
        | llm
        | StrOutputParser()
    )
    return chain, retriever