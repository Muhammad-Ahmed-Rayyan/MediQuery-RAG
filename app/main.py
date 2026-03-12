import sys
import os
sys.path.append(os.path.dirname(__file__))

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from rag_pipeline import load_vectorstore, build_vectorstore, format_docs

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

# ── Page config ───────────────────────────────────────────
st.set_page_config(
    page_title="MediQuery",
    page_icon="🏥",
    layout="wide"
)

# ── Cache vectorstore ─────────────────────────────────────
@st.cache_resource
def get_retriever():
    db = load_vectorstore()
    return db.as_retriever(search_kwargs={"k": 4})

# ── LLM ───────────────────────────────────────────────────
@st.cache_resource
def get_llm():
    return ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0
    )

# ── Rephrase follow-up into standalone question ───────────
def rephrase_question(question, chat_history, llm):
    if not chat_history:
        return question

    rephrase_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """Given a conversation history and a follow-up question, 
rephrase the follow-up into a fully standalone question.
The standalone question must make complete sense without the conversation history.
If the question is already standalone, return it as-is.
Return ONLY the rephrased question with no explanation."""
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "Follow-up question: {question}\n\nStandalone question:")
    ])

    chain = rephrase_prompt | llm | StrOutputParser()
    return chain.invoke({
        "question": question,
        "chat_history": chat_history
    })

# ── Answer using context + history ────────────────────────
def answer_question(standalone_question, chat_history, retriever, llm):
    # Retrieve relevant docs
    docs    = retriever.invoke(standalone_question)
    context = format_docs(docs)

    answer_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are MediQuery, a helpful medical assistant.
Answer questions ONLY using the retrieved context below from official FDA drug documents.
If the answer is not found in the context, say exactly:
"I don't have enough information in the documents to answer this."
Never make up medical information. Be clear and concise.

Retrieved context:
{context}"""
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])

    chain = answer_prompt | llm | StrOutputParser()

    answer = chain.invoke({
        "question":     standalone_question,
        "chat_history": chat_history,
        "context":      context
    })

    return answer, docs

# ── Initialize session state ──────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.title("🏥 MediQuery")
    st.caption("Medical Document Intelligence Assistant")
    st.divider()

    st.subheader("📂 Knowledge Base")
    st.info(
        "8 FDA Drug Labels indexed:\n\n"
        "💊 Metformin · Amoxicillin\n\n"
        "💊 Atorvastatin · Ibuprofen\n\n"
        "💊 Lisinopril · Sertraline\n\n"
        "💊 Warfarin · Omeprazole"
    )

    st.divider()

    if st.button("🔄 Rebuild Index", use_container_width=True):
        with st.spinner("Indexing PDFs... (~2 minutes)"):
            build_vectorstore(
                os.path.join(os.path.dirname(__file__), '..', 'data', 'pdfs')
            )
        st.success("Index rebuilt!")

    st.divider()

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.messages     = []
        st.rerun()

    st.divider()
    st.caption(
        "⚠️ For educational purposes only.\n"
        "Not a substitute for professional medical advice."
    )

# ── Main header ───────────────────────────────────────────
st.title("MediQuery 🏥")
st.caption("Ask questions about FDA-approved drug labels. Answers grounded in official documentation.")

# ── Example buttons ───────────────────────────────────────
st.write("**Quick questions:**")
col1, col2, col3 = st.columns(3)

example_query = None

with col1:
    if st.button("Side effects of Metformin?", use_container_width=True):
        example_query = "What are the side effects of Metformin?"
with col2:
    if st.button("Warfarin + Ibuprofen interaction?", use_container_width=True):
        example_query = "Can Warfarin and Ibuprofen be taken together?"
with col3:
    if st.button("Amoxicillin dose for children?", use_container_width=True):
        example_query = "What is the recommended dose of Amoxicillin for children?"

st.divider()

# ── Render existing chat messages ─────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg["role"] == "assistant" and "sources" in msg and msg["sources"]:
            with st.expander("📄 Sources"):
                for s in msg["sources"]:
                    st.caption(s)

# ── Chat input ────────────────────────────────────────────
query = st.chat_input("Ask about any drug in the knowledge base...") or example_query

if query:
    # Show user message immediately
    with st.chat_message("user"):
        st.write(query)

    st.session_state.messages.append({
        "role":    "user",
        "content": query
    })

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Searching documents..."):

            llm       = get_llm()
            retriever = get_retriever()

            # Step 1: rephrase if follow-up
            standalone = rephrase_question(
                query,
                st.session_state.chat_history,
                llm
            )

            # Step 2: retrieve + answer
            answer, source_docs = answer_question(
                standalone,
                st.session_state.chat_history,
                retriever,
                llm
            )

        st.write(answer)

        # Show sources
        sources = []
        seen    = set()
        for doc in source_docs:
            src  = doc.metadata.get("source", "Unknown")
            pg   = doc.metadata.get("page", "?")
            name = os.path.basename(src)
            key  = f"{name}-{pg}"
            if key not in seen:
                seen.add(key)
                sources.append(f"📄 **{name}** — Page {int(pg) + 1}")

        if sources:
            with st.expander("📄 Sources"):
                for s in sources:
                    st.caption(s)

    # Save to session state
    st.session_state.messages.append({
        "role":    "assistant",
        "content": answer,
        "sources": sources
    })

    # Update LangChain memory
    st.session_state.chat_history.extend([
        HumanMessage(content=query),
        AIMessage(content=answer)
    ])