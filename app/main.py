import sys
import os
sys.path.append(os.path.dirname(__file__))

import streamlit as st
import tempfile
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from rag_pipeline import (
    load_vectorstore,
    build_vectorstore,
    add_pdf_to_vectorstore,
    get_retriever_with_scores,
    format_docs
)

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

# ── Page config ───────────────────────────────────────────
st.set_page_config(
    page_title="MediQuery",
    page_icon="🏥",
    layout="wide"
)

# ── Cache vectorstore and LLM ─────────────────────────────
@st.cache_resource
def get_db():
    return load_vectorstore()

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

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """Given a conversation history and a follow-up question,
rephrase the follow-up into a fully standalone question that makes
complete sense without the conversation history.
Return ONLY the rephrased question, nothing else."""
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "Follow-up: {question}\n\nStandalone question:")
    ])

    return (prompt | llm | StrOutputParser()).invoke({
        "question":     question,
        "chat_history": chat_history
    })

# ── Answer using context + history ────────────────────────
def answer_question(standalone_question, chat_history, db, llm):
    docs, scores = get_retriever_with_scores(db, standalone_question, k=4)
    context      = format_docs(docs)

    prompt = ChatPromptTemplate.from_messages([
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

    answer = (prompt | llm | StrOutputParser()).invoke({
        "question":     standalone_question,
        "chat_history": chat_history,
        "context":      context
    })

    return answer, docs, scores

# ── Score to color + label helper ────────────────────────
def score_display(score: float):
    pct = round(score * 100)
    if score >= 0.75:
        color = "#22c55e"   # green
        label = "High"
    elif score >= 0.50:
        color = "#f59e0b"   # amber
        label = "Medium"
    else:
        color = "#ef4444"   # red
        label = "Low"
    return pct, color, label

# ── Initialize session state ──────────────────────────────
if "chat_history"     not in st.session_state:
    st.session_state.chat_history     = []
if "messages"         not in st.session_state:
    st.session_state.messages         = []
if "uploaded_files"   not in st.session_state:
    st.session_state.uploaded_files   = []

# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.title("🏥 MediQuery")
    st.caption("Medical Document Intelligence Assistant")
    st.divider()

    # Knowledge base info
    st.subheader("📂 Pre-loaded Knowledge Base")
    with st.expander("View 8 FDA Drug Labels"):
        st.markdown(
            "- Metformin\n"
            "- Amoxicillin\n"
            "- Atorvastatin\n"
            "- Ibuprofen\n"
            "- Lisinopril\n"
            "- Sertraline\n"
            "- Warfarin\n"
            "- Omeprazole"
        )

    st.divider()

    # Document upload section
    st.subheader("📤 Upload Your Own PDF")
    uploaded_file = st.file_uploader(
        "Upload a medical PDF",
        type=["pdf"],
        help="Upload any medical document to include it in the knowledge base"
    )

    if uploaded_file is not None:
        if uploaded_file.name not in st.session_state.uploaded_files:
            with st.spinner(f"Processing {uploaded_file.name}..."):
                # Save to temp file and add to vectorstore
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".pdf",
                    prefix=uploaded_file.name.replace(".pdf", "_")
                ) as tmp:
                    tmp.write(uploaded_file.getbuffer())
                    tmp_path = tmp.name

                chunks_added = add_pdf_to_vectorstore(tmp_path)

                # Clear cached vectorstore so it reloads with new docs
                get_db.clear()
                os.unlink(tmp_path)

            st.session_state.uploaded_files.append(uploaded_file.name)
            st.success(f"✅ Added {uploaded_file.name} ({chunks_added} chunks)")
        else:
            st.info(f"✅ {uploaded_file.name} already in knowledge base")

    # Show uploaded files list
    if st.session_state.uploaded_files:
        st.write("**Your uploads:**")
        for f in st.session_state.uploaded_files:
            st.caption(f"📄 {f}")

    st.divider()

    # Rebuild index
    if st.button("🔄 Rebuild Default Index", use_container_width=True):
        with st.spinner("Rebuilding index..."):
            build_vectorstore(
                os.path.join(os.path.dirname(__file__), '..', 'data', 'pdfs')
            )
            get_db.clear()
            st.session_state.uploaded_files = []
        st.success("Index rebuilt!")

    # Clear chat
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.messages     = []
        st.rerun()

    st.divider()
    st.caption(
        "⚠️ For educational purposes only.\n"
        "Not a substitute for professional medical advice."
    )

# ── Main area ─────────────────────────────────────────────
st.title("MediQuery 🏥")
st.caption(
    "Ask questions about FDA-approved drug labels. "
    "Answers are grounded in official documentation with confidence scores."
)

# Example buttons
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

# ── Render chat history ───────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("📄 Sources & Confidence"):
                for s in msg["sources"]:
                    st.markdown(s, unsafe_allow_html=True)

# ── Chat input ────────────────────────────────────────────
query = st.chat_input("Ask about any drug in the knowledge base...") or example_query

if query:
    # Render user message
    with st.chat_message("user"):
        st.write(query)
    st.session_state.messages.append({
        "role":    "user",
        "content": query
    })

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Searching documents..."):
            llm = get_llm()
            db  = get_db()

            # Rephrase if follow-up
            standalone = rephrase_question(
                query,
                st.session_state.chat_history,
                llm
            )

            # Get answer + sources + scores
            answer, source_docs, scores = answer_question(
                standalone,
                st.session_state.chat_history,
                db,
                llm
            )

        st.write(answer)

        # Build source cards with confidence scores
        sources = []
        seen    = set()
        for doc, score in zip(source_docs, scores):
            src  = doc.metadata.get("source", "Unknown")
            pg   = doc.metadata.get("page", "?")
            name = os.path.basename(src)
            key  = f"{name}-{pg}"
            if key not in seen:
                seen.add(key)
                pct, color, label = score_display(score)
                source_line = (
                    f'<span style="font-size:13px;">📄 <b>{name}</b> — '
                    f'Page {int(pg) + 1} &nbsp;'
                    f'<span style="background:{color}; color:white; '
                    f'padding:2px 8px; border-radius:10px; font-size:11px;">'
                    f'{label} {pct}%</span></span>'
                )
                sources.append(source_line)

        if sources:
            with st.expander("📄 Sources & Confidence"):
                for s in sources:
                    st.markdown(s, unsafe_allow_html=True)

    # Save to session state
    st.session_state.messages.append({
        "role":    "assistant",
        "content": answer,
        "sources": sources
    })

    # Update memory
    st.session_state.chat_history.extend([
        HumanMessage(content=query),
        AIMessage(content=answer)
    ])