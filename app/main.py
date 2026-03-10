import sys
import os
sys.path.append(os.path.dirname(__file__))

import streamlit as st
from dotenv import load_dotenv
from rag_pipeline import build_vectorstore, load_vectorstore, get_qa_chain

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

# ── Page config ───────────────────────────────────────────
st.set_page_config(
    page_title="MediQuery",
    page_icon="🏥",
    layout="wide"
)

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
        st.success("Index rebuilt successfully!")

    st.divider()
    st.caption(
        "⚠️ For educational purposes only.\n"
        "Not a substitute for professional medical advice."
    )

# ── Cache chain so it loads once per session ─────────────
@st.cache_resource
def get_chain():
    db = load_vectorstore()
    return get_qa_chain(db)

# ── Main UI ───────────────────────────────────────────────
st.title("MediQuery")
st.write("Ask questions about 8 FDA-approved drug labels. Every answer is grounded in official documentation with source citations.")

st.divider()

# Quick example buttons
st.write("**Try an example:**")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🔴 Side effects of Metformin?", use_container_width=True):
        st.session_state.query = "What are the side effects of Metformin?"

with col2:
    if st.button("⚠️ Warfarin + Ibuprofen interaction?", use_container_width=True):
        st.session_state.query = "Can Warfarin and Ibuprofen be taken together? What are the risks?"

with col3:
    if st.button("💊 Amoxicillin dose for children?", use_container_width=True):
        st.session_state.query = "What is the recommended dose of Amoxicillin for children?"

st.divider()

# Text input — picks up session state if example button clicked
query = st.text_input(
    "Ask your question:",
    value=st.session_state.get("query", ""),
    placeholder="e.g. What are the contraindications for Lisinopril?"
)

# Clear query from session after it loads into input
if "query" in st.session_state:
    del st.session_state["query"]

# Run the chain when query exists
if query.strip():
    with st.spinner("Searching documents..."):
        chain, retriever = get_chain()
        answer           = chain.invoke({"question": query})
        source_docs      = retriever.invoke(query)

    # Answer box
    st.subheader("Answer")
    st.success(answer)

    # Sources
    st.subheader("Sources")
    seen = set()
    for doc in source_docs:
        src  = doc.metadata.get("source", "Unknown")
        pg   = doc.metadata.get("page", "?")
        name = os.path.basename(src)
        key  = f"{name}-{pg}"
        if key not in seen:
            seen.add(key)
            with st.expander(f"📄 {name} — Page {int(pg) + 1}"):
                st.write(doc.page_content[:500] + "...")