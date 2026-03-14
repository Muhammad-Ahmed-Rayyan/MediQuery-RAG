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

# Pull secrets from Streamlit Cloud if available
import streamlit as st
if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

# ── Page config ───────────────────────────────────────────
st.set_page_config(
    page_title="MediQuery",
    page_icon="🏥",
    layout="wide"
)

# ── Custom CSS ────────────────────────────────────────────
st.markdown("""
<style>
/* Center and constrain main content always */
.main .block-container {
    max-width: 780px !important;
    width: 100% !important;
    margin: 0 auto !important;
    padding-left: 1.5rem !important;
    padding-right: 1.5rem !important;
}

/* Sidebar top padding fix */
section[data-testid="stSidebar"] > div:first-child {
    padding-top: 0rem !important;
}

/* Disclaimer at bottom */
.disclaimer-bar {
    position: fixed;
    bottom: 2px;
    left: 0;
    right: 0;
    text-align: center;
    color: #9ca3af;
    font-size: 11.5px;
    z-index: 1000;
}
div[data-testid="stChatInput"] {
    margin-bottom: 6px;
}

/* Copy button styling */
.copy-btn {
    background: none;
    border: 1px solid #e5e7eb;
    border-radius: 5px;
    cursor: pointer;
    padding: 3px 7px;
    color: #9ca3af;
    font-size: 11px;
    display: inline-flex;
    align-items: center;
    gap: 4px;
    margin-top: 6px;
    transition: all 0.15s ease;
}
.copy-btn:hover {
    background: #f3f4f6;
    color: #374151;
    border-color: #d1d5db;
}
</style>
""", unsafe_allow_html=True)

# ── Cache vectorstore and LLM ─────────────────────────────
@st.cache_resource
def get_db():
    try:
        db = load_vectorstore()
        return db
    except FileNotFoundError:
        # Auto-build on first run (Streamlit Cloud)
        with st.spinner("Building knowledge base for first time... (~2 minutes)"):
            build_vectorstore(
                os.path.join(os.path.dirname(__file__), '..', 'data', 'pdfs')
            )
        return load_vectorstore()
    except RuntimeError as e:
        return None

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

# ── Score to color + label helper ─────────────────────────
def score_display(score: float):
    pct = round(score * 100)
    if score >= 0.75:
        color = "#22c55e"
        label = "High"
    elif score >= 0.50:
        color = "#f59e0b"
        label = "Medium"
    else:
        color = "#ef4444"
        label = "Low"
    return pct, color, label

# ── Initialize session state ──────────────────────────────
if "chat_history"   not in st.session_state:
    st.session_state.chat_history   = []
if "messages"       not in st.session_state:
    st.session_state.messages       = []
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
if "chat_started"   not in st.session_state:
    st.session_state.chat_started   = False

# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
        <div style="display:flex; align-items:center; gap:10px; padding: 4px 0 8px 0;">
            <span style="font-size:26px;">🏥</span>
            <div>
                <div style="font-size:18px; font-weight:700; line-height:1.2;">MediQuery</div>
                <div style="font-size:11px; color:#6b7280; line-height:1.4;">
                    Medical Document Intelligence Assistant
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Pre-loaded knowledge base
    st.markdown("""
        <div style="display:flex; align-items:center; gap:8px; margin-bottom:8px;">
            <svg xmlns="http://www.w3.org/2000/svg" width="15" height="15"
                 viewBox="0 0 24 24" fill="none" stroke="currentColor"
                 stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <ellipse cx="12" cy="5" rx="9" ry="3"/>
                <path d="M3 5v14c0 1.66 4.03 3 9 3s9-1.34 9-3V5"/>
                <path d="M3 12c0 1.66 4.03 3 9 3s9-1.34 9-3"/>
            </svg>
            <span style="font-weight:600; font-size:14px;">Pre-loaded Knowledge Base</span>
        </div>
    """, unsafe_allow_html=True)

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


    # Upload section
    st.markdown("""
        <div style="display:flex; align-items:center; gap:8px; margin-bottom:8px;">
            <svg xmlns="http://www.w3.org/2000/svg" width="15" height="15"
                 viewBox="0 0 24 24" fill="none" stroke="currentColor"
                 stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                <polyline points="17 8 12 3 7 8"/>
                <line x1="12" y1="3" x2="12" y2="15"/>
            </svg>
            <span style="font-weight:600; font-size:14px;">Upload Your Own PDF</span>
        </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload a medical PDF",
        type=["pdf"],
        help="Upload any medical document to add it to the knowledge base",
        label_visibility="collapsed"
    )

    if uploaded_file is not None:
            if uploaded_file.name not in st.session_state.uploaded_files:
                with st.spinner(f"Processing {uploaded_file.name}..."):
                    try:
                        with tempfile.NamedTemporaryFile(
                            delete=False,
                            suffix=".pdf",
                            prefix=uploaded_file.name.replace(".pdf", "_")
                        ) as tmp:
                            tmp.write(uploaded_file.getbuffer())
                            tmp_path = tmp.name

                        chunks_added = add_pdf_to_vectorstore(tmp_path)
                        get_db.clear()
                        os.unlink(tmp_path)
                        st.session_state.uploaded_files.append(uploaded_file.name)
                        st.success(f"Added {uploaded_file.name} ({chunks_added} chunks)")

                    except RuntimeError as e:
                        st.error(f"Could not process PDF: {e}")
                    except Exception as e:
                        st.error(f"Unexpected error: {e}")
            else:
                st.info(f"{uploaded_file.name} already indexed")

    if st.session_state.uploaded_files:
        st.write("**Your uploads:**")
        for f in st.session_state.uploaded_files:
            st.caption(f"— {f}")

    st.divider()

    # Rebuild index button — icon inside button via markdown trick
    if st.button(
        "↺  Rebuild Default Index",
        use_container_width=True,
        help="Re-index all 8 pre-loaded FDA drug PDFs"
    ):
        with st.spinner("Rebuilding index..."):
            build_vectorstore(
                os.path.join(os.path.dirname(__file__), '..', 'data', 'pdfs')
            )
            get_db.clear()
            st.session_state.uploaded_files = []
        st.success("Index rebuilt!")

    # Clear chat button
    if st.button(
        "⊘  Clear Chat",
        use_container_width=True,
        help="Clear all messages and start a new conversation"
    ):
        st.session_state.chat_history = []
        st.session_state.messages     = []
        st.session_state.chat_started = False
        st.rerun()

# ── Main area ─────────────────────────────────────────────

# Welcome screen — only before chat starts
if not st.session_state.chat_started:
    st.markdown("""
            <div style="text-align:center; padding: 20px 0 10px 0;">
                <h1 style="font-size:2.2rem; margin-bottom:6px;">MediQuery 🏥</h1>
                <p style="color:#6b7280; font-size:15px;">
                    Ask questions about FDA-approved drug labels.<br>
                    Answers are grounded in official documentation with confidence scores.
                </p>
            </div>
        """, unsafe_allow_html=True)

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
else:
    example_query = None

# ── Render chat history ───────────────────────────────────
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

        # Copy button on assistant messages
        if msg["role"] == "assistant":
            # Escape backticks and newlines for JS string
            safe_text = (
                msg["content"]
                .replace("\\", "\\\\")
                .replace("`", "\\`")
                .replace("\n", "\\n")
            )
            st.markdown(
                f"""<button class="copy-btn"
                    onclick="navigator.clipboard.writeText(`{safe_text}`).then(()=>{{
                        this.innerText='✓ Copied';
                        setTimeout(()=>this.innerHTML=
                        '<svg xmlns=\\'http://www.w3.org/2000/svg\\' width=\\'11\\' height=\\'11\\' viewBox=\\'0 0 24 24\\' fill=\\'none\\' stroke=\\'currentColor\\' stroke-width=\\'2\\' stroke-linecap=\\'round\\' stroke-linejoin=\\'round\\'><rect x=\\'9\\' y=\\'9\\' width=\\'13\\' height=\\'13\\' rx=\\'2\\' ry=\\'2\\'></rect><path d=\\'M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1\\'></path></svg> Copy',
                        1500);
                    }})">
                    <svg xmlns="http://www.w3.org/2000/svg" width="11" height="11"
                         viewBox="0 0 24 24" fill="none" stroke="currentColor"
                         stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
                        <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
                    </svg>
                    Copy
                </button>""",
                unsafe_allow_html=True
            )

            if msg.get("sources"):
                with st.expander("Sources & Confidence"):
                    for s in msg["sources"]:
                        st.markdown(s, unsafe_allow_html=True)

# ── Chat input ────────────────────────────────────────────
query = st.chat_input("Ask about any drug in the knowledge base...") or example_query

# ── Handle query ──────────────────────────────────────────
if query:
    st.session_state.chat_started = True

    with st.chat_message("user"):
        st.write(query)
    st.session_state.messages.append({
        "role":    "user",
        "content": query
    })

    with st.chat_message("assistant"):
        # Check vectorstore exists before doing anything
        db = get_db()
        if db is None:
            st.error(
                "Knowledge base not found. "
                "Please click **Rebuild Default Index** in the sidebar first."
            )
            st.stop()

        try:
            with st.spinner("Searching documents..."):
                llm = get_llm()

                standalone = rephrase_question(
                    query,
                    st.session_state.chat_history,
                    llm
                )

                answer, source_docs, scores = answer_question(
                    standalone,
                    st.session_state.chat_history,
                    db,
                    llm
                )

        except Exception as e:
            error_msg = str(e)
            if "rate_limit" in error_msg.lower() or "429" in error_msg:
                st.error("Groq API rate limit reached. Please wait a few seconds and try again.")
            elif "api_key" in error_msg.lower() or "401" in error_msg:
                st.error("Invalid or missing Groq API key. Check your .env file.")
            elif "connection" in error_msg.lower():
                st.error("Connection error. Check your internet connection and try again.")
            else:
                st.error(f"Something went wrong: {error_msg}")
            st.stop()

        st.write(answer)

        # copy button
        safe_text = (
            answer
            .replace("\\", "\\\\")
            .replace("`", "\\`")
            .replace("\n", "\\n")
        )
        st.markdown(
            f"""<button class="copy-btn"
                onclick="navigator.clipboard.writeText(`{safe_text}`).then(()=>{{
                    this.innerText='✓ Copied';
                    setTimeout(()=>this.innerHTML=
                    '<svg xmlns=\\'http://www.w3.org/2000/svg\\' width=\\'11\\' height=\\'11\\' viewBox=\\'0 0 24 24\\' fill=\\'none\\' stroke=\\'currentColor\\' stroke-width=\\'2\\' stroke-linecap=\\'round\\' stroke-linejoin=\\'round\\'><rect x=\\'9\\' y=\\'9\\' width=\\'13\\' height=\\'13\\' rx=\\'2\\' ry=\\'2\\'></rect><path d=\\'M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1\\'></path></svg> Copy',
                    1500);
                }})">
                <svg xmlns="http://www.w3.org/2000/svg" width="11" height="11"
                     viewBox="0 0 24 24" fill="none" stroke="currentColor"
                     stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
                    <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
                </svg>
                Copy
            </button>""",
            unsafe_allow_html=True
        )

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
            with st.expander("Sources & Confidence"):
                for s in sources:
                    st.markdown(s, unsafe_allow_html=True)

    st.session_state.messages.append({
        "role":    "assistant",
        "content": answer,
        "sources": sources
    })

    st.session_state.chat_history.extend([
        HumanMessage(content=query),
        AIMessage(content=answer)
    ])

    st.rerun()

    # Disclaimer — absolute bottom of page
st.markdown(
    '<div class="disclaimer-bar" style="text-align:center; color:#9ca3af; '
    'font-size:11.5px; padding: 8px 0 0 0; margin-top:4px;">'
    'MediQuery can make mistakes. For educational use only — '
    'not a substitute for professional medical advice.'
    '</div>',
    unsafe_allow_html=True
)