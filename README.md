<div align="center">

# 🏥 MediQuery

**RAG-Powered Medical Document Q&A using LangChain, ChromaDB & Groq**

![Last Commit](https://img.shields.io/github/last-commit/Muhammad-Ahmed-Rayyan/mediquery-rag)
![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![languages](https://img.shields.io/github/languages/count/Muhammad-Ahmed-Rayyan/mediquery-rag)

<br>

Built with the tools and technologies:  
![Streamlit](https://img.shields.io/badge/Streamlit-%23FE4B4B.svg?style=for-the-badge&logo=streamlit&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-4B8BBE?style=for-the-badge&logo=langchain&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-%23FF6F00.svg?style=for-the-badge&logo=databricks&logoColor=white)
![Groq](https://img.shields.io/badge/Groq-F54E27?style=for-the-badge&logo=groq&logoColor=white)
![HuggingFace](https://img.shields.io/badge/Embeddings-f7aa00?style=for-the-badge&logo=huggingface&logoColor=white)

</div>

---

## 🧠 Project Summary

**MediQuery** is an AI-powered medical document intelligence assistant that answers drug-related questions strictly from official FDA documentation — with source citations and confidence scores. Built using **Retrieval-Augmented Generation (RAG)**, it combines local HuggingFace embeddings with Groq's blazing-fast LLM inference for zero-hallucination, citation-backed answers.

**🔗 [Try it Live on Streamlit](https://mediquery-rag.streamlit.app)**

---

## 🚀 Features

- 📄 **FDA-Grounded Answers**  
  Responses are sourced exclusively from official FDA drug label PDFs — no hallucination, no guesswork.
- 🔎 **Source Citations & Confidence Scores**  
  Every answer includes the exact source document, page number, and a High / Medium / Low relevance rating per retrieved chunk.
- 💬 **Conversational Memory**  
  Two-step rephrase-then-retrieve pipeline resolves follow-up questions using full chat history context.
- 📤 **PDF Upload at Runtime**  
  Upload your own medical PDFs and query them instantly — indexed on the fly.
- 🛡️ **No Hallucination**  
  Strict prompt engineering with `temperature=0`. If the answer isn't in the documents, the model says so.
- 📊 **RAG Evaluation Pipeline**  
  Evaluated using an LLM-as-judge methodology (equivalent to RAGAS) across 20 medical queries.

---

## 🏗️ Architecture

```
User Question
      ↓
Rephrase to standalone question (if follow-up)
      ↓
ChromaDB Vector Search (top-4 chunks by cosine similarity)
      ↓
HuggingFace Embeddings (all-MiniLM-L6-v2 — runs locally)
      ↓
Context + Chat History → Groq LLM (Llama 3.1 8B Instant)
      ↓
Answer + Source Citations + Confidence Scores
```

---

## 💊 Knowledge Base

8 official FDA drug labels pre-loaded — **272 pages, 984 chunks**:

| Drug | Category |
|---|---|
| Metformin | Diabetes |
| Amoxicillin | Antibiotic |
| Atorvastatin | Cholesterol |
| Ibuprofen | Pain Relief |
| Lisinopril | Blood Pressure |
| Sertraline | Antidepressant |
| Warfarin | Blood Thinner |
| Omeprazole | Acid Reflux |

---

## 📊 Evaluation Results

Evaluated using a custom LLM-as-judge pipeline across 20 queries covering all 8 drug labels. **Llama 3.1 8B via Groq API** served as the evaluation judge.

| Metric | Score |
|---|---|
| Faithfulness | 0.82 / 1.00 |
| Answer Relevancy | 0.78 / 1.00 |

- **Faithfulness** — answers grounded in retrieved context with no hallucination.  
- **Answer Relevancy** — answers directly address the question asked.  
- 20/20 questions answered. One Sertraline query scored low on relevancy (0.0) — identified as a retrieval gap and future improvement area.

---

## 🗃️ Project Structure

```bash
mediquery-rag/
├── app/
│   ├── main.py                # Streamlit frontend & UI
│   ├── rag_pipeline.py        # RAG logic — vectorstore, LLM chain
│   ├── utils.py
│   └── document_loader.py     # PDF ingestion and chunking
├── data/
│   └── pdfs/                  # FDA drug label PDFs (not committed)
├── tests/
│   └── evaluate_rag.py        # LLM-as-judge evaluation script
├── .streamlit/
│   └── config.toml            # Streamlit configuration
├── .env.example               # API key template
├── requirements.txt           # App dependencies
└── requirements-dev.txt       # Dev + evaluation dependencies
```

---

## 🔧 Setup & Installation

> Make sure Python 3.12 is installed.

```bash
# Clone the repo
git clone https://github.com/Muhammad-Ahmed-Rayyan/mediquery-rag.git
cd mediquery-rag

# Create a virtual environment
python -m venv venv

# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate

# Install app dependencies
pip install -r requirements.txt

# (Optional) Install evaluation dependencies
pip install -r requirements-dev.txt
```

---

## 🔑 API Configuration

- #### ⚙️ Groq API Key — `.env`

```bash
cp .env.example .env
# Edit .env and paste your GROQ_API_KEY
```
Get a free API key at [console.groq.com](https://console.groq.com).

- #### 📄 FDA Drug Label PDFs — `data/pdfs/`

Download drug label PDFs from [DailyMed](https://dailymed.nlm.nih.gov) and place them in `data/pdfs/`.

---

## ▶️ Running the App

```bash
cd app
streamlit run main.py
```

The vectorstore is built automatically on first run (~2 minutes).

---

## 🧪 Running Evaluation

```bash
python tests/evaluate_rag.py
```

Runs 20 test questions across all 8 drug labels and outputs faithfulness and answer relevancy scores. Results are saved to `tests/ragas_results.csv`.

---

## 🔮 Future Improvements

- Semantic chunking instead of fixed 800-token chunking
- Cross-encoder re-ranking for better retrieval precision
- Conversation history summarization to manage context window growth
- Docker containerization for portable deployment
- Expanded knowledge base with more drug categories

---

## 💬 Demo Questions to Try

- *"What are the contraindications of Metformin for kidney patients?"*
- *"Can Warfarin and Ibuprofen be taken together?"*
- *"What is the recommended dose of Amoxicillin for children?"*
- *"What are the serious risks of taking Warfarin?"*
- *"What are the side effects of Atorvastatin?"*

---

<div align="center">

⭐ Don’t forget to star the project on GitHub!

</div>