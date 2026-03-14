import os
import json
import time
import warnings
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

import requests
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
VECTOR_DIR  = os.path.join(os.path.dirname(__file__), '..', 'vectorstore')
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_URL     = "https://api.groq.com/openai/v1/chat/completions"
MODEL        = "llama-3.1-8b-instant"

TEST_QUESTIONS = [
    "What are the common side effects of Metformin?",
    "What is the maximum daily dose of Metformin for adults?",
    "What are the contraindications of Metformin in kidney patients?",
    "What infections is Amoxicillin used to treat?",
    "What is the recommended dose of Amoxicillin for children?",
    "Can patients with penicillin allergy take Amoxicillin?",
    "What are the side effects of Atorvastatin?",
    "Can Atorvastatin cause liver problems?",
    "What are the common uses of Ibuprofen?",
    "What is the maximum daily dose of Ibuprofen for adults?",
    "What are the contraindications of Lisinopril?",
    "What side effects does Lisinopril cause?",
    "What is Sertraline used for?",
    "What are the warnings associated with Sertraline in young patients?",
    "What are the serious risks of taking Warfarin?",
    "What drugs interact with Warfarin?",
    "Can Warfarin and Ibuprofen be taken together?",
    "What is Omeprazole used for?",
    "What are the side effects of Omeprazole?",
    "What is the recommended dosage of Omeprazole?"
]

def groq_call(messages, temperature=0):
    """Direct Groq API call — no LangChain dependency."""
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type":  "application/json"
    }
    payload = {
        "model":       MODEL,
        "messages":    messages,
        "temperature": temperature
    }
    for attempt in range(3):
        try:
            resp = requests.post(GROQ_URL, headers=headers, json=payload, timeout=30)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
        except requests.exceptions.HTTPError as e:
            if resp.status_code == 429:
                wait = 10 * (attempt + 1)
                print(f"    Rate limit hit — waiting {wait}s...")
                time.sleep(wait)
            else:
                raise e
        except Exception as e:
            if attempt == 2:
                raise e
            time.sleep(5)
    return ""

def get_answer_and_contexts(question, db):
    """Retrieve context and generate answer."""
    docs     = db.similarity_search(question, k=4)
    context  = "\n\n".join(doc.page_content for doc in docs)
    contexts = [doc.page_content for doc in docs]

    answer = groq_call([
        {
            "role": "system",
            "content": (
                "You are a helpful medical assistant. "
                "Answer ONLY using the context provided. "
                "If the answer is not in the context, say "
                "'I don't have enough information.' "
                "Do not make up information.\n\n"
                f"Context:\n{context}"
            )
        },
        {"role": "user", "content": question}
    ])
    return answer, contexts, context

def score_faithfulness(question, answer, context):
    """
    Faithfulness: does every claim in the answer appear in the context?
    Returns score 0.0 to 1.0
    """
    prompt = f"""You are evaluating an AI answer for faithfulness.
Faithfulness means every statement in the answer is directly supported by the context.

Question: {question}
Context: {context[:2000]}
Answer: {answer}

Instructions:
1. List each factual claim in the answer
2. Check if each claim is supported by the context
3. Calculate: supported_claims / total_claims
4. Return ONLY a JSON object like: {{"score": 0.85, "supported": 8, "total": 10}}
Return ONLY the JSON, nothing else."""

    result = groq_call([{"role": "user", "content": prompt}])
    try:
        data = json.loads(result)
        return float(data.get("score", 0.0))
    except Exception:
        # Fallback — try to extract a number
        import re
        match = re.search(r'"score"\s*:\s*([0-9.]+)', result)
        return float(match.group(1)) if match else 0.5

def score_answer_relevancy(question, answer):
    """
    Answer Relevancy: does the answer actually address the question?
    Returns score 0.0 to 1.0
    """
    prompt = f"""You are evaluating whether an AI answer is relevant to the question asked.

Question: {question}
Answer: {answer}

Rate how directly and completely the answer addresses the question.
- 1.0 = perfectly answers the question
- 0.7 = mostly answers with minor gaps
- 0.5 = partially answers
- 0.2 = barely related
- 0.0 = does not answer at all

Return ONLY a JSON object like: {{"score": 0.85, "reason": "brief reason"}}
Return ONLY the JSON, nothing else."""

    result = groq_call([{"role": "user", "content": prompt}])
    try:
        data = json.loads(result)
        return float(data.get("score", 0.0))
    except Exception:
        import re
        match = re.search(r'"score"\s*:\s*([0-9.]+)', result)
        return float(match.group(1)) if match else 0.5

def main():
    print("=" * 55)
    print("       MediQuery — RAGAS-Style Evaluation")
    print("=" * 55)

    print("\nLoading vectorstore...")
    emb = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    db  = Chroma(persist_directory=VECTOR_DIR, embedding_function=emb)
    print("Vectorstore loaded.")

    results = []

    print(f"\nStep 1/3 — Generating answers for {len(TEST_QUESTIONS)} questions...\n")

    for i, question in enumerate(TEST_QUESTIONS):
        print(f"  [{i+1:02d}/{len(TEST_QUESTIONS)}] {question}")
        try:
            answer, contexts, context = get_answer_and_contexts(question, db)
            results.append({
                "question": question,
                "answer":   answer,
                "contexts": contexts,
                "context":  context
            })
            time.sleep(0.5)   # avoid rate limits
        except Exception as e:
            print(f"         Error: {e}")
            continue

    print(f"\n  {len(results)}/{len(TEST_QUESTIONS)} answers generated.")

    print(f"\nStep 2/3 — Scoring faithfulness...\n")
    for i, r in enumerate(results):
        print(f"  [{i+1:02d}/{len(results)}] Scoring: {r['question'][:50]}...")
        r["faithfulness"] = score_faithfulness(
            r["question"], r["answer"], r["context"]
        )
        time.sleep(1.0)   # give Groq breathing room

    print(f"\nStep 3/3 — Scoring answer relevancy...\n")
    for i, r in enumerate(results):
        print(f"  [{i+1:02d}/{len(results)}] Scoring: {r['question'][:50]}...")
        r["answer_relevancy"] = score_answer_relevancy(
            r["question"], r["answer"]
        )
        time.sleep(1.0)

    # ── Compute averages ──────────────────────────────────
    avg_faith   = sum(r["faithfulness"]     for r in results) / len(results)
    avg_relev   = sum(r["answer_relevancy"] for r in results) / len(results)

    # ── Print results ─────────────────────────────────────
    print("\n" + "=" * 55)
    print("           EVALUATION RESULTS")
    print("=" * 55)
    print(f"  Faithfulness:      {avg_faith:.4f}  ({avg_faith*100:.1f}%)")
    print(f"  Answer Relevancy:  {avg_relev:.4f}  ({avg_relev*100:.1f}%)")
    print("=" * 55)
    print(f"  Questions evaluated: {len(results)}/20")
    print("=" * 55)

    print("\nPer-question breakdown:")
    print(f"  {'Question':<52} {'Faith':>6} {'Relev':>6}")
    print("  " + "-" * 66)
    for r in results:
        q = r["question"][:50] + ".." if len(r["question"]) > 50 else r["question"]
        print(f"  {q:<52} {r['faithfulness']:>6.3f} {r['answer_relevancy']:>6.3f}")

    # ── Save CSV ──────────────────────────────────────────
    import csv
    out_path = os.path.join(os.path.dirname(__file__), 'ragas_results.csv')
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'question', 'answer', 'faithfulness', 'answer_relevancy'
        ])
        writer.writeheader()
        for r in results:
            writer.writerow({
                'question':         r['question'],
                'answer':           r['answer'],
                'faithfulness':     round(r['faithfulness'], 4),
                'answer_relevancy': round(r['answer_relevancy'], 4)
            })

    print(f"\n  Results saved to tests/ragas_results.csv")
    print("\n  Add these scores to your README!")

if __name__ == "__main__":
    main()