import os
from pathlib import Path

import faiss
import numpy as np
import streamlit as st
from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="Zyro Dynamics HR Help Desk", page_icon="📘", layout="wide")
st.title("📘 Zyro Dynamics – HR Help Desk")
st.caption("Answers are grounded in the provided HR policy PDFs. Check the citations for the source.")

CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
FETCH_K = 25
TOP_K = 5
LAMBDA_MULT = 0.75
# Tune this on known in-scope and out-of-scope questions before relying on it.
MIN_SIMILARITY = 0.35
CORPUS_DIR = Path("zyro-dynamics-hr-corpus")

@st.cache_resource(show_spinner="Loading embedding model and HR policy documents...")
def build_index():
    pdf_paths = sorted(CORPUS_DIR.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(
            f"No PDFs found. Add the HR policy PDFs to: {CORPUS_DIR.resolve()}"
        )

    texts, metadata = [], []
    for pdf_path in pdf_paths:
        # This handbook belongs to a different company; do not use it as Zyro policy.
        if pdf_path.name == "01_Employee_Handbook.pdf":
            continue
        reader = PdfReader(str(pdf_path))
        for page_num, page in enumerate(reader.pages, start=1):
            page_text = (page.extract_text() or "").strip()
            if not page_text:
                continue
            start = 0
            while start < len(page_text):
                end = min(start + CHUNK_SIZE, len(page_text))
                chunk = page_text[start:end].strip()
                if chunk:
                    texts.append(chunk)
                    metadata.append({
                        "file": pdf_path.name,
                        "page": page_num,
                        "text": chunk
                    })
                if end >= len(page_text):
                    break
                start = end - CHUNK_OVERLAP

    if not texts:
        raise ValueError("No readable text was extracted from the PDFs.")

    model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    vectors = model.encode(
        texts, convert_to_numpy=True, normalize_embeddings=True,
        show_progress_bar=False
    ).astype("float32")
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return model, index, vectors, metadata

def mmr_search(question, model, index, vectors, metadata):
    q_vec = model.encode(
        [question], convert_to_numpy=True, normalize_embeddings=True
    ).astype("float32")
    scores, indices = index.search(q_vec, min(FETCH_K, len(metadata)))
    candidates = [int(i) for i in indices[0] if i >= 0]
    candidate_scores = {int(i): float(s) for i, s in zip(indices[0], scores[0]) if i >= 0}
    if not candidates:
        return [], 0.0

    selected = [max(candidates, key=lambda i: candidate_scores[i])]
    while len(selected) < min(TOP_K, len(candidates)):
        remaining = [i for i in candidates if i not in selected]
        best_i, best_score = None, -1e9
        for i in remaining:
            relevance = candidate_scores[i]
            redundancy = max(float(np.dot(vectors[i], vectors[j])) for j in selected)
            mmr_score = LAMBDA_MULT * relevance - (1 - LAMBDA_MULT) * redundancy
            if mmr_score > best_score:
                best_i, best_score = i, mmr_score
        if best_i is None:
            break
        selected.append(best_i)

    return [metadata[i] for i in selected], max(candidate_scores.values())

def answer_question(question, model, index, vectors, metadata, client):
    retrieved, top_similarity = mmr_search(question, model, index, vectors, metadata)
    if not retrieved or top_similarity < MIN_SIMILARITY:
        return (
            "I’m sorry, but I couldn’t find this information in the provided Zyro Dynamics HR policy documents. Please contact HR for assistance.",
            []
        )

    context = "\n\n".join(
        f"[Source: {d['file']}, page {d['page']}]\n{d['text']}" for d in retrieved
    )
    prompt = f"""You are Zyro Dynamics' HR policy assistant.
Answer only using the supplied HR policy context.
If the context does not directly contain enough information to answer, politely say the information is unavailable in the provided HR policy documents and suggest contacting HR.
Do not guess, invent policy details, or use outside knowledge.
Be concise but include important numbers, dates, eligibility conditions, and exceptions.

HR POLICY CONTEXT:
{context}

EMPLOYEE QUESTION:
{question}

ANSWER:"""
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=700,
    )
    return response.choices[0].message.content.strip(), retrieved

with st.sidebar:
    st.subheader("About this chatbot")
    st.write("This demo retrieves information from Zyro Dynamics HR policy PDFs.")
    st.write("Retrieval: FAISS + MMR")
    st.write("Embeddings: BAAI/bge-small-en-v1.5")
    st.write("LLM: Groq Llama 3.3 70B")
    st.caption("A similarity threshold is used as an initial out-of-scope guardrail; it should be validated against test questions.")

api_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY", ""))
if not api_key:
    st.error("GROQ_API_KEY is missing. Add it in Streamlit Cloud → App settings → Secrets.")
    st.stop()

try:
    model, index, vectors, metadata = build_index()
except Exception as e:
    st.error(str(e))
    st.info("Upload the 10 relevant Zyro Dynamics HR policy PDFs into the corpus folder in your GitHub repository. Do not include the unrelated Acrux Dynamics handbook.")
    st.stop()

client = Groq(api_key=api_key)
question = st.text_input(
    "Ask an HR policy question",
    placeholder="Example: How does my Earned Leave accrue every month?"
)
if st.button("Get answer", type="primary", disabled=not question.strip()):
    with st.spinner("Searching HR policies and generating an answer..."):
        try:
            answer, sources = answer_question(question.strip(), model, index, vectors, metadata, client)
            st.subheader("Answer")
            st.write(answer)
            if sources:
                st.subheader("Citations")
                st.caption("Open the matching PDF in the corpus and check the page and quoted passage below.")
                seen = set()
                for src in sources:
                    key = (src["file"], src["page"])
                    if key in seen:
                        continue
                    seen.add(key)
                    with st.expander(f"{src['file']} — page {src['page']}"):
                        st.write(src["text"])
        except Exception as e:
            st.error(f"Could not generate an answer: {e}")

st.divider()
st.caption("Prototype for the Intelligent RAG project. Verify policy details against the cited PDF before acting.")