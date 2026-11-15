import os
import uuid
from pathlib import Path
from datetime import datetime

import faiss
import numpy as np
import streamlit as st
from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

st.set_page_config(
    page_title="IntelligentRAG · Zyro HR",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------
# RAG CONFIGURATION — kept consistent with the original project
# ---------------------------------------------------------------------
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
FETCH_K = 25
TOP_K = 5
LAMBDA_MULT = 0.75
MIN_SIMILARITY = 0.35  # Initial guardrail; validate against test questions.
CORPUS_DIR = Path("zyro-dynamics-hr-corpus")

# ---------------------------------------------------------------------
# VISUAL DESIGN
# ---------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

    :root { --ink:#edf2ff; --muted:#9ba9c7; --line:rgba(151,170,218,.16); }
    html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
    .stApp {
        background:
          radial-gradient(ellipse at 72% 0%, rgba(93,75,210,.20), transparent 34%),
          radial-gradient(ellipse at 0% 45%, rgba(28,135,194,.12), transparent 32%),
          #080d1b;
        color:var(--ink);
    }
    [data-testid="stHeader"] { background:rgba(8,13,27,.75); }
    [data-testid="stSidebar"] {
        background:linear-gradient(180deg,#101a31 0%,#0c1428 100%);
        border-right:1px solid var(--line);
    }
    [data-testid="stSidebar"] > div { padding-top:1.2rem; }
    [data-testid="stSidebar"] * { color:#eaf0ff; }
    .brand { display:flex; align-items:center; gap:11px; margin:4px 0 22px; }
    .brand-icon {
        width:43px;height:43px;border-radius:14px;display:grid;place-items:center;
        background:linear-gradient(135deg,#6e69ff,#22c4df);
        box-shadow:0 8px 25px rgba(87,111,255,.28);font-size:23px;
    }
    .brand-name { font:700 18px 'Space Grotesk',sans-serif; letter-spacing:-.5px; }
    .brand-sub { color:#8292b5;font-size:10px;letter-spacing:1.7px;text-transform:uppercase; }
    .side-label { color:#8292b5;font-size:10px;font-weight:700;letter-spacing:1.7px;text-transform:uppercase;margin:20px 0 9px; }
    .status-pill {
        display:inline-flex;align-items:center;gap:7px;padding:7px 11px;border-radius:30px;
        color:#a8f4d1;background:rgba(32,196,137,.10);border:1px solid rgba(32,196,137,.22);
        font-size:11px;font-weight:600;
    }
    .status-dot { width:7px;height:7px;border-radius:50%;background:#35df9b;box-shadow:0 0 10px #35df9b; }
    .hero { padding:34px 8px 18px; }
    .eyebrow { color:#8e9eff;font-size:11px;letter-spacing:2.2px;font-weight:700;text-transform:uppercase;margin-bottom:12px; }
    .hero h1 { font:700 clamp(31px,4vw,48px)/1.12 'Space Grotesk',sans-serif;letter-spacing:-1.8px;margin:0;color:#f4f6ff; }
    .hero h1 span { background:linear-gradient(90deg,#9a91ff,#52d9e8);-webkit-background-clip:text;background-clip:text;color:transparent; }
    .hero p { color:#9ba9c7;font-size:14px;max-width:680px;line-height:1.7;margin-top:13px; }
    .feature-row { display:flex;gap:9px;flex-wrap:wrap;margin-top:17px; }
    .feature-chip { border:1px solid var(--line);background:rgba(255,255,255,.035);color:#b9c5df;border-radius:30px;padding:7px 11px;font-size:11px; }
    .section-title { color:#e9eeff;font:600 16px 'Space Grotesk',sans-serif;margin:25px 0 12px; }
    .suggestion-note { color:#91a0bd;font-size:12px;margin-bottom:12px; }
    div[data-testid="stChatMessage"] {
        background:rgba(18,29,53,.72);border:1px solid rgba(151,170,218,.13);
        border-radius:17px;padding:15px 17px;margin-bottom:12px;
    }
    div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        background:rgba(64,76,151,.19);border-color:rgba(132,132,255,.20);
    }
    .source-head { color:#b8c6e4;font-size:11px;font-weight:700;letter-spacing:1.2px;text-transform:uppercase;margin:12px 0 7px; }
    .source-card {
        background:rgba(28,43,72,.72);border:1px solid rgba(126,154,214,.19);
        border-radius:12px;padding:11px 13px;margin:7px 0;color:#dbe5fa;font-size:12px;
    }
    .source-meta { color:#8e9fbe;font-size:10px;margin-top:4px; }
    .welcome-card {
        border:1px solid rgba(139,151,255,.20);border-radius:22px;
        background:linear-gradient(135deg,rgba(77,81,166,.17),rgba(28,151,178,.08));
        padding:25px;margin:14px 0 20px;
    }
    .welcome-title { font:600 20px 'Space Grotesk',sans-serif;color:#f2f5ff;margin-bottom:6px; }
    .welcome-copy { color:#9eacc8;font-size:13px;line-height:1.65; }
    .robot-float {
        position:fixed;right:28px;bottom:105px;z-index:999;
        width:66px;height:66px;border-radius:22px;display:grid;place-items:center;
        font-size:35px;background:linear-gradient(145deg,#242d62,#152e4d);
        border:1px solid rgba(139,159,255,.5);
        box-shadow:0 10px 35px rgba(54,91,225,.30),inset 0 0 18px rgba(95,189,255,.13);
        animation:floaty 3.2s ease-in-out infinite;
    }
    .robot-float:after {
        content:"";position:absolute;inset:-6px;border:1px solid rgba(103,195,255,.23);
        border-radius:26px;animation:halo 2.8s ease-out infinite;
    }
    @keyframes floaty { 0%,100%{transform:translateY(0) rotate(-2deg)}50%{transform:translateY(-10px) rotate(2deg)} }
    @keyframes halo { 0%{transform:scale(.92);opacity:.7}100%{transform:scale(1.25);opacity:0} }
    .footer-note { text-align:center;color:#637391;font-size:10px;padding:25px 0 10px; }
    .stButton > button {
        border-radius:12px;border:1px solid rgba(132,153,222,.22);
        background:rgba(31,46,78,.72);color:#e8eeff;transition:all .2s ease;
    }
    .stButton > button:hover { border-color:#8c8cff;color:white;background:rgba(75,81,158,.35); }
    [data-testid="stChatInput"] {
        border:1px solid rgba(126,147,221,.35)!important;border-radius:17px!important;
        background:rgba(16,26,48,.95)!important;box-shadow:0 8px 30px rgba(0,0,0,.18);
    }
    [data-testid="stChatInput"] textarea { color:#f1f4ff!important; }
    [data-testid="stExpander"] { border:1px solid var(--line);border-radius:12px;background:rgba(17,29,52,.55); }
    .stCaption, small { color:#8493b1!important; }
    hr { border-color:var(--line)!important; }
    </style>
    <div class="robot-float" title="Zyro AI assistant">🤖</div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------
# RAG PIPELINE
# ---------------------------------------------------------------------
@st.cache_resource(show_spinner="Preparing the HR knowledge base...")
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
                    metadata.append({"file": pdf_path.name, "page": page_num, "text": chunk})
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
    candidate_scores = {
        int(i): float(s) for i, s in zip(indices[0], scores[0]) if i >= 0
    }
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
            [],
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


# ---------------------------------------------------------------------
# SESSION STATE — multiple recent chats stored for this browser session
# ---------------------------------------------------------------------
def make_conversation():
    return {
        "id": str(uuid.uuid4()),
        "title": "New conversation",
        "messages": [],
        "created": datetime.now().strftime("%I:%M %p"),
    }


if "conversations" not in st.session_state:
    first = make_conversation()
    st.session_state.conversations = [first]
    st.session_state.active_conversation = first["id"]

if "active_conversation" not in st.session_state:
    st.session_state.active_conversation = st.session_state.conversations[0]["id"]


def get_active():
    for item in st.session_state.conversations:
        if item["id"] == st.session_state.active_conversation:
            return item
    fresh = make_conversation()
    st.session_state.conversations.insert(0, fresh)
    st.session_state.active_conversation = fresh["id"]
    return fresh


def start_new_chat():
    fresh = make_conversation()
    st.session_state.conversations.insert(0, fresh)
    st.session_state.active_conversation = fresh["id"]


def choose_chat(chat_id):
    st.session_state.active_conversation = chat_id


# ---------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div class="brand">
          <div class="brand-icon">✦</div>
          <div><div class="brand-name">IntelligentRAG</div>
          <div class="brand-sub">Zyro · HR Intelligence</div></div>
        </div>
        <div class="status-pill"><span class="status-dot"></span> AI SYSTEM ONLINE</div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("＋   New chat", use_container_width=True, type="primary"):
        start_new_chat()
        st.rerun()

    st.markdown('<div class="side-label">Recent conversations</div>', unsafe_allow_html=True)
    for chat in st.session_state.conversations[:8]:
        label = chat["title"]
        if chat["id"] == st.session_state.active_conversation:
            label = "●  " + label
        else:
            label = "　" + label
        if st.button(label, key=f"chat_{chat['id']}", use_container_width=True):
            choose_chat(chat["id"])
            st.rerun()

    st.markdown("---")
    st.markdown('<div class="side-label">Knowledge engine</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div style="font-size:12px;line-height:2;color:#b7c5e2">
        <div>◈ &nbsp; Retrieval <b>FAISS + MMR</b></div>
        <div>◈ &nbsp; Embeddings <b>BGE Small EN v1.5</b></div>
        <div>◈ &nbsp; Model <b>Llama 3.3 · 70B</b></div>
        <div>◈ &nbsp; Sources <b>HR policy PDFs</b></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("Answers are grounded in Zyro Dynamics policy documents. Always verify the cited source before acting.")

# ---------------------------------------------------------------------
# LOAD API + KNOWLEDGE BASE
# ---------------------------------------------------------------------
api_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY", ""))
if not api_key:
    st.error("GROQ_API_KEY is missing. Add it in Streamlit Cloud → App settings → Secrets.")
    st.stop()

try:
    model, index, vectors, metadata = build_index()
except Exception as e:
    st.error(str(e))
    st.info(
        "Add the relevant Zyro Dynamics HR policy PDFs to the "
        "`zyro-dynamics-hr-corpus/` folder in your GitHub repository. "
        "Do not include the unrelated Acrux Dynamics handbook."
    )
    st.stop()

client = Groq(api_key=api_key)

# ---------------------------------------------------------------------
# MAIN CHAT AREA
# ---------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">✦ Your intelligent workplace companion</div>
      <h1>HR answers, <span>made effortless.</span></h1>
      <p>Ask anything about Zyro Dynamics policies. Your AI assistant searches the official HR knowledge base and responds with source-backed answers.</p>
      <div class="feature-row">
        <span class="feature-chip">✧ Policy-grounded answers</span>
        <span class="feature-chip">⌕ Smart semantic search</span>
        <span class="feature-chip">▤ PDF citations</span>
        <span class="feature-chip">◉ Privacy-conscious demo</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

active = get_active()
if not active["messages"]:
    st.markdown(
        """
        <div class="welcome-card">
          <div style="font-size:27px;margin-bottom:9px">👋 🤖</div>
          <div class="welcome-title">Hey there! I'm Zyro AI.</div>
          <div class="welcome-copy">I'm ready to help you navigate leave, work arrangements, compensation, travel, onboarding and other company policies. What would you like to know?</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="section-title">✨ Try asking me</div>', unsafe_allow_html=True)
    st.markdown('<div class="suggestion-note">Choose a prompt to start your conversation.</div>', unsafe_allow_html=True)
    prompts = [
        "How does Earned Leave accrue every month?",
        "Who is eligible to work from home?",
        "When will my monthly salary be credited?",
        "What is the travel allowance for my grade?",
    ]
    prompt_cols = st.columns(2)
    chosen_prompt = None
    for i, prompt in enumerate(prompts):
        with prompt_cols[i % 2]:
            if st.button("✧  " + prompt, key=f"suggest_{i}", use_container_width=True):
                chosen_prompt = prompt
else:
    chosen_prompt = None

# Render existing conversation
for message in active["messages"]:
    with st.chat_message(message["role"], avatar="🧑🏻‍💻" if message["role"] == "user" else "🤖"):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            st.markdown('<div class="source-head">▤ &nbsp; Sources used</div>', unsafe_allow_html=True)
            for src in message["sources"]:
                st.markdown(
                    f"""<div class="source-card">📄 &nbsp; <b>{src['file']}</b>
                    <div class="source-meta">Page {src['page']} · Retrieved policy passage</div></div>""",
                    unsafe_allow_html=True,
                )
                with st.expander("View retrieved passage"):
                    st.write(src["text"])

# Process chat input or a suggested prompt
question = st.chat_input("Message Zyro AI… ask about any HR policy")
submitted_question = question or chosen_prompt

if submitted_question:
    clean_question = submitted_question.strip()
    if clean_question:
        active["messages"].append({"role": "user", "content": clean_question})
        if active["title"] == "New conversation":
            active["title"] = clean_question[:34] + ("…" if len(clean_question) > 34 else "")
        with st.chat_message("user", avatar="🧑🏻‍💻"):
            st.markdown(clean_question)

        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Zyro AI is searching the policy knowledge base…"):
                try:
                    answer, sources = answer_question(
                        clean_question, model, index, vectors, metadata, client
                    )
                    st.markdown(answer)
                    if sources:
                        st.markdown('<div class="source-head">▤ &nbsp; Sources used</div>', unsafe_allow_html=True)
                        seen = set()
                        unique_sources = []
                        for src in sources:
                            key = (src["file"], src["page"])
                            if key in seen:
                                continue
                            seen.add(key)
                            unique_sources.append(src)
                            st.markdown(
                                f"""<div class="source-card">📄 &nbsp; <b>{src['file']}</b>
                                <div class="source-meta">Page {src['page']} · Retrieved policy passage</div></div>""",
                                unsafe_allow_html=True,
                            )
                            with st.expander("View retrieved passage"):
                                st.write(src["text"])
                    else:
                        unique_sources = []
                    active["messages"].append(
                        {"role": "assistant", "content": answer, "sources": unique_sources}
                    )
                except Exception as e:
                    error_message = f"I couldn't complete that request just now. Please try again. (Technical detail: {e})"
                    st.error(error_message)
                    active["messages"].append(
                        {"role": "assistant", "content": error_message, "sources": []}
                    )
        st.rerun()

st.markdown(
    '<div class="footer-note">POWERED BY INTELLIGENT RAG &nbsp;·&nbsp; FAISS + MMR &nbsp;·&nbsp; GROQ LLAMA 3.3 70B<br>Prototype for the Intelligent RAG project. Verify policy details against the cited PDF before acting.</div>',
    unsafe_allow_html=True,
)
