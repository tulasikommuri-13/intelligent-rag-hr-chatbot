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
