import os
import hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import streamlit as st
import anthropic

from analysis import analyze_script

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Gendered Tropes in Movies",
    page_icon="🎬",
    layout="wide",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🎬 Gendered Tropes in Movies")
    st.markdown(
        "Analyze how movie scripts describe male vs. female characters "
        "using **bigram NLP analysis** and **AI-powered interpretation**."
    )
    st.divider()

    st.subheader("Settings")
    # Use env var if set, otherwise ask the user
    env_key = os.environ.get("ANTHROPIC_API_KEY", "")
    api_key = st.text_input(
        "Anthropic API Key",
        value=env_key,
        type="password",
        help="Get your key at console.anthropic.com",
    )

    st.divider()
    st.subheader("How it works")
    st.markdown(
        """
1. Upload or paste a screenplay
2. NLP extracts every *he/she + next-word* bigram
3. Log-odds ratios reveal which words skew male or female
4. Claude interprets the patterns and answers your questions
        """
    )
    st.markdown(
        "Inspired by *[She Giggles, He Gallops](https://pudding.cool/2017/08/screen-direction/)* — The Pudding"
    )

# ── Main ──────────────────────────────────────────────────────────────────────
st.header("Upload a Screenplay")

col_upload, col_paste = st.columns(2)
with col_upload:
    uploaded_file = st.file_uploader("Upload a .txt screenplay file", type=["txt"])
with col_paste:
    pasted_text = st.text_area(
        "Or paste screenplay text here",
        height=180,
        placeholder="INT. COFFEE SHOP - DAY\n\nShe smiles. He nods...",
    )

# Resolve script text and detect changes via hash
script_text = ""
if uploaded_file:
    script_text = uploaded_file.read().decode("utf-8", errors="ignore")
    st.caption(f"Loaded: **{uploaded_file.name}** — {len(script_text):,} characters")
elif pasted_text.strip():
    script_text = pasted_text

script_hash = hashlib.md5(script_text.encode()).hexdigest() if script_text else ""

# Clear cached results when the script changes
if st.session_state.get("script_hash") != script_hash:
    for key in ["analysis_done", "ai_narrative", "chat_history", "analysis_context"]:
        st.session_state.pop(key, None)
    st.session_state["script_hash"] = script_hash

analyze_btn = st.button("Analyze Script", type="primary", disabled=not script_text)

# ── Run analysis ──────────────────────────────────────────────────────────────
if analyze_btn:
    with st.spinner("Running NLP analysis…"):
        he_counts, she_counts, top_he, top_she = analyze_script(script_text)

    if he_counts is None:
        st.error("No gendered pronouns (he / she) found. Please check your input.")
    else:
        st.session_state.update(
            analysis_done=True,
            he_counts=he_counts,
            she_counts=she_counts,
            top_he=top_he,
            top_she=top_she,
            chat_history=[],
            ai_narrative=None,
        )

# ── Display results ───────────────────────────────────────────────────────────
if st.session_state.get("analysis_done"):
    he_counts = st.session_state["he_counts"]
    she_counts = st.session_state["she_counts"]
    top_he = st.session_state["top_he"]
    top_she = st.session_state["top_she"]

    he_total = sum(he_counts.values())
    she_total = sum(she_counts.values())

    # ── Stats row ─────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Results")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("HE action words", f"{he_total:,}")
    m2.metric("SHE action words", f"{she_total:,}")
    m3.metric("HE : SHE ratio", f"{he_total / max(she_total, 1):.1f}×")
    m4.metric("Unique words analyzed", f"{len(set(he_counts) | set(she_counts)):,}")

    # ── Charts ────────────────────────────────────────────────────────────────
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("#### Words Most Associated with **HE**")
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.barh(top_he["word"], top_he["log_odds"], color="#4A90D9")
        ax.set_xlabel("Log-Odds Ratio (higher = more male)")
        ax.set_title("Male-skewed action words")
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with c2:
        st.markdown("#### Words Most Associated with **SHE**")
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.barh(top_she["word"], top_she["log_odds"].abs(), color="#E05A5A")
        ax.set_xlabel("Log-Odds Ratio magnitude (higher = more female)")
        ax.set_title("Female-skewed action words")
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ── AI narrative ──────────────────────────────────────────────────────────
    st.divider()
    st.subheader("AI Analysis")

    if not api_key:
        st.info("Enter your Anthropic API key in the sidebar to unlock AI-powered insights and chat.")
    else:
        # Build the analysis context string (reused for chat system prompt)
        he_words = ", ".join(top_he["word"].tolist())
        she_words = ", ".join(top_she["word"].tolist())
        analysis_context = f"""Script gender analysis results:
- Action words following "he": {he_total:,}
- Action words following "she": {she_total:,}
- HE:SHE ratio: {he_total / max(she_total, 1):.2f}×
- Top words associated with MALE characters (HE): {he_words}
- Top words associated with FEMALE characters (SHE): {she_words}"""

        st.session_state["analysis_context"] = analysis_context

        # Generate narrative once per script
        if not st.session_state.get("ai_narrative"):
            with st.spinner("Generating AI analysis…"):
                client = anthropic.Anthropic(api_key=api_key)
                prompt = (
                    f"{analysis_context}\n\n"
                    "Provide a 2–3 paragraph analysis of what these findings reveal about gender "
                    "representation in this script. Discuss the specific words found, what stereotypes "
                    "or patterns they suggest, and what they imply about how male vs. female characters "
                    "are portrayed. Be concrete and reference specific words from the lists."
                )
                response = client.messages.create(
                    model="claude-sonnet-4-6",
                    max_tokens=900,
                    system=(
                        "You are an expert in gender studies, media analysis, and film criticism. "
                        "You analyze movie scripts for gender representation and tropes, providing "
                        "thoughtful, evidence-based insights grounded in the data provided."
                    ),
                    messages=[{"role": "user", "content": prompt}],
                )
                st.session_state["ai_narrative"] = response.content[0].text

        st.markdown(st.session_state["ai_narrative"])

        # ── Chat ──────────────────────────────────────────────────────────────
        st.divider()
        st.subheader("Ask Follow-up Questions")

        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = []

        # Render existing messages
        for msg in st.session_state["chat_history"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if user_q := st.chat_input("e.g. What does this say about female agency in the script?"):
            st.session_state["chat_history"].append({"role": "user", "content": user_q})
            with st.chat_message("user"):
                st.markdown(user_q)

            with st.chat_message("assistant"):
                with st.spinner("Thinking…"):
                    client = anthropic.Anthropic(api_key=api_key)
                    system_prompt = (
                        "You are an expert in gender studies and film criticism. "
                        "Answer questions about the movie script's gender representation analysis "
                        "concisely and insightfully. Always ground your answers in the data provided.\n\n"
                        f"Analysis context:\n{st.session_state['analysis_context']}\n\n"
                        f"AI narrative already provided:\n{st.session_state['ai_narrative']}"
                    )
                    response = client.messages.create(
                        model="claude-sonnet-4-6",
                        max_tokens=600,
                        system=system_prompt,
                        messages=st.session_state["chat_history"],
                    )
                    answer = response.content[0].text
                    st.markdown(answer)

            st.session_state["chat_history"].append({"role": "assistant", "content": answer})
