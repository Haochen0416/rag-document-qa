"""
app.py
Streamlit UI for the RAG Document Q&A system.

Usage:
    streamlit run app.py
"""

import os
import tempfile

import streamlit as st

from rag_pipeline import load_and_split, build_vectorstore, build_qa_chain, query


st.set_page_config(
    page_title="RAG Document Q&A",
    page_icon="📄",
    layout="wide",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ Configuration")

    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        placeholder="sk-...",
        help="Your key is never stored — it lives only in this session.",
    )

    model = st.selectbox(
        "GPT Model",
        options=["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o"],
        index=0,
    )

    st.divider()
    st.markdown(
        "**How it works**\n"
        "1. Enter your OpenAI API key\n"
        "2. Upload a PDF document\n"
        "3. Ask questions about it\n\n"
        "The app splits the PDF into chunks, embeds them with "
        "`text-embedding-3-small`, stores them in a FAISS index, "
        "then uses the selected GPT model to answer your questions "
        "based only on the retrieved context."
    )
    st.divider()
    st.markdown(
        "Built by [Haochen Li](https://github.com/Haochen0416) · "
        "[Source](https://github.com/Haochen0416/rag-document-qa)"
    )

# ── Main ──────────────────────────────────────────────────────────────────────

st.title("📄 RAG Document Q&A")
st.caption("Upload a PDF and ask questions — answers are grounded in your document.")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file and api_key:
    cache_key = f"{uploaded_file.name}_{uploaded_file.size}"

    if st.session_state.get("cache_key") != cache_key:
        with st.spinner("📚 Processing document — this may take a few seconds…"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            try:
                chunks = load_and_split(tmp_path)
                vectorstore = build_vectorstore(chunks, api_key)
                chain_and_retriever = build_qa_chain(vectorstore, api_key, model)

                st.session_state["chain"] = chain_and_retriever
                st.session_state["cache_key"] = cache_key
                st.session_state["doc_name"] = uploaded_file.name
                st.session_state["num_chunks"] = len(chunks)
                st.session_state["chat_history"] = []
            except Exception as e:
                st.error(f"Failed to process document: {e}")
                st.stop()
            finally:
                os.unlink(tmp_path)

        st.success(
            f"✅ **{uploaded_file.name}** indexed — "
            f"{st.session_state['num_chunks']} chunks ready."
        )

elif uploaded_file and not api_key:
    st.warning("Please enter your OpenAI API key in the sidebar to continue.")
    st.stop()

elif not uploaded_file:
    st.info("👆 Upload a PDF to get started.")
    st.stop()

# ── Chat interface ────────────────────────────────────────────────────────────

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

for turn in st.session_state["chat_history"]:
    with st.chat_message("user"):
        st.write(turn["question"])
    with st.chat_message("assistant"):
        st.write(turn["answer"])
        if turn["sources"]:
            with st.expander("📎 Source passages"):
                for i, doc in enumerate(turn["sources"], 1):
                    page = doc.metadata.get("page", "?")
                    st.markdown(f"**Chunk {i} — page {page + 1}**")
                    st.caption(doc.page_content[:500] + ("…" if len(doc.page_content) > 500 else ""))

question = st.chat_input("Ask a question about your document…")

if question:
    if "chain" not in st.session_state:
        st.error("No document has been indexed yet.")
        st.stop()

    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                answer, sources = query(st.session_state["chain"], question)
            except Exception as e:
                st.error(f"Error querying the model: {e}")
                st.stop()

        st.write(answer)

        if sources:
            with st.expander("📎 Source passages"):
                for i, doc in enumerate(sources, 1):
                    page = doc.metadata.get("page", "?")
                    st.markdown(f"**Chunk {i} — page {page + 1}**")
                    st.caption(doc.page_content[:500] + ("…" if len(doc.page_content) > 500 else ""))

    st.session_state["chat_history"].append(
        {"question": question, "answer": answer, "sources": sources}
    )
