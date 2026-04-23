import streamlit as st

from pipeline.rag_bm25 import retrieve

st.set_page_config(page_title="NLP Paper QA (BM25 Baseline)", layout="centered")
st.title("NLP Research Assistant — BM25 Baseline")
st.caption("Enter a query and return the top-k chunks retrieved by BM25.")

query = st.chat_input("Ask a question...")

if query:
    with st.chat_message("user"):
        st.write(query)

    state = {
        "original_query": query,
        "rewritten_query": query,  # BM25 baseline uses raw query directly
        "rewritten_user_question": query,
        "retrieved_chunks": [],
        "contradiction_pairs": [],
        "answer": "",
    }

    with st.chat_message("assistant"):
        with st.status("Retrieving top-k chunks with BM25..."):
            state.update(retrieve(state))
            chunks = state["retrieved_chunks"]

            if not chunks:
                st.warning("No chunks retrieved.")
            else:
                st.write(f"Returned **{len(chunks)}** top chunks:")
                for i, chunk in enumerate(chunks):
                    title = chunk.payload.get("title", f"Paper {i + 1}")
                    abstract = chunk.payload.get("abstract", "")
                    score = chunk.score

                    with st.expander(f"[{i + 1}] {title} — BM25 score: {score:.3f}"):
                        st.write(abstract)