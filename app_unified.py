"""
app.py
------
Streamlit frontend for the NLP Research Assistant.

Provides two pipeline modes selectable from the sidebar:
    - Dense RAG QA: full pipeline with query rewriting, SPECTER 2 retrieval,
      relevance grading, contradiction detection, and answer generation.
    - BM25 Baseline: retrieval-only baseline using BM25 keyword search,
      with no rewriting, grading, or answer generation.

Run with:
    streamlit run app.py
"""

import streamlit as st

from pipeline.rewrite import rewrite_query
from pipeline.rag import retrieve as dense_retrieve
from pipeline.rag_bm25 import retrieve as bm25_retrieve
from pipeline.grade import grade_chunks
from pipeline.nli import detect_contradictions_node
from pipeline.generate import generate_answer
from pipeline.state import RAGState


def init_dense_state(query: str) -> RAGState:
    """Initialise a blank RAGState for the dense RAG pipeline."""
    return {
        "original_query": query,
        "rewritten_query": "",
        "rewritten_user_question": "",
        "retrieved_chunks": [],
        "contradiction_pairs": [],
        "answer": "",
    }


def init_bm25_state(query: str) -> RAGState:
    """
    Initialise a blank RAGState for the BM25 baseline.

    BM25 retrieval uses the raw query directly, so rewritten_query and
    rewritten_user_question are pre-populated with the original query.
    """
    return {
        "original_query": query,
        "rewritten_query": query,
        "rewritten_user_question": query,
        "retrieved_chunks": [],
        "contradiction_pairs": [],
        "answer": "",
    }


def render_dense_rag(query: str) -> None:
    """
    Render the full dense RAG pipeline response for a given query.

    Runs each pipeline stage sequentially and surfaces intermediate outputs
    via st.status blocks so the user can follow along in real time.
    """
    state = init_dense_state(query)

    with st.chat_message("assistant"):

        # Step 1: Query rewriting
        with st.status("Rewriting query..."):
            state.update(rewrite_query(state))
            st.write(f"**Retrieval query:** {state['rewritten_query']}")
            st.write(f"**Rewritten question:** {state['rewritten_user_question']}")

        # Step 2: Retrieval
        with st.status("Retrieving relevant papers..."):
            state.update(dense_retrieve(state))
            all_chunks = state["retrieved_chunks"]
            st.write(f"Found **{len(all_chunks)}** papers above similarity threshold:")
            for i, chunk in enumerate(all_chunks):
                title = chunk.payload.get("title", f"Paper {i + 1}")
                abstract = chunk.payload.get("abstract", "")
                score = chunk.score
                with st.expander(f"[{i + 1}] {title} — score: {score:.3f}"):
                    st.write(abstract)

        # Step 3: Relevance grading
        with st.status("Grading chunk relevance..."):
            state.update(grade_chunks(state))
            graded_chunks = state["retrieved_chunks"]
            graded_titles = {c.payload.get("title") for c in graded_chunks}
            passed = len(graded_chunks)
            filtered = len(all_chunks) - passed
            st.write(f"**{passed}** chunks passed · **{filtered}** filtered out")
            for chunk in all_chunks:
                title = chunk.payload.get("title", "Untitled")
                if title in graded_titles:
                    st.markdown(f":green[{title}]")
                else:
                    st.markdown(f":red[{title}]")

        # Step 4: Contradiction detection
        with st.status("Checking for contradictions..."):
            state.update(detect_contradictions_node(state))
            pairs = state["contradiction_pairs"]
            if pairs:
                st.warning(f"{len(pairs)} contradiction(s) detected:")
                for i, j in pairs:
                    title_i = graded_chunks[i].payload.get("title", f"Paper {i + 1}")
                    title_j = graded_chunks[j].payload.get("title", f"Paper {j + 1}")
                    st.markdown(f"- **[{i + 1}] {title_i}** contradicts **[{j + 1}] {title_j}**")
            else:
                st.success("No contradictions detected")

        # Step 5: Answer generation
        with st.status("Generating answer..."):
            state.update(generate_answer(state))

        st.markdown(state["answer"])


def render_bm25_baseline(query: str) -> None:
    """
    Render the BM25 retrieval-only baseline response for a given query.

    Retrieves the top-k chunks using BM25 keyword search and displays them
    with their scores. No rewriting, grading, or answer generation is performed.
    """
    state = init_bm25_state(query)

    with st.chat_message("assistant"):
        with st.status("Retrieving top-k chunks with BM25..."):
            state.update(bm25_retrieve(state))
            chunks = state["retrieved_chunks"]

            if not chunks:
                st.warning("No chunks retrieved.")
            else:
                st.markdown(f"Returned **{len(chunks)}** top chunks:")
                for i, chunk in enumerate(chunks):
                    title = chunk.payload.get("title", f"Paper {i + 1}")
                    abstract = chunk.payload.get("abstract", "")
                    score = chunk.score

                    with st.expander(f"[{i + 1}] {title} — BM25 score: {score:.3f}"):
                        st.write(abstract)


def main() -> None:
    """Configure and launch the Streamlit app."""
    st.set_page_config(page_title="NLP Paper QA", layout="centered")
    st.title("NLP Research Assistant")
    mode = st.sidebar.selectbox(
        "Choose pipeline",
        ["Dense RAG QA", "BM25 Baseline"],
    )

    if mode == "Dense RAG QA":
        st.caption("Ask a question about NLP research. The pipeline retrieves relevant papers, checks for contradictions, and generates an answer.")
    else:
        st.caption(
            "BM25 retrieval-only baseline: user query → retrieve top-k chunks with BM25."
        )

    query = st.chat_input("Ask a question...")

    if query:
        with st.chat_message("user"):
            st.write(query)

        if mode == "Dense RAG QA":
            render_dense_rag(query)
        else:
            render_bm25_baseline(query)


if __name__ == "__main__":
    main()
