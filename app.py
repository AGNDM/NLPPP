import streamlit as st

from pipeline.rewrite import rewrite_query
from pipeline.rag import retrieve
from pipeline.grade import grade_chunks
from pipeline.nli import detect_contradictions_node
from pipeline.generate import generate_answer

st.set_page_config(page_title="NLP Paper QA", layout="centered")
st.title("NLP Research Assistant")
st.caption("Ask a question about NLP research. The pipeline retrieves relevant papers, checks for contradictions, and generates an answer.")

query = st.chat_input("Ask a question...")

if query:
    with st.chat_message("user"):
        st.write(query)

    state = {
        "original_query": query,
        "rewritten_query": "",
        "rewritten_user_question": "",
        "retrieved_chunks": [],
        "contradiction_pairs": [],
        "answer": "",
    }

    with st.chat_message("assistant"):

        # Step 1: Query rewriting
        with st.status("Rewriting query..."):
            state.update(rewrite_query(state))
            st.write(f"**Retrieval query:** {state['rewritten_query']}")
            st.write(f"**Rewritten question:** {state['rewritten_user_question']}")

        # Step 2: Retrieval
        with st.status("Retrieving relevant papers..."):
            state.update(retrieve(state))
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
                    st.write(f"{title}")
                else:
                    st.write(f"{title}")

        # Step 4: Contradiction detection
        with st.status("Checking for contradictions..."):
            state.update(detect_contradictions_node(state))
            pairs = state["contradiction_pairs"]
            if pairs:
                st.warning(f"{len(pairs)} contradiction(s) detected:")
                for i, j in pairs:
                    title_i = graded_chunks[i].payload.get("title", f"Paper {i + 1}")
                    title_j = graded_chunks[j].payload.get("title", f"Paper {j + 1}")
                    st.write(f"- **[{i + 1}] {title_i}** contradicts **[{j + 1}] {title_j}**")
            else:
                st.success("No contradictions detected")

        # Step 5: Answer generation
        with st.status("Generating answer..."):
            state.update(generate_answer(state))

        st.markdown(state["answer"])