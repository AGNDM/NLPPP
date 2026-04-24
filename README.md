# NLPPP: NLP Research RAG Pipeline

A Retrieval-Augmented Generation (RAG) pipeline to answer questions about NLP questions. It features query rewriting, dual-mode retrieval (Dense & BM25), contradiction detection (NLI), and grounded answer generation with fine-tuned model.

## 🚀 Key Features

- **Unified RAG App**: A Streamlit-based UI to interact with the pipeline.
- **Dual Retrieval Modes**:
  - **Dense RAG**: Semantic search using Qdrant and Specter 2 embeddings.
  - **BM25 Baseline**: Lexical search baseline.
- **Contradiction Detection**: Uses NLI (Natural Language Inference) to identify conflicting information between retrieved papers.
- **Fine-tuned Models**: Optimized for NLP domain tasks.

## 📦 Installation

This project uses [uv](https://github.com/astral-sh/uv) for fast, reliable dependency management.

```bash
# Clone the repository
git clone https://github.com/AGNDM/NLPPP.git
cd NLPPP

# Sync dependencies
uv sync
```

## ⚙️ Configuration

1. **Environment Variables**:
   Copy `.env.example` to `.env` and provide your API keys:
   ```bash
   cp .env.example .env
   ```
   Required keys:
   - `QDRANT_URL` & `QDRANT_API_KEY`: For vector database access.
   - `OPENROUTER_API_KEY`: For LLM-based query rewriting and generation.

## 🛠️ How to Run

### 1. Unified Streamlit Application (Recommended)

The easiest way to use the pipeline is through the Streamlit UI, which allows you to toggle between Dense RAG and BM25 modes.

```bash
uv run streamlit run app_unified.py
```

### 2. CLI Pipeline

Run the LangGraph-based pipeline directly in the terminal (defaults to Dense RAG):

```bash
uv run python -m pipeline.main
```

## 🏗️ Project Structure

- `app_unified.py`: Unified Streamlit interface.
- `pipeline/`: Core RAG logic.
  - `rag.py`: Dense retrieval (semantic search).
  - `rag_bm25.py`: Lexical retrieval (BM25).
  - `nli.py`: Contradiction detection node.
  - `rewrite.py`: Query transformation for better retrieval.
  - `main.py`: LangGraph execution flow.
- `rag_QA/`: Vector database utilities and data loading.
- `evaluation_study/`: Scripts for batch inference and metric analysis.
- `finetuning/`: Training scripts for domain-specific models.

## 🤖 Fine-tuned Models

- **Qwen-0.5B**: [AGNDM/Fine-tuned_NLP_Qwen_0.5B](https://huggingface.co/AGNDM/Fine-tuned_NLP_Qwen_0.5B)
- **Tulu-Qasper-LoRA**: [AGNDM/tulu_qasper_lora_final](https://huggingface.co/AGNDM/tulu_qasper_lora_final)
