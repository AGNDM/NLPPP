# NLPPP#

```
NLPPP/
├── data/                           # Store raw and processed data  
│   ├── raw/                        # Raw QASPER data downloaded from Hugging Face  
│   └── processed/                  # Processed text chunks and index files  
├── models/                         # Store fine-tuned models  
│   └── lora_qa/                    # LoRA fine-tuned model weights  
├── notebooks/                      # Optional: Jupyter notebooks for exploratory analysis  
├── src/                            # Core source code  
│   ├── __init__.py  
│   ├── data_processing/            # Step 1: Process dataset  
│   │   ├── __init__.py  
│   │   └── build_chunks.py         # Split papers into chunks and create (Question, Chunk) pairs  
│   ├── retrieval/                  # Step 2: Retrieval module  
│   │   ├── __init__.py
│   │   ├── fuzzy_matcher.py        # Fuzzy matching as baseline  
│   │   └── vector_store.py         # Vector database construction and retrieval  
│   ├── modeling/                   # Step 3: Fine-tuning module
│   │   ├── __init__.py
│   │   ├── lora_finetune.py        # LoRA fine-tuning script
│   │   └── inference.py            # Load fine-tuned model for inference
│   ├── rag/                        # Step 4: Complete RAG pipeline
│   │   ├── __init__.py
│   │   ├── query_processor.py      # User query rewriting and embedding
│   │   ├── conflict_resolver.py    # Handle multi-source factual conflicts
│   │   └── answer_generator.py     # Integrate retrieval context with fine-tuned model to generate answers
│   └── main.py                     # Command line entry point
├── requirements.txt                # Project dependencies
├── .env                            # Store API keys (e.g., OpenAI key, if needed)
└── README.md                       # Project description
```

## 1. Create and activate Conda environment
conda create -n nlp_qa python=3.10 -y
conda activate nlp_qa

## 2. Install PyTorch (highest available version for Intel Mac)
conda install pytorch=2.2.2 torchvision torchaudio cpuonly -c pytorch -y

## 3. Navigate to project directory
cd /Users/kongcuiyuan/Desktop/2026T1/9417/project/NLPPP

## 4. Install other dependencies (using modified requirements.txt)
pip install -r requirements.txt

## 5. Set environment variable (avoid OpenMP conflict)
export KMP_DUPLICATE_LIB_OK=TRUE

## 6. Run data preprocessing script to generate training data pairs
python -m src.data_processing.build_chunks

## 7. Build vector database (for retrieval)
python -m src.retrieval.vector_store

## 8. Ensure conflict_resolver.py exists

## 9. Modify answer_generator.py to temporarily disable the fine-tuned model

## 10. Ensure all __init__.py files exist
touch src/__init__.py
touch src/data_processing/__init__.py
touch src/retrieval/__init__.py
touch src/modeling/__init__.py
touch src/rag/__init__.py

## 11. Run main program
```bash
python -m src.main --query "What are the main challenges in cross-lingual transfer learning?"  
python -m src.main --query "What are the main challenges in cross-lingual transfer learning?" --retrieval fuzzy
```