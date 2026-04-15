# --- Query Rewriting ---
# TODO: replace with fine-tuned Llama once ready
REWRITE_MODEL = "nvidia/nemotron-3-super-120b-a12b:free"

RETRIEVAL_COLLECTION = "nlp_papers"
RETRIEVAL_TOP_K = 5

# --- Contradiction Detection (NLI) ---
NLI_MODEL = "cross-encoder/nli-deberta-v3-large"
NLI_SIMILARITY_THRESHOLD = 0.65  # cosine similarity cutoff before running NLI

# --- Answer Generation ---
# TODO: replace with fine-tuned Llama once ready (and update generate.py to load locally)
GENERATE_BASE_MODEL = "allenai/Llama-3.1-Tulu-3-8B"
GENERATE_LORA_ADAPTER = "AGNDM/tulu_qasper_lora_final"
USE_LORA = False  # set to True to use the LoRA adapter (currently disabled for testing without adapter)