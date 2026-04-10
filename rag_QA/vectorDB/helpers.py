import os
import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer
from adapters import AutoAdapterModel
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

load_dotenv()

QDRANT_URL    = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# ── Load embedding model ──────────────────────────────────────────────────────

def load_embedding_model():
    """Load the SPECTER 2 model and tokenizer, and return them for reuse.

    Returns:
        tokenizer: The SPECTER 2 tokenizer.
        model: The SPECTER 2 model with the retrieval adapter loaded.
    """
    print("Loading SPECTER 2 model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("allenai/specter2_base")
    model = AutoAdapterModel.from_pretrained("allenai/specter2_base")
    model.load_adapter("allenai/specter2", source="hf", load_as="specter2", set_active=True)
    model.eval()
    print("  → Model ready\n")
    return tokenizer, model

# ── Embedding helper ──────────────────────────────────────────────────────────

def embed(texts, tokenizer, model):
    """Convert a list(!) of texts to SPECTER 2 embeddings.

    Args:
        texts (list[str]): A list of strings to embed. E.g. ["Title [SEP] Abstract", ...]
        tokenizer (AutoTokenizer): The SPECTER 2 tokenizer.
        model (AutoAdapterModel): The SPECTER 2 model with the retrieval adapter loaded.

    Returns:
        np.ndarray: A numpy array of embeddings.
    """
    # SPECTER 2 expects "title [SEP] abstract" as input
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
        return_token_type_ids=False,
    )
    with torch.no_grad():
        output = model(**inputs)
    # The embedding is the first token ([CLS]) of the last hidden state
    return output.last_hidden_state[:, 0, :].numpy()

# ── Connect to Qdrant ─────────────────────────────────────────────────────────

def get_qdrant_client():
    print("Connecting to Qdrant...")
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    print("  → Connected\n")
    return client

# ── Query VectorDB ─────────────────────────────────────────────────────────

def query_vector_db(client, query_embedding, collection_name="nlp_papers",top_k=3):
    """Query the vector database for similar papers.
    
    Args:
        client (QdrantClient): An instance of the Qdrant client.
        query_embedding (np.ndarray): The embedding vector for the user's query.
        collection_name (str): The name of the Qdrant collection to search.
        top_k (int): The number of top results to return.
        
    Returns:
        list: A list of search results with metadata.
    """
    results = client.query_points(
        collection_name=collection_name,
        query=query_embedding.tolist(),
        limit=top_k,
        with_payload=True,  # Include paper metadata
    ).points
    return results