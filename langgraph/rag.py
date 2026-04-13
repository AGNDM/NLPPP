
from qdrant_client import QdrantClient
from dotenv import load_dotenv
from transformers import AutoTokenizer
from adapters import AutoAdapterModel
import os
import torch
from state import RAGState

load_dotenv()  # load .env 

# init Qdrant client
qdrant_client = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))

COLLECTION_NAME = "nlp_papers"
TOP_K = 5

def create_embedding_model():
    model_name = "allenai/specter2_base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    embedding_model = AutoAdapterModel.from_pretrained(model_name)
    embedding_model.load_adapter("allenai/specter2", source="hf", load_as="specter2", set_active=True)
    return tokenizer, embedding_model

def text_to_embedding(text):
    tokenizer, embedding_model = create_embedding_model()
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        output = embedding_model(**inputs)
    embedding = output.last_hidden_state[:, 0, :]
    return embedding.numpy().flatten()

def query_vector_trunks(state:RAGState):
    query_vector = text_to_embedding(state["rewritten_query"])
    results = qdrant_client.query_points(
        collection_name='nlp_papers',
        query=query_vector.tolist(),
        limit=3,
        with_payload=True,  # Include paper metadata
        with_vectors=False
    ).points

    return {
        "retrieved_chunks": results
    }

