
from qdrant_client import QdrantClient
from dotenv import load_dotenv
from transformers import AutoTokenizer
from adapters import AutoAdapterModel
import os
import torch
from pipeline.state import RAGState
from pipeline.constant import RETRIEVAL_MODLE_NAME, RETRIEVAL_ADAPTER_NAME, RETRIEVAL_COLLECTION_NAME, RETRIEVAL_TOP_K
import json

load_dotenv()  # load .env 

# init Qdrant client
qdrant_client = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))

def create_embedding_model():
    model_name = RETRIEVAL_MODLE_NAME
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    embedding_model = AutoAdapterModel.from_pretrained(model_name)
    embedding_model.load_adapter(RETRIEVAL_ADAPTER_NAME, source="hf", load_as="specter2", set_active=True)
    return tokenizer, embedding_model

def text_to_embedding(text):
    tokenizer, embedding_model = create_embedding_model()
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        output = embedding_model(**inputs)
    embedding = output.last_hidden_state[:, 0, :]
    return embedding.numpy().flatten()

def qdrant_results_to_json(results):
    json_list = []
    for point in results:
        json_list.append({
            "id": point.id,
            "score": point.score,
            "payload": point.payload
        })
    return json_list

def query_vector_trunks(state:RAGState):
    query_vector = text_to_embedding(state["rewritten_query"])
    results = qdrant_client.query_points(
        collection_name=RETRIEVAL_COLLECTION_NAME,
        query=query_vector.tolist(),
        limit=RETRIEVAL_TOP_K,
        with_payload=True,  # Include paper metadata
        with_vectors=True
    ).points

    # json_data = qdrant_results_to_json(results)

    return {
        "retrieved_chunks": results
    }

