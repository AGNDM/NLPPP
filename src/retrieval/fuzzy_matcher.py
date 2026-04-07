import json
from rapidfuzz import fuzz, process
from typing import List, Tuple

class FuzzyRetriever:
    def __init__(self, chunks_file: str = "data/processed/train_pairs.json"):
        with open(chunks_file, "r") as f:
            self.pairs = json.load(f)
        # 匹配问题，而不是 chunk
        self.questions = [pair["question"] for pair in self.pairs]
        self.metadata = [
            {
                "question": pair["question"], 
                "answer": pair["answer"], 
                "paper_id": pair["paper_id"],
                "chunk": pair["chunk"]
            } 
            for pair in self.pairs
        ]

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[str, float, dict]]:
        """匹配最相似的问题，返回对应的答案"""
        results = process.extract(
            query, 
            self.questions,  # 匹配问题，不是 chunk
            scorer=fuzz.token_set_ratio,
            limit=top_k
        )
        retrieved = []
        for question, score, idx in results:
            retrieved.append((self.metadata[idx]["chunk"], score, self.metadata[idx]))
        return retrieved