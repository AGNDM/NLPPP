from typing import List, Dict
import numpy as np

class ConflictResolver:
    def __init__(self, method="keyword_boosted"):
        self.method = method
    
    def resolve(self, query: str, chunks: List[Dict]) -> str:
        print(f"\n[DEBUG] Resolving with method: {self.method}")
        print(f"[DEBUG] Query: {query}")
        print(f"[DEBUG] Number of chunks: {len(chunks)}")
        
        if self.method == "keyword_boosted":
            result = self._keyword_boosted_selection(query, chunks)
            print(f"[DEBUG] Keyword boosted selected: {result[:100]}...")
            return result
        elif self.method == "first":
            result = chunks[0]["metadata"].get("answer", "") if chunks else ""
            print(f"[DEBUG] First selected: {result[:100]}...")
            return result
        else:
            result = chunks[0]["metadata"].get("answer", "") if chunks else ""
            print(f"[DEBUG] Default selected: {result[:100]}...")
            return result
    
    def _keyword_boosted_selection(self, query: str, chunks: List[Dict]) -> str:
        """
        Prioritize answers that match the query keywords
        """
        # Extract keywords from query
        query_keywords = set(query.lower().split())
        # Add synonyms/related words
        if "challenges" in query_keywords:
            related_words = ["challenge", "limitation", "difficulty", "problem", "issue", "cannot", "not directly usable"]
            query_keywords.update(related_words)
        
        best_answer = None
        best_score = -1
        
        for chunk in chunks:
            answer = chunk["metadata"].get("answer", "")
            chunk_text = chunk["chunk"].lower()
            retrieval_score = chunk.get("score", 0.0)
            
            # Calculate keyword matching score
            keyword_matches = 0
            for kw in query_keywords:
                if kw in chunk_text or kw in answer.lower():
                    keyword_matches += 1
            
            # Total score = retrieval score + keyword matching score * 2
            total_score = retrieval_score + (keyword_matches * 2.0)
            
            if total_score > best_score:
                best_score = total_score
                best_answer = answer
        
        return best_answer if best_answer else (chunks[0]["metadata"].get("answer", "") if chunks else "")