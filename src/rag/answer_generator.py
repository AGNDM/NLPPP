from src.retrieval.vector_store import load_vector_db
from src.modeling.inference import load_lora_model
from src.rag.conflict_resolver import ConflictResolver

class RAGAnswerGenerator:
    def __init__(self):
        # Load vector database
        self.vectorstore = load_vector_db()
        
        # Check if vector database loaded successfully
        if self.vectorstore is None:
            print("⚠️ Warning: Vector database failed to load, using fallback mode")
            self.retriever = None
        else:
            self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 10})
        
        # Load LoRA model (if not exists, return None)
        self.generator = None
        self.tokenizer = None
        # try:
        #     self.generator, self.tokenizer = load_lora_model("./models/lora_qa_final")
        # except Exception as e:
        #     print(f"⚠️ Warning: LoRA model loading failed: {e}")
        #     self.generator = None
        #     self.tokenizer = None
        
        # self.conflict_resolver = ConflictResolver(method="weighted_vote")
        self.conflict_resolver = ConflictResolver(method="keyword_boosted")

    def rewrite_query(self, query: str) -> str:
        """Rewrite user query into a more search-friendly format"""
        # Optimization for "what is" questions
        if query.lower().startswith("what is"):
            return query + " definition description"
        return query
    
    def generate_answer(self, query: str) -> str:
        # Check if retriever is available
        if self.retriever is None:
            # Fallback answer when vector database is unavailable
            return self._fallback_answer(query)
        
        # Normal workflow
        try:
            rewritten_query = self.rewrite_query(query)
            # 1. Vector retrieval
            retrieved_docs = self.retriever.invoke(rewritten_query)

            # Debug: print retrieved documents
            print(f"\nRetrieved {len(retrieved_docs)} documents:")
            for i, doc in enumerate(retrieved_docs):
                print(f"--- Document {i+1} ---")
                print(f"Content preview: {doc.page_content[:200]}...")
                print(f"Answer: {doc.metadata.get('answer', 'None')}")
            
            if not retrieved_docs:
                return self._fallback_answer(query)
            
            # 2. Build format for conflict resolver
            chunks_for_resolver = []
            for doc in retrieved_docs:
                chunks_for_resolver.append({
                    "chunk": doc.page_content,
                    "score": doc.metadata.get("score", 0.0),
                    "metadata": {"answer": doc.metadata.get("answer", "")}
                })
            
            # 3. Resolve conflicts to get final answer text
            final_answer_text = self.conflict_resolver.resolve(query, chunks_for_resolver)
            
            # 4. If fine-tuned model is available, refine the answer
            if self.generator is not None and self.tokenizer is not None:
                try:
                    input_text = f"Question: {query}\nProposed answer: {final_answer_text}\nRefine the answer to be more accurate and concise:"
                    inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
                    outputs = self.generator.generate(**inputs, max_new_tokens=128)
                    refined_answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    return refined_answer
                except Exception as e:
                    print(f"Refinement failed: {e}")
                    return final_answer_text
            else:
                return final_answer_text
                
        except Exception as e:
            print(f"Error generating answer: {e}")
            return self._fallback_answer(query)
    
    def _fallback_answer(self, query: str) -> str:
        """Fallback answer when vector database is unavailable"""
        return f"""Unable to answer your question: "{query}"

Reason: Vector database not loaded correctly.

Please fix by following these steps:
1. Run python -m src.retrieval.vector_store to build the vector database
2. Ensure necessary packages are installed: pip install sentence-transformers
3. Verify that the chroma_db directory exists and contains data
"""