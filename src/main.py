import argparse
from src.rag.answer_generator import RAGAnswerGenerator

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--query", type=str, required=True, help="User question about NLP papers")
#     args = parser.parse_args()

#     print(args.query)
    
#     rag = RAGAnswerGenerator()
#     answer = rag.generate_answer(args.query)
#     print(f"Question: {args.query}")
#     print(f"Answer: {answer}")

# if __name__ == "__main__":
#     main()

from src.retrieval.fuzzy_matcher import FuzzyRetriever

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, required=True, help="User question about NLP papers")
    parser.add_argument("--retrieval", type=str, default="vector", 
                        choices=["vector", "fuzzy"],
                        help="Retrieval method: vector (semantic search) or fuzzy (string matching)")
    parser.add_argument("--top_k", type=int, default=10, 
                        help="Number of documents to retrieve (default: 10)")
    args = parser.parse_args()

    print(f"Query: {args.query}")
    print(f"Retrieval method: {args.retrieval}")
    print(f"Top K: {args.top_k}")
    print("-" * 50)
    
    # Choose retrieval method
    if args.retrieval == "fuzzy":
        # Use fuzzy matching
        print("Using fuzzy matching retrieval...")
        retriever = FuzzyRetriever("data/processed/train_pairs.json")
        results = retriever.retrieve(args.query, top_k=args.top_k)
        
        print(f"\nRetrieved {len(results)} documents:")
        for i, (chunk, score, metadata) in enumerate(results):
            print(f"--- Document {i+1} ---")
            print(f"Similarity score: {score:.2f}")
            print(f"Question: {metadata['question']}")
            print(f"Answer: {metadata['answer']}")
            print(f"Chunk preview: {chunk[:200]}...")
            print()
        
        # Simple answer extraction: return the best match's answer
        if results:
            best_answer = results[0][2]['answer']
            print(f"Answer: {best_answer}")
        else:
            print("Answer: No relevant documents found.")
    
    else:
        # Use vector search (original RAG pipeline)
        print("Using vector search RAG pipeline...")
        rag = RAGAnswerGenerator()
        answer = rag.generate_answer(args.query)
        print(f"Answer: {answer}")

if __name__ == "__main__":
    main()