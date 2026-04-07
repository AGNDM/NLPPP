from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
import json
import os

def build_vector_db(chunks_file: str, persist_dir: str = "chroma_db"):
    """Read processed (question, chunk) pairs and build vector database (with progress display)"""
    print("=" * 50)
    print("Starting to build vector database...")
    
    # 1. Check if file exists
    if not os.path.exists(chunks_file):
        print(f"Error: File does not exist - {chunks_file}")
        print("Please run build_chunks.py first to generate data file")
        return None
    
    # 2. Read data
    print(f"Reading file: {chunks_file}")
    with open(chunks_file, "r") as f:
        pairs = json.load(f)
    
    print(f"Successfully read {len(pairs)} QA pairs")
    
    # 3. Check if data is empty
    if len(pairs) == 0:
        print("Error: QA pair list is empty, please check if build_chunks.py generated data correctly")
        return None
    
    # 4. Display first 3 samples
    print("\nFirst 3 sample examples:")
    for i, pair in enumerate(pairs[:3]):
        print(f"\n--- Sample {i+1} ---")
        print(f"Question: {pair['question'][:100]}...")
        print(f"Chunk length: {len(pair['chunk'])} characters")
        print(f"Answer: {pair['answer'][:100]}...")
        print(f"Paper ID: {pair['paper_id']}")
        print(f"Title: {pair['title'][:80]}...")
    
    # 5. Create Document objects
    print("\nCreating Document objects...")
    documents = []
    for idx, pair in enumerate(pairs):
        content = f"Question: {pair['question']}\nContext: {pair['chunk']}"
        doc = Document(
            page_content=content,
            metadata={
                "answer": pair["answer"], 
                "paper_id": pair["paper_id"], 
                "title": pair["title"]
            }
        )
        documents.append(doc)
        
        if (idx + 1) % 1000 == 0:
            print(f"  Processed {idx + 1}/{len(pairs)} documents")
    
    print(f"Successfully created {len(documents)} Document objects")
    
    # 6. Load embedding model
    print("\nLoading embedding model (BAAI/bge-large-en-v1.5)...")
    print("First download may take a few minutes, please be patient...")
    try:
        embedding_model = HuggingFaceEmbeddings(
            model_name="BAAI/bge-large-en-v1.5",
            model_kwargs={'device': 'cpu'}
        )
        print("Embedding model loaded successfully")
    except Exception as e:
        print(f"Embedding model loading failed: {e}")
        return None
    
    # 7. Build vector database (batch addition with progress)
    print(f"\nBuilding vector database, saving to: {persist_dir}")
    print("This may take a few minutes...")
    
    # Delete existing database directory (optional, to avoid residue)
    import shutil
    if os.path.exists(persist_dir):
        print(f"Deleting existing database directory: {persist_dir}")
        shutil.rmtree(persist_dir)
    
    # Batch parameters
    batch_size = 100
    total_batches = (len(documents) + batch_size - 1) // batch_size
    
    print(f"Total documents: {len(documents)}")
    print(f"Batch size: {batch_size}")
    print(f"Total batches: {total_batches}")
    print("-" * 40)
    
    try:
        # Use batch version of Chroma.from_documents
        from langchain_community.vectorstores import Chroma
        
        # Method: Add in batches, display progress each time
        vectorstore = None
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(documents))
            batch_docs = documents[start_idx:end_idx]
            
            if batch_idx == 0:
                # First batch: create database
                vectorstore = Chroma.from_documents(
                    batch_docs,
                    embedding_model,
                    persist_directory=persist_dir,
                    collection_name="qasper_docs"
                )
            else:
                # Subsequent batches: add to existing database
                vectorstore.add_documents(batch_docs)
            
            # Display progress
            percent = (end_idx / len(documents)) * 100
            print(f"✅ Batch {batch_idx + 1}/{total_batches} | "
                  f"Added {end_idx}/{len(documents)} documents | "
                  f"Progress: {percent:.1f}%")
        
        # Persist save
        vectorstore.persist()
        print(f"\nVector database built successfully! Saved to: {persist_dir}")
        
    except Exception as e:
        print(f"Vector database building failed: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # 8. Verify
    print("\nVerifying vector database...")
    test_query = "What is machine learning?"
    try:
        test_results = vectorstore.similarity_search(test_query, k=1)
        if test_results:
            print(f"Test query '{test_query}' returned results successfully")
        else:
            print("Warning: Test query returned no results")
    except Exception as e:
        print(f"Verification query failed: {e}")
    
    print("=" * 50)
    return vectorstore

def load_vector_db(persist_dir: str = "chroma_db"):
    """Load the built vector database"""
    print("=" * 50)
    print(f"Loading vector database: {persist_dir}")
    
    if not os.path.exists(persist_dir):
        print(f"Error: Vector database directory does not exist - {persist_dir}")
        return None
    
    print("Loading embedding model...")
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        embedding_model = HuggingFaceEmbeddings(
            model_name="BAAI/bge-large-en-v1.5",
            model_kwargs={'device': 'cpu'}
        )
        print("Embedding model loaded successfully")
    except Exception as e:
        print(f"Embedding model loading failed: {e}")
        return None
    
    print("Loading Chroma database...")
    try:
        from langchain_community.vectorstores import Chroma
        # Important: Specify the correct collection_name
        vectorstore = Chroma(
            persist_directory=persist_dir, 
            embedding_function=embedding_model,
            collection_name="qasper_docs"  # Add this line!
        )
        
        # Verify document count
        collection = vectorstore._collection
        print(f"Number of documents in database: {collection.count()}")
        print("Vector database loaded successfully")
    except Exception as e:
        print(f"Vector database loading failed: {e}")
        return None
    
    print("=" * 50)
    return vectorstore

if __name__ == "__main__":
    """Build vector database when running this script directly"""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks_file", type=str, default="data/processed/train_pairs.json", 
                        help="Input QA pair file path")
    parser.add_argument("--persist_dir", type=str, default="chroma_db",
                        help="Vector database save directory")
    args = parser.parse_args()
    
    print("=" * 60)
    print("Vector Database Building Tool")
    print("=" * 60)
    
    # Build vector database
    vectorstore = build_vector_db(args.chunks_file, args.persist_dir)
    
    if vectorstore:
        print("\n✅ Build completed!")
        print(f"   Data file: {args.chunks_file}")
        print(f"   Database location: {args.persist_dir}")
    else:
        print("\n❌ Build failed, please check the error messages above")