import json
from datasets import load_dataset
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm
from typing import List, Dict, Any
import os

def load_qasper_split(split: str = "train"):
    """Load the specified split of QASPER dataset (train/validation)"""
    dataset = load_dataset("allenai/qasper", split=split)
    return dataset

def extract_paragraphs_from_paper(paper: Dict) -> List[str]:
    """Extract all paragraph texts from the full_text of a paper (in order)"""
    full_text = paper["full_text"]
    paragraphs = full_text["paragraphs"]
    # paragraphs is a list of lists: each element corresponds to paragraphs under a section
    all_paras = []
    for section_paras in paragraphs:
        all_paras.extend(section_paras)
    return all_paras

def create_question_chunk_pairs(paper: Dict, chunk_size: int = 1000, overlap: int = 200):
    """
    Generate training data (question, text_chunk) -> answer for each paper.
    Strategy: Split the paper into chunks, then for each question, find the evidence paragraphs
    containing the answer, and pair the question with the chunks containing those paragraphs.
    If the answer spans multiple paragraphs, pair with multiple chunks.
    """
    paper_id = paper["id"]
    title = paper["title"]
    paragraphs = extract_paragraphs_from_paper(paper)
    
    # 1. Concatenate paragraphs into text and split into chunks using LangChain splitter (preserve paragraph boundaries)
    full_text = "\n\n".join(paragraphs)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    chunks = splitter.create_documents([full_text], metadatas=[{"paper_id": paper_id, "title": title}])
    
    # 2. Build mapping from paragraph text to chunk index
    para_to_chunk_map = {}
    for idx, chunk in enumerate(chunks):
        chunk_text = chunk.page_content
        for para in paragraphs:
            if para in chunk_text and para not in para_to_chunk_map:
                para_to_chunk_map[para] = idx
    
    # 3. Process each question
    qas = paper["qas"]
    question_list = qas["question"]
    answer_list = qas["answers"]
    pairs = []  # Store (question, chunk_text, answer_text)
    
    for q_idx, question in enumerate(question_list):
        if q_idx >= len(answer_list):
            continue
            
        answer_entry = answer_list[q_idx]
        
        # Check if answer entry exists
        if not answer_entry:
            continue
        
        # QASPER's actual structure: answer_entry is a dictionary containing an 'answer' key
        # The value of the 'answer' key is the actual answer content (could be a list, may contain multiple answers)
        if isinstance(answer_entry, dict) and 'answer' in answer_entry:
            answer_content = answer_entry['answer']
            
            # Handle multiple answers (take the first one)
            if isinstance(answer_content, list) and len(answer_content) > 0:
                first_answer = answer_content[0]
            elif isinstance(answer_content, dict):
                first_answer = answer_content
            else:
                continue
        else:
            continue
        
        # Extract answer text
        answer_text = ""
        if not first_answer.get("unanswerable", False):
            if first_answer.get("free_form_answer"):
                answer_text = first_answer["free_form_answer"]
            elif first_answer.get("extractive_spans"):
                answer_text = " ".join(first_answer["extractive_spans"])
            elif first_answer.get("yes_no") is not None:
                answer_text = "Yes" if first_answer["yes_no"] else "No"
        
        if not answer_text:
            continue
        
        # Get evidence paragraphs
        evidence_paras = first_answer.get("evidence", [])
        evidence_paras = [e for e in evidence_paras if "FLOAT" not in e]
        
        # Pair question with chunks containing evidence
        paired_chunks = set()
        for para in evidence_paras:
            if para in para_to_chunk_map:
                chunk_idx = para_to_chunk_map[para]
                paired_chunks.add(chunk_idx)
        
        for chunk_idx in paired_chunks:
            pairs.append({
                "question": question,
                "chunk": chunks[chunk_idx].page_content,
                "answer": answer_text,
                "paper_id": paper_id,
                "title": title
            })
    
    return pairs

def build_dataset(split: str = "train", chunk_size: int = 1000, output_path: str = "data/processed/qasper_chunk_pairs.json"):
    """Main function: process the entire split and save the results"""
    dataset = load_qasper_split(split)

    # # Debug: Check the structure of the first paper
    # first_paper = dataset[0]
    # print("=" * 50)
    # print("First paper ID:", first_paper["id"])
    # print("First paper title:", first_paper["title"])
    # print("Number of questions:", len(first_paper["qas"]["question"]))
    # print("Number of answers:", len(first_paper["qas"]["answers"]))
    
    # # Check the answer structure of the first question
    # first_answers = first_paper["qas"]["answers"][0]
    # print("\nFirst answer structure:")
    # print("Type of answers:", type(first_answers))
    # if isinstance(first_answers, list) and len(first_answers) > 0:
    #     print("First answer keys:", first_answers[0].keys())
    #     print("unanswerable:", first_answers[0].get("unanswerable"))
    #     print("free_form_answer:", first_answers[0].get("free_form_answer", "")[:100])
    #     print("extractive_spans:", first_answers[0].get("extractive_spans", []))
    #     print("evidence:", first_answers[0].get("evidence", [])[:2])
    # elif isinstance(first_answers, dict):
    #     print("Answer keys:", first_answers.keys())
    
    # print("=" * 50)

    all_pairs = []
    for paper in tqdm(dataset, desc=f"Processing {split} papers"):
        pairs = create_question_chunk_pairs(paper, chunk_size=chunk_size)
        all_pairs.extend(pairs)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(all_pairs, f, indent=2)
    print(f"Saved {len(all_pairs)} (question, chunk) pairs to {output_path}")

if __name__ == "__main__":
    # Process training set and validation set
    build_dataset(split="train", output_path="data/processed/train_pairs.json")
    build_dataset(split="validation", output_path="data/processed/val_pairs.json")