"""
Load LoRA fine-tuned model for inference
"""

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
import os


def load_lora_model(model_path: str = "./models/lora_qa_final"):
    """Load LoRA fine-tuned model"""
    
    if not os.path.exists(model_path):
        print(f"Warning: LoRA model path does not exist - {model_path}")
        return None, None
    
    try:
        # Read adapter_config to get base model name
        import json
        config_path = os.path.join(model_path, "adapter_config.json")
        with open(config_path, "r") as f:
            config = json.load(f)
            base_model_name = config.get("base_model_name", "google/flan-t5-small")
        
        print(f"Base model (read from config): {base_model_name}")
        
        print("Loading base model...")
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float32,
            device_map="cpu"
        )
        
        print("Loading LoRA adapter...")
        model = PeftModel.from_pretrained(model, model_path)
        print("LoRA model loaded successfully")
        
        return model, tokenizer
        
    except Exception as e:
        print(f"LoRA model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def generate_answer(model, tokenizer, question: str, max_length: int = 128) -> str:
    """
    Generate answer using fine-tuned model
    
    Args:
        model: Loaded model
        tokenizer: Tokenizer
        question: User question
        max_length: Maximum generation length
    
    Returns:
        Generated answer text
    """
    inputs = tokenizer(question, return_tensors="pt", truncation=True, max_length=512)
    
    # If GPU is available, move inputs to GPU
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            do_sample=False,  # Use greedy decoding for more deterministic results
            num_beams=4,      # Use beam search to improve quality
        )
    
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer