import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def load_model_and_tokenizer(base_model_id, peft_model_id):
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    
    print(f"Loading LoRA weights from {peft_model_id}...")
    model = PeftModel.from_pretrained(base_model, peft_model_id)
    model.eval()
    return model, tokenizer

def generate_answer(model, tokenizer, prompt, max_new_tokens=512):
    messages = [
        {"role": "user", "content": prompt}
    ]
    
    # Format according to Tulu/Llama-3 chat template
    prompt_text = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
        
    # extract only the generated text (ignoring the prompt)
    generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return generated_text

if __name__ == "__main__":
    base_model_id = "allenai/Llama-3.1-Tulu-3-8B"
    peft_model_id = "./tulu_qasper_lora_final"
    
    model, tokenizer = load_model_and_tokenizer(base_model_id, peft_model_id)
    
    print("\nModel loaded successfully! Enter your prompt below (or 'quit' to exit).")
    while True:
        user_prompt = input("\nPrompt: ")
        if user_prompt.lower() in ['quit', 'exit']:
            break
            
        print("\nGenerating answer...")
        answer = generate_answer(model, tokenizer, user_prompt)
        print("\nAnswer:")
        print(answer)
