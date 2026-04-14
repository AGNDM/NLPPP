import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

def main():
    # The output directory of your fully merged model
    model_dir = "outputs/llama-3-8b-NLPPP-general-full"
    
    if not os.path.exists(model_dir):
        print(f"Error: Model directory '{model_dir}' not found.")
        print("Make sure you have finished running ft_scratch.py and the merged model is saved.")
        return

    print(f"Loading model and tokenizer from '{model_dir}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    
    # Ensure model knows the pad token (Llama 3 generation requires it)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    print("\n" + "="*50)
    print("Model loaded successfully! Type 'exit' or 'quit' to end.")
    print("="*50 + "\n")

    # Keep a small conversation history if needed, or just do single turns
    # We will do single turns here for simple testing
    while True:
        try:
            user_input = input("\nUser: ")
            if user_input.lower() in ["exit", "quit"]:
                break
            if not user_input.strip():
                continue

            # Format the prompt using the chat template learned during training
            messages = [
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": user_input}
            ]
            
            prompt_ids = tokenizer.apply_chat_template(
                messages, 
                tokenize=True, 
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(model.device)

            # Generate response
            with torch.no_grad():
                output_ids = model.generate(
                    prompt_ids,
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    repetition_penalty=1.05,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            # Slicing the output to only return the assistant's new generated tokens
            generated_ids = output_ids[0][prompt_ids.shape[1]:]
            response = tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            print(f"\nAssistant: {response.strip()}")

        except KeyboardInterrupt:
            # Handle Ctrl+C smoothly
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    main()
