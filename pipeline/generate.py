import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

from pipeline.state import RAGState
from pipeline.constants import GENERATE_BASE_MODEL, GENERATE_LORA_ADAPTER, USE_LORA


def _load_model():
    """
    Load the base Tulu model and attach the LoRA adapter.
    """
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"[generate] using device: {device}")
    print(f"[generate] loading base model: {GENERATE_BASE_MODEL}")

    tokenizer = AutoTokenizer.from_pretrained(GENERATE_BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if device == "cuda":
        # 4-bit quantisation only supported on CUDA
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            GENERATE_BASE_MODEL,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            GENERATE_BASE_MODEL,
            torch_dtype=torch.bfloat16,  # load on CPU first
        )
        base_model = base_model.to(device)  # then move to MPS (or stay on CPU)

    print(f"[generate] attaching LoRA adapter: {GENERATE_LORA_ADAPTER}")
    
    if USE_LORA:
        model = PeftModel.from_pretrained(base_model, GENERATE_LORA_ADAPTER)
    else:
        model = base_model

    model.eval()

    print("[generate] model ready")
    return model, tokenizer


# Loaded once at import time — reloading on every query would take minutes.
_model, _tokenizer = _load_model()


def _run_inference(prompt: str, max_new_tokens: int = 512) -> str:
    """Tokenize the prompt, run inference, and decode only the newly generated tokens."""
    messages = [{"role": "user", "content": prompt}]
    prompt_text = _tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = _tokenizer(prompt_text, return_tensors="pt").to(_model.device)

    with torch.no_grad():
        outputs = _model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=_tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )

    # Slice off the input tokens — we only want the newly generated part
    generated_tokens = outputs[0][inputs.input_ids.shape[1]:]
    return _tokenizer.decode(generated_tokens, skip_special_tokens=True)


def _build_prompt(
    query: str,
    abstracts: list[str],
    contradiction_pairs: list[tuple[int, int]],
) -> str:
    context_block = "\n\n".join(
        f"[{i + 1}] {abstract}" for i, abstract in enumerate(abstracts)
    )

    if contradiction_pairs:
        callout_lines = []
        for i, j in contradiction_pairs:
            callout_lines.append(
                f"Paper [{i + 1}] and Paper [{j + 1}] were found to contain contradicting statements:\n"
                f"  Paper [{i + 1}]: {abstracts[i]}\n"
                f"  Paper [{j + 1}]: {abstracts[j]}"
            )
        contradiction_block = (
            "Note — our evaluation identified the following contradicting claims "
            "among the retrieved papers. Please consider this when generating your answer "
            "and do not present either claim as definitive fact:\n\n"
            + "\n\n".join(callout_lines)
        )
    else:
        contradiction_block = ""

    conciseness_instruction = (
        ""
        if USE_LORA
        else "Keep your answer concise and intuitive — avoid unnecessary technical depth.\n"
    )

    prompt = f"""\
    You are a scientific assistant specialising in NLP research.

    Answer the question using the retrieved papers as your primary source. \
    If the papers are relevant, ground your answer in them. \
    If they are irrelevant or insufficient, use your own knowledge instead and IGNORE THEM. \
    If you genuinely do not know, say so. \
    Never mention the papers, the retrieval process, or your sources in your answer.
    {conciseness_instruction}
    Question:
    {query}

    Retrieved papers:
    {context_block}
    """

    if contradiction_block:
        prompt += f"\n{contradiction_block}\n"

    prompt += "\nAnswer:"
    return prompt


def generate_answer(state: RAGState) -> dict:
    """LangGraph node: generates the final answer using the fine-tuned Tulu model."""
    chunks = state["retrieved_chunks"]
    abstracts = [chunk.payload["abstract"] for chunk in chunks]

    prompt = _build_prompt(
        query=state["rewritten_user_question"],
        abstracts=abstracts,
        contradiction_pairs=state["contradiction_pairs"],
    )


    print(prompt)
    print("\n")
    print("\n")
    answer = _run_inference(prompt)
    print(f"[generate] answer generated ({len(answer)} chars)")

    return {"answer": answer}