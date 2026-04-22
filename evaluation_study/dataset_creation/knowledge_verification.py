"""
verify_parametric_knowledge.py

Reads a JSON file of NLP questions, queries LLaMA 3.1 8B via OpenRouter,
checks whether the model's answer contains any expected keyword, and saves
passing questions to a JSON file.

Setup:
    pip install openai

    Get a free API key at https://openrouter.ai
    Then set it:  export OPENROUTER_API_KEY="your_key_here"

Input JSON format:
    [
      {
        "id": "A01",
        "question": "What architecture is BERT?",
        "expected_answers": ["encoder-only", "encoder only"]
      },
      ...
    ]

Usage:
    python verify_parametric_knowledge.py --input questions.json
    python verify_parametric_knowledge.py --input questions.json --output results.json
"""

import json
import os
import time
from openai import OpenAI

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL = "meta-llama/llama-3.1-8b-instruct"
DEFAULT_OUTPUT_FILE = "verified_questions.json"
SLEEP_BETWEEN_REQUESTS = 1.0  # seconds — stay within free tier rate limits

SYSTEM_PROMPT = (
    "You are a knowledgeable NLP researcher. "
    "Answer the following question with a short, direct answer. "
    "Do not explain or elaborate — just state the answer concisely."
)

# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def query_llama(client, question: str) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        temperature=0.0,
        max_tokens=100,
    )
    return response.choices[0].message.content.strip()


def check_answer(model_output: str, expected_answers: list) -> bool:
    output_lower = model_output.lower()
    return any(ans.lower() in output_lower for ans in expected_answers)


def run_verification(client, questions: list) -> tuple:
    correctly_answered, failed = [], []

    print(f"\nVerifying {len(questions)} questions against {MODEL}\n")
    print(f"{'ID':<8} {'RESULT':<10} MODEL ANSWER")
    print("-" * 80)

    for i, entry in enumerate(questions):
        qid = entry.get("id", str(i))

        try:
            model_answer = query_llama(client, entry["question"])
        except Exception as e:
            print(f"{qid:<8} {'ERROR':<10} {e}")
            failed.append({**entry, "model_answer": f"ERROR: {e}", "passed": False})
            time.sleep(SLEEP_BETWEEN_REQUESTS)
            continue

        passed = check_answer(model_answer, entry["expected_answers"])
        status = "PASS ✓" if passed else "FAIL ✗"
        print(f"{qid:<8} {status:<10} {model_answer.replace(chr(10), ' ')[:80]}")

        (correctly_answered if passed else failed).append(
            {**entry, "model_answer": model_answer, "passed": passed}
        )

        if i < len(questions) - 1:
            time.sleep(SLEEP_BETWEEN_REQUESTS)

    return correctly_answered, failed


def save_results(correctly_answered: list, failed: list, output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(correctly_answered, f, indent=2, ensure_ascii=False)

    total = len(correctly_answered) + len(failed)
    print("\n" + "=" * 80)
    print(f"SUMMARY: {len(correctly_answered)} / {total} questions passed")
    print(f"Verified questions saved to: {output_path}")

    if failed:
        print(f"\nFailed questions ({len(failed)}):")
        for entry in failed:
            print(f"  [{entry.get('id', '?')}] Expected: {entry['expected_answers']}")
            print(f"         Got:      {entry['model_answer'][:100]}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )

    with open("data/questions.json", "r", encoding="utf-8") as f:
        questions = json.load(f)

    correctly_answered, failed = run_verification(client, questions)
    save_results(correctly_answered, failed, "data/verified_questions.json")