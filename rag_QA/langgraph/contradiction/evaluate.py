import json
import argparse
from pathlib import Path
from nli import _load_nli_model, _run_nli

def load_test_cases(path: Path) -> list[dict]:
    """
    Loads NLI test cases from a JSON file.

    Each test case is expected to have the following fields:
        id:         Unique identifier string.
        category:   Category label for grouped reporting (e.g. 'domain_specific',
                    'standard_contradiction').
        abstract_a: First chunk text.
        abstract_b: Second chunk text.
        expected:   Ground truth label ('contradiction', 'entailment', 'neutral').
    """
    with open(path) as f:
        return json.load(f)


def save_results(results: dict, path: Path) -> None:
    """Serialises evaluation results to a JSON file."""
    with open(path, "w") as f:
        json.dump(results, f, indent=2)


def evaluate(test_cases: list[dict], model_name: str) -> dict:
    """
    Runs NLI evaluation over a set of labelled test cases and returns
    structured results.

    For each test case, runs the NLI model on the (abstract_a, abstract_b)
    pair and compares the predicted label against the expected label.
    Results are aggregated overall and broken down by category.
    """
    model = _load_nli_model(model_name)
    pairs = [(case["abstract_a"], case["abstract_b"]) for case in test_cases]
    labels = _run_nli(pairs, model)
    
    results = {
        "model": model_name,
        "overall": {"correct": 0, "total": len(test_cases)},
        "by_category": {},
        "failures": []
    }
    
    for case, label in zip(test_cases, labels):
        category = case["category"]
        expected = case["expected"]
        correct = label == expected
        
        if category not in results["by_category"]:
            results["by_category"][category] = {"correct": 0, "total": 0}
        
        results["by_category"][category]["total"] += 1
        
        if correct:
            results["overall"]["correct"] += 1
            results["by_category"][category]["correct"] += 1
        else:
            results["failures"].append({
                "id": case["id"],
                "category": category,
                "expected": expected,
                "got": label
            }) 
    return results


def print_results(results: dict) -> None:
    """
    Prints a summary of evaluation results to stdout.

    Outputs overall accuracy, per-category breakdown, and a list of
    failed predictions with their expected and actual labels.
    """
    total = results["overall"]["total"]
    correct = results["overall"]["correct"]
    print(f"\nModel: {results['model']}")
    print(f"Overall accuracy: {correct}/{total} ({100*correct/total:.1f}%)\n")
    print("By category:")
    for category, scores in results["by_category"].items():
        c, t = scores["correct"], scores["total"]
        print(f"  {category:<20} {c}/{t} ({100*c/t:.1f}%)")
    if results["failures"]:
        print("\nFailures:")
        for f in results["failures"]:
            print(f"  {f['id']} — expected: {f['expected']}, got: {f['got']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="HuggingFace model name")
    parser.add_argument("--test_cases", default="test_cases.json", help="Path to test cases JSON")
    parser.add_argument("--output", default=None, help="Path to save results as JSON")
    args = parser.parse_args()
    
    test_cases = load_test_cases(Path(args.test_cases))
    results = evaluate(test_cases, args.model)
    print_results(results)

    if args.output:
        save_results(results, Path(args.output))
