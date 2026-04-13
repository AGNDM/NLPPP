import numpy as np
from sentence_transformers import CrossEncoder
from qdrant_client.models import ScoredPoint


def _compute_similarity(vec_a:np.ndarray, vec_b: np.ndarray) -> float:
    return float(vec_a.dot(vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b)))


def _get_label(scores):
    label_mapping = ['contradiction', 'entailment', 'neutral']
    labels = [label_mapping[score_max] for score_max in scores.argmax(axis=1)]
    return labels


def _load_nli_model(model_name: str) -> CrossEncoder:
    return CrossEncoder(model_name)


def _find_candidates(results: list[ScoredPoint], similarity_threshold: float = 0.65) -> list[tuple[int, int, float]]:
    candidates: list[tuple[int, int, float]] = []
    if len(results) > 1:
        for i, a in enumerate(results):
            for j, b in enumerate(results[i+1:], start=i+1):
                sim = _compute_similarity(np.array(a.vector), np.array(b.vector))
                if sim > similarity_threshold:
                    candidates.append((i, j, sim))
    return candidates


def _get_contradiction_pairs(candidates: list[tuple[int, int, float]], labels: list[str]) -> list[tuple[int, int]]:
    return [
        (i, j)
        for (i, j, _), label in zip(candidates, labels)
        if label == "contradiction"
    ]


def _run_nli(pairs: list[tuple[str, str]], model: CrossEncoder) -> list[str]:
    scores = model.predict(pairs)
    return _get_label(scores)


def _build_pairs(results: list[ScoredPoint], candidates: list[tuple[int, int, float]]) -> list[tuple[str, str]]:
    return [
        (results[i].payload["abstract"], results[j].payload["abstract"])
        for i, j, _ in candidates
    ]


def test_contradiction_pipeline(pairs: list[tuple[str, str]], nli_model_name: str) -> list[tuple[int, int]]:
    model = _load_nli_model(nli_model_name)
    return _run_nli(pairs, model)
    

def detect_contradictions(results: list[ScoredPoint], nli_model_name: str) -> list[tuple[int, int]]:
    """
    Detect contradicting pairs among retrieved chunks.
    
    Args:
        results (list[ScoredPoint]): Retrieved chunks from query_vector_db. 
            Must be called with `with_vectors=True`, otherwise cosine similarity 
            cannot be computed and no candidates will be found.
        nli_model_name (str): HuggingFace model name for the NLI CrossEncoder
            e.g. 'cross-encoder/nli-deberta-v3-large'
    
    Returns:
        list[tuple[int, int]]: Index pairs into results where a contradiction 
            was detected. Empty list if no contradictions found.
    """
    model = _load_nli_model(nli_model_name)
    candidates = _find_candidates(results)
    pairs = _build_pairs(results, candidates)
    labels = _run_nli(pairs, model)
    return _get_contradiction_pairs(candidates, labels)


if __name__ == "__main__":
    # model_name =  "cross-encoder/nli-deberta-v3-large"
    # model_name = "cross-encoder/nli-deberta-v3-small"
    model_name = "cross-encoder/nli-MiniLM2-L6-H768"

    abstract_a = "In this work we analyze the role of attention weights in Transformer-based models. Our experiments on BERT demonstrate that attention weights are strongly correlated with token importance and are therefore reliable indicators of model reasoning. We show that removing high-attention tokens leads to a significant drop in performance of 23.4 F1 on SQuAD, confirming that attention weights provide a faithful explanation of model behaviour."
    abstract_b = "We investigate whether attention weights constitute faithful explanations in Transformer-based models. Contrary to prior claims, our experiments on BERT show that attention weights are poorly correlated with token importance and do not reliably indicate model reasoning. Removing high-attention tokens leads to negligible performance degradation of 1.2 F1 on SQuAD, confirming that attention weights are not faithful explanations of model behaviour."
    
    abstract_c = "Multi-head attention scales quadratically with sequence length"
    abstract_d = "Sparse multi-head attention scales with O(n log n)."
    abstract_e = "Multi-head attention scales linearly with sequence length, as demonstrated by our experiments on sequences up to 100k tokens, confirming that standard attention mechanisms are efficient even for very long documents."
    print(
        test_contradiction_pipeline(
            [
                (abstract_a, abstract_b), # Contradiction
                (abstract_c, abstract_d), # Neutral
                (abstract_c, abstract_e), # Contradiction
                (abstract_d, abstract_e), # # Neutral
            ], 
            model_name
        )
    )
