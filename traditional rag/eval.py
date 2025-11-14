from typing import List, Dict, Tuple
from collections import Counter
import numpy as np


def normalize_answer(s: str) -> str:
    """
    Lowercase and strip unnecessary characters to normalize answers.
    """
    import re
    s = s.lower()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)  # Remove articles
    s = re.sub(r'[^a-z0-9\s]', '', s)  # Remove punctuation
    s = re.sub(r'\s+', ' ', s).strip()  # Remove extra whitespaces
    return s


def compute_exact_match(gold_answers: List[str], predicted_answer: str) -> float:
    """
    Compute Exact Match (EM) score for a single prediction.
    """
    predicted = normalize_answer(predicted_answer)
    for gold in gold_answers:
        if normalize_answer(gold) == predicted:
            return 1.0
    return 0.0


def compute_f1(gold_answers: List[str], predicted_answer: str) -> float:
    """
    Compute F1 score for a single prediction.
    """
    def f1_score(predicted: str, gold: str) -> float:
        predicted_tokens = normalize_answer(predicted).split()
        gold_tokens = normalize_answer(gold).split()
        common = Counter(predicted_tokens) & Counter(gold_tokens)
        num_same = sum(common.values())

        if num_same == 0:
            return 0.0

        precision = num_same / len(predicted_tokens)
        recall = num_same / len(gold_tokens)
        return 2 * (precision * recall) / (precision + recall)

    predicted = normalize_answer(predicted_answer)
    f1_scores = [f1_score(predicted, gold) for gold in gold_answers]
    return max(f1_scores) if f1_scores else 0.0


def evaluate_qa(
    gold_answers: List[List[str]],
    predicted_answers: List[str]
) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    """
    Evaluate QA performance using Exact Match (EM) and F1 scores.

    Args:
        gold_answers (List[List[str]]): List of lists containing ground truth answers.
        predicted_answers (List[str]): List of predicted answers.

    Returns:
        Tuple[Dict[str, float], List[Dict[str, float]]]:
            - A dictionary with overall EM and F1 scores.
            - A list of dictionaries with EM and F1 scores for each example.
    """
    assert len(gold_answers) == len(predicted_answers), "Mismatched lengths of gold and predicted answers."

    total_em, total_f1 = 0.0, 0.0
    example_results = []

    for gold_list, predicted in zip(gold_answers, predicted_answers):
        em = compute_exact_match(gold_list, predicted)
        f1 = compute_f1(gold_list, predicted)

        example_results.append({"ExactMatch": em, "F1": f1})
        total_em += em
        total_f1 += f1

    overall_results = {
        "ExactMatch": total_em / len(gold_answers) if gold_answers else 0.0,
        "F1": total_f1 / len(gold_answers) if gold_answers else 0.0
    }

    return overall_results, example_results