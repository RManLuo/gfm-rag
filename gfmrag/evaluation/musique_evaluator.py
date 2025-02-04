# Adapt from: https://github.com/OSU-NLP-Group/HippoRAG/blob/main/src/qa/musique_evaluation.py
import collections
import re
import string
from collections.abc import Callable

from gfmrag.evaluation.base_evaluator import BaseEvaluator


def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles, and extra whitespace."""

    def remove_articles(text: str) -> str:
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_tokens(s: str) -> list:
    if not s:
        return []
    return normalize_answer(s).split()


def compute_exact(a_gold: str, a_pred: str) -> int:
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def compute_f1(a_gold: str, a_pred: str) -> tuple:
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return (
            int(gold_toks == pred_toks),
            int(gold_toks == pred_toks),
            int(gold_toks == pred_toks),
        )
    if num_same == 0:
        return 0, 0, 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def metric_max_over_ground_truths(
    metric_fn: Callable, prediction: str, ground_truths: list
) -> int:
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def metric_max_f1_over_ground_truths(
    metric_fn: Callable, prediction: str, ground_truths: list
) -> tuple:
    max_f1, max_precision, max_recall = 0, 0, 0
    for ground_truth in ground_truths:
        f1, prec, recal = metric_fn(prediction, ground_truth)
        if f1 > max_f1:
            max_f1 = f1
            max_precision = prec
            max_recall = recal
    return max_f1, max_precision, max_recall


class MusiqueEvaluator(BaseEvaluator):
    """
    MusiqueEvaluator
    """

    def evaluate(self) -> dict:
        metrics = {"em": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0}

        for pred in self.data:
            if "Answer: " in pred["response"]:
                pre_ans = pred["response"].split("Answer:")[1].strip()
            else:
                pre_ans = pred["response"]
            gold_answers = [pred["answer"]] + pred["answer_aliases"]
            em = metric_max_over_ground_truths(compute_exact, pre_ans, gold_answers)
            (
                f1,
                precision,
                recall,
            ) = metric_max_f1_over_ground_truths(compute_f1, pre_ans, gold_answers)
            metrics["em"] += float(em)
            metrics["f1"] += f1
            metrics["precision"] += precision
            metrics["recall"] += recall

        n = len(self.data)
        for k in metrics.keys():
            metrics[k] /= n
        return metrics
