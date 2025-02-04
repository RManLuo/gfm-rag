from statistics import mean

from gfmrag.evaluation.base_evaluator import BaseEvaluator


class RetrievalEvaluator(BaseEvaluator):
    def evaluate(self, k: tuple = (1, 2, 5, 10)) -> dict:
        metrics: dict[str, list] = {f"recall@{i}": [] for i in k}
        for pred in self.data:
            gold_docs = pred["supporting_facts"]
            sorted_retrieved_docs = sorted(
                pred["retrieved_docs"], key=lambda x: x["score"], reverse=True
            )
            sorted_retrieved_docs = [doc["title"] for doc in sorted_retrieved_docs]
            for i in k:
                recall = len(set(sorted_retrieved_docs[:i]) & set(gold_docs)) / len(
                    set(gold_docs)
                )
                metrics[f"recall@{i}"].append(recall)
        for key in metrics.keys():
            metrics[key] = mean(metrics[key])
        return metrics
