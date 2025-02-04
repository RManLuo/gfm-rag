import json
from abc import ABC, abstractmethod


class BaseEvaluator(ABC):
    """Base evaluator class for evaluation tasks.

    This abstract base class provides a foundation for implementing evaluators
    that assess model predictions. It handles loading prediction data from a JSON
    lines file where each line contains a single JSON object.

    Args:
        prediction_file (str): Path to the JSON lines prediction file to evaluate.
            Each line should contain a valid JSON object.

    Attributes:
        data (List[dict]): List of prediction data loaded from the JSON lines file.

    Examples:
        >>> evaluator = MyEvaluator("predictions.jsonl")
        >>> results = evaluator.evaluate()

    Note:
        Subclasses must implement the `evaluate()` method to define evaluation logic.
    """

    def __init__(self, prediction_file: str) -> None:
        super().__init__()
        with open(prediction_file) as f:
            self.data = [json.loads(line) for line in f]

    @abstractmethod
    def evaluate(self) -> dict:
        pass
