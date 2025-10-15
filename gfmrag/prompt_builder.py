from omegaconf import DictConfig


class QAPromptBuilder:
    """A class for building prompts for question-answering tasks.

    This class constructs formatted prompts for Q&A systems using a configuration-based approach.
    It supports system prompts, document contexts, questions, and optional few-shot examples.

    Args:
        prompt_cfg (DictConfig): Configuration dictionary containing:
            - system_prompt: The system instruction prompt
            - doc_prompt: Template for formatting document context
            - question_prompt: Template for formatting the question
            - examples: List of few-shot examples (optional)

    Methods:
        build_input_prompt(question, retrieved_docs, thoughts=None):
            Builds a formatted prompt list for the Q&A system.

            Args:
                question (str): The input question to be answered
                retrieved_docs (list): List of dictionaries containing document information
                    with 'title' and 'content' keys
                thoughts (list, optional): Additional thought process or reasoning steps

            Returns:
                list: A list of dictionaries containing the formatted prompt with roles
                    and content for each component
    """

    def __init__(self, prompt_cfg: DictConfig) -> None:
        self.cfg = prompt_cfg
        self.system_prompt = self.cfg.system_prompt
        self.doc_prompt = self.cfg.doc_prompt
        self.question_prompt = self.cfg.question_prompt
        self.examples = self.cfg.examples

    def build_input_prompt(
        self,
        question: str,
        retrieved_result: dict[str, list[dict]],
        thoughts: list | None = None,
    ) -> list:
        prompt = [
            {"role": "system", "content": self.system_prompt},
        ]
        context = ""
        for target_type, retrieved_items in retrieved_result.items():
            if len(retrieved_items) > 0:
                target_type_context = f"## Relevant {target_type}:\n\n"
                target_type_context += "\n\n".join(
                    "\n".join([f"{key}: {value}" for key, value in item.items()])
                    for item in retrieved_items
                )
                context += target_type_context + "\n\n"

        question = self.question_prompt.format(question=question)
        if thoughts is not None:
            question += " ".join(thoughts)

        if len(self.examples) > 0:
            for example in self.examples:
                prompt.extend(
                    [
                        {"role": "user", "content": example["input"]},
                        {"role": "assistant", "content": example["response"]},
                    ]
                )
        prompt.append(
            {
                "role": "user",
                "content": context + question,
            }
        )

        return prompt
