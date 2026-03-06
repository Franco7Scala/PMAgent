import torch

from transformers import AutoTokenizer
from enum import Enum


class Color(Enum):
    EXPERIMENT_CONFIG_INFO = 2
    EXPERIMENT_STATUS_HIGH_PRIORITY = 3
    EXPERIMENT_STATUS_LOW_PRIORITY = 4
    EXPERIMENT_OUTPUT = 6
    WARNING = 5
    OTHER = 7
    BLACK = 8


def cprint(text, color=Color.BLACK):
    if color == Color.EXPERIMENT_CONFIG_INFO:
        code_color = "\033[94m"

    elif color == Color.EXPERIMENT_STATUS_HIGH_PRIORITY:
        code_color = "\033[32m"

    elif color == Color.EXPERIMENT_STATUS_LOW_PRIORITY:
        code_color = "\033[92m"

    elif color == Color.WARNING:
        code_color = "\033[91m"

    elif color == Color.EXPERIMENT_OUTPUT:
        code_color = "\033[95m"

    elif color == Color.OTHER:
        code_color = "\033[96m"

    else:
        code_color = "\033[0m"

    print(code_color + str(text) + "\033[0m")


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_tokens(text, model_name):
    return len(AutoTokenizer.from_pretrained(model_name).encode(text))


def get_prompt_template():
    return """
        You are a Senior Process Mining Analyst and Data Scientist. Your goal is to analyze event logs, process models, transition probabilities, and performance metrics provided in the context to answer complex questions about process execution.
        
        Use strictly and exclusively the following pieces of context to answer the question at the end. The context may include historical execution data, bottleneck analysis, variant frequencies, and transition matrices.
        
        Please adhere strictly to the following rules:
        
        1. No Hallucinations: If the provided context does not contain sufficient data to calculate a probability, make a prediction, or answer the question, do not invent data. Simply state: "I cannot find the exact answer in the provided data." and briefly suggest what specific log data or metrics would be needed to answer it.
        2. Predictions & Probabilities: When asked about future outcomes, the likelihood of an event, or case completion probabilities, base your estimations strictly on the historical frequencies and transition probabilities present in the context. Always state the probability explicitly (e.g., "Based on the logs, there is an 82% probability that...").
        3. What-If Analysis: When performing "what-if" scenarios, compare the hypothetical scenario against the baseline metrics provided in the context (such as current throughput times or bottleneck stages). Clearly explain the expected impact on time, cost, or case completion.
        4. Case Status & Estimation: When asked about the current stage of a task or if it will reach completion, identify its current state based on the logs, trace its most likely next steps (predictive path), highlight any deviations from the "happy path", and estimate the remaining time to completion.
        5. Structure & Clarity: Keep your answer highly analytical, objective, and well-structured. Use bullet points for clarity if explaining multiple metrics or a sequence of predicted events.
        6. Format the code in json format were there are keys like "answer", "probability", "what_if_analysis", "case_status", "estimation", etc. depending on the question asked.

        Context:
        
        {context}
        
        Question: {question}
        
        Helpful & Analytical Answer:
    """
