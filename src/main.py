from types import SimpleNamespace
from src.pm_agent import PMAgent
from src.utils import get_prompt_template, cprint, Color, count_tokens
from transformers import logging as hf_logging


if __name__ == "__main__":
    hf_logging.set_verbosity_error()
    file_paths = ["/home/jovyan/projects/PMAgent/datasets/helpdesk.csv"]

    splitter_model_name = "BAAI/bge-small-en-v1.5"
    chunk_size = 700
    chunk_overlap = 50

    reasoner_model_name = "google/gemma-3-4b-it"
    max_new_tokens=2000
    temperature=0.7
    do_sample=True
    top_p=0.95

    k_retriever = 7

    prompt_template = get_prompt_template()

    config = SimpleNamespace(
        splitter_model_name=splitter_model_name,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        reasoner_model_name=reasoner_model_name,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=do_sample,
        top_p=top_p,
        k_retriever=k_retriever
    )

    pm_agent = PMAgent(file_paths, prompt_template, config)

    while True:
        query = input("Enter your question (or 'exit' to quit): ")
        if query.lower() == "exit":
            break

        cprint(f"Tokens per question: {count_tokens(query, reasoner_model_name)}\n", Color.EXPERIMENT_OUTPUT)
        context, answer = pm_agent.ask(query)
        joined_context = '\n'.join(context)
        cprint(f"Context:\n{joined_context}\n", Color.EXPERIMENT_CONFIG_INFO)
        cprint(f"Answer:\n{answer}\n", Color.EXPERIMENT_OUTPUT)
