from types import SimpleNamespace
from src.pm_agent import PMAgent
from src.utils import get_prompt_template


if __name__ == "__main__":
    file_paths = ["/home/jovyan/projects/PMAgent/datasets/helpdesk.csv"]

    splitter_model_name = "BAAI/bge-small-en-v1.5"
    chunk_size = 700
    chunk_overlap = 50

    reasoner_model_name = "mistralai/Mistral-7B-v0.1"
    max_new_tokens=300
    temperature=0.7
    do_sample=True
    top_p=0.95

    k_retriever = 3

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
        if query.lower() == 'exit':
            break

        answer = pm_agent.ask(query)
        print(f"Answer: {answer}\n")
