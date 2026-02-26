from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from transformers import GenerationConfig


def get_model(model_name, max_new_tokens=300, temperature=0.7, do_sample=True,top_p=0.95):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.generation_config.max_new_tokens = max_new_tokens
    model.generation_config.temperature = temperature
    model.generation_config.do_sample = do_sample
    model.generation_config.top_p = top_p
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.eos_token_id = tokenizer.eos_token_id
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    return HuggingFacePipeline(pipeline=pipe)
