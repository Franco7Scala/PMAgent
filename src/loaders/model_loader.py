import torch

from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, TextStreamer


def get_model(model_name, max_new_tokens=300, temperature=0.7, do_sample=True, top_p=0.95):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", dtype=torch.bfloat16, attn_implementation="sdpa")
    terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<end_of_turn>")]
    #streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=False)
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        max_length=None,
        temperature=temperature,
        top_p=top_p,
        do_sample=do_sample,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=terminators,
        return_full_text=False,
        #streamer=streamer
    )
    return HuggingFacePipeline(pipeline=pipe)
