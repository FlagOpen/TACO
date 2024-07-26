from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel, PreTrainedTokenizerBase, LlamaForCausalLM, LlamaTokenizer
import transformers
import torch

import os
import re
import json
from tqdm import tqdm
from utils import get_taco_dataset
from prompts import *
EOF_STRINGS = ["\nQUESTION", "\n---", "\nANSWER", "<|endoftext|>", "<|EOT|>", "<｜end▁of▁sentence｜>"]

def truncate_after_eof_strings(text):
    pattern = '|'.join(re.escape(s) for s in EOF_STRINGS)
    match = re.search(pattern, text)
    
    if match:
        return text[:match.start()]
    else:
        return text

import random, numpy as np
def set_random_seed(seed):
    """Set random seed for reproducibility."""
    if seed is not None and seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

def decode(tokenizer, raw_text_len, output):
    sents = []
    for tokens in output:
        tokens = tokens.cpu().numpy().tolist()
        sent = tokenizer.decode(tokens[raw_text_len:], skip_special_tokens=True)
        sents.append(sent)
    return sents

def predict(device, model: LlamaForCausalLM, tokenizer: LlamaTokenizer, prompt, seed, topp, t, max_length=2048):
    set_random_seed(seed)
    _prompt = tokenizer(prompt, truncation=False, return_tensors="pt")
    input_ids = _prompt["input_ids"]
    raw_text_len = len(input_ids[0])
    with torch.no_grad():
        _prompt = _prompt.to(device)
        output = model.generate(
            **_prompt,
            do_sample=True,
            temperature = t,
            top_p = topp,
            max_new_tokens = max_length,
            eos_token_id = tokenizer.eos_token_id,
            pad_token_id = tokenizer.pad_token_id,
        )
        output = decode(tokenizer, raw_text_len, output) 
    return output[0]




# Initialize model and tokenizer
from configs import paths
model_path = paths.model_path
model_name = 'models--deepseek-ai--deepseek-coder-7b-instruct-v1.5' # 'deepseek-ai/deepseek-coder-6.7b-instruct'
tokenizer = AutoTokenizer.from_pretrained(model_path+model_name)
model = AutoModelForCausalLM.from_pretrained(model_path+model_name)
device = "cuda:0"
model = model.to(device)
print(tokenizer)


# Initialize evaluation dataset 
# difficulties = ['ALL']
# difficulties = ["EASY", "MEDIUM", "MEDIUM_HARD", "HARD", "VERY_HARD"] 
# skills = ['ALL']
# skills = ["Data structures", "Sorting", "Range queries", "Complete search", "Amortized analysis", "Dynamic programming", "Bit manipulation", "Greedy algorithms"]

# from datasets import load_dataset
# taco = load_dataset('BAAI/TACO', split='test', difficulties=difficulties)
# taco = load_dataset('BAAI/TACO', split='test', skills=skills)
taco = get_taco_dataset(split='test', difficulties=["EASY"])

output_file = 'results/generations/deepseek-coder-7b-instruct-v1.5/EASY-prompttest/n10_t0.5/generations.json'
prompt_func = get_prompt_code_test
os.makedirs(os.path.dirname(output_file), exist_ok=True)

# setting up times of run
n_samples = 10
temperature = 0.5
top_p = 0.95 


output = []
prompt_sample = prompt_func(taco[0], tokenizer)
print(prompt_sample)
for idx, sample in tqdm(enumerate(taco), total=len(taco), desc="Generating", position=0, leave=True):
    prompt = prompt_func(sample, tokenizer)
    results = {"task_id": idx, "prompt": prompt}
    generations = []
    for i in tqdm(range(n_samples), desc=f"Sampling {n_samples} samples", position=1, leave=False):
        seed = i
        generation = predict(device, model, tokenizer, prompt, seed, top_p, temperature, max_length=2048)
        clean_code = truncate_after_eof_strings(generation)
        generations.append(clean_code)
    # lens = [len(g) for g in generations]
    # print(lens)
    results["output"] = generations
    output.append(results)

with open(output_file, 'w') as f:
    json.dump(output, f, indent=4)


