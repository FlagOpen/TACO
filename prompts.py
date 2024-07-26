from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel, PreTrainedTokenizerBase, LlamaForCausalLM, LlamaTokenizer

import os
import re
import json

def get_prompt_base(sample:dict):
    prompt = "\nQUESTION:\n"
    prompt += sample["question"]
    starter_code = None if len(sample["starter_code"]) == 0 else sample["starter_code"]
    try:
        input_outpout = json.loads(sample["input_output"])
        fn_name = (
            None if not input_outpout.get("fn_name") else input_outpout["fn_name"]
        )
    except ValueError:
        fn_name = None
    if starter_code:
        prompt += starter_code
    if (not fn_name) and (not starter_code):
        call_format = "\nUse Standard Input format"
        prompt += call_format
    else:
        call_format = "\nUse Call-Based format"
        prompt += call_format
    prompt += "\nANSWER:\n"
    return prompt

def get_prompt_chat_base(sample:dict, tokenizer: LlamaTokenizer):
    prompt = "\nQUESTION:\n"
    prompt += sample["question"]
    starter_code = None if len(sample["starter_code"]) == 0 else sample["starter_code"]
    try:
        input_outpout = json.loads(sample["input_output"])
        fn_name = (
            None if not input_outpout.get("fn_name") else input_outpout["fn_name"]
        )
    except ValueError:
        fn_name = None
    if starter_code:
        prompt += starter_code
    if (not fn_name) and (not starter_code):
        call_format = "\nUse Standard Input format"
        prompt += call_format
    else:
        call_format = "\nUse Call-Based format"
        prompt += call_format

    messages = [
    {"role": "user", "content": prompt},
    ]
    chat = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    return chat

def msg_to_prompt(messages, tokenizer:LlamaTokenizer):
    chat = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    return chat

def get_prompt_code_test(sample:dict, tokenizer: LlamaTokenizer):
    system_prompt = """
You are an exceptionally intelligent coding assistant who consistently delivers accurate and reliable responses to user instructions. When facing a programming problem, you will first analyze the problem to derive the correct solution, figure out its time complexity, and finally provide a {language} code to solve the problem. 
If the problem uses Standard Input format, your code should be a complete program that reads from standard input and outputs answers to standard output. In this case, please make sure your program follows the input/output format provided in the problem description. If the problem uses Call-Based format, you should write a single function, that receives parameters as inputs and returns the corresponding answer. In this case, the first line of the function, containing the function name and required arguments, would be provided at the end of the problem description, you should copy this line and complete the following code in your response.
Please return all completed codes in one code block. This code block should be in the following format:
```{language_md}
# Your codes here, for Standard Input format
```
or
```{language_md}
def function_name(arg1, arg2, ...):
    # Your codes here, for Call-Based format
    return ...
```
""".lstrip().format(language="Python", language_md="python")

    starter_code = None if len(sample["starter_code"]) == 0 else sample["starter_code"]
    try:
        input_outpout = json.loads(sample["input_output"])
        fn_name = (
            None if not input_outpout.get("fn_name") else input_outpout["fn_name"]
        )
    except ValueError:
        fn_name = None

    call_format = "Standard Input format" if (not fn_name) and (not starter_code) else "Call-Based format"
    prompt = "\nSolve the below question in {format}:\n".format(format=call_format)
    prompt += sample["question"]
    if starter_code:
        prompt += "\nFunction start:\n"
        prompt += starter_code

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    chat = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    return chat