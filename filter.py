import jsonlines
import os
import json
import numpy as np
import re
from functools import partial
from tqdm import tqdm
from yapf.yapflib.yapf_api import FormatCode
from datasets import load_from_disk
from compute_metric import evaluate_ref_codes

def check_code(content:str):
    if "```" in content:
        code = ''               
        for c in content.split('```')[1::2]:
            if 'ython\n' in c:
                code += c.split('ython\n')[1]
            # else:
            #     code += c
        try:
            compile(code, "<string>", "exec")
        except Exception as e:
            # return repr(e).replace("'<string>',", "line")
            return False
        
    return True

def check_include_code(content):
    key_word_list = [
        'def',
        'class',
        'static',
        '```',
        '=',
        'string',
        'int',
        'for',
        'print',
        '.h',
        '#include',
    ]
    for key_word in key_word_list:
        if key_word in content:
            return True

    return False


def filter_samples(data:dict, difficulties=None, skills=None, standard_only=False, min_num_sols=0):
    if difficulties is not None and data['difficulty'] not in difficulties:
        return False
    if skills is not None and data['skill'] not in skills:
        return False
    if standard_only and len(data['starter_code'])>0:
        return False
    if len(json.loads(data['solutions'])) < min_num_sols:
        return False
    return True


def filter_invalid_code(data:dict, max_num_sols=-1, with_testcases=False):
    try:
        solutions = eval(data['solutions'])
    except:
        solutions = []
        print(data['solutions'])
        raise ValueError('solutions is not a list')
    count_case_syntax = 0
    count_case_res_short = 0
    count_4ws_code = 0
    count_reformated_code = 0
    # if data['starter_code']:
        # if is_double_tab_code(data['starter_code']):
        #     data['starter_code'] = data['starter_code'].replace('\t\t', '\t')
        # elif is_double_4_white_space(data['starter_code']):
        #     data['starter_code'] = data['starter_code'].replace('        ', '\t')
        # import ipdb; ipdb.set_trace()
        # data['starter_code'], changed = FormatCode(data['starter_code'])

    filtered_solutions = []
    # c. 剔除response过短（小于20个单词）和过长（大于2000个单词）的
    for solution in solutions:
        # replace \t\t to 4 white spaces
        # if is_double_tab_code(solution):
        #     count_reformated_code += 1
        #     solution = solution.replace('\t\t', '\t')
        # elif is_double_4_white_space(solution):
        #     count_4ws_code += 1
        #     solution = solution.replace('        ', '\t')

        # solution = solution.replace('\n\t\t\n', '\n')
        # solution = solution.replace('\n\t\n', '\n')

        if not check_include_code(solution):
            if len(re.findall(r'\w+', solution)) < 20:
                # f.write(json.dumps(data, ensure_ascii=False) + '\n')
                count_case_res_short += 1
                continue
        
        # taco only contain python solutions
        if not check_code(solution):
            count_case_syntax += 1
            continue
        
        try:
            new_solution, changed = FormatCode(solution)
        except:
            count_4ws_code += 1
            continue
        if changed:
            # import ipdb; ipdb.set_trace()
            count_reformated_code += 1

        new_solution = '```python\n' + new_solution + '```\n'
        filtered_solutions.append(new_solution)

    if len(filtered_solutions) != len(solutions):
        print('filtered_solutions:', len(filtered_solutions), 'total solutions:', len(solutions))

    data['solutions'] = json.dumps(filtered_solutions[:max_num_sols])
    if not with_testcases:
        input_output = json.loads(data['input_output'])
        input_output.pop('inputs', None)
        input_output.pop('outputs', None)
        data['input_output'] = json.dumps(input_output)
    return data

prefix = "~/datasets/raw/"
TACO_PATH=prefix + "BAAI-TACO/"
SAVE_PATH=prefix + "BAAI-TACO-filtered/ALL/withIO/"
taco = load_from_disk(TACO_PATH)
print(taco)
# taco.shuffle(seed=42)
taco = taco.map(
    partial(filter_invalid_code, max_num_sols=32, with_testcases=True), 
    num_proc=60
)
taco_filtered = taco.filter(
    partial(filter_samples, standard_only=True, min_num_sols=1),
    num_proc=32
)
flag = True
if flag:
    for split in ['test']:
        taco_filtered[split] = taco_filtered[split].map(
            partial(evaluate_ref_codes, max_used_sols=16),
            num_proc=1
        )
else:
    taco_p = load_from_disk(SAVE_PATH)
    for split in ['test']:
        ref_code_results = taco_p[split]['ref_code_results']
        taco_filtered[split] = taco_filtered[split].add_column("ref_code_results", ref_code_results)

os.makedirs(SAVE_PATH, exist_ok=True)
taco_filtered.save_to_disk(SAVE_PATH)
import pdb; pdb.set_trace()
print(taco_filtered)
from collections import Counter
difficulty_counter = Counter(taco_filtered['train']['difficulty'])
print(difficulty_counter)


# TACO_PATH=prefix + "BAAI-TACO/"
# SAVE_PATH=prefix + "BAAI-TACO-filtered/ALL/formatted/"
# taco = load_from_disk(TACO_PATH)
# print(taco)
# # taco.shuffle(seed=42)
# taco = taco.map(
#     partial(filter_invalid_code, max_num_sols=32, with_testcases=False), 
#     num_proc=60
# )
# taco_filtered = taco.filter(
#     partial(filter_samples, standard_only=True, min_num_sols=1),
#     num_proc=32
# )
# os.makedirs(SAVE_PATH, exist_ok=True)
# taco_filtered.save_to_disk(SAVE_PATH)
# taco_filtered['train'].to_json(path_or_buf=prefix + "BAAI-TACO-filtered/ALL/formatted_jsonl/train.jsonl", lines=True)
# import pdb; pdb.set_trace()
# print(taco_filtered)
# from collections import Counter
# difficulty_counter = Counter(taco_filtered['train']['difficulty'])
# print(difficulty_counter)