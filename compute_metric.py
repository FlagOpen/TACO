from metrics.testing_util import run_test
import json, os
import multiprocessing
import numpy as np
from typing import Dict
from datasets import load_dataset, load_from_disk
import argparse

TIMEOUT = 10

parser = argparse.ArgumentParser()
parser.add_argument('--generation', type=str, default="/share/project/bowen/taco2/taco_gpt4_result_format.jsonl")
parser.add_argument('--result', type=str, default='./')
parser.add_argument('--idx', type=int, default=None)

args = parser.parse_args()
print(args)

def check_correctness(sample, generation, timeout, debug=True):
    """Check correctness of code generation with a global timeout.
    The global timeout is to catch some extreme/rare cases not handled by the timeouts
    inside `run_test`"""
    def _temp_run(sample, generation, debug, result):
        result.append(run_test(sample, test=generation, debug=debug))

    manager = multiprocessing.Manager()
    result = manager.list()
    p = multiprocessing.Process(target=_temp_run, args=(sample, generation, debug, result))
    p.start()
    p.join(timeout=timeout + 1)
    if p.is_alive():
        p.kill()
    if not result:
        in_outs = json.loads(sample["input_output"])
        # consider that all tests failed
        result = [[-1 for i in range(len(in_outs["inputs"]))]]
        if debug:
            print(f"global timeout")
    return result[0]

def load_taco_test(file='/share/project/bowen/copy/lra/cache_data/taco'):
    ds = load_from_disk(file)
    ds_test = ds['test']
    benchmark = {}
    statistics = {
        "difficulty": {},
        "raw": {},
        "tag": {},
        "skill": {},
    }
    for i, data in enumerate(ds_test):
        task_id = f'TACO/TEST_{i}'
        difficulty = data['difficulty']
        raw_tags = data['raw_tags']
        tags = data['tags']
        skill_types = data['skill_types']
        benchmark[task_id] = data
        statistics["difficulty"][difficulty] = statistics["difficulty"].get(difficulty, [])
        statistics["difficulty"][difficulty].append(task_id)
        for raw in raw_tags:
            statistics["raw"][raw] = statistics["raw"].get(raw, [])
            statistics["raw"][raw].append(task_id)
        for tag in tags:
            statistics["tag"][tag] = statistics["tag"].get(tag, [])
            statistics["tag"][tag].append(task_id)
        for skill in skill_types:
            statistics["skill"][skill] = statistics["skill"].get(skill, [])
            statistics["skill"][skill].append(task_id)
    return benchmark, statistics

def load_generation(input_file):
    generations = {}
    with open(input_file, 'r') as f:
        results = json.load(f)
        for i, res in enumerate(results):
            task_id = res['task_id']
            if isinstance(task_id, int):
                task_id = f"TACO/TEST_{task_id}"
            output = res['output']
            generations[task_id] = output
    return generations

def evaluate_generations(generations, samples, idx=None, debug=False):
    task_ids = samples.keys()
    results = {}
    for task_id, problem_generations in generations.items():
        if idx is not None and task_id != f'TACO/TEST_{idx}':
            continue
        assert task_id in task_ids
        sample = samples[task_id]
        res = []
        # loop over the generations
        for o_idx, o in enumerate(problem_generations):
            curr_res = [-2]
            try:
                curr_res = check_correctness(sample, o, timeout=TIMEOUT, debug=debug)
                if debug:
                    print(f"\nSuccessful compilation of task {index}!")
                fixed = []
                for e in curr_res:
                    if isinstance(e, np.ndarray):
                       e = e.item(0)
                    if isinstance(e, np.bool_):
                        e = bool(e)
                    fixed.append(e)
                curr_res = fixed
                if not np.all(curr_res):
                    if debug:
                        print(f"Results were not True for all test cases")
            except Exception as e:
                if debug:
                    print(f"Compilation failed, test framework exception = {repr(e)}{e}\n")
                break
            finally:
                assert isinstance(curr_res, list)
                res.append(curr_res)
        results[task_id] = res
    return results

def load_local_results(results_dir):
    results = {}
    for file in os.listdir(results_dir):
        with open(os.path.join(results_dir, file), 'r') as f:
            results.update(json.load(f))
    return results

def estimate_pass_at_k(num_samples, num_correct, k):
    """Estimates pass@k of each problem and returns them in an array."""

    def estimator(n: int, c: int, k: int) -> float:
        """Calculates 1 - comb(n - c, k) / comb(n, k)."""
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    return np.array([estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)])

def generate_report(metrics, statistics, report_types=['difficulty', 'skill'], difficulty=None, raw=None, tag=None, skill=None):
    def single_report(filtered_metrics, filtered_statistics):
        nums_eval = [f"{metric}: {len(scores.values())}" for metric, scores in filtered_metrics.items()]
        nums_eval = '/'.join(nums_eval) + f'/{len(filtered_statistics)}'
        report = {metric: round(sum(scores.values())*100.0/len(scores.values()), 2) for metric, scores in filtered_metrics.items()}
        report.update({"info": nums_eval})
        return report
    def filter(metrics, statistics, report_type, report_value):
        filtered_statistics = []
        if not report_type:
            filtered_statistics = [f"TACO/TEST_{task_id}" for task_id in range(1000)]
        else:
            assert report_type in ['difficulty', 'raw', 'tag', 'skill']
            filtered_statistics = statistics[report_type][report_value]
        filtered_metrics = {k:{task_id: score for task_id, score in v.items() if task_id in filtered_statistics} for k,v in metrics.items()}
        return filtered_metrics, filtered_statistics
    full_metrics, full_statistics = filter(metrics, statistics, report_type=None, report_value=None)
    reports = {
        'mean' : single_report(full_metrics, full_statistics)
    }
    for report_type in report_types:
        valid_values = statistics[report_type].keys()
        # print(report_type, valid_values)
        if report_type == 'difficulty' and difficulty is not None:
            valid_values = difficulty
        elif report_type == 'raw' and raw is not None:
            valid_values = raw
        elif report_type == 'tag' and tag is not None:
            valid_values = tag
        elif report_type == 'skill' and skill is not None:
            valid_values = skill
        
        reports[report_type] = reports.get(report_type, {})
        if not valid_values:
            continue
        for report_value in valid_values:
            filtered_metrics, filtered_statistics = filter(metrics, statistics, report_type, report_value)
            reports[report_type][report_value] = single_report(filtered_metrics, filtered_statistics)
    return reports
        

def compute_metrics(results, statistics, k_list=[1, 10, 100], report_type=['difficulty', 'skill'], difficulty=None, raw=None, tag=None, skill=None):
    total = []
    correct = []
    task_ids = []
    for task_id, res in results.items():
        all_correct = []
        for generation in res:
            gen = np.array(generation)
            all_correct.append(np.all(gen>0))
        task_ids.append(task_id)
        total.append(len(all_correct))
        correct.append(sum(all_correct))
    total = np.array(total)
    correct = np.array(correct)
    ks = k_list
    pass_at_k = {f"pass@{k}": estimate_pass_at_k(total, correct, k).tolist() for k in ks if (total >= k).all()}
    metrics = {k:dict(zip(task_ids, v)) for k, v in pass_at_k.items()}
    reports = generate_report(metrics, statistics, report_types=['difficulty','skill'])
    return reports



def main():
    taco, statistics = load_taco_test()
    generations = load_generation(args.generation)
    # print(taco.keys())
    # print(generations.keys())
    valid_tasks = generations.keys()
    samples = {k:v for k, v in taco.items() if k in valid_tasks}
    # results = evaluate_generations(generations, samples, idx=args.idx)
    # output_path = os.path.join(args.result, 'tmp_result2')
    # if not os.path.exists(output_path):
    #     os.makedirs(output_path)
    # if args.idx is not None:
    #     filename = f'{args.idx}.json'
    # else:
    #     filename = 'tmp.json'
    # fo = open(output_path+'/'+filename, 'w')
    # json.dump(results, fo, indent=4)
    results = load_local_results('tmp_result2')
    reports = compute_metrics(results, statistics)
    json.dump(reports, open('gpt4_taco_report.json', 'w'), indent=4)

if __name__ == "__main__":
    main()
