from metrics.testing_util import run_test
import json, os
import time
import multiprocessing
import numpy as np
from typing import Dict
from datasets import load_dataset
from utils import get_taco_dataset
from tqdm import tqdm
import re
import argparse

TIMEOUT = 10


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
    p.join()
    if p.is_alive():
        p.kill()
    if not result:
        in_outs = json.loads(sample["input_output"])
        # consider that all tests failed
        result = [[-1 for i in range(len(in_outs["inputs"]))]]
        if debug:
            print(f"global timeout")
    return result[0]

def load_generation(input_file):
    generations = {}
    with open(input_file, 'r') as f:
        results = json.load(f)
        for _, res in enumerate(results):
            task_id = res['task_id']
            output = res['output']
            generations[task_id] = output
    return generations

def extract_code(generation:str):
    code_blocks = re.findall(r'```(?:\w+)?\n?([^`]+)```', generation, re.DOTALL)
    if len(code_blocks)==0:
        return generation
    return code_blocks[0]

def get_score(res:list):
    # get number of True in list
    return np.sum([r for r in res if not r<0])

def process_generation(args):
    task_id, sample, problem_generations, debug, ref_code_fix = args
    res = []
    if ref_code_fix:
        ref_code_results = sample['ref_code_results']
        ref_score = max(1, int(ref_code_results[len(ref_code_results)*3//4]))
    # sample_copy = sample.copy()
    # sample_copy.pop("input_output")
    # sample_copy['solutions'] = eval(sample_copy['solutions'])[:1]
    # print(json.dumps(sample_copy, indent=4, ensure_ascii=False))
    # import pdb; pdb.set_trace()
    # loop over the generations
    for o_idx, o in enumerate(problem_generations):
        # import pdb; pdb.set_trace()
        curr_res = [-2]
        try:
            code = extract_code(o)
        except Exception as e:
            if debug:
                print(f"Failed to extract code from generation {o_idx}, exception = {repr(e)}{e}\n")
            code = ""
        try:
            curr_res = check_correctness(sample, code, timeout=TIMEOUT, debug=debug)
            if debug:
                print(f"\nSuccessful compilation of task {o_idx}!")
            fixed = []
            for e in curr_res:
                if isinstance(e, np.ndarray):
                    e = e.item(0)
                if isinstance(e, np.bool_):
                    e = bool(e)
                fixed.append(e)
            curr_res = fixed
            if not np.all(curr_res):
                if ref_code_fix and get_score(curr_res) >= ref_score:
                    print(f"Results were not True for all test cases, but the score is higher than the reference code ({get_score(curr_res)} >= {ref_score})")
                    curr_res = [True]*ref_score
                else:
                    print(get_score(curr_res), ref_score)
                # import pdb; pdb.set_trace()
                if debug:
                    print(f"Results were not True for all test cases")
        except Exception as e:
            if debug:
                print(f"Compilation failed, test framework exception = {repr(e)}{e}\n")
            break
        finally:
            assert isinstance(curr_res, list)
            res.append(curr_res)
    return task_id, res

def evaluate_generations(generations, samples, idx=None, debug=False, ref_code_fix=True):
    assert len(generations.keys()) == len(samples)
    results = {}
    idx = 0
    for task_id, problem_generations in tqdm(generations.items(), desc=f"Evaluating {len(generations.keys())} problems", position=0, leave=True):
        sample = samples[task_id]
        args = (task_id, sample, problem_generations, debug, ref_code_fix)
        _, res = process_generation(args)
        # print(f"Problem {task_id} evaluation finished, result: {res}")
        results[task_id] = res
        idx += 1
    return results

def evaluate_generations_parallel(generations, samples, idx=None, debug=False, ref_code_fix=True, nproc = -1):
    assert len(generations.keys()) == len(samples)
    import multiprocessing as mp
    def mp_worker(task, args, semaphore, result_queue, info = None):
        with semaphore:
            task_id = args[0]
            print(f"evaluating {task_id}-th problem")
            result = task(args)
            result_queue.put(result)
            print(f"Problem {task_id} evaluation finished, result: {result}")
            time.sleep(1)

    if nproc == -1:
        nproc = mp.cpu_count()
    semaphore = multiprocessing.Semaphore(nproc)
    args_list = [(task_id, samples[task_id], problem_generations, debug, ref_code_fix) for i, (task_id, problem_generations) in enumerate(generations.items())]
    result_queue = multiprocessing.Queue()
    processes = []
    for args in args_list:
        p = multiprocessing.Process(target=mp_worker, args=(process_generation, args, semaphore, result_queue))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()
    
    results = {}
    while not result_queue.empty():
        task_id, res = result_queue.get()
        results[task_id]=res
    return results

def evaluate_ref_codes(sample, max_used_sols=-1):
    # evaluate reference codes to see how many test cases they pass
    ref_codes = json.loads(sample['solutions'])
    ref_code_results = []
    for ref_code in ref_codes[:max_used_sols]:
        try:
            code = extract_code(ref_code)
        except Exception as e:
            code = "" 
        try:
            curr_res = check_correctness(sample, code, timeout=TIMEOUT, debug=False)
            fixed = []
            for e in curr_res:
                if isinstance(e, np.ndarray):
                    e = e.item(0)
                if isinstance(e, np.bool_):
                    e = bool(e)
                fixed.append(e)
            curr_res = fixed
            ref_code_results.append(get_score(curr_res))
        except Exception as e:
            ref_code_results.append(0)
    ref_code_results = sorted(ref_code_results)
    sample['ref_code_results'] = ref_code_results
    return sample

def estimate_pass_at_k(num_samples, num_correct, k):
    """Estimates pass@k of each problem and returns them in an array."""

    def estimator(n: int, c: int, k: int) -> float:
        """Calculates 1 - comb(n - c, k) / comb(n, k)."""
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))
    import itertools
    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    return np.array([estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)])
        

def compute_metrics(results, k_list=[1, 10, 100]):
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
    print(total)
    ks = k_list
    detail_pass_at_k = {f"pass@{k}": estimate_pass_at_k(total, correct, k).tolist() for k in ks if (total >= k).all()}
    pass_at_k = {f"pass@{k}": estimate_pass_at_k(total, correct, k).mean() for k in ks if (total >= k).all()}
    detail_metrics = {k:dict(zip(task_ids, v)) for k, v in detail_pass_at_k.items()}
    pass_at_k["detail"] = detail_metrics
    return pass_at_k

def by_difficulties(results, taco):
    detail_metrics = results["detail"]
    results["by_difficulties"] = {}
    for metric, result in detail_metrics.items():
        stats = {}
        stats_nospj = {}
        for task_id, val in result.items():
            difficulty = taco[int(task_id)]['difficulty']
            if difficulty not in stats:
                stats[difficulty] = []
                if taco[int(task_id)]['special_judge_case'] != 1:
                    stats_nospj[difficulty] = []
            stats[difficulty].append(val)
            if taco[int(task_id)]['special_judge_case'] != 1:
                stats_nospj[difficulty].append(val)
        results["by_difficulties"][metric] = {
            k: np.mean(v) for k, v in stats.items()
        }
        results["by_difficulties"][metric+"_nospj"] = {
            k: np.mean(v) for k, v in stats_nospj.items()
        }
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_path', type=str, help='path to the results')
    parser.add_argument('--dataset_config', type=str, help='path to TACO dataset configs')
    parser.add_argument('--split', type=str, default='test', help='split of the dataset (train, val, test)')
    parser.add_argument('--debug', action='store_true', help='print debug information')
    parser.add_argument('--skip', action='store_true', help='skip')
    parser.add_argument('--nproc', type=int, default=1, help='num procs for evalutaion')
    args = parser.parse_args()

    with open(args.dataset_config, 'r') as f:
        dataset_config = json.load(f)
    taco = get_taco_dataset(split=args.split, **dataset_config)

    generation_file = os.path.join(args.result_path, 'generations.json')
    generations = load_generation(generation_file)

    if args.skip and os.path.exists(os.path.join(args.result_path, 'taco_metrics.json')):
        metrics = json.load(open(os.path.join(args.result_path, 'taco_metrics.json')))
    else:
        if args.nproc == 1:
            results = evaluate_generations(generations, taco, debug=args.debug)
        else:
            # You can use evaluate_generations_parallel to parallel executing multiple outputs for each problem
            results = evaluate_generations_parallel(generations, taco, debug=args.debug, nproc = args.nproc)
        metrics = compute_metrics(results)
    metrics = by_difficulties(metrics, taco)

    json.dump(metrics, open(os.path.join(args.result_path, 'taco_metrics.json'), 'w'), indent=4)

if __name__ == "__main__":
    main()
