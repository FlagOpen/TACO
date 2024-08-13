# modifed from https://github.com/hendrycks/apps/blob/main/eval/testing_util.py to fix some evaluation bugs and add instructions

from .pyext2 import RuntimeModule
import signal
import numpy as np

# used for debugging to time steps
from datetime import datetime
import time

import os, sys, json, shutil
import faulthandler

import subprocess
import tempfile

from enum import Enum
class CODE_TYPE(Enum):
    call_based = 0
    standard_input = 1

# to run the solution files we're using a timing based approach
import signal
# stuff for setting up signal timer
class TimeoutException(Exception):
    pass
def timeout_handler(signum, frame):
    print("alarm went off")
    #return
    raise TimeoutException
signal.signal(signal.SIGALRM, timeout_handler)
TIMEOUT = 4  # seconds

EXECUTION_RESULTS = {1: "passed", 0: "false", -1: "timeout", -2: "runtime_error", -3: "returncode:{code}", -4: "compile_error"}

def run_test(sample, test=None, debug=False):
    """
    if test(generated_code) is not None it'll try to run the code.
    otherwise it'll just return an input and output pair.
    """
    
    if debug:
        print(f"start = {datetime.now().time()}")

    try:
        in_outs = json.loads(sample["input_output"])
    except ValueError:
        in_outs = None
    
    if in_outs:
        if in_outs.get("fn_name") is None:
            which_type = CODE_TYPE.standard_input  # Standard input
            method_name = None
        else:
            which_type = CODE_TYPE.call_based  # Call-based
            method_name = in_outs["fn_name"]
    inputs_list = []
    outputs_list = []
    for index, inputs in enumerate(in_outs["inputs"]):
        outputs = in_outs["outputs"][index]
        inputs, outputs = process_input_output(inputs, outputs)
        inputs_list.append(inputs)
        outputs_list.append(outputs)

    if debug:
        print(f"loaded input_output = {datetime.now().time()}")

    if test is None:
        return None
    elif test is not None:
        results = []
        if debug:
            print(f"loading test code = {datetime.now().time()}")
        if which_type == CODE_TYPE.call_based:
            synthesized_code = synthesize_cb_code(test, debug)
            method_func = compile_and_get_func(synthesized_code, which_type, method_name, timeout=TIMEOUT, debug=debug)
        elif which_type == CODE_TYPE.standard_input:
            synthesized_code, exec_code = synthesize_std_code(test, debug)
            method_func = compile_and_get_func(synthesized_code, which_type, method_name, timeout=TIMEOUT, debug=debug)
        if not method_func:
            results.append(-2)
            return results
        else:
            if which_type == CODE_TYPE.call_based:  # Call-based
                detail_results, debug_infos = execute_cb_code(method_func, inputs_list, outputs_list, timeout=TIMEOUT, early_stop=False, debug=debug)
            elif which_type == CODE_TYPE.standard_input:
                detail_results = execute_std_code(exec_code, inputs_list, outputs_list, timeout=TIMEOUT, early_stop=False, debug=debug)
                debug_infos = detail_results.get('debug', None)
                detail_results = {k:v for k, v in detail_results.items() if k!='debug'}
                if set(detail_results.values()) == {(False, 'returncode:1')}:
                    detail_results = execute_std_code(synthesized_code+'\ncode()\n', inputs_list, outputs_list, timeout=TIMEOUT, early_stop=False, debug=debug)
                
        if isinstance(detail_results, list):
            if len(detail_results) == 1:
                detail_results = detail_results * len(inputs_list)
            detail_results = dict(zip([i for i in range(len(inputs_list))], detail_results))
        for test_id, test_result in detail_results.items():
            if test_result[1] == "passed":
                results.append(True)
            elif test_result[1] == "false":
                results.append(False)
            elif test_result[1] == "timeout":
                results.append(-1)
            else:
                results.append(-3)
        return results

def process_input_output(inputs, outputs):
    # JSON forces dictionaries to have string keys; this undoes this (assuming a singleton list)
    try:
        if isinstance(inputs[0], dict):
            inputs = [{int(k): v for k,v in inputs[0].items()}]
    except:
        True
    
    try:
        if isinstance(outputs, dict):
            outputs = [{int(k): v for k,v in outputs.items()}]
    except:
        True

    try:
        if isinstance(outputs[0], dict):
            outputs = [{int(k): v for k,v in outputs[0].items()}]
    except:
        True
    
    return inputs, outputs

def compile_and_get_func(program, which_type, method_name, timeout, debug):
    signal.alarm(timeout)
    try:
        tmp_sol = RuntimeModule.from_string("tmp_sol", "", program)
        if which_type == CODE_TYPE.call_based and "class Solution" in program:
            tmp = tmp_sol.Solution()
        else:
            tmp = tmp_sol
        signal.alarm(0)
    except Exception as e:
        signal.alarm(0)
        if debug:
            print(f"compilation error = {e}")
        return False
    signal.alarm(0)
    
    if which_type == CODE_TYPE.call_based:
        assert isinstance(method_name, str)
    else:
        method_name = "code"
    
    try:
        method = getattr(tmp, method_name)  # get_attr second arg must be str
    except:
        signal.alarm(0)
        signal.alarm(0)
        e = sys.exc_info()
        if debug:
            print(f"unable to get function error = {e}")
        return False
    return method

import re
def convert_main_func(code:str):
    pattern = r'__name__\s*==\s*["\']__main__["\']'
    return re.sub(pattern, 'True', code)

def synthesize_cb_code(raw_code, debug=False):
    sol = "import sys\nimport time\nimport itertools\nfrom itertools import accumulate, product, permutations, combinations\nimport collections\nfrom collections import Counter, OrderedDict, deque, defaultdict, ChainMap\nfrom functools import lru_cache\nimport math\nfrom math import sqrt, sin, cos, tan, ceil, fabs, floor, gcd, exp, log, log2\nimport fractions\nfrom typing import List, Tuple\nimport numpy as np\nimport random\nimport heapq\nfrom heapq import *\n"
    if debug:
        print(f"loading test code = {datetime.now().time()}")
    sol += raw_code
    return sol

def synthesize_std_code(raw_code, debug=False):
    normal_import_lines = "import sys\nimport time\nimport itertools\nfrom itertools import accumulate, product, permutations, combinations\nimport collections\nfrom collections import Counter, OrderedDict, deque, defaultdict, ChainMap\nfrom functools import lru_cache\nimport math\nfrom math import sqrt, sin, cos, tan, ceil, fabs, floor, gcd, exp, log, log2\nimport fractions\nfrom typing import List, Tuple\nimport numpy as np\nimport random\nimport heapq\nfrom heapq import *\n"
    if debug:
        print(f"loading test code = {datetime.now().time()}")
    
    sol = "" # code for compile
    sol2 = "" # code for execute

    tmp_test = convert_main_func(raw_code).split("\n")
    # define the code line type, 1 for import lines, 2 for import * lines with indent, 0 for normal codes
    code_types = [] 


    for x in tmp_test:
        if 'import *' in x:
            code_types.append(2)
        elif x.startswith("from ") or x.startswith("import "):
            code_types.append(1) 
        else:
            code_types.append(0)
    
    started = False

    special_import_lines = [i.lstrip('\t') for idx, i in enumerate(tmp_test) if code_types[idx]==2]
    special_import_lines = '\n'.join(special_import_lines)

    for idx, i in enumerate(tmp_test):
        code_type = code_types[idx]
        if code_type == 0 and not started:
            sol2 += normal_import_lines
            sol2 += "\nstdin = sys.stdin\nstdout = sys.stdout\n"
            sol2 += f"{i}\n"

            sol += normal_import_lines
            sol += special_import_lines
            sol += "\nstdin = sys.stdin\nstdout = sys.stdout\n"
            sol += "def code():\n"
            sol += f"\t{i}\n"
            started = True
        else:
            sol2 += f"{i}\n"
            if code_type < 2:
                if started:
                    sol += '\t'
                sol += f"{i}\n"
    
    if debug:
        print(f"sol = {sol}")
        print(f"sol2 = {sol2}")
    
    return sol, sol2

def execute_cb_code(method, inputs_list, outputs_list, timeout, early_stop=False, debug=False):
    # Disable functionalities that can make destructive changes to the test.
    reliability_guard()
    results = []
    debug_infos = {}
    for index, inputs in enumerate(inputs_list):
        if debug:
            debug_infos[index] = {}
        signal.alarm(timeout) 
        faulthandler.enable()
        outputs = outputs_list[index]
        try:
            exec_outputs = method(*inputs)
        except Exception as e:
            signal.alarm(0)
            faulthandler.disable()
            if debug:
                print(f"Standard input runtime error = {e}")
            results.append((False, EXECUTION_RESULTS[-2]))
            continue
        try:
            # ground truth sequences are not tuples
            if isinstance(exec_outputs, tuple):
                exec_outputs = list(exec_outputs)
            
            tmp_result = exec_outputs == outputs
            if isinstance(outputs, list) and outputs:
                tmp_result = tmp_result or (exec_outputs == outputs[0])

            # ground truth sequences are not tuples
            try:
                if isinstance(exec_outputs[0], tuple):
                    exec_outputs = [list(x) for x in exec_outputs]
                    tmp_result = tmp_result or (exec_outputs == outputs[0])
            except:
                True
            if tmp_result:
                results.append((True, EXECUTION_RESULTS[1]))
            else:
                results.append((False, EXECUTION_RESULTS[0]))
            
            # reset the alarm
            signal.alarm(0)
        except Exception as e:
            signal.alarm(0)
            faulthandler.disable()
            if debug:
                print(f"Standard input time limit exceeded error = {e}")
            results.append((False, EXECUTION_RESULTS[-1]))
            continue
        faulthandler.disable()
        signal.alarm(0)
        if debug:
            print(f"outputs = {exec_outputs}, test outputs = {outputs}, inputs = {inputs}, {type(inputs)}, {exec_outputs == [outputs]}")
            debug_infos[index] = {
                    'inputs': inputs,
                    'gt_outputs': outputs,
                    'exec_outputs': exec_outputs
                }
    return results, debug_infos

def remove_tmp_files(path=None):
    if path is not None:
        if os.path.exists(path):
            shutil.rmtree(path)
            return None
    tmp_files = ['input.txt', 'output.txt']
    for tmp_file in tmp_files:
        if tmp_file in os.listdir('.'):
            os.remove(tmp_file)

def execute_std_code(synthesized_code, inputs_list, outputs_list, timeout, early_stop=False, debug=False):
    temp_program_path = create_temp_file(synthesized_code)
    if debug:
        print("Test program:", temp_program_path)
    assert isinstance(inputs_list, list) and isinstance(outputs_list, list)
    assert len(inputs_list) == len(outputs_list)
    exec_results = {}
    if debug:
        exec_results['debug'] = {}
    stamp = int(time.time()*1000000)
    for i, inputs in enumerate(inputs_list):
        remove_tmp_files(f'./tmp/{stamp}')
        outputs = outputs_list[i]
        if isinstance(inputs, list):
            inputs = "\n".join(inputs)
        if isinstance(outputs, list):
            outputs = "\n".join(outputs)
        try:
            result = subprocess.run(['python', temp_program_path], input=inputs, text=True, capture_output=True, timeout=timeout)  
            exec_code = 999
        except subprocess.TimeoutExpired:
            exec_code = -1
        except Exception as e:
            print(e)
            exec_code = -2

        
        if exec_code > 0:
            if result.returncode != 0:
                try:
                    inputs_tmp_file = open(create_temp_file(inputs), 'r')
                    result = subprocess.run(['python', temp_program_path], stdin=inputs_tmp_file, text=True, capture_output=True, timeout=timeout)
                    assert result.returncode == 0
                    if compare_std_results(result.stdout, outputs, debug):
                        exec_code = 1
                    else:
                        exec_code = 0
                except:
                    try:
                        stamp = int(time.time()*1000000)
                        os.makedirs(f'./tmp/{stamp}', exist_ok=True)
                        inputs_tmp_file = f'./tmp/{stamp}/input_{i}.txt'
                        with open(inputs_tmp_file, 'w') as ftemp:
                            ftemp.write(inputs)
                        result = subprocess.run(['python', temp_program_path], text=True, timeout=timeout)
                        assert result.returncode == 0
                        if compare_std_results(open(f'./tmp/{stamp}/output_{i}.txt').read(), outputs, debug):
                            exec_code = 1
                        else:
                            exec_code = 0
                        
                    except:
                        # print('!!!!!!!!!!!!!')
                        exec_code = -3
            elif compare_std_results(result.stdout, outputs, debug):
                exec_code = 1
            else:
                exec_code = 0
        exec_results[i] = (exec_code==1, EXECUTION_RESULTS[exec_code] if exec_code>-3 else EXECUTION_RESULTS[exec_code].format(code=result.returncode))
        if exec_code >= 0:
            if debug:
                print_debug_info(inputs=inputs, outputs=outputs, exec_outputs=result.stdout)
                exec_results['debug'][i] = {
                    'inputs': inputs,
                    'gt_outputs': outputs,
                    'exec_outputs': result.stdout
                }
        if early_stop and exec_code<=0:
            break
    return exec_results

def print_debug_info(inputs, outputs, exec_outputs):
    nl = "\n"
    if not isinstance(inputs, list):
        print(f"exec output = {exec_outputs}, test outputs = {outputs}, inputs = {inputs.replace(nl,' new-line ')}, {type(inputs)}, {exec_outputs == [outputs]}")
    else:
        print(f"exec output = {exec_outputs}, test outputs = {outputs}, inputs = {inputs}, {type(inputs)}, {exec_outputs == [outputs]}")

def create_temp_file(content):
    with tempfile.NamedTemporaryFile(delete=False, mode='w', encoding='utf-8') as temp_file:
        temp_file.write(content)
        temp_file_path = temp_file.name
    return temp_file_path

def compare_std_results(exec_outputs, outputs, debug=False):
    if stripped_string_compare(exec_outputs, outputs):
        return True
    
    if isinstance(exec_outputs, list):
        output_1 = "\n".join(exec_outputs)
        if stripped_string_compare(output_1, outputs):
            return True
    
    if isinstance(exec_outputs, list):
        output_2 = [o.lstrip().rstrip() for o in exec_outputs]
        output_2 = "\n".join(output_2)
        if stripped_string_compare(output_2, outputs):
            return True
    
    tmp_result = False
    # ground truth sequences are expressed as lists not tuples
    if isinstance(outputs, tuple):
        outputs = list(outputs)
    
    try:
        tmp_result = (exec_outputs == [outputs])
        if isinstance(outputs, list):
            tmp_result = tmp_result or (exec_outputs == outputs)
            if isinstance(exec_outputs[0], str):
                tmp_result = tmp_result or ([e.strip() for e in exec_outputs] == outputs)
    except Exception as e:
        if debug:
            print(f"Failed check1 exception = {e}")
        pass
    if tmp_result:
        return True
    
    # try one more time without \n
    if isinstance(outputs, list):
        for tmp_index, i in enumerate(outputs):
            outputs[tmp_index] = i.split("\n")
            outputs[tmp_index] = [x.strip() for x in outputs[tmp_index] if x]
    else:
        outputs = outputs.split("\n")
        outputs = list(filter(len, outputs))
        outputs = list(map(lambda x:x.strip(), outputs))
    
    try:
        tmp_result = (exec_outputs == [outputs])
        if isinstance(outputs, list):
            tmp_result = tmp_result or (exec_outputs == outputs)
    except Exception as e:
        if debug:
            print(f"Failed check2 exception = {e}")
        pass
    if tmp_result:
        return True
    
    # try by converting the output into a split up list too
    if isinstance(exec_outputs, list):
        exec_outputs = list(filter(len, exec_outputs))
    try:
        tmp_result = (exec_outputs == [outputs])
        if isinstance(outputs, list):
            tmp_result = tmp_result or (exec_outputs == outputs)
    except Exception as e:
        if debug:
            print(f"Failed check3 exception = {e}")
        pass
    if tmp_result:
        return True
    

    try:
        output_float = [float(e) for e in exec_outputs]
        gt_float = [float(e) for e in outputs]
        tmp_result = tmp_result or ((len(output_float) == len(gt_float)) and np.allclose(output_float, gt_float))
    except Exception as e:
        pass
    try:
        if isinstance(exec_outputs[0], list):
            output_float = [float(e) for e in exec_outputs[0]]
            gt_float = [float(e) for e in outputs[0]]
            tmp_result = tmp_result or ((len(output_float) == len(gt_float)) and np.allclose(output_float, gt_float))
    except Exception as e:
        pass
    if tmp_result:
        return True

    
    if isinstance(outputs, list):
        for tmp_index, i in enumerate(outputs):
            outputs[tmp_index] = set(i.split())
    else:
        outputs = set(outputs.split())

    try:
        tmp_result = (exec_outputs == outputs)
    except Exception as e:
        if debug:
            print(f"Failed check4 exception = {e}")
    if tmp_result:
        return True
    
    # try by converting the output into a split up list too
    if isinstance(exec_outputs, list):
        for tmp_index, i in enumerate(exec_outputs):
            exec_outputs[tmp_index] = i.split()
        exec_outputs = list(filter(len, exec_outputs))
        for tmp_index, i in enumerate(exec_outputs):
            exec_outputs[tmp_index] = set(i)    
    else:
        exec_outputs = exec_outputs.split()
        exec_outputs = list(filter(len, exec_outputs))
        exec_outputs = set(exec_outputs)
    try:
        tmp_result = (set(frozenset(s) for s in exec_outputs) == set(frozenset(s) for s in outputs))
    except Exception as e:
        if debug:
            print(f"Failed check5 exception = {e}")
    
    # if they are all numbers, round so that similar numbers are treated as identical
    try:
        tmp_result = tmp_result or (set(frozenset(round(float(t),3) for t in s) for s in exec_outputs) ==\
            set(frozenset(round(float(t),3) for t in s) for s in outputs))
    except Exception as e:
        if debug:
            print(f"Failed check6 exception = {e}")

    if tmp_result:
        return True
    
    return False

def stripped_string_compare(s1, s2):
    s1 = s1.lstrip().rstrip()
    s2 = s2.lstrip().rstrip()
    return s1 == s2

def reliability_guard(maximum_memory_bytes=None):
    """
    This disables various destructive functions and prevents the generated code
    from interfering with the test (e.g. fork bomb, killing other processes,
    removing filesystem files, etc.)
    WARNING
    This function is NOT a security sandbox. Untrusted code, including, model-
    generated code, should not be blindly executed outside of one. See the
    Codex paper for more information about OpenAI's code sandbox, and proceed
    with caution.
    """

    if maximum_memory_bytes is not None:
        import resource

        resource.setrlimit(resource.RLIMIT_AS, (maximum_memory_bytes, maximum_memory_bytes))
        resource.setrlimit(resource.RLIMIT_DATA, (maximum_memory_bytes, maximum_memory_bytes))
        if not platform.uname().system == "Darwin":
            resource.setrlimit(resource.RLIMIT_STACK, (maximum_memory_bytes, maximum_memory_bytes))

    faulthandler.disable()

    import builtins

    builtins.exit = None
    builtins.quit = None

    import os

    os.environ["OMP_NUM_THREADS"] = "1"

    os.kill = None
    os.system = None
    os.putenv = None
    os.remove = None
    os.removedirs = None
    os.rmdir = None
    os.fchdir = None
    os.setuid = None
    os.fork = None
    os.forkpty = None
    os.killpg = None
    os.rename = None
    os.renames = None
    os.truncate = None
    os.replace = None
    os.unlink = None
    os.fchmod = None
    os.fchown = None
    os.chmod = None
    os.chown = None
    os.chroot = None
    os.fchdir = None
    os.lchflags = None
    os.lchmod = None
    os.lchown = None
    os.getcwd = None
    os.chdir = None

    import shutil

    shutil.rmtree = None
    shutil.move = None
    shutil.chown = None

    import subprocess

    subprocess.Popen = None  # type: ignore

    __builtins__["help"] = None

    import sys

    sys.modules["ipdb"] = None
    sys.modules["joblib"] = None
    sys.modules["resource"] = None
    sys.modules["psutil"] = None
    sys.modules["tkinter"] = None
