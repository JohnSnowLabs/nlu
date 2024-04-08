import argparse
import json
import os
import subprocess
import sys
from typing import List

import colorama


sys.path.append(os.getcwd())
sys.path.append(os.getcwd() + '/tests')
sys.path.append(os.getcwd() + '/tests/utils')
from utils import all_tests
from tests.utils import one_per_lib
_TEST_TIMEOUT = 60*60*20

def run_cmd_and_check_succ(
        args: List[str],
        log=True,
        timeout=60
) -> bool:
    print(f"👷 Executing {colorama.Fore.LIGHTGREEN_EX}{args}{colorama.Fore.RESET}")
    try:
        r = subprocess.run(args, capture_output=True, timeout=timeout)
        was_suc = process_was_suc(r)
        if was_suc:
            print(
                f"{colorama.Fore.LIGHTGREEN_EX}✅ Success running {args}{colorama.Fore.RESET}"
            )
        else:
            print(
                f"{colorama.Fore.LIGHTRED_EX}❌ Failure running {args}{colorama.Fore.RESET}"
            )
        if not was_suc and log:
            log_process(r)
        return was_suc
    except subprocess.TimeoutExpired:
        try:
            log_process(r)
        except:
            print("No logs to print")
        print(f"{colorama.Fore.LIGHTRED_EX}❌ Timeout running {args}{colorama.Fore.RESET}")
        return False


def process_was_suc(
        result: subprocess.CompletedProcess,
) -> bool:
    return result.returncode == 0


def log_process(result: subprocess.CompletedProcess):
    print("______________STDOUT:")
    print(result.stdout.decode())
    print("______________STDERR:")
    print(result.stderr.decode())


if __name__ == '__main__':
    # Workaround until logging issue with pytest-xdist is fixed
    # https://github.com/pytest-dev/pytest-xdist/issues/402
    # We need to launch every test in a separate process
    # because we cannot de-allocate models from JVM from within a test
    # So to prevent JVM-OOM we need to run each test in a separate process

    parser = argparse.ArgumentParser(description='Testing CLI')
    parser.add_argument('test_type', choices=['all', 'one_per_lib'], default='all', help='Type of test to run')
    args = parser.parse_args()
    logs = {}
    tests_to_execute = all_tests

    if args.test_type == 'all':
        tests_to_execute = all_tests
    elif args.test_type == 'one_per_lib':
        tests_to_execute = one_per_lib
    total_tests = len(tests_to_execute)


    print(f'Running Tests: {tests_to_execute}')
    for i, test_params in enumerate(tests_to_execute):
        if i % 10 == 0:
            # Delete models so we dont run out of diskspace
            os.system('rm -r ~/cache_pretrained')
        print(f"{'#' * 10} Running test {i} of {total_tests}  with config {test_params} {'#' * 10}")
        logs[i] = {}
        logs[i]['test_data'] = test_params
        print(f"Running test {i} {test_params}")
        py_path = 'python'
        with open('test.json', 'w') as json_file:
            json.dump(test_params.dict(), json_file)
        cmd = [py_path, 'tests/utils/run_test.py', 'test.json']
        logs[i]['success'] = run_cmd_and_check_succ(cmd, timeout=_TEST_TIMEOUT)

    print(f"{'#' * 10} Failed tests {'#' * 10}")
    failed = 0
    for test_idx in logs:
        if not logs[test_idx]['success']:
            failed += 1
            print(logs[test_idx])
    print(f"{'#' * 10} {failed} of {total_tests} failed {'#' * 10}")

