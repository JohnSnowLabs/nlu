import json
import os
import sys

sys.path.append(os.getcwd())
from test_utils import model_and_output_levels_test

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python run_test.py <json_file>")
        sys.exit(1)

    with open(sys.argv[1], 'r') as json_file:
        test_data = json.load(json_file)

    # Call your test function with the test data from the JSON file
    print('Running test with data: ', type(test_data), test_data)
    model_and_output_levels_test(**test_data)
