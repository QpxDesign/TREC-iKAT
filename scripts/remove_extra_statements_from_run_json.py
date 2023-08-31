import json
import sys

# RUN LIKE: python3 scripts/remove_extra_statements_from_run_json.py <name of run JSON file>.json

if __name__ == '__main__':
    with open(f'./output/{sys.argv[1]}', 'r') as f:
        data = json.load(f)
        correct_output = {
            "run_name": data['run_name'],
            "run_type": data['run_type'],
            "turns": []
        }
        for topic_subarray in data['turns']:
            correct_output['turns'] += topic_subarray
        with open(f"./output/{sys.argv[1].split('.')[0]}_CF-2.json", 'a') as f2:
            f2.write(json.dumps(correct_output))
