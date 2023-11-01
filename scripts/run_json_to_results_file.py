import json
import sys

if __name__ == '__main__':  # run like python3 run_json_to_results_file.py <run_json> <output_file_name>
    output = []
    with open(f'./{sys.argv[1]}', 'r') as f:
        data = json.load(f)
        for turn in data["turns"]:
            for passage in turn['responses'][0]['passage_provenance']:
                if passage['used'] == True:
                    output.append(
                        f"{turn['turn_id']}   0   {passage['id']}   0   0   a")
        final = ""
        for o in output:
            final += o + "\n"
        with open(f"./{sys.argv[2]}", 'a') as f2:
            f2.write(final)
