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
        for t in data['turns']:
            correct_turn = {
                "turn_id": t["turn_id"],
                "responses": [
                    {
                        "rank":1,
                        "text": t['responses'][0]['text'],
                        "ptkb_provenance": t['responses'][0]['ptkb_provenance'],
                        "passage_provenance": t['responses'][0]['passage_provenance']
                    }
                ]
            }
            correct_output['turns'].append(correct_turn) 
        with open(f"./output/{sys.argv[1]}_CF.json", 'a') as f2:
            f2.write(json.dumps(correct_output))
