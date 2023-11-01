import json
import sys
# RUN LIKE: python3 scripts/remove_extra_statements_from_run_json.py <name of run JSON file>.json MODE ('normal','fix_ptkb_index')

if __name__ == '__main__':
    with open(f'./output/{sys.argv[1]}', 'r') as f:
        data = json.load(f)
        correct_output = {
            "run_name": data['run_name'],
            "run_type": data['run_type'],
            "turns": []
        }
        for t in data['turns']:
            ptkbs = t['responses'][0]['ptkb_provenance']
            new_ptkbs = []
            if sys.argv[2] == 'fix_ptkb_index':
                for ptkb_entry in ptkbs:
                    new_ptkbs.append({
                        "id": str(int(ptkb_entry['id'])+1),
                        "text": ptkb_entry['text'],
                        "score": ptkb_entry['score']
                    })

            correct_turn = {
                "turn_id": t["turn_id"],
                "responses": [
                    {
                        "rank": 1,
                        "text": t['responses'][0]['text'],
                        "ptkb_provenance": t['responses'][0]['ptkb_provenance'] if not sys.argv[2] == 'fix_ptkb_index' else new_ptkbs,
                        "passage_provenance": t['responses'][0]['passage_provenance']
                    }
                ]
            }
            correct_output['turns'].append(correct_turn)
        with open(f"./output/{sys.argv[1].split('.')[0]}-2.json", 'a') as f2:
            f2.write(json.dumps(correct_output))
