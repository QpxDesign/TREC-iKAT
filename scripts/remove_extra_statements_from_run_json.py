import json

if __name__ == '__main__':
    with open('./output/AUG24_RUN_4.json', 'r') as f:
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
        with open(f"./output/AUG24_RUN_4_CF.json", 'a') as f2:
            f2.write(json.dumps(correct_output))
