import json

# count total number of turns in topics.json file
with open('./data/2023_test_topics.json', 'r') as f: 
    total_turns = 0
    data = json.load(f)
    for obj in data:
        total_turns +=  len(obj['turns'])
    print(total_turns)

