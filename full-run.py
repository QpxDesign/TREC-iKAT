import utils.llama2 as llama2
import utils.pktb_similarity as pktb_similarity
import utils.genPromptFromPTKBsAndQuestion
import utils.trimPKTB
import json
import time 
import sys
sys.path.append('../')

def run(topic_obj): # outputs JSON that fufils all requirements (ranked PTKBs from )
    data = {}
    PTKBs = [None] * len(topic_obj['ptkb'])
    index = 0
    while index < len(topic_obj['ptkb']):
        PTKBs[index] = topic_obj['ptkb'][str(index+1)]
        index += 1
    turn_index = 2
    a = pktb_similarity.rankPTKBS(PTKBs,topic_obj["turns"][turn_index]["resolved_utterance"])
    data["user utterance"] = topic_obj["turns"][turn_index]["resolved_utterance"]
    data["ranked_ptkbs"] = utils.trimPKTB.trimPKTB(a)
    
    prompt = utils.genPromptFromPTKBsAndQuestion.genPromptFromPTKBAndQuestion(data["ranked_ptkbs"],topic_obj["turns"][turn_index]["resolved_utterance"])
    print("prompt: " + prompt)
    b = llama2.gen_response(prompt,topic_obj["turns"][0:turn_index])
    print(b)
    """ 
    filename = f"run-{time.time()}.json"
    with open(f"./output/{filename}", 'a') as f2:
        f2.write(json.dumps(data))
        """


if __name__ == '__main__':
    with open('./data/2023_train_topics.json', 'r') as f: 
        data = json.load(f)
        index = 0
        run(data[index])
        
