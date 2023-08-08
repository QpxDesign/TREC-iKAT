import utils.llama2 as llama2
import utils.pktb_similarity as pktb_similarity
import utils.gen_prompt_from_ptkbs_and_question
import utils.trim_PTKB
import utils.json_ptkb_dict_to_array
import utils.get_passages
import json
import sys
import time
sys.path.append('../')

start_time = time.time()
total_turns = 0
def run(topic_obj): # outputs JSON that fufils all requirements (ranked PTKBs from )
    global total_turns
    PTKBs = utils.json_ptkb_dict_to_array.format(topic_obj)
    turn_index = 0
    ouput = {
        "run_name":"georgetown_infosense_run",
        "run_type": "automatic",
        "turns" : []
    }
    turn_outputs = []
    for obj in topic_obj["turns"]:
        total_turns += 1

        a = pktb_similarity.rankPTKBS(PTKBs,obj["resolved_utterance"])
        ranked_ptkbs = utils.trim_PTKB.trim(a)
        
        prompt = utils.gen_prompt_from_ptkbs_and_question.gen(ranked_ptkbs,obj["resolved_utterance"])
        print("prompt: " + prompt)
        b = llama2.gen_response(prompt,topic_obj["turns"][0:turn_index])
        print(b)
        ptkb_provenance_objs = []
        
        for ptkb in ranked_ptkbs:
            ptkb_provenance_objs.append({
                "id": PTKBs.index(ptkb[0]),
                "text":ptkb,
                "score":ptkb[1]

            })
        passage_provenance_objs = []
        passages = utils.get_passages.getPassagesFromSearchQuery(b)
        for passage in passages:
            passage_provenance_objs.append({
                "id":passage.docid,
                "text":json.loads(passage.raw)["contents"],
                "score":passage.score
            })
        turn_outputs.append({
            "turn_id":f"{topic_obj['number']}_{obj['turn_id']}",
            "responses": [
                {
                    "rank":1,
                    "text":b,
                    "ptkb_provenance":ptkb_provenance_objs,
                    "passage_provenance":passage_provenance_objs
                }

            ]
        })
        turn_index += 1
        print(f"STATUS UPDATE: FINISHED TURN {turn_index}/{len(topic_obj['turns'])} - TOPIC {topic_obj['number']} @ {(time.time()-start_time)/60}min elapsed - {total_turns}/332 DONE")
    
    return turn_outputs

        


if __name__ == '__main__':
    with open('./data/2023_test_topics.json', 'r') as f: 
        data = json.load(f)
        index = 0
        final = []
        for o in data:
            final.append(run(o))
        filename = f"run-{time.time()}.json"
        with open(f"./output/{filename}", 'a') as f2:
            f2.write(json.dumps(final))

        
