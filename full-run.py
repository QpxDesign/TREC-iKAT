import utils.llama2 as llama2
import utils.pktb_similarity as pktb_similarity
import utils.gen_prompt_from_ptkbs_and_question
import utils.trim_PTKB
from utils.trim_passages import trim_passages
import utils.json_ptkb_dict_to_array
import utils.get_passages
import json
import time
import math
from utils.fastchat import summarize_with_fastchat
import utils.chatgpt as chatgpt
from utils.extract_keywords import extract_keywords

start_time = time.time()
total_turns = 0
def run(topic_obj): # outputs JSON that fufils all requirements (ranked PTKBs from )
    global total_turns
    PTKBs = utils.json_ptkb_dict_to_array.format(topic_obj)
    turn_index = 0
    turn_outputs = []
    for obj in topic_obj["turns"]:
        total_turns += 1

        a = pktb_similarity.rankPTKBS(PTKBs,obj["resolved_utterance"])
        ranked_ptkbs = utils.trim_PTKB.trim(a)
        
        prompt = utils.gen_prompt_from_ptkbs_and_question.gen(ranked_ptkbs,obj["resolved_utterance"])
        print("prompt: " + prompt)
        preliminary_response = llama2.gen_response(prompt,topic_obj["turns"][0:turn_index])
        ptkb_provenance_objs = []
        
        for ptkb in ranked_ptkbs:
            ptkb_provenance_objs.append({
                "id": str(PTKBs.index(ptkb[0])),
                "text":ptkb[0],
                "score":ptkb[1]

            })

        passage_provenance_objs = []
        combined_passage_summaries = ""
        passages = utils.get_passages.getPassagesFromSearchQuery(preliminary_response,100)
        passages = trim_passages(passages, preliminary_response, obj["resolved_utterance"])
        combined_passage_summaries = ""
        if len(passages) == 0: # BODGE
            keywords = extract_keywords(text=preliminary_response)
            for keyword in keywords:
                a = utils.get_passages.getPassagesFromSearchQuery(keyword[0], 15,True,True)
                a = trim_passages(a,preliminary_response, obj["resolved_utterance"])
                passages += a

        for passage in passages:
            summary = summarize_with_fastchat(json.loads(passage.raw)['contents'],prompt)
            combined_passage_summaries += summary
            passage_provenance_objs.append({
                "id":passage.docid,
                "text":json.loads(passage.raw)["contents"],
                "score":passage.score,
                "used":True
            })
        final_ans = llama2.answer_question_from_passage(combined_passage_summaries, prompt,topic_obj["turns"][0:turn_index])
        #final_ans = chatgpt.answer_question_from_passage(combined_passage_summaries, prompt,topic_obj["turns"][0:turn_index])
        print(f"Final Answer: {final_ans}")
        turn_outputs.append({
            "turn_id":f"{topic_obj['number']}_{obj['turn_id']}",
            "responses": [
                {
                    "rank":1,
                    "user_utterance": obj["resolved_utterance"],
                    "generated_prompt": prompt,
                    "text":final_ans,
                    "preliminary_response": preliminary_response,
                    "combined_passage_summaries":combined_passage_summaries,
                    "ptkb_provenance":ptkb_provenance_objs,
                    "passage_provenance":passage_provenance_objs
                }

            ]
        })
        turn_index += 1
        runtime_min = math.floor((time.time()-start_time)/60)
        print(f"STATUS UPDATE: FINISHED TURN {turn_index}/{len(topic_obj['turns'])} - TOPIC {topic_obj['number']} @ {runtime_min}min elapsed - {total_turns}/332 DONE (EST. {math.floor((runtime_min/(total_turns + .00001)) * (332-total_turns))}min remaining)")
    
    return turn_outputs
        


if __name__ == '__main__':
    with open('./data/2023_test_topics.json', 'r') as f: 
        data = json.load(f)
        index = -1
        output = {
            "run_name":"georgetown_infosense_run",
            "run_type": "automatic",
            "internal_id":"3 Passages, No Score Threshold, Llama2 13B GPU CAPABLE, .25 PTKB Threshold, NOT Using Kaggle Articles in passage classification, One Shot Approach w/ChatGPT Relevance Verification, YAKE Keyword Extraction Fallback, Checks for question type and for passage user utterance relevance",
            "turns" : []
        }
        #for o in data:
            #output['turns'] += run(o)
        output = run(data[index])
        filename = f"AUG24_RUN_1.json"
        with open(f"./output/{filename}", 'a') as f2:
            f2.write(json.dumps(output))

        
