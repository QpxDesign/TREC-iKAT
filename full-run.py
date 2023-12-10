import utils.llama2 as llama2
import utils.ptkb_similarity as ptkb_similarity
import utils.gen_prompt_from_ptkbs_and_question
from utils.trim_passages import trim_passages
import utils.json_ptkb_dict_to_array
import utils.get_passages
import json
import time
import math
from utils.fastchat import summarize_with_fastchat
import utils.chatgpt as chatgpt
from utils.extract_keywords import extract_keywords
from utils.sort_passage_provenances import sort_passage_provenances
from utils.determine_output_name import determine_output_name
from utils.prevent_trail_off import prevent_trail_off
from utils.remove_extra_spaces import remove_extra_spaces

start_time = time.time()
total_turns = 0

AUTOMATIC_RUN = True


def run(topic_obj):  # outputs JSON that fufils all requirements (ranked PTKBs from )
    global total_turns
    global AUTOMATIC_RUN
    PTKBs = utils.json_ptkb_dict_to_array.format(topic_obj)
    turn_index = 0
    turn_outputs = []
    for obj in topic_obj["turns"]:
        N_SHOTS = 1
        total_turns += 1
        ranked_ptkbs = list(filter(
            lambda a: a[1] > .25, ptkb_similarity.rankPTKBS(PTKBs, obj["utterance"])))
        if not AUTOMATIC_RUN:
            used_ptkbs = obj["ptkb_provenance"]
            ranked_ptkbs = []
            for up in used_ptkbs:
                a = (PTKBs[up-1], -1)
                ranked_ptkbs.append(a)

        prompt = utils.gen_prompt_from_ptkbs_and_question.gen(
            ranked_ptkbs, obj["utterance"]) if AUTOMATIC_RUN else utils.gen_prompt_from_ptkbs_and_question.gen(
            ranked_ptkbs, obj["resolved_utterance"])
        print("prompt: " + prompt)
       # preliminary_response = llama2.gen_response(
        #  prompt, topic_obj["turns"][0:turn_index])
        ptkb_provenance_objs = []

        for ptkb in ranked_ptkbs:
            ptkb_provenance_objs.append({
                "id": str(PTKBs.index(ptkb[0])+1),
                "text": ptkb[0],
                "score": ptkb[1]

            })
        answer = prompt
        passage_provenance_objs = []
        while N_SHOTS > 0:
            combined_passage_summaries = ""
            passages = utils.get_passages.getPassagesFromSearchQuery(
                answer, 100)
            used_passages = trim_passages(
                passages, answer, obj["utterance"]) if AUTOMATIC_RUN else trim_passages(
                passages, answer, obj["resolved_utterance"])

            if len(passages) == 0:  # BODGE
                keywords = extract_keywords(text=answer)
                for keyword in keywords:
                    keyword_passages = utils.get_passages.getPassagesFromSearchQuery(
                        keyword[0], 15, True)
                    passages += keyword_passages
                    trimmed_keyword_passages = trim_passages(
                        keyword_passages, answer, obj["utterance"])
                    used_passages += trimmed_keyword_passages

            print(f"LEN OF USED PASSAGES: {len(used_passages)}")

            for passage in passages:
                passageWasUsed = False
                for used_passage in used_passages:
                    if used_passage["result"].docid == passage["result"].docid:
                        passageWasUsed = True
                        summary = summarize_with_fastchat(
                            json.loads(passage["doc"].raw())['contents'], prompt)
                        combined_passage_summaries += summary
                        break
                if N_SHOTS == 1:
                    passage_provenance_objs.append({
                        "id": passage["result"].docid,
                        "text": json.loads(passage["doc"].raw())["contents"],
                        "score": passage["result"].score + 10_000 if passageWasUsed else passage["result"].score,
                        "used": passageWasUsed
                    })
            if N_SHOTS-1 != 0:
                answer = llama2.answer_question_from_passage(
                    combined_passage_summaries, prompt, topic_obj["turns"][0:turn_index])
            N_SHOTS = N_SHOTS - 1

        # final_ans = chatgpt.answer_question_from_passage(combined_passage_summaries, prompt,topic_obj["turns"][0:turn_index])
        answer = remove_extra_spaces(
            prevent_trail_off(combined_passage_summaries))
        print(f"Final Answer: {answer}")
        turn_outputs.append({
            "turn_id": f"{topic_obj['number']}_{obj['turn_id']}",
            "responses": [
                {
                    "rank": 1,
                    "user_utterance": obj["utterance"],
                    "generated_prompt": prompt,
                    "text": answer,
                    "ptkb_provenance": ptkb_provenance_objs,
                    "passage_provenance": sort_passage_provenances(passage_provenance_objs)
                }

            ]
        })
        turn_index += 1
        runtime_min = math.floor((time.time()-start_time)/60)
        print(
            f"STATUS UPDATE: FINISHED TURN {turn_index}/{len(topic_obj['turns'])} - TOPIC {topic_obj['number']} @ {runtime_min}min elapsed - {total_turns}/332 DONE (EST. {math.floor((runtime_min/(total_turns + .00001)) * (332-total_turns))}min remaining)")

    return turn_outputs


if __name__ == '__main__':
    with open('./data/2023_test_topics.json', 'r') as f:
        data = json.load(f)
        index = 0
        output = {
            "run_name": "georgetown_infosense_run",
            "run_type": "automatic",
            "internal_id": "BM --> LLM, no LLama",
            "turns": []
        }
        for o in data:
            # output['turns'].append(run(o))
            output['turns'] += run(o)
        # output = run(data[index])
        # output['turns'] += run(data[index])
        filename = determine_output_name()
        with open(f"./output/{filename}", 'a') as f2:
            f2.write(json.dumps(output))
