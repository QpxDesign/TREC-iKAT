# import utils.llama2 as llama2
import utils.chatgpt as chatgpt
from utils.fastchat import summarize_with_fastchat
from utils.classify_passages import determinePassageReliability
import json


def filterOutUnreliablePassages(passages):
    final = []
    for passage in passages:
        predicted_label = determinePassageReliability(
            json.loads(passage.raw)["contents"])
        if predicted_label == 'reliable':
            final.append(passage)
    return final


def trim_passages(passages, response, userUtterance):
    NUMBER_OF_PASSAGES = 5
    # filtered_passages = filterOutUnreliablePassages(passages)
    return passages[:NUMBER_OF_PASSAGES]


"""    # filtered_passages = passages
    refined_passages = []

    for passage in filtered_passages:
        passage_summary = summarize_with_fastchat(
            json.loads(passage.raw)['contents'], userUtterance)
        if len(refined_passages) == NUMBER_OF_PASSAGES:
            return refined_passages
        if chatgpt.determine_passage_relevance(passage_summary=passage_summary, statement=response, userUtterance=userUtterance):
            print(passage)
            refined_passages.append({
                "docid": passage.docid,
                "summary": passage_summary
            })
    return refined_passages[:NUMBER_OF_PASSAGES]"""
