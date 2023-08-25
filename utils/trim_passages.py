# import utils.llama2 as llama2
import utils.chatgpt as chatgpt
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
    a = filterOutUnreliablePassages(passages)
    return a[:3]
    """
    refined_passages = []
    for passage in passages:
        if len(refined_passages) == 3:
            return refined_passages
        if chatgpt.determine_passage_relevance(passage=passage, statement=response, userUtterance=userUtterance):
            print(passage)
            refined_passages.append(passage)
    return refined_passages[:3]
    """
