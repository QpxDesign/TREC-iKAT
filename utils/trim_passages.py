#import utils.llama2 as llama2
import utils.chatgpt as chatgpt

def trim_passages(passages,response, userUtterance):
    refined_passages = []
    for passage in passages:
        if len(refined_passages) == 3:
            return refined_passages
        if chatgpt.determine_passage_relevance(passage=passage, statement=response, userUtterance=userUtterance):
            print(passage)
            refined_passages.append(passage)
    return refined_passages[:3]
    """cutoffValue = 10 # TO-DO: FIND GOOD CUT OFF VALUE
    for i in range(len(passages)):
        if passages[i].score < cutoffValue:
            return passages[0:i]
    return passages
"""