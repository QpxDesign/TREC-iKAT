from llama_cpp import Llama
import sys
llm = Llama(model_path="./models/llama-2-13b-chat.ggmlv3.q4_1.bin", n_ctx=10240, n_threads=16)
#llm = Llama(model_path="./models/llama-2-70b-chat.ggmlv3.q4_1.bin", n_ctx=10240)
from utils.rank_passage_sentances import rank
import time

def gen_response(prompt,previous_chats):
    full_prompt = ""
    for pc in previous_chats:
        full_prompt += f"Q: {pc['resolved_utterance']} A: {pc['response']}"

    full_prompt += f" Q: {prompt} A: "
    output = llm(full_prompt,max_tokens=1024, stop=["Q:"], echo=True) # 250 tokens is TREC iKAT limit
    ans = output["choices"][-1]["text"]
    ans = ans.split(" A: ")[-1]
    return ans

def gen_summary(passage,question):
    MAX_PASSAGE_LENGTH = 512 #num characters
    prompt = f"Q: Summerize this passage in 1 sentance: {rank(passage, question)[:MAX_PASSAGE_LENGTH]} A:"
    output = llm(prompt,max_tokens=250, stop=["Q:"], echo=True)
    ans = output["choices"][-1]["text"]
    ans = ans.split(" A: ")[-1]
    ans = ans.replace("Sure! Here is a summary of the passage in one sentence:",'')
    ans = ans.replace('Here is a summary of the passage in one sentence:','')
    ans = ans.replace('Here is the summary in one sentence based on the information from the passages:')
    return ans

def answer_question_from_passage(passage,question):
    summary = gen_summary(passage,question)
    prompt = f"Q: Answer this question {question} using information from these passages - {summary} A:"
    output = llm(prompt,max_tokens=250, stop=["Q:"], echo=True)
    ans = output["choices"][-1]["text"]
    ans = ans.split(" A: ")[-1]
    ans = ans.replace("Sure! Here is a summary of the passage in one sentence:",'')
    ans = ans.replace('Here is a summary of the passage in one sentence:','')
    ans = ans.replace('Here is the summary in one sentence based on the information from the passages:')
    ans = ans.replace("\n","")
    return ans