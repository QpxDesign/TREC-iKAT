from llama_cpp import Llama
import sys
#llm = Llama(model_path="./models/llama-2-13b-chat.ggmlv3.q4_1.bin", n_ctx=10240, n_threads=16)
llm = Llama(model_path="./models/llama-2-70b-chat.ggmlv3.q4_1.bin")
from utils.rank_passage_sentances import rank
from utils.prevent_trail_off import prevent_trail_off
import time
import re

def gen_response(prompt,previous_chats):
    full_prompt = ""
    for pc in previous_chats:
        full_prompt += f"Q: {pc['resolved_utterance']} A: {pc['response']}"

    full_prompt += f" Q: {prompt} A: "
    print(f'generating response: {full_prompt}')
    output = llm(full_prompt,max_tokens=256, stop=["Q:"], echo=True) 
    print(output)
    ans = output["choices"][-1]["text"]
    ans = ans.split(" A: ")[-1]
    return ans

def gen_summary(passage,question):
    MAX_PASSAGE_LENGTH = 512 #num characters
    prompt = f"Q: Summarize this passage in 1 sentence: {rank(passage, question)[:MAX_PASSAGE_LENGTH]} A: "
    output = llm(prompt,max_tokens=250, stop=["Q:"], echo=True)
    ans = output["choices"][-1]["text"]
    ans = ans.split(" A: ")[-1]
    ans = ans.replace("Sure! Here is a summary of the passage in one sentence:",'')
    ans = ans.replace('Here is a summary of the passage in one sentence:','')
    ans = ans.replace('Here is the summary in one sentence based on the information from the passages:','')
    return ans

def answer_question_from_passage(passage,question,previous_chats):
    prompt = f"Answer this question - {question} - using this information - {remove_extra_spaces(passage)} and your own knowledge."
    full_prompt = ""
    for pc in previous_chats:
        full_prompt += f"Q: {pc['resolved_utterance']} A: {pc['response']}"

    full_prompt += f" Q: {prompt} A: "
    output = llm(full_prompt,max_tokens=250, stop=["Q:"], echo=True)
    ans = output["choices"][-1]["text"]
    ans = ans.split(" A: ")[-1]
    ans = ans.replace("Sure! Here is a summary of the passage in one sentence:",'')
    ans = ans.replace('Here is a summary of the passage in one sentence:','')
    ans = ans.replace('Here is the summary in one sentence based on the information from the passages:','')
    ans = ans.replace('Sure! Based on the information provided, here is a summary of the passage in one sentence:','')
    ans = ans.replace("\n","")
    return prevent_trail_off(ans)

def remove_extra_spaces(input_string):
    cleaned_string = re.sub(r'\s+', ' ', input_string)
    cleaned_string = cleaned_string.replace("\n","")
    return cleaned_string

